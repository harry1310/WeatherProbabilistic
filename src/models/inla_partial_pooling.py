"""INLA backend for Phase 5a's partial-pooled hierarchical logreg.

Drop-in replacement for src.models.phase2_partial_pooling.fit_partial_pooling.
Returns the same PartialPoolingFit shape so extend_5a + predict_5a
consume the result unchanged.

Engine swap rationale (2026-05-11): PyMC + blackjax NUTS on 102k rows
takes ~3h locally / 4h+ CI. INLA fits the random-intercept-only
variant in ~11s with identical Brier and CI widths within 10% of
PyMC's. Random-slopes INLA takes ~80s for zero additional skill.
Smoke comparisons in scripts/smoke_inla_5a.py + memo
project_inla_5a_smoke_2026-05-11.md.

Architecture:
  - R-INLA does the fit + posterior sampling via subprocess. Python
    builds the data inputs, writes a self-contained R script to a
    temp file, runs Rscript, reads the per-draw samples back as a
    CSV.
  - The resulting samples are wrapped in an arviz InferenceData with
    the SAME variable shapes the deployed predict_partial_pooling
    code expects:
      posterior.intercept_s  shape (chain=1, draw=N, station=S)
      posterior.beta_s       shape (chain=1, draw=N, station=S, feature=F)
    For the random-intercept-only model, beta_s is broadcast across
    the station dimension (all stations share the same fixed
    coefficient draws). intercept_s is the per-station partially-
    pooled intercept = fixed_intercept + station_random_effect.
  - N (number of posterior draws) is a CLI parameter; default 1000.
    PyMC defaults to 4 chains × 2000 draws = 8000 samples, but for
    fitted-value posterior summaries (which is all extend_5a does)
    ~1000 samples is overkill already.

Requirements:
  - R 4.x with the INLA package installed from r-inla.org/R/stable.
  - Local: install via R console (see project_inla_5a_smoke memo).
  - CI: install step in retrain-python.yml gated on run_5a.
"""
from __future__ import annotations

import io
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# arviz + xarray for the InferenceData wrapping. Kept as imports rather
# than inlined `import` inside the function because predict_5a et al
# already require arviz at module-import time.
import arviz as az
import xarray as xr


@dataclass
class PartialPoolingFit:
    """Mirror of src.models.phase2_partial_pooling.PartialPoolingFit so
    extend_5a + predict_5a can consume InteractiveData unchanged."""
    idata: object
    feature_names: list[str]
    station_codes: list[str]


def _default_r_exe() -> str:
    """Locate Rscript: PATH first (works on both Linux CI runners and
    Windows dev boxes), then platform-specific fallback.

    Pre-2026-05-11 this was a hard-coded `C:\\Program Files\\R\\R-4.6.0\\
    bin\\x64\\R.exe` Windows path — fine for local dev, broke instantly
    in CI as soon as run_phase5_bayesian.py reached the subprocess call
    (run 25689518652, FileNotFoundError on the Windows path). Override
    via ``INLA_R_EXE`` env var for either platform."""
    found = shutil.which("Rscript")
    if found:
        return found
    if platform.system() == "Windows":
        return r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe"
    return "/usr/bin/Rscript"


def _default_r_libs_user() -> str:
    """Default R user library path. Linux CI sets R_LIBS_USER via the
    workflow env; this fallback only fires for local dev or unusual
    setups where the env var isn't set."""
    if platform.system() == "Windows":
        return r"C:\Users\rhcsl\R\win-library\4.6"
    return os.path.expanduser("~/R/x86_64-pc-linux-gnu-library/4.4")


# R script that fits INLA, samples N posterior draws, and dumps them as
# a CSV with columns: draw, fixed_intercept, x1..xN, station_intercept_1..S.
# Each row = one joint posterior sample. inla.posterior.sample returns
# joint draws (not marginals), so the across-column correlations within
# a row are preserved — matters for predict_partial_pooling's logit
# computation per row.
_R_SCRIPT = r"""
suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(INLA)
})

train_csv   <- "%TRAIN_CSV%"
samples_csv <- "%SAMPLES_CSV%"
n_features  <- %N_FEATURES%
n_stations  <- %N_STATIONS%
n_draws     <- %N_DRAWS%

df <- read.csv(train_csv)

# Random-intercepts + random-slopes hierarchical model. The 10-feat
# smoke 2026-05-11 showed random-slopes adds ~70s of compute for zero
# Brier gain over random-intercept-only, but the user opted in for
# random slopes as the production setup ahead of adding more features
# (full 20-feature 3a parity from 2026-05-11). With more features
# random slopes start mattering — each feature gets its own per-station
# partial-pool σ, so a feature like cape_mean can have a station-
# specific slope without imposing it on every other coefficient.
# f(slope_idx_k, x_k, model='iid') = random slope for feature k keyed
# by station_idx (slope_idx_k columns are duplicate-name aliases of
# station_idx — INLA requires distinct index names per random-effect
# block but they all map to the same station factor).
feat_terms <- paste(paste0("x", 1:n_features), collapse = " + ")
slope_terms <- paste0(
  "f(slope_idx_", 1:n_features, ", x", 1:n_features, ", model = 'iid')",
  collapse = " + "
)
formula_str <- paste0(
  "y ~ ", feat_terms, " + f(station_idx, model = 'iid') + ", slope_terms
)
cat("Formula:", formula_str, "\n")

t0 <- Sys.time()
fit <- inla(
  formula = as.formula(formula_str),
  family  = "binomial",
  data    = df,
  control.compute = list(config = TRUE),
  verbose = FALSE
)
cat(sprintf("INLA fit wall: %.1fs\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

# Joint posterior samples. config=TRUE in control.compute is required
# for inla.posterior.sample. Returns a list of length n_draws; each
# element has $latent (named vector of all latent values for one
# sample). Names look like:
#   (Intercept):1, x1:1, ..., xN:1                      (fixed effects)
#   station_idx:1, station_idx:2, ..., station_idx:S    (random intercepts per station)
#   slope_idx_k:1, slope_idx_k:2, ..., slope_idx_k:S    (random slope DEVIATIONS, one block per feature k)
# Each ":N" suffix means "for unique index N" — fixed effects always :1,
# random effects use the station_idx values 1..S.
t1 <- Sys.time()
samples <- inla.posterior.sample(n = n_draws, fit)
cat(sprintf("Sampled %d joint posterior draws in %.1fs\n",
            n_draws, as.numeric(difftime(Sys.time(), t1, units = "secs"))))

# Per-draw row in `flat`:
#   fixed_intercept                                     (1 col)
#   x1..xN  (fixed-effect coefficients, population-mean) (n_features cols)
#   station_intercept_1..S                              (n_stations cols)
#   station_slope_k_s  (deviation from x_k for station s) (n_features × n_stations cols)
intercept_row_name  <- "(Intercept):1"
beta_row_names      <- paste0("x", 1:n_features, ":1")
station_row_names   <- paste0("station_idx:", 1:n_stations)
slope_row_names <- lapply(1:n_features, function(k)
  paste0("slope_idx_", k, ":", 1:n_stations))

# Sanity-check expected names exist before we start slicing.
expected <- c(intercept_row_name, beta_row_names, station_row_names,
              unlist(slope_row_names))
missing  <- setdiff(expected, rownames(samples[[1]]$latent))
if (length(missing) > 0) {
  stop("inla.posterior.sample latent missing expected names: ",
       paste(missing, collapse = ", "))
}

n_slope_cols <- n_features * n_stations
flat <- matrix(NA_real_, nrow = n_draws,
               ncol = 1 + n_features + n_stations + n_slope_cols)
slope_col_names <- as.vector(outer(
  paste0("station_slope_"), 1:n_features,
  function(a, b) paste0(a, b, "_")))
# Flatten the (feature, station) grid into "station_slope_k_s" labels.
slope_col_names <- unlist(lapply(1:n_features, function(k)
  paste0("station_slope_", k, "_", 1:n_stations)))
colnames(flat) <- c("fixed_intercept",
                    paste0("x", 1:n_features),
                    paste0("station_intercept_", 1:n_stations),
                    slope_col_names)

for (i in seq_len(n_draws)) {
  lat <- samples[[i]]$latent
  flat[i, "fixed_intercept"]                              <- lat[intercept_row_name, 1]
  flat[i, paste0("x", 1:n_features)]                      <- lat[beta_row_names, 1]
  flat[i, paste0("station_intercept_", 1:n_stations)]     <- lat[station_row_names, 1]
  for (k in 1:n_features) {
    flat[i, paste0("station_slope_", k, "_", 1:n_stations)] <- lat[slope_row_names[[k]], 1]
  }
}

write.csv(flat, samples_csv, row.names = FALSE)
cat("Wrote samples →", samples_csv, "\n")
"""


def fit_partial_pooling(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    station_idx_train: np.ndarray,
    station_codes: list[str],
    feature_names: list[str],
    *,
    # Kept as kwargs even though INLA doesn't use them, so the
    # signature matches phase2_partial_pooling.fit_partial_pooling
    # for drop-in replacement. Caller may pass them but they're
    # ignored for the INLA backend.
    draws: int = 1000,
    tune: int = None,             # noqa: ARG001 — kept for sig compat
    chains: int = None,           # noqa: ARG001 — kept for sig compat
    target_accept: float = None,  # noqa: ARG001 — kept for sig compat
    random_seed: int = None,      # noqa: ARG001 — kept for sig compat
    progressbar: bool = False,    # noqa: ARG001 — kept for sig compat
    chain_method: str = None,     # noqa: ARG001 — kept for sig compat
) -> PartialPoolingFit:
    """INLA backend. `draws` is the number of joint posterior samples
    drawn from INLA's approximated posterior — defaults to 1000, which
    is well above what predict_partial_pooling needs (CI band quantiles
    converge well below that)."""
    r_exe       = os.environ.get("INLA_R_EXE") or _default_r_exe()
    r_libs_user = os.environ.get("R_LIBS_USER") or _default_r_libs_user()

    n_features = X_train_s.shape[1]
    n_stations = len(station_codes)

    feature_cols = [f"x{i+1}" for i in range(n_features)]
    train_df = pd.DataFrame(X_train_s, columns=feature_cols)
    # 1-based station_idx for R (R is 1-indexed for factor levels).
    train_df["station_idx"] = np.asarray(station_idx_train, dtype=int) + 1
    # slope_idx_k columns are aliases of station_idx — INLA's
    # f(slope_idx_k, x_k, model='iid') needs a distinct index column
    # name per random-slope block, even though they all map to the
    # same station factor.
    for k in range(1, n_features + 1):
        train_df[f"slope_idx_{k}"] = train_df["station_idx"]
    train_df["y"] = np.asarray(y_train, dtype=int)

    with tempfile.TemporaryDirectory(prefix="inla_5a_") as tmpdir:
        train_csv   = Path(tmpdir) / "train.csv"
        samples_csv = Path(tmpdir) / "samples.csv"
        r_script    = Path(tmpdir) / "fit_inla.R"

        train_df.to_csv(train_csv, index=False)
        r_text = (_R_SCRIPT
                  .replace("%TRAIN_CSV%",   train_csv.as_posix())
                  .replace("%SAMPLES_CSV%", samples_csv.as_posix())
                  .replace("%N_FEATURES%",  str(n_features))
                  .replace("%N_STATIONS%",  str(n_stations))
                  .replace("%N_DRAWS%",     str(draws)))
        r_script.write_text(r_text, encoding="utf-8")

        env = os.environ.copy()
        env["R_LIBS_USER"] = r_libs_user

        print(f"[{time.strftime('%H:%M:%S')}] INLA: calling R subprocess "
              f"(n_train={len(train_df):,}, n_features={n_features}, "
              f"n_stations={n_stations}, n_draws={draws})", flush=True)
        t0 = time.time()
        # Rscript handles non-interactive mode by default (no banner, no
        # prompt) — invocation collapses to `Rscript fit_inla.R`. If the
        # resolved r_exe is `R` rather than `Rscript`, fall back to the
        # legacy --no-save --slave -f form so the call still works.
        r_name = Path(r_exe).name.lower()
        if r_name.startswith("rscript"):
            r_argv = [r_exe, str(r_script)]
        else:
            r_argv = [r_exe, "--no-save", "--slave", "-f", str(r_script)]
        result = subprocess.run(
            r_argv,
            capture_output=True, text=True, env=env,
        )
        wall = time.time() - t0
        # Pass R stdout through — it's the only progress signal.
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"  [R] {line}", flush=True)
        if result.returncode != 0:
            print(f"\n[INLA] R subprocess failed (exit {result.returncode})")
            if result.stderr.strip():
                print(f"  stderr: {result.stderr}")
            raise RuntimeError(
                f"INLA subprocess failed (exit {result.returncode}). "
                f"See stdout/stderr above for the R-side traceback.")
        print(f"[{time.strftime('%H:%M:%S')}] INLA subprocess wall: {wall:.1f}s", flush=True)

        # Read samples + reshape into (chain=1, draw, station, feature)
        # arrays so the resulting InferenceData has the same variable
        # shapes PyMC's posterior had.
        samples = pd.read_csv(samples_csv)
        if len(samples) != draws:
            raise RuntimeError(
                f"INLA returned {len(samples)} samples but {draws} were "
                f"requested. Posterior sampling truncated unexpectedly.")

        fixed_intercept = samples["fixed_intercept"].values             # (draw,)
        fixed_beta      = samples[feature_cols].values                  # (draw, feature)
        station_random  = samples[[f"station_intercept_{s+1}" for s in range(n_stations)]].values
                                                                        # (draw, station)
        # Random-slope deviations: column names are "station_slope_k_s"
        # for feature k (1..F) and station s (1..S). Reshape into
        # (draw, station, feature) so it broadcasts cleanly with
        # fixed_beta below.
        slope_random = np.empty((draws, n_stations, n_features), dtype="float64")
        for k in range(1, n_features + 1):
            for s in range(1, n_stations + 1):
                slope_random[:, s - 1, k - 1] = samples[f"station_slope_{k}_{s}"].values

    # Construct intercept_s and beta_s with the shapes predict_partial_pooling
    # expects: (chain=1, draw, station) and (chain=1, draw, station, feature).
    # Random intercepts + random slopes:
    #   intercept_s[d, s] = fixed_intercept[d] + station_random[d, s]
    #   beta_s[d, s, k]   = fixed_beta[d, k] + slope_random[d, s, k]
    intercept_s = (fixed_intercept[:, None] + station_random)           # (draw, station)
    intercept_s = intercept_s[None, :, :]                                # (chain=1, draw, station)
    beta_s = (fixed_beta[:, None, :] + slope_random)                    # (draw, station, feature)
    beta_s = beta_s[None, :, :, :]                                       # (chain=1, draw, station, feature)

    posterior = xr.Dataset(
        {
            "intercept_s": (("chain", "draw", "station"), intercept_s),
            "beta_s":      (("chain", "draw", "station", "feature"), beta_s),
        },
        coords={
            "chain":   [0],
            "draw":    np.arange(draws),
            "station": station_codes,
            "feature": feature_names,
        },
    )
    idata = az.InferenceData(posterior=posterior)
    return PartialPoolingFit(
        idata=idata,
        feature_names=list(feature_names),
        station_codes=list(station_codes),
    )


def predict_partial_pooling(
    fit: PartialPoolingFit,
    X_test_s: np.ndarray,
    station_idx_test: np.ndarray,
) -> np.ndarray:
    """Mirror of phase2_partial_pooling.predict_partial_pooling — kept
    here so callers can import either backend transparently. The math
    is identical (the InferenceData shape is what was carefully
    arranged in fit_partial_pooling)."""
    intercept_s = fit.idata.posterior["intercept_s"].values
    beta_s = fit.idata.posterior["beta_s"].values

    out = np.empty(len(X_test_s), dtype="float64")
    for s_idx in range(intercept_s.shape[-1]):
        mask = station_idx_test == s_idx
        if not mask.any():
            continue
        ints = intercept_s[..., s_idx]
        bets = beta_s[..., s_idx, :]
        logit = ints[..., None] + np.einsum("nf,cdf->cdn", X_test_s[mask], bets)
        out[mask] = (1.0 / (1.0 + np.exp(-logit))).mean(axis=(0, 1))
    return out
