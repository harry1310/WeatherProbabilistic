"""INLA 5a smoke test — uncommitted (per overnight-plan).

Asks: can INLA (R-INLA) fit a hierarchical logistic regression on 5a's
production dataset (102k rows × 10 features × 3-station random
intercept) faster than the current PyMC + blackjax stack while
preserving competitive Brier?

CLI:
    smoke_inla_5a.py [--random-slopes]

--random-slopes adds `f(slope_idx_k, x_k, model='iid')` per feature on
top of the random intercept — a direct mirror of the deployed PyMC 5a
hierarchy. Without the flag, the smoke runs random-intercept-only
(the simpler model that produced the -8.6% Brier win in the first
smoke run 2026-05-11).

Status going into this: PyMC + blackjax NUTS on JAX-pmap takes ~3h
locally (4-vCPU with real physical-core parallelism), 4h+ on CI
where JAX-pmap doesn't deliver real OS-level parallelism. The CI
multiprocess fallback (4 chains × shell `&` workers) hit a
hyperthreading ceiling at 3.23× single-chain wall (smoke run
2026-05-10). Inference-side approximation via INLA would side-step
the whole MCMC parallelism question if it converges fast enough
and matches PyMC's posterior shape closely enough.

This smoke does the MINIMAL viable comparison:
  - Random intercept per station, FIXED (non-pooled) slopes.
    A simplification vs 5a's full random-slopes hierarchical model,
    but the right starting point for a timing + feasibility check.
  - Spread-features variant (10 features) to match what 5a ships
    with in production after the 2026-05-10 promotion.
  - Same prepare_phase3_dataset call as run_phase5_bayesian.py so
    the inputs are byte-identical.

NOT committed — overnight uncommitted exploration. If INLA looks
promising the productionised version replaces run_phase5_bayesian's
sampler. If it doesn't, this script captures the negative result
for the memory log.

Outputs printed to stdout:
  - Fit wall time (s)
  - Per-row test predictions → Brier
  - Reference PyMC 5a Brier from the most-recent live bundle (if
    available on disk) for side-by-side
  - Posterior summary (station intercepts ± their 95% CI widths)

Usage: python scripts/smoke_inla_5a.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import MODELS_NO_UKMO, prepare_phase3_dataset  # noqa: E402


R_EXE = r"C:\Program Files\R\R-4.6.0\bin\x64\R.exe"
R_LIBS_USER = r"C:\Users\rhcsl\R\win-library\4.6"


# R-side INLA fit, written to a temp .R file and invoked via Rscript.
# Reads:
#   - csv with columns [y, station_idx, slope_idx_1..N (= station_idx duplicates), x1..xN]
#   - csv with test rows (same columns, no y is OK — we set NA in R)
# Writes:
#   - csv of per-test-row p_wet predictions WITH credible-interval columns
#     (mean + sd + q05, q10, q50, q90, q95) so the smoke validates CI
#     extraction end-to-end, not just trusts the docs.
#   - a json summary file with wall time + posterior summaries
R_SCRIPT_TEMPLATE = r"""
suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(INLA)
})

train_csv       <- "%TRAIN_CSV%"
test_csv        <- "%TEST_CSV%"
n_features      <- %N_FEATURES%
n_stations      <- %N_STATIONS%
random_slopes   <- as.logical("%RANDOM_SLOPES%")
predictions_csv <- "%PREDICTIONS_CSV%"
summary_json    <- "%SUMMARY_JSON%"

train_df <- read.csv(train_csv)
test_df  <- read.csv(test_csv)

# Formula construction:
#   - Random intercept per station: f(station_idx, model='iid')
#   - Random slopes (if enabled): f(slope_idx_k, x_k, model='iid') for each k.
#     slope_idx_k columns are duplicates of station_idx; INLA requires
#     distinct index names per random-effect block.
feat_terms <- paste(paste0("x", 1:n_features), collapse = " + ")
re_terms <- "f(station_idx, model = 'iid')"
if (random_slopes) {
  slope_terms <- paste0(
    "f(slope_idx_", 1:n_features, ", x", 1:n_features, ", model = 'iid')",
    collapse = " + "
  )
  re_terms <- paste(re_terms, "+", slope_terms)
}
formula_str <- paste0("y ~ ", feat_terms, " + ", re_terms)
cat("Formula:", formula_str, "\n")

# Append test rows with y = NA so INLA computes predictive means.
# control.predictor: link = 1 means apply the binomial logit inverse-
# link to get probabilities back on the [0,1] scale. compute = TRUE
# triggers per-row fitted-value posteriors.
test_df_predict <- test_df
test_df_predict$y <- NA
combined <- rbind(train_df, test_df_predict)
n_train  <- nrow(train_df)
n_test   <- nrow(test_df)

# Custom quantile set matches what 5a writes today
# (q05, q10, q50, q90, q95 — predict_5a + the site read these directly).
# Quantiles is a TOP-LEVEL inla() argument; in recent versions putting
# it inside control.compute prints a warning + ignores it (verified
# 2026-05-11 with INLA 25.10.19).
quantiles_to_compute <- c(0.05, 0.10, 0.50, 0.90, 0.95)

t0 <- Sys.time()
fit <- inla(
  formula = as.formula(formula_str),
  family  = "binomial",
  data    = combined,
  control.predictor = list(link = 1, compute = TRUE),
  control.compute   = list(config = TRUE),
  quantiles         = quantiles_to_compute,
  verbose = FALSE
)
t1 <- Sys.time()
wall_s <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("INLA fit wall: %.1fs\n", wall_s))

# Per-test-row posteriors — mean + sd + 5 quantiles direct from INLA.
# Column names depend on the quantiles argument: 0.05 → "0.05quant",
# 0.1 → "0.1quant", etc. (INLA trims trailing zeros after the leading
# decimal: 0.10 → "0.1quant" not "0.10quant"; 0.5 stays "0.5quant".)
fv <- fit$summary.fitted.values
test_slice <- (n_train + 1):(n_train + n_test)
cat("fitted.values columns:", paste(colnames(fv), collapse = ", "), "\n")
pred_out <- data.frame(
  p_wet  = fv$mean[test_slice],
  p_sd   = fv$sd[test_slice],
  q05    = fv[test_slice, "0.05quant"],
  q10    = fv[test_slice, "0.1quant"],
  q50    = fv[test_slice, "0.5quant"],
  q90    = fv[test_slice, "0.9quant"],
  q95    = fv[test_slice, "0.95quant"]
)
write.csv(pred_out, predictions_csv, row.names = FALSE)

# Posterior summary for the diagnostics: fixed effect coefficients +
# station-intercept random effect 95% CIs.
station_summary <- fit$summary.random$station_idx
station_ci_widths <- station_summary[, "0.95quant"] - station_summary[, "0.05quant"]

# Random-slope hyperprior precisions (one per feature when --random-slopes
# is on) tell us how much per-station variation each feature's slope shows.
# Higher precision = tighter shrinkage to the population mean (less
# per-station differentiation). Useful for "is this feature contributing
# to the partial-pooling story?" diagnostics.
hyper_summary <- if (random_slopes) {
  list(
    intercept = as.list(fit$summary.hyperpar["Precision for station_idx", "mean"]),
    slopes    = as.list(setNames(
      sapply(1:n_features, function(k)
        fit$summary.hyperpar[paste0("Precision for slope_idx_", k), "mean"]),
      paste0("x", 1:n_features)
    ))
  )
} else {
  list(intercept = as.list(fit$summary.hyperpar["Precision for station_idx", "mean"]))
}

summary <- list(
  wall_s = wall_s,
  formula = formula_str,
  random_slopes = random_slopes,
  n_train = n_train,
  n_test  = n_test,
  n_features = n_features,
  n_stations = n_stations,
  fixed_effects = as.list(fit$summary.fixed$mean),
  station_intercepts_mean = as.list(station_summary[, "mean"]),
  station_intercepts_ci_widths = as.list(station_ci_widths),
  hyperpriors = hyper_summary
)
writeLines(jsonlite::toJSON(summary, auto_unbox = TRUE, pretty = TRUE),
           summary_json)
cat("Wrote predictions →", predictions_csv, "\n")
cat("Wrote summary →", summary_json, "\n")
"""


def _brier(p, y):
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


def _read_pymc_baseline_brier() -> float | None:
    """Grab the most-recent PyMC 5a test Brier from the spread-features
    pilot output if it's on disk, for a side-by-side reference. Returns
    None if not available — script still reports INLA's number alone.
    """
    candidates = [
        ROOT / "reports" / "phase5a_artefacts_spread" / "predictions" / "test_predictions.parquet",
        ROOT / "reports" / "phase5a_artefacts" / "predictions" / "test_predictions.parquet",
    ]
    for c in candidates:
        if c.exists():
            try:
                df = pd.read_parquet(c)
                if "p_wet" in df.columns and "observed_wet" in df.columns:
                    b = _brier(df["p_wet"].values, df["observed_wet"].values)
                    print(f"  PyMC 5a reference (from {c.name}): n={len(df):,}, Brier={b:.4f}")
                    return b
            except Exception as e:
                print(f"  could not read {c}: {e}")
    print("  No PyMC 5a reference on disk; INLA-only result reported.")
    return None


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--random-slopes", action="store_true",
                   help="Add per-feature random slopes per station — direct mirror "
                        "of the deployed PyMC 5a hierarchy. Default is "
                        "random-intercept-only (the simpler model that produced "
                        "the -8.6%% Brier win in the first smoke).")
    args = p.parse_args()
    mode = "random slopes + intercepts" if args.random_slopes else "random intercept only"
    print(f"[{time.strftime('%H:%M:%S')}] INLA 5a smoke — 10-feat spread variant, {mode}")

    t0 = time.time()
    ds = prepare_phase3_dataset(
        models=MODELS_NO_UKMO,
        lead_as_feature=True,
        add_spread_features=True,
        verbose=False,
    )
    print(f"  dataset prep: {time.time() - t0:.1f}s")
    print(f"  train rows: {len(ds.X_train):,}  test rows: {len(ds.X_test):,}")
    print(f"  features ({len(ds.feature_names)}): {ds.feature_names}")
    print(f"  stations: {ds.station_codes}")

    n_features = ds.X_train_s.shape[1]
    feature_cols = [f"x{i+1}" for i in range(n_features)]

    train_df = pd.DataFrame(ds.X_train_s, columns=feature_cols)
    train_df["station_idx"] = np.asarray(ds.station_idx_train) + 1   # 1-based for R
    # slope_idx_k columns are exact duplicates of station_idx — INLA
    # requires distinct index names per random-effect block, but they
    # all key on the same station grouping.
    if args.random_slopes:
        for k in range(1, n_features + 1):
            train_df[f"slope_idx_{k}"] = train_df["station_idx"]
    train_df["y"] = ds.y_train.values.astype(int) if hasattr(ds.y_train, "values") else np.asarray(ds.y_train, dtype=int)

    test_df = pd.DataFrame(ds.X_test_s, columns=feature_cols)
    test_df["station_idx"] = np.asarray(ds.station_idx_test) + 1
    if args.random_slopes:
        for k in range(1, n_features + 1):
            test_df[f"slope_idx_{k}"] = test_df["station_idx"]
    test_df["y"] = ds.y_test.values.astype(int) if hasattr(ds.y_test, "values") else np.asarray(ds.y_test, dtype=int)

    with tempfile.TemporaryDirectory(prefix="inla_smoke_") as tmpdir:
        train_csv = Path(tmpdir) / "train.csv"
        test_csv  = Path(tmpdir) / "test.csv"
        pred_csv  = Path(tmpdir) / "predictions.csv"
        summary_json = Path(tmpdir) / "summary.json"
        r_script_path = Path(tmpdir) / "fit_inla.R"

        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        r_script = (R_SCRIPT_TEMPLATE
                    .replace("%TRAIN_CSV%", train_csv.as_posix())
                    .replace("%TEST_CSV%", test_csv.as_posix())
                    .replace("%PREDICTIONS_CSV%", pred_csv.as_posix())
                    .replace("%SUMMARY_JSON%", summary_json.as_posix())
                    .replace("%N_FEATURES%", str(n_features))
                    .replace("%N_STATIONS%", str(len(ds.station_codes)))
                    .replace("%RANDOM_SLOPES%", "TRUE" if args.random_slopes else "FALSE"))
        r_script_path.write_text(r_script, encoding="utf-8")

        # Make sure jsonlite is available — it ships with most R installs
        # but not always. cheap check before running the main script.
        env = os.environ.copy()
        env["R_LIBS_USER"] = R_LIBS_USER

        print(f"\n[{time.strftime('%H:%M:%S')}] Calling R-INLA...")
        t1 = time.time()
        result = subprocess.run(
            [R_EXE, "--no-save", "--slave", "-f", str(r_script_path)],
            capture_output=True, text=True, env=env,
        )
        r_wall = time.time() - t1
        print(result.stdout, end="")
        if result.returncode != 0:
            print(f"\n!!! R script failed (exit {result.returncode}) !!!")
            print(result.stderr)
            sys.exit(1)
        if result.stderr.strip():
            print(f"  stderr: {result.stderr[:400]}")
        print(f"\n[{time.strftime('%H:%M:%S')}] R subprocess wall (incl. startup): {r_wall:.1f}s")

        # Read predictions + compute Brier
        pred_df = pd.read_csv(pred_csv)
        p_inla = pred_df["p_wet"].values
        y_test = test_df["y"].values
        brier_inla = _brier(p_inla, y_test)
        print(f"\nINLA test Brier (all rows aggregated): {brier_inla:.4f} (n_test={len(y_test):,})")

        # CI feasibility check: verify the quantile columns came back +
        # report mean CI widths so we can sanity-check that the per-row
        # bands aren't all 0 (which would mean we'd somehow lost the
        # uncertainty signal).
        if all(c in pred_df.columns for c in ["q05", "q10", "q50", "q90", "q95"]):
            mean_80_width = float((pred_df["q90"] - pred_df["q10"]).mean())
            mean_90_width = float((pred_df["q95"] - pred_df["q05"]).mean())
            print(f"\nPer-row CI feasibility (out of {len(pred_df):,} test rows):")
            print(f"  Mean 80% width (q90-q10): {mean_80_width:.4f}")
            print(f"  Mean 90% width (q95-q05): {mean_90_width:.4f}")
            print(f"  Min/Max p_wet posterior SD: {pred_df['p_sd'].min():.4f} / {pred_df['p_sd'].max():.4f}")
            # Show a few sample rows so we can eyeball the shape
            print(f"\n  Sample test rows:")
            print(pred_df.head(5).to_string(float_format=lambda x: f"{x:.4f}"))
        else:
            print(f"\n!!! CI columns missing from predictions parquet — got: {list(pred_df.columns)}")

        # Per-station + per-lead breakdown — useful to see whether the
        # simplified random-intercept model already captures the
        # production differentiation 5a achieves with random slopes.
        df = pd.DataFrame({
            "p_wet":        p_inla,
            "observed_wet": y_test,
            "station_idx":  test_df["station_idx"].values,
        })
        # Station back-mapping
        for i, code in enumerate(ds.station_codes):
            sub = df[df["station_idx"] == i + 1]
            if len(sub) == 0:
                continue
            b = _brier(sub["p_wet"].values, sub["observed_wet"].values)
            wet_rate = float(sub["observed_wet"].mean())
            print(f"  {code:<28}  n={len(sub):>6,}  Brier={b:.4f}  wet={wet_rate:.2%}")

        # PyMC reference (if disk has it)
        print()
        print("Reference (existing PyMC 5a on disk):")
        baseline = _read_pymc_baseline_brier()
        if baseline is not None:
            delta_pct = 100.0 * (brier_inla - baseline) / baseline
            print(f"\n  INLA vs PyMC 5a: {delta_pct:+.1f}% Brier  "
                  f"({'INLA wins' if delta_pct < 0 else 'PyMC wins'})")

        # Summary diagnostics
        if summary_json.exists():
            print("\nINLA posterior summary:")
            summary = json.loads(summary_json.read_text())
            print(f"  Fit wall (R-INLA inla() call): {summary['wall_s']:.1f}s")
            print(f"  Station-intercept means: {summary['station_intercepts_mean']}")
            print(f"  Station-intercept 95% CI widths: {summary['station_intercepts_ci_widths']}")
            # Hyperprior precision for the station random-effect:
            # higher precision = tighter cross-station shrinkage.
            print(f"  Intercept hyperpriors (log-precision posterior means): "
                  f"{summary['intercept_hyperprior_log_precision']}")


if __name__ == "__main__":
    main()
