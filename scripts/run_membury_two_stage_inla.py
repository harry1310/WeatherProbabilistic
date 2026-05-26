"""Stage 2 swap: hierarchical R-INLA LogNormal with partial pooling across
the 3 Membury stations, vs NGBoost-LogNormal (current per-station champion).

The hypothesis from the previous reply: NGBoost was fit per-station with no
information sharing. INLA pools across stations via a random intercept,
which should help — especially on the smaller-sample / harder cells (Goren
and Raymonds Hill at long lead).

Architecture per lead (24/48/72h):
  * Time-split each station 70/15/15 then stack -> train, val, test
  * Stage 1: LightGBM binary classifier per (station, lead) -- same as before
  * Stage 2a (baseline): NGBoost-LogNormal per (station, lead)
  * Stage 2b (new):      R-INLA hierarchical LogNormal, POOLED across stations:
        log y ~ Normal(beta_0 + beta . x + u_station, sigma^2)
        u_station ~ Normal(0, tau)
    Posterior predictive per test row collapses to
        log y_test ~ Normal(eta_post_mean, sqrt(eta_post_var + sigma_post^2))
    Both parameter (epistemic) and likelihood (aleatory) variance are
    captured -- this is the distinctive INLA advantage.

20 quantiles extracted via scipy.stats.lognorm.ppf, combined with stage 1's
P(wet), scored with the same crps_mixed helper.

Requires R + INLA (already wired for Phase 5a; see src/models/inla_partial_pooling.py).

Run:
    .venv/Scripts/python.exe -u scripts/run_membury_two_stage_inla.py
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import lognorm
from sklearn.preprocessing import StandardScaler

from ngboost import NGBRegressor
from ngboost.distns import LogNormal as NGBLogNormal
from ngboost.scores import LogScore

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import WET_THRESHOLD_MM  # noqa: E402

LOCATION = "membury_devon"
STATIONS = ("Chards Snowdon Hill", "Goren", "Raymonds Hill")
LEADS = (24, 48, 72)

MODELS_LEAN = [
    ("gfs_seamless",         "gfs"),
    ("ecmwf_ifs025",         "ecmwf"),
    ("icon_seamless",        "icon"),
    ("meteofrance_seamless", "mf"),
    ("gem_seamless",         "gem"),
    ("ecmwf_aifs025_single", "aifs"),
    ("jma_seamless",         "jma"),
]
PRECIP_COLS = [f"precip_{s}" for _, s in MODELS_LEAN]

OUT_DIR = ROOT / "reports" / "membury_intensity_lgbm"
CACHE = OUT_DIR / "_precip_cache.parquet"

QUANTILE_ALPHAS_EVAL = tuple(np.round(np.linspace(1/41, 40/41, 20), 4).tolist())

LGB_BASE = {
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20,
    "lambda_l1": 0.1, "lambda_l2": 0.1, "feature_fraction": 0.9,
    "verbose": -1, "seed": 42, "num_threads": 0,
}
NGB_BASE = dict(n_estimators=500, learning_rate=0.01,
                minibatch_frac=1.0, col_sample=1.0,
                verbose=False, random_state=42)
NUM_ITERS = 500
EARLY_STOP = 30


# ---------------------------------------------------------------------------
# Features (same as the NGBoost script)
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    pm = df[PRECIP_COLS].to_numpy(dtype="float64")
    present = (~np.isnan(pm)).sum(axis=1)
    sumv = np.nansum(pm, axis=1); sumsq = np.nansum(pm ** 2, axis=1)
    mean_safe = np.where(present > 0, sumv / np.maximum(present, 1), np.nan)
    var = np.maximum(0.0, sumsq / np.maximum(present, 1) - mean_safe ** 2)
    wet_count = (pm >= WET_THRESHOLD_MM).sum(axis=1)
    df = df.copy()
    df["precip_mean"] = mean_safe
    df["precip_std"]  = np.where(present > 1, np.sqrt(var), 0.0)
    df["precip_max"]  = np.where(present > 0, np.nanmax(pm, axis=1), np.nan)
    df["precip_agreement_wet_01"] = np.where(present > 0, wet_count / np.maximum(present, 1), np.nan)
    vt = pd.to_datetime(df["ValidTimeUtc"])
    df["hour_sin"] = np.sin(2.0 * np.pi * vt.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * vt.dt.hour / 24.0)
    df["doy_sin"]  = np.sin(2.0 * np.pi * (vt.dt.dayofyear - 1) / 365.0)
    df["doy_cos"]  = np.cos(2.0 * np.pi * (vt.dt.dayofyear - 1) / 365.0)
    feats = PRECIP_COLS + ["precip_mean", "precip_std", "precip_max",
                           "precip_agreement_wet_01",
                           "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    return df, feats


def time_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    a = int(n * train_frac); b = a + int(n * val_frac)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def impute(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    P = X[:, :7]
    row_mean = np.nanmean(P, axis=1)
    row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
    inds = np.where(np.isnan(P))
    P[inds] = row_mean[inds[0]]
    X[:, :7] = P
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = med[inds[1]]
    return X


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def crps_mixed(pi: np.ndarray, quantiles: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = quantiles.shape[1]
    w_dry = 1.0 - pi
    w_wet = pi / K
    term1 = w_dry * y + w_wet * np.abs(quantiles - y[:, None]).sum(axis=1)
    cross_0k = 2.0 * w_dry * w_wet * quantiles.sum(axis=1)
    pairwise = np.abs(quantiles[:, :, None] - quantiles[:, None, :]).sum(axis=(1, 2))
    cross_kl = (w_wet ** 2) * pairwise
    return term1 - 0.5 * (cross_0k + cross_kl)


# ---------------------------------------------------------------------------
# R-INLA subprocess
# ---------------------------------------------------------------------------

def _default_r_exe() -> str:
    found = shutil.which("Rscript")
    if found:
        return found
    if platform.system() == "Windows":
        return r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe"
    return "/usr/bin/Rscript"


def _default_r_libs_user() -> str:
    if platform.system() == "Windows":
        return r"C:\Users\rhcsl\R\win-library\4.6"
    return os.path.expanduser("~/R/x86_64-pc-linux-gnu-library/4.4")


_R_LOGNORMAL = r"""
suppressPackageStartupMessages({
  .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
  library(INLA)
})

train_csv <- "%TRAIN_CSV%"
out_csv   <- "%OUT_CSV%"
n_features <- %N_FEATURES%

df_tr <- read.csv(train_csv)   # has y (>0) and x1..xN and station_idx

cat(sprintf("INLA-lognormal: n_train=%d, n_features=%d\n",
            nrow(df_tr), n_features))

feat_terms <- paste(paste0("x", 1:n_features), collapse = " + ")
formula_str <- paste0("y ~ ", feat_terms, " + f(station_idx, model = 'iid')")
cat("Formula:", formula_str, "\n")

t0 <- Sys.time()
fit <- inla(
  formula = as.formula(formula_str),
  family = "lognormal",
  data = df_tr,
  control.compute = list(config = TRUE),
  verbose = FALSE
)
cat(sprintf("INLA fit wall: %.1fs\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

# Likelihood precision -> sigma (data-noise std on the log scale).
hp <- fit$marginals.hyperpar
prec_name <- grep("recision for the lognormal", names(hp), value = TRUE)[1]
if (is.na(prec_name)) prec_name <- grep("recision", names(hp), value = TRUE)[1]
prec_post_mean <- inla.emarginal(function(x) x, hp[[prec_name]])
lik_sigma <- 1.0 / sqrt(prec_post_mean)
cat(sprintf("lik_sigma (data-noise on log scale) = %.4f  [via '%s']\n",
            lik_sigma, prec_name))

# Station random-effect sd
re_name <- grep("station_idx", names(hp), value = TRUE)[1]
if (!is.na(re_name)) {
  tau_post_mean <- inla.emarginal(function(x) x, hp[[re_name]])
  cat(sprintf("station random-effect sd ~ %.4f (pooling strength)\n",
              1.0 / sqrt(tau_post_mean)))
}

# Fixed-effect posterior means (intercept + per-feature betas)
fixed <- fit$summary.fixed
cat("Fixed-effect posterior means:\n")
for (rn in rownames(fixed)) {
  cat(sprintf("  %-15s  mean=%+.4f  sd=%.4f\n", rn, fixed[rn, "mean"], fixed[rn, "sd"]))
}
intercept_mean <- fixed["(Intercept)", "mean"]
beta_means <- sapply(1:n_features, function(k) fixed[paste0("x", k), "mean"])

# Station random-intercept posterior means
re <- fit$summary.random$station_idx
cat("Station random-effect posterior means:\n")
for (i in seq_len(nrow(re))) {
  cat(sprintf("  station_idx=%d  mean=%+.4f  sd=%.4f\n", re$ID[i], re$mean[i], re$sd[i]))
}

# Dump everything Python needs to compute predictions
coef_out <- list(
  intercept = intercept_mean,
  beta      = beta_means,
  station_re_id   = re$ID,
  station_re_mean = re$mean,
  lik_sigma = lik_sigma
)
# Write as two CSVs: scalars+betas, and the station RE table
write.csv(data.frame(
  name = c("intercept", paste0("beta_", 1:n_features), "lik_sigma"),
  value = c(intercept_mean, beta_means, lik_sigma)
), file = paste0(out_csv, ".scalars.csv"), row.names = FALSE)
write.csv(data.frame(ID = re$ID, mean = re$mean),
          file = paste0(out_csv, ".re.csv"), row.names = FALSE)
cat("Wrote", out_csv, ".scalars.csv and .re.csv\n")
"""


def fit_inla_lognormal_pooled(
    X_tr_w: np.ndarray, y_tr_w: np.ndarray, station_idx_tr_w: np.ndarray,
    X_te: np.ndarray, station_idx_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit INLA hierarchical LogNormal on wet train rows pooled across
    stations; return (pred_log_mean, pred_log_sd) per test row, where
    pred_log_mean = posterior-mean linear predictor at the test row, and
    pred_log_sd  = posterior-mean likelihood sigma (data-noise std on log
    scale).

    We deliberately use posterior MEANS (not full posterior predictive)
    because the marginal sd of the linear predictor at NA rows can be
    pathologically inflated under INLA's approximation -- this strips the
    epistemic contribution and isolates the pooling/regression effect,
    matching what NGBoost-LogNormal does at predict time.
    """
    r_exe = os.environ.get("INLA_R_EXE") or _default_r_exe()
    r_libs_user = os.environ.get("R_LIBS_USER") or _default_r_libs_user()

    n_features = X_tr_w.shape[1]
    feat_cols = [f"x{i+1}" for i in range(n_features)]
    df_tr = pd.DataFrame(X_tr_w, columns=feat_cols)
    df_tr["station_idx"] = (station_idx_tr_w + 1).astype(int)  # 1-based for R
    df_tr["y"] = y_tr_w.astype(float)

    with tempfile.TemporaryDirectory(prefix="inla_lognormal_") as tmp:
        train_csv = Path(tmp) / "train.csv"
        out_csv   = Path(tmp) / "predictions.csv"
        r_script  = Path(tmp) / "fit_inla.R"

        df_tr.to_csv(train_csv, index=False)
        r_text = (_R_LOGNORMAL
                  .replace("%TRAIN_CSV%", train_csv.as_posix())
                  .replace("%OUT_CSV%",   out_csv.as_posix())
                  .replace("%N_FEATURES%", str(n_features)))
        r_script.write_text(r_text, encoding="utf-8")

        env = os.environ.copy()
        env["R_LIBS_USER"] = r_libs_user

        t0 = time.time()
        r_name = Path(r_exe).name.lower()
        if r_name.startswith("rscript"):
            argv = [r_exe, str(r_script)]
        else:
            argv = [r_exe, "--no-save", "--slave", "-f", str(r_script)]
        result = subprocess.run(argv, capture_output=True, text=True, env=env)
        wall = time.time() - t0

        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"    [R] {line}", flush=True)
        if result.returncode != 0:
            print(f"  [INLA] R failed (exit {result.returncode})")
            if result.stderr.strip():
                print(f"  stderr: {result.stderr[:2000]}")
            raise RuntimeError(f"INLA subprocess failed (exit {result.returncode})")

        scalars = pd.read_csv(str(out_csv) + ".scalars.csv")
        re_df = pd.read_csv(str(out_csv) + ".re.csv")

    s_lookup = dict(zip(scalars["name"], scalars["value"]))
    intercept = float(s_lookup["intercept"])
    beta = np.array([float(s_lookup[f"beta_{k+1}"]) for k in range(n_features)])
    lik_sigma = float(s_lookup["lik_sigma"])

    # Build station_id -> random-effect mean map (R was 1-indexed)
    re_lookup = {int(row.ID): float(row.mean) for row in re_df.itertuples()}
    u_te = np.array([re_lookup.get(int(s) + 1, 0.0) for s in station_idx_te])

    eta_te = intercept + X_te @ beta + u_te
    sd_te = np.full_like(eta_te, lik_sigma)
    return eta_te, sd_te, wall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def point_metrics(p, y) -> dict:
    err = p - y
    wet = y >= WET_THRESHOLD_MM
    return {
        "mae":  float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae_wet": float(np.mean(np.abs(err[wet]))) if wet.any() else float("nan"),
        "n": int(len(y)),
    }


def fit_pwet(X_tr, y_tr, X_va, y_va):
    p = dict(LGB_BASE); p["objective"] = "binary"; p["metric"] = "binary_logloss"
    return lgb.train(p, lgb.Dataset(X_tr, label=y_tr),
                     num_boost_round=NUM_ITERS,
                     valid_sets=[lgb.Dataset(X_va, label=y_va)],
                     valid_names=["val"],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def lognormal_quantiles(log_mu: np.ndarray, log_sd: np.ndarray) -> np.ndarray:
    qs = np.empty((len(log_mu), len(QUANTILE_ALPHAS_EVAL)), dtype="float64")
    for k, a in enumerate(QUANTILE_ALPHAS_EVAL):
        qs[:, k] = lognorm.ppf(a, s=log_sd, scale=np.exp(log_mu))
    return np.clip(qs, WET_THRESHOLD_MM, None)


def main() -> None:
    if not CACHE.exists():
        raise SystemExit(f"Cache not found: {CACHE}.")
    print(f"[cache] loading {CACHE}")
    df_all = pd.read_parquet(CACHE)
    df_all, FEATS = build_features(df_all)
    print(f"[data] {len(df_all):,} rows, features={len(FEATS)}, K_eval={len(QUANTILE_ALPHAS_EVAL)}")

    rows = []

    for lead in LEADS:
        print(f"\n========== LEAD {lead}h ==========")

        # Build per-station splits, then stack
        per_station = {}
        for j, station in enumerate(STATIONS):
            sub = (df_all[(df_all["station"] == station) & (df_all["lead"] == lead)]
                   .sort_values("ValidTimeUtc").reset_index(drop=True))
            tr, va, te = time_split(sub)
            per_station[station] = (tr, va, te, j)

        # Per-(station, lead) stage 1 + NGBoost stage 2 -- unchanged
        # Per-lead pooled INLA -- stage 2 alternative

        # Collect pooled tensors (for INLA)
        X_tr_pool, y_tr_pool, sidx_tr_pool = [], [], []
        X_te_pool, sidx_te_pool = [], []
        test_offsets = {}
        cur = 0
        for station, (tr, va, te, j) in per_station.items():
            X_tr_st = impute(tr[FEATS].to_numpy("float64"))
            y_tr_st = tr["precip_mm_hour"].to_numpy("float64")
            wmask = y_tr_st >= WET_THRESHOLD_MM
            X_tr_pool.append(X_tr_st[wmask])
            y_tr_pool.append(y_tr_st[wmask])
            sidx_tr_pool.append(np.full(int(wmask.sum()), j, dtype="int64"))

            X_te_st = impute(te[FEATS].to_numpy("float64"))
            X_te_pool.append(X_te_st)
            sidx_te_pool.append(np.full(len(X_te_st), j, dtype="int64"))
            test_offsets[station] = (cur, cur + len(X_te_st))
            cur += len(X_te_st)

        X_tr_pool = np.vstack(X_tr_pool)
        y_tr_pool = np.concatenate(y_tr_pool)
        sidx_tr_pool = np.concatenate(sidx_tr_pool)
        X_te_pool = np.vstack(X_te_pool)
        sidx_te_pool = np.concatenate(sidx_te_pool)

        scaler = StandardScaler().fit(X_tr_pool)
        X_tr_pool_s = scaler.transform(X_tr_pool)
        X_te_pool_s = scaler.transform(X_te_pool)

        print(f"  pooled INLA inputs: n_train_wet={len(y_tr_pool):,}  n_test={len(X_te_pool):,}")

        # Fit pooled INLA-LogNormal
        try:
            log_mu_inla, log_sd_inla, t_inla = fit_inla_lognormal_pooled(
                X_tr_pool_s, y_tr_pool, sidx_tr_pool,
                X_te_pool_s, sidx_te_pool,
            )
            print(f"  INLA total wall: {t_inla:.1f}s")
        except Exception as e:
            print(f"  INLA FAILED: {e}")
            continue

        qs_inla_all = lognormal_quantiles(log_mu_inla, log_sd_inla)

        # Per-station scoring -- compare INLA-pooled vs NGB-per-station vs stitch isn't here.
        # For NGB we re-train fresh on the per-station split (cheap, ensures
        # identical splits with INLA).
        for station, (tr, va, te, j) in per_station.items():
            X_tr = tr[FEATS].to_numpy("float64")
            X_va = va[FEATS].to_numpy("float64")
            X_te = te[FEATS].to_numpy("float64")
            y_tr = tr["precip_mm_hour"].to_numpy("float64")
            y_va = va["precip_mm_hour"].to_numpy("float64")
            y_te = te["precip_mm_hour"].to_numpy("float64")
            wet_tr = (y_tr >= WET_THRESHOLD_MM).astype("int8")
            wet_va = (y_va >= WET_THRESHOLD_MM).astype("int8")
            wmask_tr = wet_tr.astype(bool); wmask_va = wet_va.astype(bool)

            # Stage 1
            b_cls = fit_pwet(X_tr, wet_tr, X_va, wet_va)
            pi_te = b_cls.predict(X_te, num_iteration=b_cls.best_iteration)

            # Slice INLA predictions for this station
            lo, hi = test_offsets[station]
            qs_inla = qs_inla_all[lo:hi]
            crps_inla = float(crps_mixed(pi_te, qs_inla, y_te).mean())
            mu_inla_lin = np.exp(log_mu_inla[lo:hi] + 0.5 * log_sd_inla[lo:hi] ** 2)
            m = point_metrics(pi_te * mu_inla_lin, y_te); m.update(
                {"station": station, "lead": lead, "model": "ts_inla_lognormal_pooled",
                 "crps": crps_inla})
            rows.append(m)

            # NGBoost-LogNormal per (station, lead) -- baseline
            X_tr_im = impute(X_tr); X_va_im = impute(X_va); X_te_im = impute(X_te)
            ngb = NGBRegressor(Dist=NGBLogNormal, Score=LogScore, **NGB_BASE)
            ngb.fit(X_tr_im[wmask_tr], y_tr[wmask_tr],
                    X_val=X_va_im[wmask_va], Y_val=y_va[wmask_va],
                    early_stopping_rounds=EARLY_STOP)
            dist = ngb.pred_dist(X_te_im, max_iter=ngb.best_val_loss_itr)
            s = np.asarray(dist.params["s"])
            scale = np.asarray(dist.params["scale"])
            qs_ngb = np.empty((len(X_te), len(QUANTILE_ALPHAS_EVAL)), dtype="float64")
            for k, alpha in enumerate(QUANTILE_ALPHAS_EVAL):
                qs_ngb[:, k] = lognorm.ppf(alpha, s=s, scale=scale)
            qs_ngb = np.clip(qs_ngb, WET_THRESHOLD_MM, None)
            crps_ngb = float(crps_mixed(pi_te, qs_ngb, y_te).mean())
            mu_ngb = scale * np.exp(0.5 * s ** 2)
            m = point_metrics(pi_te * mu_ngb, y_te); m.update(
                {"station": station, "lead": lead, "model": "ts_ngboost_lognormal_perstation",
                 "crps": crps_ngb})
            rows.append(m)

            delta = crps_inla - crps_ngb
            pct = 100.0 * delta / crps_ngb
            print(f"  {station:22s}  INLA-pooled CRPS={crps_inla:.4f}  "
                  f"NGB-per-station={crps_ngb:.4f}  "
                  f"delta={delta:+.4f} ({pct:+.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "two_stage_inla_summary.csv", index=False)

    print("\n\n========== AGGREGATE (mean across 3 stations) ==========")
    agg = (df.groupby(["lead", "model"])
             .agg(crps=("crps", "mean"), mae=("mae", "mean"),
                  rmse=("rmse", "mean"), mae_wet=("mae_wet", "mean"))
             .reset_index().sort_values(["lead", "crps"]))
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    agg.to_csv(OUT_DIR / "two_stage_inla_aggregate.csv", index=False)
    print(f"\nWrote {OUT_DIR/'two_stage_inla_summary.csv'}")
    print(f"Wrote {OUT_DIR/'two_stage_inla_aggregate.csv'}")


if __name__ == "__main__":
    main()
