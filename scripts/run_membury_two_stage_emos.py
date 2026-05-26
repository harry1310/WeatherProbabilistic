"""Stage 2 swap: parametric distributional regression (EMOS-style) vs the
quantile-stitching from run_membury_two_stage.py.

Stage 1 (P(wet)) stays the same — LightGBM binary classifier on the lean
15-feature set, fit per (station, lead). Three stage-2 variants compete:

  * quantile_stitch  — 9 independent LightGBM quantile regressors at
                       alpha in {0.1, ..., 0.9}, sorted post-hoc. The
                       previous-run baseline.
  * emos_gamma       — Gamma(shape=alpha, scale=mu/alpha) with
                       log-link mean (log mu = X @ beta) and a constant
                       shape, jointly fit by MLE on the wet training rows.
  * emos_lognormal   — LogNormal(mu, sigma) with linear-link log-mean
                       (log y ~ N(X @ beta, sigma^2)) — closed form via OLS
                       on log-truth plus residual sigma.

For each variant the wet-conditional distribution is sampled at K=20 equally
spaced quantiles and combined with stage-1's P(wet) into the same mixed
(1-pi)*delta_0 + pi*G(x|wet) distribution. CRPS is computed by the same
discrete-distribution closed form as the previous script.

Run:
    .venv/Scripts/python.exe -u scripts/run_membury_two_stage_emos.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm

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

QUANTILE_ALPHAS_FIT = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# Use K=20 evenly-spaced quantiles to represent any wet-conditional
# distribution. Bigger K gives finer CRPS but O(K^2) memory in crps_mixed.
QUANTILE_ALPHAS_EVAL = tuple(np.round(np.linspace(1/41, 40/41, 20), 4).tolist())

LGB_BASE = {
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20,
    "lambda_l1": 0.1, "lambda_l2": 0.1, "feature_fraction": 0.9,
    "bagging_fraction": 1.0, "verbose": -1, "seed": 42, "num_threads": 0,
}
NUM_ITERS = 500
EARLY_STOP = 30


# ---------------------------------------------------------------------------
# Features
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


def point_metrics(p, y) -> dict:
    err = p - y
    wet = y >= WET_THRESHOLD_MM
    return {
        "mae":  float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae_wet": float(np.mean(np.abs(err[wet]))) if wet.any() else float("nan"),
        "n": int(len(y)),
    }


# ---------------------------------------------------------------------------
# Stage 1 (P(wet)) — LightGBM binary
# ---------------------------------------------------------------------------

def fit_pwet(X_tr, y_tr, X_va, y_va):
    params = dict(LGB_BASE); params["objective"] = "binary"; params["metric"] = "binary_logloss"
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    return lgb.train(params, train_set, num_boost_round=NUM_ITERS,
                     valid_sets=[val_set], valid_names=["val"],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


# ---------------------------------------------------------------------------
# Stage 2a — quantile stitching (LightGBM, K models)
# ---------------------------------------------------------------------------

def fit_quantile_stitch(X_tr_w, y_tr_w, X_va_w, y_va_w, X_te):
    qs = np.empty((len(X_te), len(QUANTILE_ALPHAS_FIT)), dtype="float64")
    for k, alpha in enumerate(QUANTILE_ALPHAS_FIT):
        p = dict(LGB_BASE); p["objective"] = "quantile"; p["metric"] = "quantile"; p["alpha"] = alpha
        b = lgb.train(p, lgb.Dataset(X_tr_w, label=y_tr_w), num_boost_round=NUM_ITERS,
                      valid_sets=[lgb.Dataset(X_va_w, label=y_va_w)],
                      valid_names=["val"],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        qs[:, k] = b.predict(X_te, num_iteration=b.best_iteration)
    qs = np.sort(qs, axis=1)
    return np.clip(qs, WET_THRESHOLD_MM, None)


# ---------------------------------------------------------------------------
# Stage 2b — EMOS-Gamma (joint MLE: log-link mean, constant shape)
# ---------------------------------------------------------------------------

def fit_emos_gamma(X_tr_w, y_tr_w):
    """Fit Gamma(shape=alpha, scale=mu/alpha) with log mu = X @ beta.
    Includes intercept column (caller passes X already augmented).
    Joint MLE over (beta, log_alpha) via L-BFGS-B."""
    n, p = X_tr_w.shape
    y = y_tr_w
    log_y = np.log(np.clip(y, 1e-6, None))

    def nll(theta):
        beta = theta[:p]
        log_alpha = theta[p]
        alpha = np.exp(log_alpha)
        eta = X_tr_w @ beta
        # Stabilize: clip eta to avoid overflow in exp
        eta = np.clip(eta, -20.0, 20.0)
        mu = np.exp(eta)
        # Gamma logpdf with mean mu and shape alpha (scale = mu/alpha):
        #   ll = (alpha-1) log y - y*alpha/mu - alpha*log(mu/alpha) - lgamma(alpha)
        ll = ((alpha - 1.0) * log_y
              - y * alpha / mu
              - alpha * (eta - log_alpha)
              - special.gammaln(alpha))
        return -float(np.sum(ll))

    # Init: intercept = log(mean), other betas = 0, alpha = 1
    theta0 = np.zeros(p + 1)
    theta0[0] = float(np.log(np.clip(y.mean(), 1e-3, None)))
    theta0[p] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(nll, theta0, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-9})
    beta = res.x[:p]
    alpha = float(np.exp(res.x[p]))
    return beta, alpha, res


def predict_emos_gamma_quantiles(X_te, beta, alpha):
    eta = np.clip(X_te @ beta, -20.0, 20.0)
    mu = np.exp(eta)
    scale = mu / alpha  # per-row
    qs = np.empty((len(X_te), len(QUANTILE_ALPHAS_EVAL)), dtype="float64")
    for k, a in enumerate(QUANTILE_ALPHAS_EVAL):
        qs[:, k] = gamma_dist.ppf(a, a=alpha, scale=scale)
    return np.clip(qs, WET_THRESHOLD_MM, None), mu, alpha


# ---------------------------------------------------------------------------
# Stage 2c — EMOS-LogNormal (OLS on log y, residual sigma)
# ---------------------------------------------------------------------------

def fit_emos_lognormal(X_tr_w, y_tr_w):
    log_y = np.log(np.clip(y_tr_w, 1e-6, None))
    beta, *_ = np.linalg.lstsq(X_tr_w, log_y, rcond=None)
    resid = log_y - X_tr_w @ beta
    # Use ML estimator (1/n), not unbiased (1/(n-p)) — matches the LogNormal MLE.
    sigma = float(np.sqrt(np.mean(resid ** 2)))
    return beta, sigma


def predict_emos_lognormal_quantiles(X_te, beta, sigma):
    mu_log = X_te @ beta  # log-scale mean
    qs = np.empty((len(X_te), len(QUANTILE_ALPHAS_EVAL)), dtype="float64")
    for k, a in enumerate(QUANTILE_ALPHAS_EVAL):
        qs[:, k] = lognorm.ppf(a, s=sigma, scale=np.exp(mu_log))
    # E[Y] for LogNormal = exp(mu + sigma^2/2)
    mu_lin = np.exp(mu_log + 0.5 * sigma ** 2)
    return np.clip(qs, WET_THRESHOLD_MM, None), mu_lin


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CACHE.exists():
        raise SystemExit(f"Cache not found: {CACHE}.\nRun run_membury_member_weighted.py first.")
    print(f"[cache] loading {CACHE}")
    df_all = pd.read_parquet(CACHE)
    df_all, FEATS = build_features(df_all)
    print(f"[data] {len(df_all):,} rows, features={len(FEATS)}, "
          f"K_eval={len(QUANTILE_ALPHAS_EVAL)}")

    rows = []

    for station in STATIONS:
        for lead in LEADS:
            sub = (df_all[(df_all["station"] == station) & (df_all["lead"] == lead)]
                   .sort_values("ValidTimeUtc").reset_index(drop=True))
            tr, va, te = time_split(sub)

            X_tr = tr[FEATS].to_numpy(dtype="float64")
            X_va = va[FEATS].to_numpy(dtype="float64")
            X_te = te[FEATS].to_numpy(dtype="float64")
            y_tr = tr["precip_mm_hour"].to_numpy(dtype="float64")
            y_va = va["precip_mm_hour"].to_numpy(dtype="float64")
            y_te = te["precip_mm_hour"].to_numpy(dtype="float64")
            wet_tr = (y_tr >= WET_THRESHOLD_MM).astype("int8")
            wet_va = (y_va >= WET_THRESHOLD_MM).astype("int8")
            wmask_tr = wet_tr.astype(bool)
            wmask_va = wet_va.astype(bool)

            print(f"\n=== {station} | lead {lead}h ===")
            print(f"  n_train={len(y_tr):,}  n_test={len(y_te):,}  "
                  f"wet_tr={wmask_tr.sum()}  wet_va={wmask_va.sum()}  "
                  f"test_wet_rate={(y_te>=WET_THRESHOLD_MM).mean()*100:.1f}%")

            # NaN handling: LightGBM eats NaN natively; EMOS GLMs do not.
            # For EMOS we use row-mean imputation on the precip columns (NaN
            # in spread/calendar features is already 0 by construction).
            def emos_impute(X):
                X = X.copy()
                # First 7 columns are per-NWP precip; impute NaN with row-mean of present
                P = X[:, :7]
                row_mean = np.nanmean(P, axis=1)
                row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
                inds = np.where(np.isnan(P))
                P[inds] = row_mean[inds[0]]
                X[:, :7] = P
                # Fill any remaining NaN with column median
                med = np.nanmedian(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = med[inds[1]]
                return X

            # Augment X with intercept column for EMOS GLMs
            def with_intercept(X):
                return np.hstack([np.ones((len(X), 1)), X])

            X_tr_im = emos_impute(X_tr)
            X_va_im = emos_impute(X_va)
            X_te_im = emos_impute(X_te)
            X_tr_int = with_intercept(X_tr_im)
            X_va_int = with_intercept(X_va_im)
            X_te_int = with_intercept(X_te_im)

            # --- Stage 1: P(wet) ---
            t0 = time.time()
            b_cls = fit_pwet(X_tr, wet_tr, X_va, wet_va)
            pi_te = b_cls.predict(X_te, num_iteration=b_cls.best_iteration)
            t_s1 = time.time() - t0

            # --- Stage 2a: LightGBM quantile stitching ---
            t0 = time.time()
            qs_stitch = fit_quantile_stitch(X_tr[wmask_tr], y_tr[wmask_tr],
                                            X_va[wmask_va], y_va[wmask_va], X_te)
            t_stitch = time.time() - t0
            crps_stitch = float(crps_mixed(pi_te, qs_stitch, y_te).mean())
            yhat_stitch = pi_te * qs_stitch.mean(axis=1)
            m = point_metrics(yhat_stitch, y_te); m.update(
                {"station": station, "lead": lead, "model": "ts_quantile_stitch",
                 "crps": crps_stitch, "fit_s": round(t_stitch, 1)})
            rows.append(m)

            # --- Stage 2b: EMOS-Gamma ---
            t0 = time.time()
            beta_g, alpha_g, res_g = fit_emos_gamma(X_tr_int[wmask_tr], y_tr[wmask_tr])
            qs_gamma, mu_g, _ = predict_emos_gamma_quantiles(X_te_int, beta_g, alpha_g)
            t_gamma = time.time() - t0
            crps_gamma = float(crps_mixed(pi_te, qs_gamma, y_te).mean())
            yhat_gamma = pi_te * mu_g
            m = point_metrics(yhat_gamma, y_te); m.update(
                {"station": station, "lead": lead, "model": "ts_emos_gamma",
                 "crps": crps_gamma, "fit_s": round(t_gamma, 1),
                 "extra": f"alpha={alpha_g:.3f} converged={res_g.success}"})
            rows.append(m)

            # --- Stage 2c: EMOS-LogNormal ---
            t0 = time.time()
            beta_ln, sigma_ln = fit_emos_lognormal(X_tr_int[wmask_tr], y_tr[wmask_tr])
            qs_ln, mu_ln = predict_emos_lognormal_quantiles(X_te_int, beta_ln, sigma_ln)
            t_ln = time.time() - t0
            crps_ln = float(crps_mixed(pi_te, qs_ln, y_te).mean())
            yhat_ln = pi_te * mu_ln
            m = point_metrics(yhat_ln, y_te); m.update(
                {"station": station, "lead": lead, "model": "ts_emos_lognormal",
                 "crps": crps_ln, "fit_s": round(t_ln, 1),
                 "extra": f"sigma={sigma_ln:.3f}"})
            rows.append(m)

            print(f"  stage1 LGBM     {t_s1:.1f}s  "
                  f"stitch {t_stitch:.1f}s  gamma {t_gamma:.1f}s  lognormal {t_ln:.1f}s")
            for r in rows[-3:]:
                print(f"   {r['model']:22s} CRPS={r['crps']:.4f}  "
                      f"MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
                      f"MAE_wet={r['mae_wet']:.3f}"
                      + (f"  ({r.get('extra','')})" if r.get('extra') else ""))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "two_stage_emos_summary.csv", index=False)

    print("\n\n========== AGGREGATE (mean across 3 stations) ==========")
    agg = (df.groupby(["lead", "model"])
             .agg(crps=("crps", "mean"), mae=("mae", "mean"),
                  rmse=("rmse", "mean"), mae_wet=("mae_wet", "mean"),
                  fit_s=("fit_s", "mean"))
             .reset_index().sort_values(["lead", "crps"]))
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    agg.to_csv(OUT_DIR / "two_stage_emos_aggregate.csv", index=False)

    print(f"\nWrote {OUT_DIR/'two_stage_emos_summary.csv'}")
    print(f"Wrote {OUT_DIR/'two_stage_emos_aggregate.csv'}")


if __name__ == "__main__":
    main()
