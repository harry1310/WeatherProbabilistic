"""Add NGBoost (feature-dependent distribution parameters) to the two-stage
EMOS comparison.

Five stage-2 variants now compete:
  * quantile_stitch    — 9 independent LightGBM quantile regressors (prior winner).
  * emos_gamma         — Joint MLE Gamma with log-link mean, CONSTANT shape.
  * emos_lognormal     — OLS on log y, CONSTANT sigma.
  * ngboost_gamma      — Gradient boosting where each leaf updates both
                         shape AND scale parameters of a Gamma. Feature-dependent
                         heteroscedasticity is the whole point.
  * ngboost_lognormal  — Same but for LogNormal(mu, sigma).

NGBoost trains via natural-gradient boosting on the log-score by default
(CRPScore is supported for LogNormal but not Gamma in ngboost 0.5).

Run:
    .venv/Scripts/python.exe -u scripts/run_membury_two_stage_ngboost.py
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

from ngboost import NGBRegressor
from ngboost.distns import Gamma as NGBGamma
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

QUANTILE_ALPHAS_FIT = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QUANTILE_ALPHAS_EVAL = tuple(np.round(np.linspace(1/41, 40/41, 20), 4).tolist())

LGB_BASE = {
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20,
    "lambda_l1": 0.1, "lambda_l2": 0.1, "feature_fraction": 0.9,
    "verbose": -1, "seed": 42, "num_threads": 0,
}
NUM_ITERS = 500
EARLY_STOP = 30

# NGBoost: fewer / shallower trees per stage, slower LR. Defaults are reasonable.
NGB_BASE = dict(
    n_estimators=500, learning_rate=0.01,
    minibatch_frac=1.0, col_sample=1.0, verbose=False, random_state=42,
)
NGB_EARLY_STOP = 30


# ---------------------------------------------------------------------------
# Features (same as before)
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


def emos_impute(X: np.ndarray) -> np.ndarray:
    """Row-mean impute NWP precip cols, column-median fill anything else."""
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
# Metrics
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
# Stage 1 (P(wet)) and Stage 2 fitters
# ---------------------------------------------------------------------------

def fit_pwet(X_tr, y_tr, X_va, y_va):
    params = dict(LGB_BASE); params["objective"] = "binary"; params["metric"] = "binary_logloss"
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    return lgb.train(params, train_set, num_boost_round=NUM_ITERS,
                     valid_sets=[val_set], valid_names=["val"],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


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


def fit_emos_gamma(X_tr_w, y_tr_w):
    n, p = X_tr_w.shape
    y = y_tr_w
    log_y = np.log(np.clip(y, 1e-6, None))
    def nll(theta):
        beta = theta[:p]; log_alpha = theta[p]
        alpha = np.exp(log_alpha)
        eta = np.clip(X_tr_w @ beta, -20.0, 20.0)
        mu = np.exp(eta)
        ll = ((alpha - 1.0) * log_y - y * alpha / mu
              - alpha * (eta - log_alpha) - special.gammaln(alpha))
        return -float(np.sum(ll))
    theta0 = np.zeros(p + 1); theta0[0] = float(np.log(np.clip(y.mean(), 1e-3, None)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(nll, theta0, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-9})
    return res.x[:p], float(np.exp(res.x[p]))


def predict_emos_gamma_quantiles(X_te, beta, alpha):
    eta = np.clip(X_te @ beta, -20.0, 20.0)
    mu = np.exp(eta); scale = mu / alpha
    qs = np.empty((len(X_te), len(QUANTILE_ALPHAS_EVAL)), dtype="float64")
    for k, a in enumerate(QUANTILE_ALPHAS_EVAL):
        qs[:, k] = gamma_dist.ppf(a, a=alpha, scale=scale)
    return np.clip(qs, WET_THRESHOLD_MM, None), mu


def fit_emos_lognormal(X_tr_w, y_tr_w):
    log_y = np.log(np.clip(y_tr_w, 1e-6, None))
    beta, *_ = np.linalg.lstsq(X_tr_w, log_y, rcond=None)
    sigma = float(np.sqrt(np.mean((log_y - X_tr_w @ beta) ** 2)))
    return beta, sigma


def predict_emos_lognormal_quantiles(X_te, beta, sigma):
    mu_log = X_te @ beta
    qs = np.empty((len(X_te), len(QUANTILE_ALPHAS_EVAL)), dtype="float64")
    for k, a in enumerate(QUANTILE_ALPHAS_EVAL):
        qs[:, k] = lognorm.ppf(a, s=sigma, scale=np.exp(mu_log))
    mu_lin = np.exp(mu_log + 0.5 * sigma ** 2)
    return np.clip(qs, WET_THRESHOLD_MM, None), mu_lin


# ---------------------------------------------------------------------------
# NGBoost — feature-dependent shape AND scale (the whole point)
# ---------------------------------------------------------------------------

def fit_ngboost_gamma(X_tr_w, y_tr_w, X_va_w, y_va_w):
    ngb = NGBRegressor(Dist=NGBGamma, Score=LogScore, **NGB_BASE)
    ngb.fit(X_tr_w, y_tr_w, X_val=X_va_w, Y_val=y_va_w,
            early_stopping_rounds=NGB_EARLY_STOP)
    return ngb


def predict_ngboost_quantiles(ngb, X_te, dist_kind: str):
    """Pull predicted distribution per row, extract quantiles + mean."""
    dist = ngb.pred_dist(X_te, max_iter=ngb.best_val_loss_itr)
    qs = np.empty((len(X_te), len(QUANTILE_ALPHAS_EVAL)), dtype="float64")
    if dist_kind == "gamma":
        # NGBoost Gamma params are 'alpha' (shape) and 'beta' (rate).
        # scipy.stats.gamma uses (a=shape, scale=1/rate). Mean = alpha/beta.
        shape = np.asarray(dist.params["alpha"])
        rate = np.asarray(dist.params["beta"])
        scale = 1.0 / np.clip(rate, 1e-6, None)
        for k, a in enumerate(QUANTILE_ALPHAS_EVAL):
            qs[:, k] = gamma_dist.ppf(a, a=shape, scale=scale)
        mu_pred = shape * scale
    elif dist_kind == "lognormal":
        # NGB LogNormal params: 's' (sigma, log-std) and 'scale' (exp(mu))
        s = np.asarray(dist.params["s"])
        scale = np.asarray(dist.params["scale"])
        for k, a in enumerate(QUANTILE_ALPHAS_EVAL):
            qs[:, k] = lognorm.ppf(a, s=s, scale=scale)
        mu_pred = scale * np.exp(0.5 * s ** 2)
    else:
        raise ValueError(dist_kind)
    return np.clip(qs, WET_THRESHOLD_MM, None), mu_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CACHE.exists():
        raise SystemExit(f"Cache not found: {CACHE}.")
    print(f"[cache] loading {CACHE}")
    df_all = pd.read_parquet(CACHE)
    df_all, FEATS = build_features(df_all)
    print(f"[data] {len(df_all):,} rows, features={len(FEATS)}, K_eval={len(QUANTILE_ALPHAS_EVAL)}")

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
            wmask_tr = wet_tr.astype(bool); wmask_va = wet_va.astype(bool)

            print(f"\n=== {station} | lead {lead}h ===  "
                  f"wet_tr={wmask_tr.sum()}  wet_va={wmask_va.sum()}")

            X_tr_im = emos_impute(X_tr); X_va_im = emos_impute(X_va); X_te_im = emos_impute(X_te)
            X_tr_int = np.hstack([np.ones((len(X_tr_im), 1)), X_tr_im])
            X_va_int = np.hstack([np.ones((len(X_va_im), 1)), X_va_im])
            X_te_int = np.hstack([np.ones((len(X_te_im), 1)), X_te_im])

            # Stage 1
            b_cls = fit_pwet(X_tr, wet_tr, X_va, wet_va)
            pi_te = b_cls.predict(X_te, num_iteration=b_cls.best_iteration)

            # Stage 2 variants
            def record(tag, qs, mu, fit_s, extra=""):
                crps = float(crps_mixed(pi_te, qs, y_te).mean())
                yhat = pi_te * mu
                m = point_metrics(yhat, y_te); m.update(
                    {"station": station, "lead": lead, "model": tag,
                     "crps": crps, "fit_s": round(fit_s, 1), "extra": extra})
                rows.append(m)
                print(f"   {tag:22s} CRPS={crps:.4f}  MAE={m['mae']:.4f}  "
                      f"RMSE={m['rmse']:.4f}  MAE_wet={m['mae_wet']:.3f}"
                      + (f"  ({extra})" if extra else ""))

            t0 = time.time()
            qs = fit_quantile_stitch(X_tr[wmask_tr], y_tr[wmask_tr],
                                     X_va[wmask_va], y_va[wmask_va], X_te)
            record("ts_quantile_stitch", qs, qs.mean(axis=1), time.time() - t0)

            t0 = time.time()
            beta, alpha = fit_emos_gamma(X_tr_int[wmask_tr], y_tr[wmask_tr])
            qs, mu = predict_emos_gamma_quantiles(X_te_int, beta, alpha)
            record("ts_emos_gamma", qs, mu, time.time() - t0, f"alpha={alpha:.3f}")

            t0 = time.time()
            beta, sigma = fit_emos_lognormal(X_tr_int[wmask_tr], y_tr[wmask_tr])
            qs, mu = predict_emos_lognormal_quantiles(X_te_int, beta, sigma)
            record("ts_emos_lognormal", qs, mu, time.time() - t0, f"sigma={sigma:.3f}")

            # NGBoost-Gamma
            try:
                t0 = time.time()
                ngb = fit_ngboost_gamma(X_tr_im[wmask_tr], y_tr[wmask_tr],
                                        X_va_im[wmask_va], y_va[wmask_va])
                qs, mu = predict_ngboost_quantiles(ngb, X_te_im, "gamma")
                record("ts_ngboost_gamma", qs, mu, time.time() - t0,
                       f"best_iter={ngb.best_val_loss_itr}")
            except Exception as e:
                print(f"   ngboost_gamma FAILED: {e}")

            # NGBoost-LogNormal
            try:
                t0 = time.time()
                ngb = NGBRegressor(Dist=NGBLogNormal, Score=LogScore, **NGB_BASE)
                ngb.fit(X_tr_im[wmask_tr], y_tr[wmask_tr],
                        X_val=X_va_im[wmask_va], Y_val=y_va[wmask_va],
                        early_stopping_rounds=NGB_EARLY_STOP)
                qs, mu = predict_ngboost_quantiles(ngb, X_te_im, "lognormal")
                record("ts_ngboost_lognormal", qs, mu, time.time() - t0,
                       f"best_iter={ngb.best_val_loss_itr}")
            except Exception as e:
                print(f"   ngboost_lognormal FAILED: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "two_stage_ngboost_summary.csv", index=False)

    print("\n\n========== AGGREGATE (mean across 3 stations) ==========")
    agg = (df.groupby(["lead", "model"])
             .agg(crps=("crps", "mean"), mae=("mae", "mean"),
                  rmse=("rmse", "mean"), mae_wet=("mae_wet", "mean"),
                  fit_s=("fit_s", "mean"))
             .reset_index().sort_values(["lead", "crps"]))
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    agg.to_csv(OUT_DIR / "two_stage_ngboost_aggregate.csv", index=False)
    print(f"\nWrote {OUT_DIR/'two_stage_ngboost_summary.csv'}")
    print(f"Wrote {OUT_DIR/'two_stage_ngboost_aggregate.csv'}")


if __name__ == "__main__":
    main()
