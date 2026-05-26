"""Membury mm/h via two-stage (P(wet) gate × conditional intensity) and proper
distributional scoring via CRPS.

Why two-stage:
  Earlier sweeps showed unconditional bias correction *hurts* MAE because the
  question is dominated by dry hours (~15% wet rate). A two-stage decomposition
  routes the dry-hour question through P(wet) and lets the intensity model
  focus on the wet-conditional distribution.

Why CRPS:
  For a deterministic point forecast, CRPS reduces to MAE -- so for the
  NWP-mean / constrained-Tweedie baselines CRPS adds nothing. The two-stage
  model naturally produces a *mixed distribution*
  ((1-pi)*delta_0 + pi*G(x | wet)), and a 7-member NWP ensemble is also a
  distribution; CRPS scores both fairly against the same observation.

Architecture (per station, per lead):
  Stage 1 (P(wet)):     LightGBM binary classifier on 15 features.
  Stage 2 (intensity):  LightGBM Tweedie regressor (for the conditional mean)
                        plus 9 LightGBM quantile regressors at alphas
                        {0.1, 0.2, ..., 0.9}, both trained on WET rows only.
  Point forecast:       pi_hat * E[mm | wet]
  Mixed distribution:   atoms [0, q1, ..., q9] with weights [(1-pi_hat),
                        pi_hat/9, ..., pi_hat/9].

Baselines for context (all on the same test split):
  * equal_mean       -- 7-NWP unweighted mean, point forecast (CRPS == MAE).
  * ensemble_7nwp    -- 7 NWPs read as a 7-member ensemble distribution.
  * lgbm_tweedie     -- single-stage Tweedie LightGBM on all-hours, point.

Features (15): 7 raw NWP precip + 4 spread aggregates (mean/std/max/agreement)
+ 4 cyclical calendar (hour_sin/cos, doy_sin/cos). Lean -- if two-stage shows
promise, adding the 39 extra rich-feature columns is a follow-up.

Run:
    .venv/Scripts/python.exe -u scripts/run_membury_two_stage.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

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

QUANTILE_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

LGB_BASE = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "feature_fraction": 0.9,
    "bagging_fraction": 1.0,
    "verbose": -1,
    "seed": 42,
    "num_threads": 0,
}
NUM_ITERS = 500
EARLY_STOP = 30


# ---------------------------------------------------------------------------
# Feature build (reuses the cached 7-NWP parquet from the member-weighted run)
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add 4 spread aggregates + 4 cyclical features. Returns (df, feature_names).

    NaN-safe spread features mirror PrecipFeatureBuilder.ComposeRow (C#).
    """
    pm = df[PRECIP_COLS].to_numpy(dtype="float64")
    present = (~np.isnan(pm)).sum(axis=1)
    sumv = np.nansum(pm, axis=1)
    sumsq = np.nansum(pm ** 2, axis=1)
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
# Metrics
# ---------------------------------------------------------------------------

def crps_mixed(pi: np.ndarray, quantiles: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per row for mixed predictive distribution:
        (1 - pi) at atom 0  +  pi/K mass at each of the K wet quantiles.

    Uses the discrete-distribution closed form
        CRPS = sum_a w_a |z_a - y|  -  0.5 sum_a sum_b w_a w_b |z_a - z_b|

    pi:        (n,)   probability of wet
    quantiles: (n, K) wet-conditional quantiles, sorted ascending per row
    y:         (n,)   observed mm/h, >= 0
    """
    K = quantiles.shape[1]
    w_dry = 1.0 - pi
    w_wet = pi / K

    # Term 1: E|X - y|.  y >= 0 so |0 - y| = y.
    term1 = w_dry * y + w_wet * np.abs(quantiles - y[:, None]).sum(axis=1)

    # Term 2: E|X - X'| over all ordered pairs of atoms.
    #   * (0, 0): contributes 0
    #   * (0, q_k) and (q_k, 0): together 2 * w_dry * w_wet * q_k
    #   * (q_k, q_l): w_wet^2 * |q_k - q_l|  (all ordered pairs)
    cross_0k = 2.0 * w_dry * w_wet * quantiles.sum(axis=1)
    pairwise = np.abs(quantiles[:, :, None] - quantiles[:, None, :]).sum(axis=(1, 2))
    cross_kl = (w_wet ** 2) * pairwise
    e_xxp = cross_0k + cross_kl

    return term1 - 0.5 * e_xxp


def crps_ensemble(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of an equally-weighted ensemble (samples), one CRPS per row.
    samples: (n, M); rows with NaN columns are treated as M_eff = sum of non-NaN."""
    n, M = samples.shape
    # Treat NaN as "no atom" by masking
    mask = ~np.isnan(samples)
    M_eff = mask.sum(axis=1).astype("float64")
    M_eff = np.where(M_eff > 0, M_eff, 1.0)  # avoid /0 for all-NaN rows (rare)
    s = np.where(mask, samples, 0.0)
    # Term 1: mean over present members of |s_m - y|
    term1 = np.where(mask, np.abs(s - y[:, None]), 0.0).sum(axis=1) / M_eff
    # Term 2: 0.5 * mean over (m, m') of |s_m - s_m'|; pairs of NaN drop.
    pair_mask = mask[:, :, None] & mask[:, None, :]
    pair_count = pair_mask.sum(axis=(1, 2)).astype("float64")
    pair_count = np.where(pair_count > 0, pair_count, 1.0)
    pair_abs = np.where(pair_mask,
                        np.abs(s[:, :, None] - s[:, None, :]),
                        0.0).sum(axis=(1, 2))
    e_xxp = pair_abs / pair_count
    return term1 - 0.5 * e_xxp


def point_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    wet = y >= WET_THRESHOLD_MM
    mae_wet = float(np.mean(np.abs(err[wet]))) if wet.any() else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mae_wet": mae_wet, "n": int(len(y))}


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------

def fit_lgb(X_tr, y_tr, X_va, y_va, params, num_iters=NUM_ITERS, early=EARLY_STOP):
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    booster = lgb.train(
        params, train_set,
        num_boost_round=num_iters,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(early, verbose=False)],
    )
    return booster


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CACHE.exists():
        raise SystemExit(f"Cache not found: {CACHE}.\nRun run_membury_member_weighted.py first to build it.")
    print(f"[cache] loading {CACHE}")
    df_all = pd.read_parquet(CACHE)
    df_all, FEATS = build_features(df_all)
    print(f"[data] {len(df_all):,} rows, features={len(FEATS)}")

    rows = []

    for station in STATIONS:
        for lead in LEADS:
            sub = (df_all[(df_all["station"] == station) & (df_all["lead"] == lead)]
                   .sort_values("ValidTimeUtc").reset_index(drop=True))
            tr, va, te = time_split(sub)

            y_tr = tr["precip_mm_hour"].to_numpy(dtype="float64")
            y_va = va["precip_mm_hour"].to_numpy(dtype="float64")
            y_te = te["precip_mm_hour"].to_numpy(dtype="float64")
            X_tr = tr[FEATS].to_numpy(dtype="float64")
            X_va = va[FEATS].to_numpy(dtype="float64")
            X_te = te[FEATS].to_numpy(dtype="float64")
            wet_tr = (y_tr >= WET_THRESHOLD_MM).astype("int8")
            wet_va = (y_va >= WET_THRESHOLD_MM).astype("int8")

            print(f"\n=== {station} | lead {lead}h ===")
            print(f"  n_train={len(y_tr):,}  n_test={len(y_te):,}  "
                  f"wet_rate_train={wet_tr.mean()*100:.1f}%  "
                  f"wet_rate_test={(y_te>=WET_THRESHOLD_MM).mean()*100:.1f}%")

            # --- Stage 1: P(wet) ---
            t0 = time.time()
            cls_params = dict(LGB_BASE); cls_params["objective"] = "binary"; cls_params["metric"] = "binary_logloss"
            b_cls = fit_lgb(X_tr, wet_tr, X_va, wet_va, cls_params)
            pi_te = b_cls.predict(X_te, num_iteration=b_cls.best_iteration)
            t_stage1 = time.time() - t0

            # --- Stage 2a: conditional mean (Tweedie on wet rows only) ---
            wet_mask_tr = (y_tr >= WET_THRESHOLD_MM)
            wet_mask_va = (y_va >= WET_THRESHOLD_MM)
            n_wet_tr = int(wet_mask_tr.sum())
            if n_wet_tr < 200:
                print(f"  SKIP: only {n_wet_tr} wet train rows")
                continue
            t0 = time.time()
            mean_params = dict(LGB_BASE); mean_params["objective"] = "tweedie"
            mean_params["metric"] = "tweedie"; mean_params["tweedie_variance_power"] = 1.5
            b_mean = fit_lgb(X_tr[wet_mask_tr], y_tr[wet_mask_tr],
                             X_va[wet_mask_va], y_va[wet_mask_va], mean_params)
            mu_wet_te = np.clip(b_mean.predict(X_te, num_iteration=b_mean.best_iteration),
                                WET_THRESHOLD_MM, None)
            t_stage2_mean = time.time() - t0

            # --- Stage 2b: quantile distribution (one model per alpha) ---
            t0 = time.time()
            qs = np.empty((len(X_te), len(QUANTILE_ALPHAS)), dtype="float64")
            for k, alpha in enumerate(QUANTILE_ALPHAS):
                qparams = dict(LGB_BASE); qparams["objective"] = "quantile"
                qparams["metric"] = "quantile"; qparams["alpha"] = alpha
                bq = fit_lgb(X_tr[wet_mask_tr], y_tr[wet_mask_tr],
                             X_va[wet_mask_va], y_va[wet_mask_va], qparams)
                qs[:, k] = bq.predict(X_te, num_iteration=bq.best_iteration)
            # Ensure monotone non-decreasing quantiles per row (fix any crossing)
            qs = np.sort(qs, axis=1)
            qs = np.clip(qs, WET_THRESHOLD_MM, None)
            t_stage2_q = time.time() - t0

            # --- Two-stage point forecast ---
            yhat_two_stage = pi_te * mu_wet_te
            m_ts = point_metrics(yhat_two_stage, y_te)
            # For a point forecast, CRPS == MAE
            m_ts["crps"] = m_ts["mae"]
            m_ts.update({"station": station, "lead": lead, "model": "two_stage_point"})
            rows.append(m_ts)

            # --- Two-stage distributional forecast ---
            crps_ts = float(crps_mixed(pi_te, qs, y_te).mean())
            # Distribution's mean as a separate point forecast (sanity check; usually similar)
            mean_from_dist = (1.0 - pi_te) * 0.0 + pi_te * qs.mean(axis=1)
            m_tsd = point_metrics(mean_from_dist, y_te)
            m_tsd["crps"] = crps_ts
            m_tsd.update({"station": station, "lead": lead, "model": "two_stage_distribution"})
            rows.append(m_tsd)

            # --- Baseline: equal_mean (NWP mean, point) ---
            pm_te = te[PRECIP_COLS].to_numpy(dtype="float64")
            train_mean = float(y_tr.mean())
            eq_mean = np.nanmean(pm_te, axis=1)
            eq_mean = np.where(np.isfinite(eq_mean), eq_mean, train_mean)
            m_eq = point_metrics(eq_mean, y_te)
            m_eq["crps"] = m_eq["mae"]
            m_eq.update({"station": station, "lead": lead, "model": "equal_mean (point)"})
            rows.append(m_eq)

            # --- Baseline: ensemble_7nwp (NWPs as 7-member ensemble distribution) ---
            crps_ens = float(crps_ensemble(pm_te, y_te).mean())
            m_ens = point_metrics(eq_mean, y_te)  # point summary == mean
            m_ens["crps"] = crps_ens
            m_ens.update({"station": station, "lead": lead, "model": "ensemble_7nwp"})
            rows.append(m_ens)

            # --- Baseline: single-stage LightGBM Tweedie on all rows ---
            t0 = time.time()
            ss_params = dict(LGB_BASE); ss_params["objective"] = "tweedie"
            ss_params["metric"] = "tweedie"; ss_params["tweedie_variance_power"] = 1.5
            b_ss = fit_lgb(X_tr, y_tr, X_va, y_va, ss_params)
            yhat_ss = np.clip(b_ss.predict(X_te, num_iteration=b_ss.best_iteration), 0.0, None)
            t_ss = time.time() - t0
            m_ss = point_metrics(yhat_ss, y_te)
            m_ss["crps"] = m_ss["mae"]
            m_ss.update({"station": station, "lead": lead, "model": "lgbm_tweedie_single_stage"})
            rows.append(m_ss)

            print(f"  trained in {t_stage1 + t_stage2_mean + t_stage2_q + t_ss:.1f}s "
                  f"(stage1={t_stage1:.1f}s, stage2_mean={t_stage2_mean:.1f}s, "
                  f"stage2_q={t_stage2_q:.1f}s, single_stage={t_ss:.1f}s)")
            for r in rows[-5:]:
                print(f"   {r['model']:30s} MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
                      f"R2={r['r2']:+.3f}  MAE_wet={r['mae_wet']:.3f}  CRPS={r['crps']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "two_stage_summary.csv", index=False)

    print("\n\n========== AGGREGATE (mean across 3 stations, per lead) ==========")
    agg = (df.groupby(["lead", "model"])
             .agg(mae=("mae", "mean"), rmse=("rmse", "mean"),
                  r2=("r2", "mean"), mae_wet=("mae_wet", "mean"),
                  crps=("crps", "mean"))
             .reset_index()
             .sort_values(["lead", "crps"]))
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    agg.to_csv(OUT_DIR / "two_stage_aggregate.csv", index=False)

    print(f"\nWrote {OUT_DIR/'two_stage_summary.csv'}")
    print(f"Wrote {OUT_DIR/'two_stage_aggregate.csv'}")


if __name__ == "__main__":
    main()
