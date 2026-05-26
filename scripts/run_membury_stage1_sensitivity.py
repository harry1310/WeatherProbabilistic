"""How much does stage 1 P(wet) choice matter for the two-stage CRPS?

Same NGBoost-LogNormal stage 2 (current champion). Four stage-1 variants:
  * lgbm_raw       -- LightGBM binary, no calibration (current baseline; ~3a-equivalent on
                      the lean 15-feature set)
  * lgbm_pav       -- Same LightGBM + PAV (isotonic) calibration fitted on val
  * oracle         -- True wet/dry from observation: pi = 1{y >= 0.1}.
                      Lower bound on CRPS achievable with a perfect classifier.
  * climatology    -- Constant pi = train-set base rate. Upper bound on CRPS
                      from a totally uninformative classifier.

If oracle - lgbm_raw is small (e.g. < 0.01 CRPS), stage 1 is essentially saturated and
swapping for 3c / 3e / 4a is not worth the effort. If oracle - lgbm_raw is big,
there is genuine room to improve by picking a better stage 1.

Reports Brier score of each stage 1 too, so we can see whether the calibrators
move the needle.

Run:
    .venv/Scripts/python.exe -u scripts/run_membury_stage1_sensitivity.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import lognorm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

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
# Features (same as before)
# ---------------------------------------------------------------------------

def build_features(df):
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


def time_split(df, train_frac=0.70, val_frac=0.15):
    n = len(df)
    a = int(n * train_frac); b = a + int(n * val_frac)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def impute(X):
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


def crps_mixed(pi, quantiles, y):
    K = quantiles.shape[1]
    w_dry = 1.0 - pi
    w_wet = pi / K
    term1 = w_dry * y + w_wet * np.abs(quantiles - y[:, None]).sum(axis=1)
    cross_0k = 2.0 * w_dry * w_wet * quantiles.sum(axis=1)
    pairwise = np.abs(quantiles[:, :, None] - quantiles[:, None, :]).sum(axis=(1, 2))
    cross_kl = (w_wet ** 2) * pairwise
    return term1 - 0.5 * (cross_0k + cross_kl)


def fit_pwet(X_tr, y_tr, X_va, y_va):
    p = dict(LGB_BASE); p["objective"] = "binary"; p["metric"] = "binary_logloss"
    return lgb.train(p, lgb.Dataset(X_tr, label=y_tr), num_boost_round=NUM_ITERS,
                     valid_sets=[lgb.Dataset(X_va, label=y_va)],
                     valid_names=["val"],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def main():
    if not CACHE.exists():
        raise SystemExit(f"Cache not found: {CACHE}.")
    print(f"[cache] loading {CACHE}")
    df_all = pd.read_parquet(CACHE)
    df_all, FEATS = build_features(df_all)

    rows = []

    for station in STATIONS:
        for lead in LEADS:
            sub = (df_all[(df_all["station"] == station) & (df_all["lead"] == lead)]
                   .sort_values("ValidTimeUtc").reset_index(drop=True))
            tr, va, te = time_split(sub)
            X_tr = tr[FEATS].to_numpy("float64"); X_va = va[FEATS].to_numpy("float64"); X_te = te[FEATS].to_numpy("float64")
            y_tr = tr["precip_mm_hour"].to_numpy("float64")
            y_va = va["precip_mm_hour"].to_numpy("float64")
            y_te = te["precip_mm_hour"].to_numpy("float64")
            wet_tr = (y_tr >= WET_THRESHOLD_MM).astype("int8")
            wet_va = (y_va >= WET_THRESHOLD_MM).astype("int8")
            wet_te = (y_te >= WET_THRESHOLD_MM).astype("int8")
            wmask_tr = wet_tr.astype(bool); wmask_va = wet_va.astype(bool)
            X_tr_im = impute(X_tr); X_va_im = impute(X_va); X_te_im = impute(X_te)

            base_rate = float(wet_tr.mean())
            print(f"\n=== {station} | lead {lead}h ===  base_rate={base_rate:.3f}  "
                  f"n_test={len(y_te):,}")

            # Stage 2: NGBoost-LogNormal on wet rows (same for all stage-1 variants)
            ngb = NGBRegressor(Dist=NGBLogNormal, Score=LogScore, **NGB_BASE)
            ngb.fit(X_tr_im[wmask_tr], y_tr[wmask_tr],
                    X_val=X_va_im[wmask_va], Y_val=y_va[wmask_va],
                    early_stopping_rounds=EARLY_STOP)
            dist = ngb.pred_dist(X_te_im, max_iter=ngb.best_val_loss_itr)
            s = np.asarray(dist.params["s"]); scale = np.asarray(dist.params["scale"])
            qs = np.empty((len(X_te), len(QUANTILE_ALPHAS_EVAL)), dtype="float64")
            for k, alpha in enumerate(QUANTILE_ALPHAS_EVAL):
                qs[:, k] = lognorm.ppf(alpha, s=s, scale=scale)
            qs = np.clip(qs, WET_THRESHOLD_MM, None)

            # Stage 1 variant: lgbm_raw
            b = fit_pwet(X_tr, wet_tr, X_va, wet_va)
            pi_lgbm = b.predict(X_te, num_iteration=b.best_iteration)
            pi_lgbm_val = b.predict(X_va, num_iteration=b.best_iteration)

            # Stage 1 variant: lgbm + PAV calibration
            iso = IsotonicRegression(out_of_bounds="clip").fit(pi_lgbm_val, wet_va)
            pi_pav = iso.transform(pi_lgbm)

            # Stage 1 variant: oracle (true wet/dry)
            pi_oracle = wet_te.astype("float64")

            # Stage 1 variant: climatology (constant base rate)
            pi_clim = np.full_like(pi_lgbm, base_rate)

            for tag, pi in [("lgbm_raw", pi_lgbm),
                            ("lgbm_pav", pi_pav),
                            ("oracle",   pi_oracle),
                            ("climatology", pi_clim)]:
                crps = float(crps_mixed(pi, qs, y_te).mean())
                # Brier on the observed wet/dry label
                brier = float(brier_score_loss(wet_te, np.clip(pi, 0, 1)))
                rows.append({"station": station, "lead": lead, "stage1": tag,
                             "brier": brier, "crps": crps, "n": int(len(y_te))})
                print(f"  {tag:12s}  Brier={brier:.4f}  CRPS={crps:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "stage1_sensitivity_summary.csv", index=False)

    print("\n\n========== AGGREGATE (mean across 3 stations, per lead) ==========")
    agg = (df.groupby(["lead", "stage1"])
             .agg(brier=("brier", "mean"), crps=("crps", "mean"))
             .reset_index().sort_values(["lead", "crps"]))
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n========== STAGE-1 SENSITIVITY DELTAS (vs lgbm_raw, per lead) ==========")
    pivot = agg.pivot(index="lead", columns="stage1", values="crps")
    for lead in LEADS:
        baseline = pivot.loc[lead, "lgbm_raw"]
        deltas = {k: (pivot.loc[lead, k] - baseline) for k in pivot.columns}
        delta_pct = {k: 100 * v / baseline for k, v in deltas.items()}
        line = f"  lead={lead}h baseline lgbm_raw CRPS={baseline:.4f}  "
        for k in ["lgbm_pav", "oracle", "climatology"]:
            line += f"| {k}: {deltas[k]:+.4f} ({delta_pct[k]:+.1f}%)  "
        print(line)

    agg.to_csv(OUT_DIR / "stage1_sensitivity_aggregate.csv", index=False)
    print(f"\nWrote {OUT_DIR/'stage1_sensitivity_summary.csv'}")
    print(f"Wrote {OUT_DIR/'stage1_sensitivity_aggregate.csv'}")


if __name__ == "__main__":
    main()
