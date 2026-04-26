"""Phase 4 native 27-feature LightGBM trainer (supporting context).

Replicates WeatherBlend's PrecipFeatureBuilder feature set (5 per-model
precip + 5 per-model prob (all-NaN) + 4 ensemble spread + 7 meteo covariates
+ 4 calendar = 25 features for the 5-model variant) but evaluates on the
**identical** Phase3Dataset test rows used by both the stripped LightGBM
and the Bayesian 5-model. This makes (a) supporting context truly
apples-to-apples with the headline (b) comparison.

Why 25 not 27: the 5-model variant drops UKMO, so the 2 UKMO columns
(precip_ukmo, prob_ukmo) WeatherBlend keeps as always-NaN are simply not
present here. LightGBM's behaviour is identical either way (constant-NaN
features carry zero information gain).

Same hyperparameters as the stripped variant. Different feature inputs.

Run with:

    .venv/Scripts/python.exe -u scripts/run_phase4_lightgbm_native.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    LOCATION, MODELS_NO_UKMO, PHASE3_LEAD_HOURS, STATIONS, WET_THRESHOLD_MM,
    _load_all_forecasts_multi_lead_rich, _load_rainfall_truth,
)

OUT_DIR = ROOT / "reports" / "phase4_artefacts" / "lightgbm_predictions" / "native_25feature"

# Mirrors run_phase4_lightgbm.py — same hyperparameters as stripped, only
# difference is the feature inputs.
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "verbose": -1,
    "seed": 42,
    "num_threads": 0,
}
NUM_ITERATIONS = 500
EARLY_STOPPING = 30


def build_native_dataset(test_fraction: float = 0.2, verbose: bool = True):
    """Build a Phase3Dataset-shape with 25 native features, identical splits to
    the stripped Phase4 variant."""
    forecasts, precip_cols = _load_all_forecasts_multi_lead_rich(
        PHASE3_LEAD_HOURS, verbose=verbose, models=MODELS_NO_UKMO,
    )

    # Feature list: per-model precip + per-model prob + 4 ensemble spread +
    # 7 covariates + 4 calendar = 25 features.
    prob_cols = [f"prob_{m}" for m in MODELS_NO_UKMO]
    spread_cols = ["precip_mean", "precip_std", "precip_max", "precip_agreement_wet01"]
    covariate_cols = ["rh_mean", "dew_depression_mean",
                      "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean",
                      "cape_mean", "wind_speed_mean"]
    calendar_cols = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    feature_names = precip_cols + prob_cols + spread_cols + covariate_cols + calendar_cols

    train_frames, test_frames = [], []
    station_codes = [code for _, code in STATIONS]
    lead_to_idx = {l: i for i, l in enumerate(PHASE3_LEAD_HOURS)}

    for s_idx, (full_name, code) in enumerate(STATIONS):
        if verbose:
            print(f"\nLoading rainfall truth: {full_name}")
        truth = _load_rainfall_truth(full_name)
        df = truth[["ValidTimeUtc", "observed_wet"]].merge(
            forecasts, on="ValidTimeUtc", how="inner",
        )
        # Drop rows with any null precip — same row-drop policy as Phase 3.
        before = len(df)
        df = df.dropna(subset=precip_cols).copy()
        if verbose:
            print(f"  rows after precip-not-null: {len(df):,}/{before:,}")
        # Fill NaN in covariates with column mean (rare; only some cape/cloud
        # values may have NaN in a tiny fraction of rows).
        for c in covariate_cols + spread_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mean())
        # prob_* are all-NaN — fill with column mean which is NaN; LightGBM
        # treats remaining NaN as missing (same as constant-NaN behaviour).

        df = df.sort_values(["ValidTimeUtc", "LeadHours"]).reset_index(drop=True)
        df["station_idx"] = s_idx
        df["station"] = code
        df["lead_idx"] = df["LeadHours"].map(lead_to_idx).astype("int64")

        unique_vt = np.sort(df["ValidTimeUtc"].unique())
        cut_pos = int(len(unique_vt) * (1 - test_fraction))
        cut_vt = unique_vt[cut_pos]
        train_frames.append(df[df["ValidTimeUtc"] < cut_vt].copy())
        test_frames.append(df[df["ValidTimeUtc"] >= cut_vt].copy())

        if verbose:
            print(f"  {code}: train {len(train_frames[-1]):,}  test {len(test_frames[-1]):,}")

    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)

    X_train = train[feature_names].to_numpy(dtype="float64")
    X_test = test[feature_names].to_numpy(dtype="float64")
    # Pooled standardisation on combined training set (matches Phase3Dataset)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    return {
        "feature_names": feature_names,
        "X_train_s": X_train_s,
        "X_test_s": X_test_s,
        "y_train": train["observed_wet"].to_numpy(),
        "y_test": test["observed_wet"].to_numpy(),
        "valid_time_test": train["ValidTimeUtc"], # unused but keep symmetric
        "valid_time_test_arr": test["ValidTimeUtc"].to_numpy(),
        "station_idx_train": train["station_idx"].to_numpy(),
        "station_idx_test": test["station_idx"].to_numpy(),
        "lead_idx_train": train["lead_idx"].to_numpy(),
        "lead_idx_test": test["lead_idx"].to_numpy(),
        "station_codes": station_codes,
        "lead_hours": list(PHASE3_LEAD_HOURS),
    }


def _per_lead_split(ds, lead_idx: int):
    tr_mask = ds["lead_idx_train"] == lead_idx
    te_mask = ds["lead_idx_test"] == lead_idx
    X_tr_full = ds["X_train_s"][tr_mask]
    y_tr_full = ds["y_train"][tr_mask]
    X_te = ds["X_test_s"][te_mask]
    y_te = ds["y_test"][te_mask]
    valid_te = ds["valid_time_test_arr"][te_mask]
    station_idx_te = ds["station_idx_test"][te_mask]
    cut = int(len(X_tr_full) * (15.0 / 80.0))
    cut = max(cut, 10)
    X_tr, X_val = X_tr_full[:-cut], X_tr_full[-cut:]
    y_tr, y_val = y_tr_full[:-cut], y_tr_full[-cut:]
    return X_tr, y_tr, X_val, y_val, X_te, y_te, valid_te, station_idx_te


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 5-model native 25-feature dataset (Phase 4 supporting context)...")
    ds = build_native_dataset(verbose=True)
    print(f"\nfeatures ({len(ds['feature_names'])}): {ds['feature_names']}")
    print(f"train: {len(ds['X_train_s']):,}  test: {len(ds['X_test_s']):,}")

    for li, lead in enumerate(ds["lead_hours"]):
        print(f"\n--- Lead {lead}h ---")
        X_tr, y_tr, X_val, y_val, X_te, y_te, valid_te, station_idx_te = _per_lead_split(ds, li)
        print(f"  train: {len(X_tr):,}  val: {len(X_val):,}  test: {len(X_te):,}  wet train: {y_tr.mean():.3f}")

        train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=ds["feature_names"])
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=ds["feature_names"], reference=train_set)
        t0 = time.time()
        booster = lgb.train(
            LGB_PARAMS, train_set,
            num_boost_round=NUM_ITERATIONS,
            valid_sets=[val_set], valid_names=["val"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        print(f"  trained in {time.time() - t0:.1f}s, best iter = {booster.best_iteration}")
        p_test = booster.predict(X_te, num_iteration=booster.best_iteration)
        brier = brier_score_loss(y_te, p_test)
        ll = log_loss(y_te, np.clip(p_test, 1e-9, 1 - 1e-9))
        clim = y_te.mean()
        bss = 1.0 - brier / float(np.mean((clim - y_te) ** 2))
        print(f"  test Brier={brier:.4f}  BSS={bss:+.3f}  log_loss={ll:.4f}  n={len(y_te):,}")

        out = pd.DataFrame({
            "valid_time": pd.to_datetime(valid_te),
            "station": [ds["station_codes"][i] for i in station_idx_te],
            "lead": lead,
            "p_wet": p_test,
            "observed_wet": y_te.astype("int8"),
        })
        out.to_parquet(OUT_DIR / f"lead_{lead}h.parquet", index=False)
        print(f"  wrote {len(out):,} rows to {OUT_DIR / f'lead_{lead}h.parquet'}")

    print(f"\nNative 25-feature done. Predictions saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
