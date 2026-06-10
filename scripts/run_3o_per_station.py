"""Test 2: per-station 3o LightGBM vs pooled 3o LightGBM, both via Python
LightGBM on the existing rich-oro dumps.

Production 3o is ML.NET LightGBM trained pooled across the 4 Bonehill stations.
Today's per-station 4a result suggested pooling-vs-per-station is the dominant
architectural lever — this script answers the same question for 3o.

Important caveat: this uses Python lightgbm 4.x, NOT ML.NET LightGBM. Absolute
Brier here won't match production 3o's numbers; what matters is the
per-station-vs-pooled DELTA within this script. Same package, same hypers,
same data — just the train-row scope differs.

Hypers chosen to roughly match the ML.NET LightGbm defaults the C# bake-off
uses (Phase3cOroBakeoffCommand.PrecipOccurrenceTrainer.Hyperparameters):
  num_leaves=31, learning_rate=0.2, min_data_in_leaf=20, n_estimators=100.
Binary log-loss objective with Platt-style sigmoid output.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the dump loader + time_split from the BART runner so the test set is
# bit-identical to what the rich-pooled-terrain BART arms saw earlier today.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase6_dbarts_pooled_oro import load_dump_wide, STATIONS  # noqa: E402
from _shared import time_split  # noqa: E402
import lightgbm as lgb  # noqa: E402

LEADS = [24, 48, 72]
OUT_DIR = ROOT / "reports" / "pooled_oro_4a_bakeoff_2026-05-29"

LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    num_leaves=31,
    learning_rate=0.2,
    min_data_in_leaf=20,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    verbosity=-1,
)
N_ESTIMATORS = 100


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) -
                          np.asarray(y, dtype=np.float64)) ** 2))


def fit_and_score(X_train, y_train, X_val, y_val, X_test, y_test) -> tuple[float, int]:
    train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, free_raw_data=False)
    model = lgb.train(
        LGB_PARAMS, train_ds, num_boost_round=N_ESTIMATORS,
        valid_sets=[val_ds], callbacks=[lgb.early_stopping(10, verbose=False)],
    )
    p_test = model.predict(X_test, num_iteration=model.best_iteration)
    return brier(p_test, y_test), int(model.best_iteration or N_ESTIMATORS)


def main() -> None:
    rows = []
    for lead in LEADS:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Lead {lead}h ===", flush=True)

        # Build per-station train/val/test wide frames; same split as the BART runs.
        per_station = {}
        feat_names = None
        for slug in STATIONS:
            wide, names, _sidecar = load_dump_wide(slug, lead, "rich-oro")
            if feat_names is None:
                feat_names = names
            train_df, val_df, test_df = time_split(wide)
            per_station[slug] = (train_df, val_df, test_df)
            print(f"  {slug}: train={len(train_df)} val={len(val_df)} test={len(test_df)}",
                  flush=True)

        # --- POOLED: one LightGBM fit on all 4 stations combined ---
        pooled_train = pd.concat([per_station[s][0] for s in STATIONS], ignore_index=True)
        pooled_val = pd.concat([per_station[s][1] for s in STATIONS], ignore_index=True)
        X_train = pooled_train[feat_names].to_numpy(dtype="float64")
        y_train = pooled_train["wet"].to_numpy(dtype="float64")
        X_val = pooled_val[feat_names].to_numpy(dtype="float64")
        y_val = pooled_val["wet"].to_numpy(dtype="float64")
        # impute pooled-train median for NaN (LightGBM handles natively, but
        # match scaler conventions used elsewhere — actually skip imputation,
        # let LightGBM's missing-value handling do its thing).
        train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, free_raw_data=False)
        pooled_model = lgb.train(
            LGB_PARAMS, train_ds, num_boost_round=N_ESTIMATORS,
            valid_sets=[val_ds], callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        pooled_best_iter = int(pooled_model.best_iteration or N_ESTIMATORS)
        print(f"  pooled fit done (best_iter={pooled_best_iter})", flush=True)

        # Score pooled per-station on identical test rows
        for slug in STATIONS:
            test_df = per_station[slug][2]
            X_test = test_df[feat_names].to_numpy(dtype="float64")
            y_test = test_df["wet"].to_numpy(dtype="float64")
            p_test = pooled_model.predict(X_test, num_iteration=pooled_best_iter)
            b = brier(p_test, y_test)
            rows.append({"lead": lead, "station": slug, "mode": "pooled",
                         "brier": round(b, 4), "n_test": int(len(y_test))})
            print(f"    pooled {slug}: Brier {b:.4f}", flush=True)

        # --- PER-STATION: one LightGBM fit per station, only that station's data ---
        for slug in STATIONS:
            train_df, val_df, test_df = per_station[slug]
            X_tr = train_df[feat_names].to_numpy(dtype="float64")
            y_tr = train_df["wet"].to_numpy(dtype="float64")
            X_va = val_df[feat_names].to_numpy(dtype="float64")
            y_va = val_df["wet"].to_numpy(dtype="float64")
            X_te = test_df[feat_names].to_numpy(dtype="float64")
            y_te = test_df["wet"].to_numpy(dtype="float64")
            b, best_iter = fit_and_score(X_tr, y_tr, X_va, y_va, X_te, y_te)
            rows.append({"lead": lead, "station": slug, "mode": "per-station",
                         "brier": round(b, 4), "n_test": int(len(y_te))})
            print(f"    per-station {slug}: Brier {b:.4f} (best_iter={best_iter})", flush=True)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "test2_3o_per_station_vs_pooled.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Pivot for a tidy summary table
    pivot = df.pivot_table(index=["station", "lead"], columns="mode", values="brier").reset_index()
    pivot["delta_brier"] = pivot["per-station"] - pivot["pooled"]
    pivot["delta_pct"] = (pivot["delta_brier"] / pivot["pooled"] * 100).round(2)
    print("\nPer-cell pooled vs per-station:")
    print(pivot.to_string(index=False))

    # Aggregate
    agg = pivot.groupby("lead", as_index=False).agg(
        mean_pooled=("pooled", "mean"),
        mean_per_station=("per-station", "mean"),
        mean_delta_pct=("delta_pct", "mean"),
    )
    print("\nAggregate per lead (mean across 4 stations):")
    print(agg.round(4).to_string(index=False))

    overall = pd.DataFrame([{
        "mean_pooled":      pivot["pooled"].mean(),
        "mean_per_station": pivot["per-station"].mean(),
        "mean_delta_pct":   pivot["delta_pct"].mean(),
    }]).round(4)
    print("\nOverall (mean across all stations × leads):")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
