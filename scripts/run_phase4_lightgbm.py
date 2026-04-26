"""Phase 4 stripped 7-feature LightGBM trainer.

Trains one LightGBM binary classifier per lead {24, 48, 72}h on the 5-model
Phase3Dataset (5 precip + hour_sin + hour_cos features — identical to the
Bayesian 5-model variant's input). Evaluates on the held-out test split and
saves per-row test predictions to
``reports/phase4_artefacts/lightgbm_predictions/stripped_7feature/lead_{N}h.parquet``
with columns ``valid_time, station, lead, p_wet, observed_wet``.

Hyperparameters mirror WeatherBlend's 3a-lean: regression tree with binary
cross-entropy, 500 iterations max, early stopping on val log-loss, 31 leaves,
LR 0.05, min 20 per leaf, seed 42.

Run with:

    .venv/Scripts/python.exe -u scripts/run_phase4_lightgbm.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import MODELS_NO_UKMO, prepare_phase3_dataset  # noqa: E402

OUT_DIR = ROOT / "reports" / "phase4_artefacts" / "lightgbm_predictions" / "stripped_7feature"

# LightGBM hyperparameters mirroring WeatherBlend 3a-lean — see
# WeatherBlend/src/WeatherBlend/Train/PrecipOccurrenceTrainer.cs.
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
    "num_threads": 0,  # auto
}
NUM_ITERATIONS = 500
EARLY_STOPPING = 30


def _per_lead_split(ds, lead_idx: int):
    """Filter the dataset's train/test rows to a single lead, then carve val out
    of the tail of train for early stopping."""
    tr_mask = ds.lead_idx_train == lead_idx
    te_mask = ds.lead_idx_test == lead_idx

    X_tr_full = ds.X_train_s[tr_mask]
    y_tr_full = ds.y_train.values[tr_mask]
    X_te = ds.X_test_s[te_mask]
    y_te = ds.y_test.values[te_mask]
    valid_te = ds.valid_time_test.values[te_mask]
    station_idx_te = ds.station_idx_test[te_mask]

    # Carve val out of the chronological tail of the train slice (matches the
    # spirit of WeatherBlend's 70/15/15 chronological split — we do the
    # equivalent within the per-lead train slice).
    cut = int(len(X_tr_full) * (15.0 / 80.0))  # last 15/80 of train → val
    cut = max(cut, 10)
    X_tr, X_val = X_tr_full[:-cut], X_tr_full[-cut:]
    y_tr, y_val = y_tr_full[:-cut], y_tr_full[-cut:]

    return X_tr, y_tr, X_val, y_val, X_te, y_te, valid_te, station_idx_te


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 5-model Phase3Dataset (UKMO removed)...")
    ds = prepare_phase3_dataset(models=MODELS_NO_UKMO, verbose=False)
    print(f"  train rows: {len(ds.X_train):,}  test rows: {len(ds.X_test):,}")
    print(f"  features ({len(ds.feature_names)}): {ds.feature_names}")
    print(f"  stations: {ds.station_codes}")
    print(f"  leads: {ds.lead_hours}")

    summary_rows = []
    for li, lead in enumerate(ds.lead_hours):
        print(f"\n--- Lead {lead}h ---")
        X_tr, y_tr, X_val, y_val, X_te, y_te, valid_te, station_idx_te = _per_lead_split(ds, li)
        print(f"  train: {len(X_tr):,}  val: {len(X_val):,}  test: {len(X_te):,}  wet train: {y_tr.mean():.3f}")

        train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=ds.feature_names)
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=ds.feature_names, reference=train_set)

        t0 = time.time()
        booster = lgb.train(
            LGB_PARAMS,
            train_set,
            num_boost_round=NUM_ITERATIONS,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        elapsed = time.time() - t0
        print(f"  trained in {elapsed:.1f}s, best iter = {booster.best_iteration}")

        p_test = booster.predict(X_te, num_iteration=booster.best_iteration)
        brier = brier_score_loss(y_te, p_test)
        ll = log_loss(y_te, np.clip(p_test, 1e-9, 1 - 1e-9))
        clim = y_te.mean()
        brier_clim = float(np.mean((clim - y_te) ** 2))
        bss = 1.0 - brier / brier_clim if brier_clim > 0 else float("nan")
        print(f"  test Brier={brier:.4f}  BSS={bss:+.3f}  log_loss={ll:.4f}  n={len(y_te):,}")

        # Per-row test predictions to parquet for the comparison step.
        out = pd.DataFrame({
            "valid_time": pd.to_datetime(valid_te),
            "station": [ds.station_codes[i] for i in station_idx_te],
            "lead": lead,
            "p_wet": p_test,
            "observed_wet": y_te.astype("int8"),
        })
        out_path = OUT_DIR / f"lead_{lead}h.parquet"
        out.to_parquet(out_path, index=False)
        print(f"  wrote {len(out):,} rows to {out_path}")

        summary_rows.append({
            "lead": lead, "n_test": len(y_te), "brier": brier, "bss": bss, "log_loss": ll,
        })

    # Per-station × lead breakdown
    print("\n--- Per-(station, lead) test metrics ---")
    rows = []
    for li, lead in enumerate(ds.lead_hours):
        df = pd.read_parquet(OUT_DIR / f"lead_{lead}h.parquet")
        for st in ds.station_codes:
            sub = df[df["station"] == st]
            if len(sub) == 0:
                continue
            b = brier_score_loss(sub["observed_wet"], sub["p_wet"])
            rows.append({"station": st, "lead": lead, "n": len(sub), "brier": b})
    perdf = pd.DataFrame(rows)
    print(perdf.to_string(index=False))
    perdf.to_csv(OUT_DIR / "per_cell_brier.csv", index=False)

    print("\nLightGBM stripped 7-feature done.")
    print(f"Predictions saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
