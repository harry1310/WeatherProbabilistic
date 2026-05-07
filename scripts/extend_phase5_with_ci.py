"""Phase 5 sibling of `extend_phase4_with_ci.py` — Phase 5's posterior is
a SINGLE lead-as-feature partial-pool fit (vs Phase 4's three independent
per-lead fits), so the live-bundle layout is correspondingly simpler:
one posterior NetCDF, one StandardScaler that includes the lead column.

What this writes
----------------
1. ``reports/phase5_artefacts/live_bundle/{scaler.pkl, metadata.json}``
   — the same format as Phase 4's live_bundle, with an extra
   ``lead_feature_index`` field marking which feature column carries
   the standardised lead value (so the live predict path can replace
   that one column when scoring at a new lead without re-running the
   full StandardScaler pipeline).
2. ``reports/phase5_artefacts/predictions/test_predictions_with_ci.parquet``
   — augments the existing test predictions with mean / std / quantiles
   / ci80_width / ci90_width columns. Same shape as Phase 4's
   ``lead_{N}h_with_ci.parquet`` but one file (no per-lead split).

Reconciliation check: posterior-mean from the new summary should match
the test-predictions parquet's ``p_wet`` column to within float noise.

Run with:
    PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -u scripts/extend_phase5_with_ci.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import arviz as az  # noqa: E402

from src.data import MODELS_NO_UKMO, prepare_phase3_dataset  # noqa: E402
from src.models.phase2_partial_pooling import (  # noqa: E402
    PartialPoolingFit, predict_partial_pooling_summary,
)

ART_DIR = ROOT / "reports" / "phase5_artefacts"
POSTERIOR_DIR = ART_DIR / "posteriors"
LIVE_BUNDLE_DIR = ART_DIR / "live_bundle"
PREDICTIONS_DIR = ART_DIR / "predictions"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def brier(prob: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((prob - truth) ** 2))


def save_live_bundle(ds) -> None:
    """Persist scaler + metadata for the live predict path. Mirrors Phase 4
    save_live_bundle but adds ``lead_feature_index`` so the live predict
    knows which column is the standardised lead."""
    import json
    import pickle

    LIVE_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_BUNDLE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(ds.scaler, f)

    if "lead" not in ds.feature_names:
        raise RuntimeError(
            "Phase 5 dataset must include 'lead' as a feature — pass "
            "lead_as_feature=True to prepare_phase3_dataset."
        )
    lead_feature_index = ds.feature_names.index("lead")

    metadata = {
        "phase": "5",
        "feature_names": list(ds.feature_names),
        "station_codes": list(ds.station_codes),
        "station_full_names": list(ds.station_full_names),
        "lead_hours": [int(l) for l in ds.lead_hours],
        "lead_feature_index": lead_feature_index,
        "scaler_mean": [float(x) for x in ds.scaler.mean_.tolist()],
        "scaler_scale": [float(x) for x in ds.scaler.scale_.tolist()],
        "scaler_var": [float(x) for x in ds.scaler.var_.tolist()],
        "scaler_n_samples_seen": int(ds.scaler.n_samples_seen_),
    }
    (LIVE_BUNDLE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  wrote live bundle (scaler + metadata) to {LIVE_BUNDLE_DIR}")
    print(f"  lead is feature index {lead_feature_index} "
          f"(standardised mean={ds.scaler.mean_[lead_feature_index]:.3f}, "
          f"scale={ds.scaler.scale_[lead_feature_index]:.3f})")


def load_fit_from_disk(feature_names: list[str], station_codes: list[str]) -> PartialPoolingFit:
    nc_path = POSTERIOR_DIR / "lead_feature.nc"
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Phase 5 posterior not found at {nc_path}. "
            f"Run scripts/run_phase5_bayesian.py first."
        )
    return PartialPoolingFit(
        idata=az.from_netcdf(nc_path),
        feature_names=feature_names,
        station_codes=station_codes,
    )


def main() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Loading Phase 3 dataset (5-model, lead-as-feature)")
    ds = prepare_phase3_dataset(
        models=MODELS_NO_UKMO, lead_as_feature=True, verbose=False,
    )
    print(f"  test rows: {len(ds.X_test):,}  features: {len(ds.feature_names)}")
    save_live_bundle(ds)

    print(f"[{time.strftime('%H:%M:%S')}] Loading saved posterior from {POSTERIOR_DIR}")
    fit = load_fit_from_disk(ds.feature_names, ds.station_codes)

    print(f"[{time.strftime('%H:%M:%S')}] Computing per-row posterior summary")
    t0 = time.time()
    summary = predict_partial_pooling_summary(
        fit, ds.X_test_s, ds.station_idx_test, quantiles=QUANTILES,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    # Reconciliation: summary["mean"] should match the existing test
    # predictions parquet's p_wet column to within float noise.
    existing_path = PREDICTIONS_DIR / "test_predictions.parquet"
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        max_diff = float(np.max(np.abs(existing["p_wet"].to_numpy() - summary["mean"])))
        print(f"  posterior-mean reconciliation: max |Δ| vs existing parquet = {max_diff:.2e}")

    # Augmented predictions with full CI columns.
    out = pd.DataFrame({
        "valid_time": pd.to_datetime(ds.valid_time_test.values),
        "station": [ds.station_codes[i] for i in ds.station_idx_test],
        "lead": [ds.lead_hours[i] for i in ds.lead_idx_test],
        "p_wet_mean": summary["mean"],
        "p_wet_std": summary["std"],
        "observed_wet": ds.y_test.values.astype("int8"),
    })
    for q in QUANTILES:
        out[f"p_wet_q{q:g}"] = summary[f"q{q:g}"]
    out["ci80_width"] = summary["q0.9"] - summary["q0.1"]
    out["ci90_width"] = summary["q0.95"] - summary["q0.05"]
    out_path = PREDICTIONS_DIR / "test_predictions_with_ci.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  wrote {len(out):,} rows to {out_path}")

    # Per-(station, lead) Brier × CI quartile.
    print(f"\n[{time.strftime('%H:%M:%S')}] Narrow-CI vs wide-CI Brier check (per lead)")
    print(f"{'lead':>5} {'quartile':>10} {'n':>6} {'mean ci80':>10} {'Brier':>8} {'p_wet mean':>11}")
    for lead in ds.lead_hours:
        sub_lead = out[out["lead"] == lead].copy()
        sub_lead["ci80_quartile"] = pd.qcut(sub_lead["ci80_width"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        for q, sub in sub_lead.groupby("ci80_quartile", observed=True):
            b = brier(sub["p_wet_mean"].to_numpy(), sub["observed_wet"].to_numpy().astype("float64"))
            print(f"{lead:>5} {q:>10} {len(sub):>6} {sub['ci80_width'].mean():>10.4f} "
                  f"{b:>8.4f} {sub['p_wet_mean'].mean():>11.4f}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
