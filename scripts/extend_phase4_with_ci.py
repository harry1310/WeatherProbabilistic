"""Re-predict Phase 4 Bayesian on the same test slice but emit posterior
summary stats (mean, std, quantiles) per row instead of just the posterior
mean.

Why this exists
---------------
Phase 4's `run_phase4_bayesian.py` already trained per-lead Bayesian models
and saved their posteriors as NetCDF + per-row mean P(wet) parquets. The
predictions parquet has columns
    valid_time, station, lead, p_wet, observed_wet
which is enough for Brier scoring but throws away the *posterior width* —
the per-row credible interval that's the actual confidence signal we want
to ship downstream (Option 2 of the WeatherBlend uncertainty discussion).

This script reloads the saved posteriors (no refit), re-runs prediction
on the same test rows with the new `predict_per_lead_summary` helper that
keeps the full posterior over (chain × draw), and writes an augmented
parquet at
    reports/phase4_artefacts/bayesian_predictions/lead_{N}h_with_ci.parquet
containing additional columns:
    p_wet_mean, p_wet_std,
    p_wet_q0.05, p_wet_q0.1, p_wet_q0.5, p_wet_q0.9, p_wet_q0.95,
    ci80_width  (= q0.9 - q0.1),
    ci90_width  (= q0.95 - q0.05).

Verifies the Phase 4 finding "narrow-CI rows have lower Brier than wide-CI
rows" by binning the test set by ci80 width quartile and reporting Brier
per quartile.

Run with:

    .venv/Scripts/python.exe -u scripts/extend_phase4_with_ci.py
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
from src.models.phase2_partial_pooling import PartialPoolingFit  # noqa: E402
from src.models.phase3a_per_lead import PerLeadFit, predict_per_lead_summary  # noqa: E402

ART_DIR = ROOT / "reports" / "phase4_artefacts" / "bayesian_predictions"
POSTERIOR_DIR = ART_DIR / "posteriors"
LIVE_BUNDLE_DIR = ART_DIR / "live_bundle"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def load_per_lead_fit_from_disk(
    lead_hours: list[int],
    station_codes: list[str],
    feature_names: list[str],
) -> PerLeadFit:
    """Reconstruct a PerLeadFit from the NetCDF files Phase 4 saved.

    Per-lead NetCDFs at `posteriors/lead_{N}h.nc` were written by
    `run_phase4_bayesian.py`. Each is a full ArviZ InferenceData object
    with the variables predict_partial_pooling_summary needs
    (`intercept_s`, `beta_s`).
    """
    fits: dict[int, PartialPoolingFit] = {}
    for lh in lead_hours:
        path = POSTERIOR_DIR / f"lead_{lh}h.nc"
        if not path.exists():
            raise FileNotFoundError(
                f"Phase 4 posterior not found at {path}. "
                f"Run scripts/run_phase4_bayesian.py first to generate it."
            )
        idata = az.from_netcdf(path)
        fits[lh] = PartialPoolingFit(
            idata=idata,
            feature_names=feature_names,
            station_codes=station_codes,
        )
    return PerLeadFit(
        fits_by_lead=fits,
        feature_names=feature_names,
        station_codes=station_codes,
        lead_hours=list(lead_hours),
    )


def brier(prob: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((prob - truth) ** 2))


def save_live_bundle(ds) -> None:
    """Persist everything live-predict needs, alongside the saved posteriors.

    The live predict path mustn't re-run `prepare_phase3_dataset` (~40s and
    loads ~2y of historical parquets just to refit a StandardScaler). Saving
    the scaler params + feature names + station codes + lead hours once
    here lets live predict bootstrap in well under a second.
    """
    import json
    import pickle

    LIVE_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_BUNDLE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(ds.scaler, f)

    metadata = {
        "feature_names": list(ds.feature_names),
        "station_codes": list(ds.station_codes),
        "station_full_names": list(ds.station_full_names),
        "lead_hours": [int(l) for l in ds.lead_hours],
        # Scaler parameters in JSON form too, so a non-Python consumer (or a
        # reproducibility audit) can verify the standardisation without
        # unpickling. Pickle is the canonical loader; JSON is a sanity check.
        "scaler_mean": [float(x) for x in ds.scaler.mean_.tolist()],
        "scaler_scale": [float(x) for x in ds.scaler.scale_.tolist()],
        "scaler_var": [float(x) for x in ds.scaler.var_.tolist()],
        "scaler_n_samples_seen": int(ds.scaler.n_samples_seen_),
    }
    (LIVE_BUNDLE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  wrote live bundle (scaler + metadata) to {LIVE_BUNDLE_DIR}")


def main() -> None:
    print(f"[{time.strftime('%H:%M:%S')}] Loading Phase 3 dataset (5-model variant)")
    ds = prepare_phase3_dataset(models=MODELS_NO_UKMO, verbose=False)
    print(f"  test rows: {len(ds.X_test):,}  features: {len(ds.feature_names)}")
    save_live_bundle(ds)

    print(f"[{time.strftime('%H:%M:%S')}] Loading saved posteriors from {POSTERIOR_DIR}")
    fit = load_per_lead_fit_from_disk(
        lead_hours=list(ds.lead_hours),
        station_codes=ds.station_codes,
        feature_names=ds.feature_names,
    )

    print(f"[{time.strftime('%H:%M:%S')}] Computing per-row posterior summary "
          f"(mean, std, quantiles {QUANTILES})")
    t0 = time.time()
    summary = predict_per_lead_summary(
        fit,
        ds.X_test_s,
        ds.station_idx_test,
        ds.lead_idx_test,
        quantiles=QUANTILES,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    # Sanity check: posterior mean from summary should match the existing
    # `lead_{N}h.parquet` p_wet column to within float noise — they're computed
    # the same way (mean over chain × draw of the sigmoid).
    for li, lead in enumerate(ds.lead_hours):
        mask = ds.lead_idx_test == li
        existing_path = ART_DIR / f"lead_{lead}h.parquet"
        if existing_path.exists():
            existing = pd.read_parquet(existing_path)
            existing_p = existing["p_wet"].to_numpy()
            new_p = summary["mean"][mask]
            if len(existing_p) == len(new_p):
                max_diff = float(np.max(np.abs(existing_p - new_p)))
                print(f"  lead {lead}h posterior-mean reconciliation: "
                      f"max |Δ| vs existing parquet = {max_diff:.2e}")

    # Write augmented per-lead parquets with CI columns alongside the
    # existing files. Different filename so we don't clobber the original
    # until consumers migrate.
    for li, lead in enumerate(ds.lead_hours):
        mask = ds.lead_idx_test == li
        out = pd.DataFrame({
            "valid_time": pd.to_datetime(ds.valid_time_test.values[mask]),
            "station": [ds.station_codes[i] for i in ds.station_idx_test[mask]],
            "lead": int(lead),
            "p_wet_mean": summary["mean"][mask],
            "p_wet_std": summary["std"][mask],
            "observed_wet": ds.y_test.values[mask].astype("int8"),
        })
        for q in QUANTILES:
            out[f"p_wet_q{q:g}"] = summary[f"q{q:g}"][mask]
        out["ci80_width"] = summary["q0.9"][mask] - summary["q0.1"][mask]
        out["ci90_width"] = summary["q0.95"][mask] - summary["q0.05"][mask]

        out_path = ART_DIR / f"lead_{lead}h_with_ci.parquet"
        out.to_parquet(out_path, index=False)
        print(f"  lead {lead}h: wrote {len(out):,} rows to {out_path}")

    # Replicate the Phase 4 finding by binning rows by CI width quartile
    # within each lead and reporting Brier per quartile. If the headline
    # "narrow CI -> lower Brier" pattern holds, ci width is a usable
    # transferable confidence signal for downstream consumers (WeatherBlend's
    # site cells, dry-window blender gating, etc.).
    print(f"\n[{time.strftime('%H:%M:%S')}] Verifying narrow-CI vs wide-CI Brier pattern")
    print("Quartile binning is by ci80 width (q0.9 - q0.1). Q1=narrowest, Q4=widest.")
    print(f"{'lead':>5} {'quartile':>10} {'n':>6} {'mean ci80':>10} {'Brier':>8} {'p_wet mean':>11}")
    for lead in ds.lead_hours:
        out_path = ART_DIR / f"lead_{lead}h_with_ci.parquet"
        df = pd.read_parquet(out_path)
        df["ci80_quartile"] = pd.qcut(df["ci80_width"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        for q, sub in df.groupby("ci80_quartile", observed=True):
            b = brier(sub["p_wet_mean"].to_numpy(), sub["observed_wet"].to_numpy().astype("float64"))
            print(f"{lead:>5} {q:>10} {len(sub):>6} {sub['ci80_width'].mean():>10.4f} "
                  f"{b:>8.4f} {sub['p_wet_mean'].mean():>11.4f}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
