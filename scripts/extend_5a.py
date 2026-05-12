"""Phase 5a sibling of `extend_phase4_with_ci.py` — 5a's posterior is
a SINGLE lead-as-feature partial-pool fit (vs Phase 4's three independent
per-lead fits), so the live-bundle layout is correspondingly simpler:
one posterior NetCDF, one StandardScaler that includes the lead column.

What this writes
----------------
1. ``reports/phase5a_artefacts/live_bundle/{scaler.pkl, metadata.json}``
   — the same format as Phase 4's live_bundle, with an extra
   ``lead_feature_index`` field marking which feature column carries
   the standardised lead value (so the live predict path can replace
   that one column when scoring at a new lead without re-running the
   full StandardScaler pipeline).
2. ``reports/phase5a_artefacts/predictions/test_predictions_with_ci.parquet``
   — augments the existing test predictions with mean / std / quantiles
   / ci80_width / ci90_width columns. Same shape as Phase 4's
   ``lead_{N}h_with_ci.parquet`` but one file (no per-lead split).

Reconciliation check: posterior-mean from the new summary should match
the test-predictions parquet's ``p_wet`` column to within float noise.

Run with:
    PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -u scripts/extend_5a.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import arviz as az  # noqa: E402

from src.data import (  # noqa: E402
    MODELS_NO_UKMO,
    PHASE5A_LEAD_HOURS,
    WEATHERBLEND_DATA_ROOT,
    prepare_phase3_dataset,
)
from src.models.phase2_partial_pooling import (  # noqa: E402
    PartialPoolingFit, predict_partial_pooling_summary,
)

from _shared import resolve_station  # noqa: E402

ART_DIR = ROOT / "reports" / "phase5a_artefacts"
POSTERIOR_DIR = ART_DIR / "posteriors"
LIVE_BUNDLE_DIR = ART_DIR / "live_bundle"
PREDICTIONS_DIR = ART_DIR / "predictions"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
PHASE = "5a"


def brier(prob: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((prob - truth) ** 2))


def save_live_bundle(ds, version: str, trained_at_utc: str) -> None:
    """Persist scaler + metadata for the live predict path. Mirrors Phase 4
    save_live_bundle but adds ``lead_feature_index`` so the live predict
    knows which column is the standardised lead.

    ``version`` and ``trained_at_utc`` are baked into the bundle so every
    predict tick downstream sees the same train-time stamps (matches 4a's
    pattern where train_4a.py owns the version and predict_4a.py just
    reads it). Pre-2026-05-11 5a generated a fresh version per predict
    tick — now frozen at train time."""
    import json
    import pickle

    LIVE_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_BUNDLE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(ds.scaler, f)

    if "lead" not in ds.feature_names:
        raise RuntimeError(
            "Phase 5a dataset must include 'lead' as a feature — pass "
            "lead_as_feature=True to prepare_phase3_dataset."
        )
    lead_feature_index = ds.feature_names.index("lead")

    metadata = {
        "Version": version,
        "TrainedAtUtc": trained_at_utc,
        "phase": "5a",
        "feature_names": list(ds.feature_names),
        "station_codes": list(ds.station_codes),
        "station_full_names": list(ds.station_full_names),
        "lead_hours": [int(l) for l in ds.lead_hours],
        "lead_feature_index": lead_feature_index,
        "scaler_mean": [float(x) for x in ds.scaler.mean_.tolist()],
        "scaler_scale": [float(x) for x in ds.scaler.scale_.tolist()],
        "scaler_var": [float(x) for x in ds.scaler.var_.tolist()],
        "scaler_n_samples_seen": int(ds.scaler.n_samples_seen_),
        # Per-feature median used to impute NaN cells on the predict side
        # (mirrors the training-time imputation in data.prepare_phase3_dataset).
        # Empty for non-full feature sets where outer-join is not used.
        "feature_medians": {k: float(v) for k, v in ds.feature_medians.items()},
    }
    (LIVE_BUNDLE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  wrote live bundle (scaler + metadata) to {LIVE_BUNDLE_DIR}")
    print(f"  lead is feature index {lead_feature_index} "
          f"(standardised mean={ds.scaler.mean_[lead_feature_index]:.3f}, "
          f"scale={ds.scaler.scale_[lead_feature_index]:.3f})")


def write_per_station_metadata(models_root: Path, ds, version: str, trained_at_utc: str) -> int:
    """Write training_metadata.json + feature_schema.json under
    ``models_root/precipitation/{station}/{version}/`` — one shadow per
    station so WeatherBlend's LoadModelSummaries + spec page + verify
    pipeline pick 5a up. Mirrors train_4a.py's pattern: metadata is
    canonical at train time, not predict time. Returns the number of
    stations written.

    The per-station shadow carries the SAME version + TrainedAtUtc that
    live in the bundle's metadata.json — so a reviewer comparing the two
    locations sees one consistent timestamp instead of a moving target.
    Pre-2026-05-11, predict_5a.py wrote these per-tick with a fresh
    `datetime.now()` and accumulated dozens of `v..._phase5a/` dirs on
    R2; that's now one dir per training run, matching 4a."""
    import json

    nwp_models = list(MODELS_NO_UKMO)
    feature_names = list(ds.feature_names)
    lead_hours = [int(l) for l in ds.lead_hours]
    station_full_names = list(ds.station_full_names)

    per_lead = {
        str(L): {
            "LeadHours":   L,
            "TestRows":    0,
            "BlendTestMae": float("nan"),
        }
        for L in lead_hours
    }
    metadata = {
        "Version":     version,
        "Target":      "precipitation",
        "Phase":       PHASE,
        "DataSource":  "open_meteo + bayesian_partial_pooling",
        "TrainedAtUtc": trained_at_utc,
        "Hyperparameters": {
            "library":          "r-inla (via Rscript subprocess)",
            "model":            "lead-as-feature partial-pooling Bayesian logistic regression "
                                "with random intercepts + random slopes per (station × feature)",
            "leadAsFeature":    True,
            "stations":         "partial-pooled per station (varying intercept + slopes)",
            "featureSet":       "full-21feat-rslopes (5 per-model precip + 4 precip-spread "
                                "stats + 4 atmospheric ensemble means + 4 cyclic + lead)",
        },
        "DeviationsFromBrief": [
            "Bayesian logistic regression fit via R-INLA's deterministic "
            "Laplace approximation (replaced PyMC + blackjax NUTS 2026-05-11 "
            "after a smoke run delivered ~1000x speedup at lower Brier). "
            "Posterior sampled in ~36 min wall on 175k training rows; 1000 "
            "joint samples persisted as `posteriors/lead_feature.nc` and "
            "scored online.",
            "Five-model precip feature set (excludes ukmo_seamless because "
            "the lead-as-feature design conflicts with UKMO's hybrid "
            "UKV+UM-Global lead profile).",
            "Leads 24/48/72/96/120 pooled into a single posterior with "
            "`lead` as a standardised feature column. At leads 96/120, "
            "meteofrance_seamless precip is missing (OM previous_runs "
            "archive caps mf at 72h) and median-imputed from the training "
            "distribution; random slopes per (station × feature) absorb "
            "the imputed-constant signal at long leads.",
            "Predict-time evaluation only — TrainedAtUtc above is the actual "
            "train timestamp (moved here from predict time on 2026-05-11 so "
            "every cron tick downstream sees the same stamp).",
            "PerLead.BlendTestMae is NaN here because no test slice runs "
            "per cycle; the verify pipeline backfills real Brier numbers "
            "as EA gauge truth lands.",
        ],
        "PerLead": per_lead,
    }
    schema_per_lead = {
        str(L): {
            "Target":         "precipitation",
            "FeatureSet":     f"phase5a-blr-l{L:02}",
            "LeadHours":      L,
            "RequiredModels": [],
            "OptionalModels": nwp_models,
            "Models":         nwp_models,
            "FeatureNames":   feature_names,
            "DataSource":     "open_meteo + bayesian_partial_pooling",
            "Tier":           "5a-blr",
            "UkvStrategy":    None,
        }
        for L in lead_hours
    }
    schema = {"Leads": schema_per_lead}

    written = 0
    for full_name in station_full_names:
        station_slug, _ = resolve_station(full_name)
        out_dir = models_root / "precipitation" / station_slug / version
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str))
        (out_dir / "feature_schema.json").write_text(
            json.dumps(schema, indent=2))
        print(f"  shadow metadata → {out_dir}")
        written += 1
    return written


def load_fit_from_disk(feature_names: list[str], station_codes: list[str]) -> PartialPoolingFit:
    nc_path = POSTERIOR_DIR / "lead_feature.nc"
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Phase 5a posterior not found at {nc_path}. "
            f"Run scripts/run_phase5_bayesian.py first."
        )
    return PartialPoolingFit(
        idata=az.from_netcdf(nc_path),
        feature_names=feature_names,
        station_codes=station_codes,
    )


def main() -> None:
    # 2026-05-11: switched to feature_set='full' (20 features pre-lead,
    # 21 with lead-as-feature) to match the new INLA backend in
    # run_phase5_bayesian.py. The old --add-spread-features flag is
    # retained for back-compat but is a no-op — the full feature set
    # already includes precip_max + precip_agreement_wet_01 plus the
    # missing 11 features (precip_mean/std, 7 atmospheric ensemble
    # means, doy_sin/cos) that the prior 10-feat 5a was missing.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--add-spread-features",
        action="store_true",
        help="(deprecated, no-op) Always uses the full 20+1-feature set.",
    )
    args = p.parse_args()
    _ = args  # silence linter — args.add_spread_features intentionally ignored.

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Loading Phase 3 dataset "
          f"(5-model, lead-as-feature, full-21feat for INLA backend)")
    ds = prepare_phase3_dataset(
        models=MODELS_NO_UKMO, lead_as_feature=True,
        leads=PHASE5A_LEAD_HOURS,
        feature_set="full",
        verbose=False,
    )
    print(f"  test rows: {len(ds.X_test):,}  features: {len(ds.feature_names)}")

    # Version + train-time stamp generated here, used for both the live
    # bundle metadata AND the per-station shadow metadata under
    # WeatherBlend's data/models/ tree. Frozen at train time so every
    # downstream predict tick sees one consistent set of stamps.
    now_utc = datetime.now(timezone.utc)
    version = now_utc.strftime("v%Y-%m-%d_%H%M%S_phase5a")
    trained_at_utc = now_utc.isoformat()
    print(f"  version: {version}")
    save_live_bundle(ds, version=version, trained_at_utc=trained_at_utc)

    # Per-station shadow metadata under WeatherBlend's data/models/ tree.
    # WeatherBlend's LoadModelSummaries reads this path; the shadow makes
    # 5a a first-class phase on the Models page + Spec page + verify
    # pipeline alongside 3a/4a. Path mirrors 4a's bundle dir but carries
    # only metadata JSON (5a's actual posterior lives in the singleton
    # reports/phase5a_artefacts/ tree because it's one-shared-across-
    # stations rather than per-cell).
    models_root = WEATHERBLEND_DATA_ROOT / "models"
    n_shadows = write_per_station_metadata(
        models_root, ds, version=version, trained_at_utc=trained_at_utc,
    )
    print(f"  {n_shadows} per-station shadow metadata files written under "
          f"{models_root / 'precipitation'}")

    # Promote into MANIFEST.json's Active. Mirrors train_4a's promote
    # call (added 2026-05-12 — same "no 4a on the site" symptom would
    # have hit 5a sooner or later if it weren't already exempt from the
    # Models-card filtering by role=confidence). Belt-and-braces: keep
    # 5a's version visible in Active so verify/spec/admin tooling
    # discovers it consistently. Phase tag = "phase5a".
    from src.manifest_promote import promote_station_version
    for full_name in ds.station_full_names:
        station_slug, _ = resolve_station(full_name)
        new_active = promote_station_version(
            models_root=models_root,
            target="precipitation",
            station_slug=station_slug,
            version=version,
            phase_tag="phase5a",
            role="challenger",
        )
        print(f"  promoted into manifest → {station_slug} Active = {new_active['Active']}")

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
