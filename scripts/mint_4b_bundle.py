"""Phase 4b — one-time bundle minter for the 2-way mean of 4a + 3e.

4b isn't a trained model — it's the arithmetic mean of phase 4a (BART
per-cell) and phase 3e (TorchSharp MLP) predictions, which the
2026-05-12 LightGBM-meta-learner bake-off identified as the best
production stack across the three Bonehill stations (mean Brier
0.0830, beats best single by 1.8%, wins 14/15 per-(station,lead)
cells). See ``project_lgbm_meta_bakeoff_2026-05-12.md`` for the
result; this script ships that finding as a first-class phase so
render / verify / Models page treat 4b like any other bundle.

What this script does (once per Bonehill station):
  1. Read the latest 4a + 3e ``test_predictions.parquet`` from the
     bundles already on disk under ``data/models/precipitation/
     {station}/{version}/``. Inner-join on (valid_time, lead).
  2. Compute p_wet_4b = mean(p_wet_4a, p_wet_3e).
  3. Mint a new bundle directory at ``data/models/precipitation/
     {station}/v{ts}_phase4b/`` containing:
       - ``test_predictions.parquet`` — the joined+averaged rows.
       - ``training_metadata.json`` — Phase=4b, LocationName,
         per-lead Brier computed from the joined parquet.
       - ``feature_schema.json`` — names p_4a + p_3e as inputs.
       - ``climatology.json`` — copied from the station's 3a
         bundle (4b serves the same target → same climatology).
       - ``training_summary.json`` — minimal stub for RetrainGuard.

Usage (typically once per Sunday auto-retrain, after 4a + 3e have
retrained and pushed)::

    python scripts/mint_4b_bundle.py \\
        --models-root /path/to/WeatherBlend/data/models/precipitation \\
        --stations ea_bellever_dartmoor,ea_bovey_tracey,ea_dartmoor_nr_hexworthy

Per-cycle LIVE predictions are NOT this script's job — that's
``predict_4b.py``, which reads the latest 4a + 3e prediction
parquets and synthesises the cycle's 4b predictions.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
DEFAULT_LOCATION = "bonehill_rocks"
PHASE = "4b"


def find_latest_bundle(station_dir: Path, phase_suffix: str | None) -> Path | None:
    """Return the newest version dir under ``station_dir`` whose name
    matches the phase suffix (e.g. ``"phase4a"``) and that contains a
    ``test_predictions.parquet`` file. Returns None if nothing matches.

    ``phase_suffix=None`` selects bundles WITHOUT any phase suffix —
    i.e. the unsuffixed 3a champion convention from pre-2026-05-12.
    """
    if not station_dir.is_dir():
        return None
    candidates = []
    for d in station_dir.iterdir():
        if not d.is_dir():
            continue
        if phase_suffix is None:
            if "phase" in d.name:
                continue
        else:
            if phase_suffix not in d.name:
                continue
        if (d / "test_predictions.parquet").exists():
            candidates.append(d)
    return max(candidates, default=None, key=lambda d: d.name)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def mint_for_station(
    station_dir: Path,
    station: str,
    location: str,
    version: str,
) -> tuple[bool, str]:
    """Mint a 4b bundle for one station. Returns ``(ok, note)``."""
    bundle_4a = find_latest_bundle(station_dir, "phase4a")
    bundle_3e = find_latest_bundle(station_dir, "phase3e")
    bundle_3a = find_latest_bundle(station_dir, None)
    if bundle_4a is None:
        return False, f"no 4a bundle with test_predictions under {station_dir}"
    if bundle_3e is None:
        return False, f"no 3e bundle with test_predictions under {station_dir}"
    if bundle_3a is None:
        return False, f"no 3a bundle (needed for climatology.json copy) under {station_dir}"

    df_4a = pd.read_parquet(bundle_4a / "test_predictions.parquet").rename(columns={"p_wet": "p_4a"})
    df_3e = pd.read_parquet(bundle_3e / "test_predictions.parquet").rename(columns={"p_wet": "p_3e"})
    joined = df_4a.merge(
        df_3e.drop(columns=["observed_wet"]),
        on=["valid_time", "station", "lead"], how="inner",
    )
    if joined.empty:
        return False, f"inner-join of 4a + 3e test_predictions produced no rows for {station}"

    # 4b prediction = arithmetic mean of the two components' p_wet.
    joined["p_wet"] = (joined["p_4a"] + joined["p_3e"]) / 2.0
    test_predictions = joined[["valid_time", "station", "lead", "p_wet", "observed_wet"]].copy()

    # Per-lead Brier from the joined slice — same convention as
    # training_metadata.PerLead.BlendTestMae for the binary phases
    # (3a/3c/3d/3e/4a all use that field to carry Brier).
    per_lead: dict[str, dict] = {}
    for lead, grp in joined.groupby("lead"):
        y = grp["observed_wet"].astype("float64").to_numpy()
        p = grp["p_wet"].to_numpy()
        p_4a_only = grp["p_4a"].to_numpy()
        p_3e_only = grp["p_3e"].to_numpy()
        per_lead[str(int(lead))] = {
            "LeadHours": int(lead),
            "DataRangeTrain": "",
            "DataRangeVal": "",
            "DataRangeTest": "",
            "TrainRows": 0,
            "ValRows": 0,
            "TestRows": int(len(grp)),
            "TestCalendarMonths": 0,
            "BestSingle": "p_3e" if brier(p_3e_only, y) <= brier(p_4a_only, y) else "p_4a",
            "BestSingleValMae": None,
            "BestSingleTestMae": float(min(brier(p_3e_only, y), brier(p_4a_only, y))),
            "BlendTestMae": float(brier(p, y)),
            "BlendTestRmse": 0.0,
            "BlendTestBias": float(np.mean(p - y)),
            "CalibratedBlendTestMae": 0.0,
        }

    # Build + write the bundle directory.
    out_dir = station_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)
    test_predictions.to_parquet(out_dir / "test_predictions.parquet", index=False)

    metadata = {
        "Version":      version,
        "Target":       "precipitation",
        "Phase":        PHASE,
        "LocationName": location,
        "DataSource":   f"derived: mean(p_4a, p_3e) — 4a={bundle_4a.name}, 3e={bundle_3e.name}",
        "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
        "Hyperparameters": {
            "method":          "arithmetic_mean",
            "components":      ["p_4a", "p_3e"],
            "source_bundle_4a": bundle_4a.name,
            "source_bundle_3e": bundle_3e.name,
        },
        "TestMae": {
            f"lead_{int(L)}h_brier": stats["BlendTestMae"]
            for L, stats in per_lead.items()
        },
        "DeviationsFromBrief": [
            "Phase 4b is NOT a trained model — it is the arithmetic mean of "
            "phase 4a's BART P(wet) and phase 3e's MLP P(wet). The bundle "
            "is minted at the end of each Sunday auto-retrain so the "
            "ModelMetadata + Manifest plumbing treats it as a first-class "
            "phase (renders on rain forecasts, scored by verify, shown on "
            "the Models page). The 'training' is the join + average.",
            "test_predictions.parquet here is the inner-join of 4a and 3e's "
            "test_predictions parquets, with p_wet = (p_4a + p_3e) / 2. "
            "PerLead.BlendTestMae reports Brier on that joined slice, "
            "matching the 2026-05-12 LightGBM-meta bake-off finding.",
            "No feature schema, no LightGBM/MLP/BART training step — "
            "predict_4b.py performs the same arithmetic on the live "
            "cycle's 4a + 3e prediction parquets.",
            "PerLeadStats fields repurposed: BlendTestMae=Brier on joined "
            "test slice, BlendTestBias=mean(p − y), BlendTestRmse=0.0 "
            "(not meaningful for an unfit blend), BestSingleTestMae=Brier "
            "of the better of 4a/3e alone on the same slice.",
            "Climatology copied from the station's 3a bundle — 4b targets "
            "the same wet/dry indicator at the same station, so the "
            "climatology baseline is identical.",
        ],
        "PerLead": per_lead,
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    schema_per_lead = {
        L: {
            "Target":         "precipitation",
            "FeatureSet":     "phase4b-2way-mean",
            "LeadHours":      int(L),
            "RequiredModels": [],
            "OptionalModels": [],
            "Models":         ["phase4a", "phase3e"],
            "FeatureNames":   ["p_4a", "p_3e"],
            "DataSource":     "derived: mean(p_4a, p_3e)",
            "Tier":           "4b-synth",
            "UkvStrategy":    None,
        }
        for L in per_lead.keys()
    }
    (out_dir / "feature_schema.json").write_text(
        json.dumps({"Leads": schema_per_lead}, indent=2))

    # Copy 3a's climatology — same station + same wet indicator, so the
    # baseline is identical. Lets the precip predict path's
    # climPath check pass without bespoke handling for 4b.
    clim_src = bundle_3a / "climatology.json"
    if clim_src.exists():
        shutil.copy(clim_src, out_dir / "climatology.json")

    # Minimal training_summary stub so RetrainGuard's "previous" loader
    # has SOMETHING to compare against next mint (LocationName carried;
    # row counts mirror test_predictions size).
    summary = {
        "SchemaVersion": "1",
        "Composite": f"precipitation/{station}",
        "Phase": PHASE,
        "Version": version,
        "LocationName": location,
        "ComputedAtUtc": datetime.now(timezone.utc).isoformat(),
        "RowsTrain": 0,
        "RowsVal": 0,
        "RowsTest": int(len(joined)),
        "FeaturesEffective": 2,
        "PerFeature": {},
        "LabelRates": {
            station: float(joined["observed_wet"].mean()),
        },
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    aggregate_brier = brier(
        joined["p_wet"].to_numpy(),
        joined["observed_wet"].astype("float64").to_numpy(),
    )
    return True, (
        f"minted {version} ({len(joined):,} test rows, mean Brier {aggregate_brier:.4f}); "
        f"4a={bundle_4a.name}, 3e={bundle_3e.name}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--models-root", required=True,
        help="Path to WeatherBlend's data/models/precipitation directory "
             "(the per-station station dirs live one level below).",
    )
    ap.add_argument(
        "--stations", default=",".join(DEFAULT_STATIONS),
        help="Comma-separated EA station slugs to mint 4b for "
             "(default: 3 Bonehill stations).",
    )
    ap.add_argument(
        "--location", default=DEFAULT_LOCATION,
        help="LocationName to pin into training_metadata + summary "
             "(default: bonehill_rocks).",
    )
    ap.add_argument(
        "--version", default=None,
        help="Version string for the new bundle (default: timestamp now).",
    )
    args = ap.parse_args()

    models_root = Path(args.models_root)
    if not models_root.is_dir():
        print(f"::error::models-root not a directory: {models_root}")
        return 2

    stations = [s.strip() for s in args.stations.split(",")]
    if args.version:
        version = args.version
    else:
        version = datetime.now(timezone.utc).strftime("v%Y-%m-%d_%H%M%S_phase4b")

    print(f"Mint Phase {PHASE}: version={version}, location={args.location}")
    print(f"  models-root: {models_root}")
    print(f"  stations:    {stations}\n")

    n_ok = 0
    n_err = 0
    for station in stations:
        station_dir = models_root / station
        ok, note = mint_for_station(station_dir, station, args.location, version)
        prefix = "OK " if ok else "ERR"
        print(f"  {prefix} {station}: {note}")
        if ok:
            n_ok += 1
        else:
            n_err += 1
    print(f"\nDone. Minted: {n_ok}, failed: {n_err}.")
    return 0 if n_err == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
