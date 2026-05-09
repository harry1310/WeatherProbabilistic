"""Phase 5a — live Bayesian-logreg P(wet) predict with credible-interval
widths. Single lead-as-feature partial-pooling posterior, scored fresh
each cron tick against the live Open-Meteo forecast tree.

Why lead-as-feature: the underlying logreg pools across the full lead
horizon (one posterior, with `lead` as a standardised feature column)
rather than a per-lead posterior stack. Inference cost is one forward
pass per station rather than per (station, lead) pair.

Why a separate predict from Phase 4a (BART): 5a is the *Bayesian* P(wet)
production line. 4a is the BART blender. Both publish into the same
`data/predictions/precipitation/{station}/...` tree with distinct
`model_version=v..._phase5a` suffixes; WeatherBlend's
`LoadModelSummaries` discovers them via the metadata directory and the
prediction-line gating decides which is shown to the user.

Output schema mirrors Phase 4a's predict_4a.py: predictions partition
under `data/predictions/precipitation/{station_slug}/model_version=
v{timestamp}_phase5a/date=YYYY-MM-DD/predictions.parquet` and metadata
under `data/models/precipitation/{station_slug}/v{timestamp}_phase5a/
{training_metadata.json, feature_schema.json}`. Columns carry the full
posterior CI alongside the mean: ProbWet, ProbWetStd, ProbWetQ05/Q10/
Q50/Q90/Q95, Ci80Width, Ci90Width. Plus standard 4a-shape columns:
ModelVersion, TruthStation, PredictionMadeAtUtc, ValidTimeUtc, LeadHours.

Run:
    .venv/Scripts/python.exe -u scripts/predict_5a.py --anchor 2026-05-07
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
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
    LOCATION,
    MODELS_NO_UKMO,
    WEATHERBLEND_DATA_ROOT,
)
from src.models.phase2_partial_pooling import (  # noqa: E402
    PartialPoolingFit,
    predict_partial_pooling_summary,
)

from _shared import resolve_station  # noqa: E402

LIVE_BUNDLE_DIR = ROOT / "reports" / "phase5a_artefacts" / "live_bundle"
POSTERIOR_DIR = ROOT / "reports" / "phase5a_artefacts" / "posteriors"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
PHASE = "5a"


def _load_metadata() -> dict:
    meta_path = LIVE_BUNDLE_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Live bundle not found at {meta_path}. "
            f"Run scripts/extend_5a.py first."
        )
    return json.loads(meta_path.read_text())


def _load_scaler():
    with open(LIVE_BUNDLE_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def _load_fit(feature_names: list[str], station_codes: list[str]) -> PartialPoolingFit:
    nc_path = POSTERIOR_DIR / "lead_feature.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"Phase 5a posterior not found at {nc_path}")
    return PartialPoolingFit(
        idata=az.from_netcdf(nc_path),
        feature_names=feature_names,
        station_codes=station_codes,
    )


def _load_one_model_live_runs(model: str, window_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Same as Phase 4's helper but WITHOUT the lead filter — we keep
    every hourly forecast row in each cycle parquet."""
    model_dir = WEATHERBLEND_DATA_ROOT / "forecasts" / f"location={LOCATION}" / f"model={model}"
    frames = []
    for d in window_dates:
        date_str = d.strftime("%Y-%m-%d")
        date_dir = model_dir / f"date={date_str}"
        if not date_dir.exists():
            continue
        for path in sorted(date_dir.glob("run=*.parquet")):
            df = pd.read_parquet(
                path,
                columns=["RunTimeUtc", "ValidTimeUtc", "LeadHours", "Precipitation"],
            )
            if df.empty:
                continue
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ValidTimeUtc", "LeadHours", "RunTimeUtc", f"precip_{model}"])
    return (
        pd.concat(frames, ignore_index=True)
        # Latest cycle wins per (ValidTime, Lead) — different cycles producing
        # the SAME (valid, lead) is a refresh; older cycles' values would
        # be stale.
        .sort_values(["ValidTimeUtc", "LeadHours", "RunTimeUtc"])
        .drop_duplicates(subset=["ValidTimeUtc", "LeadHours"], keep="last")
        .rename(columns={"Precipitation": f"precip_{model}"})
        .reset_index(drop=True)
    )


def build_feature_frame(anchor: pd.Timestamp) -> pd.DataFrame:
    """Per-model hourly forecasts inner-joined on (ValidTime, Lead).
    Output columns: ValidTimeUtc, LeadHours, precip_<model>×5,
    hour_sin, hour_cos, lead. The 'lead' column duplicates LeadHours
    but in float form for the StandardScaler — keeps the column order
    matching the metadata feature_names list."""
    # 4-day-back lookback catches lead-72/96 cycles + a buffer for late
    # landings; +1d forward in case anchor is mid-day and a freshly
    # published cycle's date partition reads as 'tomorrow' UTC.
    window_dates = [anchor + pd.Timedelta(days=d) for d in range(-4, 2)]

    frames: list[pd.DataFrame] = []
    missing_models: list[str] = []
    for model in MODELS_NO_UKMO:
        df = _load_one_model_live_runs(model, window_dates)
        if df.empty:
            print(f"  WARN: no live forecasts found for {model} in window")
            missing_models.append(model)
            continue
        if "provenance_run" not in df.columns:
            df = df.rename(columns={"RunTimeUtc": "provenance_run"}) if model == MODELS_NO_UKMO[0] \
                else df.drop(columns=["RunTimeUtc"])
        frames.append(df)

    if missing_models:
        scanned = ", ".join(d.strftime("%Y-%m-%d") for d in window_dates)
        raise RuntimeError(
            f"No live forecasts found for {len(missing_models)}/{len(MODELS_NO_UKMO)} "
            f"models ({', '.join(missing_models)}) in window {scanned}."
        )

    forecasts = frames[0]
    for fc in frames[1:]:
        forecasts = forecasts.merge(fc, on=["ValidTimeUtc", "LeadHours"], how="inner")
    if forecasts.empty:
        return forecasts

    # Cyclical hour features + lead-as-feature (raw hours; the saved scaler
    # standardises it the same way it standardised the training column).
    hours = forecasts["ValidTimeUtc"].dt.hour
    forecasts["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    forecasts["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    forecasts["lead"] = forecasts["LeadHours"].astype("float64")
    return forecasts


def write_metadata(out_models_dir: Path, station_slug: str, station_full_name: str,
                   version: str, feature_names: list[str], lead_hours: list[int]) -> None:
    """Emit training_metadata.json + feature_schema.json under
    data/models/precipitation/{station}/{version}/ so WeatherBlend's
    LoadModelSummaries + spec page + verify pipeline pick 5a up.

    Phase 5a is a *predict-time* deployment of an offline-trained
    Bayesian posterior — there is no per-cycle test-set evaluation in
    this flow, so PerLead carries shape-only stubs (TestRows=0,
    BlendTestMae=NaN). The verify pipeline will populate real Brier
    numbers later as truth lands.
    """
    nwp_models = list(MODELS_NO_UKMO)

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
        "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
        "Hyperparameters": {
            "library":          "pymc + nutpie",
            "model":            "lead-as-feature partial-pooling Bayesian logistic regression",
            "leadAsFeature":    True,
            "stations":         "partial-pooled per station (varying intercept + slopes)",
        },
        "DeviationsFromBrief": [
            "Bayesian logistic regression (PyMC + nutpie NUTS sampler), "
            "partial-pooled across stations with `lead` as a continuous "
            "standardised feature column rather than a per-lead posterior stack.",
            "Five-model precip feature set (excludes ukmo_seamless because "
            "the lead-as-feature design conflicts with UKMO's hybrid "
            "UKV+UM-Global lead profile).",
            "Predict-time evaluation only — TrainedAtUtc above is the live "
            "predict timestamp; the actual posterior was sampled offline "
            "and is reused unchanged across cron ticks.",
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

    out_models_dir.mkdir(parents=True, exist_ok=True)
    (out_models_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str))
    (out_models_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2))
    print(f"  metadata → {out_models_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--anchor",
        default=pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d"),
        help="Anchor date YYYY-MM-DD (UTC). Default = today.",
    )
    p.add_argument(
        "--predictions-root",
        default=str(WEATHERBLEND_DATA_ROOT / "predictions"),
        help="Predictions tree root.",
    )
    p.add_argument(
        "--models-root",
        default=str(WEATHERBLEND_DATA_ROOT / "models"),
        help="Models tree root for metadata files.",
    )
    args = p.parse_args()

    anchor = pd.Timestamp(args.anchor, tz="UTC").normalize().tz_localize(None)
    predictions_root = Path(args.predictions_root)
    models_root = Path(args.models_root)
    version = datetime.now(timezone.utc).strftime("v%Y-%m-%d_%H%M%S_phase5a")
    print(f"[{time.strftime('%H:%M:%S')}] Phase 5a live Bayesian predict — anchor={anchor.date()}")
    print(f"  version: {version}")

    meta = _load_metadata()
    feature_names: list[str] = meta["feature_names"]
    station_codes: list[str] = meta["station_codes"]
    station_full_names: list[str] = meta["station_full_names"]
    lead_hours: list[int] = [int(L) for L in meta["lead_hours"]]
    lead_feature_index: int = meta["lead_feature_index"]
    scaler = _load_scaler()
    print(f"  loaded bundle: features={feature_names} stations={station_codes}")
    print(f"  lead feature index: {lead_feature_index} "
          f"(scaler.mean={scaler.mean_[lead_feature_index]:.3f}, "
          f"scaler.scale={scaler.scale_[lead_feature_index]:.3f})")

    fit = _load_fit(feature_names, station_codes)
    print(f"  loaded posterior {fit.idata.posterior.dims}")

    print(f"[{time.strftime('%H:%M:%S')}] Building feature frame")
    forecasts = build_feature_frame(anchor)
    if forecasts.empty:
        print("  no forecasts available — exiting.")
        return
    print(f"  {len(forecasts):,} forecast rows after inner-join across models")

    # Restrict to forward-of-anchor valid times bounded at +168h (Open-Meteo's
    # cycle horizon). A 7-day forward window covers what the user-facing site
    # cares about; rows with valid_time before anchor are kept too so the
    # chart can show recent history alongside the live forecast (the now-1h
    # filter on the chart side was dropped 2026-05-07).
    valid_lo = anchor - pd.Timedelta(days=4)
    valid_hi = anchor + pd.Timedelta(hours=168 + 24)
    forecasts = forecasts.loc[
        (forecasts["ValidTimeUtc"] >= valid_lo) & (forecasts["ValidTimeUtc"] < valid_hi)
    ].copy()
    print(f"  {len(forecasts):,} rows valid in [{valid_lo}, {valid_hi})")
    if forecasts.empty:
        print("  no forecast rows for the live window — exiting.")
        return

    # Build X in the SAME column order as training.
    X = forecasts[feature_names].to_numpy(dtype="float64")
    X_s = scaler.transform(X).astype("float64")

    rows_written = 0
    versions_emitted = 0
    prediction_made_at = datetime.now(timezone.utc).replace(tzinfo=None)

    # One posterior call per station — slice X by no condition (all rows
    # use the same posterior) but tag rows with this station's index so
    # the per-station partial-pool intercept/coefficients apply.
    for s_idx, (full_name, code) in enumerate(zip(station_full_names, station_codes)):
        station_idx = np.full(len(X_s), s_idx, dtype="int64")
        summary = predict_partial_pooling_summary(
            fit, X_s, station_idx, quantiles=QUANTILES,
        )

        # Resolve the friendly name back to the canonical station slug
        # used as the partition key in WeatherBlend's predictions tree.
        station_slug, _ = resolve_station(full_name)

        # Always emit metadata, even if no live rows for this station,
        # so LoadModelSummaries surfaces the 5a phase regardless.
        models_dir = models_root / "precipitation" / station_slug / version
        write_metadata(models_dir, station_slug, full_name, version,
                       feature_names, lead_hours)
        versions_emitted += 1

        out = pd.DataFrame({
            "ValidTimeUtc":    forecasts["ValidTimeUtc"].values,
            "LeadHours":       forecasts["LeadHours"].astype("int64").values,
            "ProbWet":         summary["mean"],
            "ProbWetStd":      summary["std"],
            "ProbWetQ05":      summary["q0.05"],
            "ProbWetQ10":      summary["q0.1"],
            "ProbWetQ50":      summary["q0.5"],
            "ProbWetQ90":      summary["q0.9"],
            "ProbWetQ95":      summary["q0.95"],
            "Ci80Width":       summary["q0.9"] - summary["q0.1"],
            "Ci90Width":       summary["q0.95"] - summary["q0.05"],
            "LocationName":    LOCATION,
            "ModelVersion":    version,
            "TruthStation":    station_slug,
            "PredictionMadeAtUtc": prediction_made_at,
        })
        if len(out) == 0:
            print(f"  {code} ({station_slug}): no rows to write")
            continue

        date_str = anchor.strftime("%Y-%m-%d")
        out_dir = (predictions_root / "precipitation" / station_slug
                   / f"model_version={version}" / f"date={date_str}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.parquet"
        out.to_parquet(out_path, index=False)
        print(f"  → {out_path}  ({len(out):,} rows, P(wet) mean {out['ProbWet'].mean():.3f})")
        rows_written += len(out)

    print()
    print(f"Phase 5a complete. Versions: {versions_emitted}, prediction rows: {rows_written:,}")
    if versions_emitted == 0:
        print("ERROR: no versions emitted — check posterior bundle presence.")
        sys.exit(1)


if __name__ == "__main__":
    main()
