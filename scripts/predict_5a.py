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

Output: predictions partition under `data/predictions/precipitation/
{station_slug}/model_version=v{timestamp}_phase5a/date=YYYY-MM-DD/
predictions.parquet`. Columns carry the full posterior CI alongside the
mean: ProbWet, ProbWetStd, ProbWetQ05/Q10/Q50/Q90/Q95, Ci80Width,
Ci90Width, plus standard 4a-shape columns: ModelVersion, TruthStation,
PredictionMadeAtUtc, ValidTimeUtc, LeadHours.

Metadata (training_metadata.json + feature_schema.json) is written at
TRAIN time by extend_5a.py — same pattern as 4a's train_4a.py. This
script reads ``Version`` + ``TrainedAtUtc`` out of the bundle's
metadata.json and reuses them unchanged across every predict tick (so
the per-station shadow under data/models/ stays stable for one training
run rather than churning per-tick the way the pre-2026-05-11 code did).

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
    WET_THRESHOLD_MM,
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

# Phase A multi-location safety (2026-05-12). Active NWP location for
# this predict invocation; the live-bundle metadata.LocationName must
# match (or be empty for legacy bundles).
ACTIVE_LOCATION = os.environ.get("WB_LOCATION", LOCATION)


def _load_metadata() -> dict:
    meta_path = LIVE_BUNDLE_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Live bundle not found at {meta_path}. "
            f"Run scripts/extend_5a.py first."
        )
    meta = json.loads(meta_path.read_text())
    # Phase A tightening (Task #21): LocationName is required. A missing
    # field is a corrupt/incomplete bundle — hard-fail rather than fall back.
    bundle_loc = (meta.get("LocationName") or "").strip()
    if not bundle_loc:
        raise ValueError(
            "Live 5a bundle has no LocationName pinned in metadata.json — "
            "required since the 2026-05-12 backfill. Rerun extend_5a.py."
        )
    if bundle_loc.lower() != ACTIVE_LOCATION.lower():
        raise ValueError(
            f"Live 5a bundle was trained on location '{bundle_loc}' but "
            f"predict is using NWP from '{ACTIVE_LOCATION}' — refusing to "
            f"score. Set WB_LOCATION={bundle_loc} or rerun extend_5a.py "
            f"with the right location."
        )
    return meta


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
    """Load per-model live-cycle parquets and produce the column set
    needed for the rich feature build: precip + temp/dewpoint + relative
    humidity + cape + wind. Columns renamed with `_<model>` suffix to
    mirror src.data._load_model_forecasts_multi_lead_rich's output, so
    the cross-model aggregation in build_feature_frame can reuse the
    same math.

    No lead filter — predict scores every hourly forecast row; the live
    bundle's lead-as-feature design means leads outside the training
    set (e.g. 144h, 168h) score correctly given the standardised lead
    column the scaler applies."""
    model_dir = WEATHERBLEND_DATA_ROOT / "forecasts" / f"location={ACTIVE_LOCATION}" / f"model={model}"
    cols = [
        "RunTimeUtc", "ValidTimeUtc", "LeadHours",
        "Precipitation",
        "Temperature2m", "DewPoint2m", "RelativeHumidity2m",
        "Cape", "WindSpeed10m",
    ]
    frames = []
    for d in window_dates:
        date_str = d.strftime("%Y-%m-%d")
        date_dir = model_dir / f"date={date_str}"
        if not date_dir.exists():
            continue
        for path in sorted(date_dir.glob("run=*.parquet")):
            df = pd.read_parquet(path, columns=cols)
            if df.empty:
                continue
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    rename = {
        "Precipitation":      f"precip_{model}",
        "Temperature2m":      f"t_{model}",
        "DewPoint2m":         f"td_{model}",
        "RelativeHumidity2m": f"rh_{model}",
        "Cape":               f"cape_{model}",
        "WindSpeed10m":       f"wind_{model}",
    }
    return (
        pd.concat(frames, ignore_index=True)
        # Latest cycle wins per (ValidTime, Lead) — different cycles producing
        # the SAME (valid, lead) is a refresh; older cycles' values would
        # be stale.
        .sort_values(["ValidTimeUtc", "LeadHours", "RunTimeUtc"])
        .drop_duplicates(subset=["ValidTimeUtc", "LeadHours"], keep="last")
        .rename(columns=rename)
        .reset_index(drop=True)
    )


def build_feature_frame(anchor: pd.Timestamp) -> pd.DataFrame:
    """Per-model hourly forecasts inner-joined on (ValidTime, Lead),
    plus the cross-model ensemble aggregates the rich 18-feature 5a
    bundle needs. Mirrors src.data._load_all_forecasts_multi_lead_rich
    column-for-column so the saved StandardScaler maps the live frame
    to the same standardised space the training data was in.

    Output columns the predict path consumes:
      - precip_<model> × 5             (raw per-model precip)
      - precip_mean / std / max / agreement_wet_01  (cross-model spread)
      - rh_mean, cape_mean, wind_speed_mean, dew_depression_mean
        (cross-model ensemble means; matches training side's covariates)
      - hour_sin, hour_cos, doy_sin, doy_cos    (cyclic calendar)
      - lead                            (raw hours; scaler z-scores it)
    """
    # 4-day-back lookback catches lead-72/96 cycles + a buffer for late
    # landings; +1d forward in case anchor is mid-day and a freshly
    # published cycle's date partition reads as 'tomorrow' UTC. Lead 120
    # extends the relevant historical window slightly — anchor-6d picks
    # up a 144h cycle published 6 days ago.
    window_dates = [anchor + pd.Timedelta(days=d) for d in range(-6, 2)]

    frames: list[pd.DataFrame] = []
    missing_models: list[str] = []
    for model in MODELS_NO_UKMO:
        df = _load_one_model_live_runs(model, window_dates)
        if df.empty:
            print(f"  WARN: no live forecasts found for {model} in window")
            missing_models.append(model)
            continue
        # Carry the lead model's RunTimeUtc as provenance; drop the
        # rest to keep the merge clean.
        if model == MODELS_NO_UKMO[0]:
            df = df.rename(columns={"RunTimeUtc": "provenance_run"})
        else:
            df = df.drop(columns=["RunTimeUtc"])
        frames.append(df)

    if missing_models:
        scanned = ", ".join(d.strftime("%Y-%m-%d") for d in window_dates)
        raise RuntimeError(
            f"No live forecasts found for {len(missing_models)}/{len(MODELS_NO_UKMO)} "
            f"models ({', '.join(missing_models)}) in window {scanned}."
        )

    forecasts = frames[0]
    for fc in frames[1:]:
        # Outer-merge so a row survives when at least one NWP has data at
        # that (validtime, lead). Matches training-side data.py's switch
        # (2026-05-11) — long-lead rows where meteofrance_seamless's
        # archive doesn't reach still produce a 5a prediction, with
        # precip_meteofrance_seamless imputed from the saved median.
        forecasts = forecasts.merge(fc, on=["ValidTimeUtc", "LeadHours"], how="outer")
    if forecasts.empty:
        return forecasts

    # Drop rows where ALL per-model precip columns are NaN — those rows
    # have no NWP signal at all and aren't predictable.
    precip_cols = [f"precip_{m}" for m in MODELS_NO_UKMO]
    forecasts = forecasts.loc[
        forecasts[precip_cols].notna().any(axis=1)
    ].reset_index(drop=True)
    if forecasts.empty:
        return forecasts

    # Cyclic calendar — hour AND day-of-year (matches the training side's
    # _load_all_forecasts_multi_lead_rich).
    hours = forecasts["ValidTimeUtc"].dt.hour
    forecasts["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    forecasts["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    doy = forecasts["ValidTimeUtc"].dt.dayofyear
    forecasts["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.0)
    forecasts["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.0)

    # Cross-model precip spread features (skip-NaN aware, same shape as
    # _load_all_forecasts_multi_lead_rich produces).
    precip_cols = [f"precip_{m}" for m in MODELS_NO_UKMO]
    pmat = forecasts[precip_cols].to_numpy(dtype="float64")
    forecasts["precip_mean"] = np.nanmean(pmat, axis=1)
    forecasts["precip_std"]  = np.nanstd(pmat, axis=1)
    forecasts["precip_max"]  = np.nanmax(pmat, axis=1)
    wet = (pmat >= WET_THRESHOLD_MM).astype("float64")
    present = (~np.isnan(pmat)).astype("float64")
    forecasts["precip_agreement_wet_01"] = np.where(
        present.sum(axis=1) > 0,
        wet.sum(axis=1) / present.sum(axis=1),
        np.nan,
    )

    # Cross-model ensemble means for the four atmospheric covariates the
    # rich training set uses. Cloud columns (cloud_low/mid/high_mean) are
    # 100% null in the OM previous_runs archive and dropped at the data.py
    # `feature_set='full'` runtime guard, so they aren't in the trained
    # feature_names list — we don't compute them here.
    for short, src in [("rh", "rh"), ("cape", "cape"), ("wind_speed", "wind")]:
        cols = [f"{src}_{m}" for m in MODELS_NO_UKMO]
        forecasts[f"{short}_mean"] = np.nanmean(
            forecasts[cols].to_numpy(dtype="float64"), axis=1
        )
    dewdep_cols = []
    for m in MODELS_NO_UKMO:
        forecasts[f"_dewdep_{m}"] = forecasts[f"t_{m}"] - forecasts[f"td_{m}"]
        dewdep_cols.append(f"_dewdep_{m}")
    forecasts["dew_depression_mean"] = np.nanmean(
        forecasts[dewdep_cols].to_numpy(dtype="float64"), axis=1
    )
    forecasts = forecasts.drop(columns=dewdep_cols)

    forecasts["lead"] = forecasts["LeadHours"].astype("float64")
    return forecasts


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
    args = p.parse_args()

    anchor = pd.Timestamp(args.anchor, tz="UTC").normalize().tz_localize(None)
    predictions_root = Path(args.predictions_root)
    print(f"[{time.strftime('%H:%M:%S')}] Phase 5a live Bayesian predict — anchor={anchor.date()}")

    meta = _load_metadata()
    # Version + TrainedAtUtc are frozen at train time by extend_5a.py
    # (2026-05-11 onwards). For bundles written before that change, fall
    # back to a fresh predict-time stamp so predict still runs — though
    # the resulting Models card will show the predict time as "trained"
    # and a fresh dir per tick on R2 until the next retrain rolls through.
    version = meta.get("Version") or datetime.now(timezone.utc).strftime(
        "v%Y-%m-%d_%H%M%S_phase5a"
    )
    print(f"  version: {version}")
    feature_names: list[str] = meta["feature_names"]
    station_codes: list[str] = meta["station_codes"]
    station_full_names: list[str] = meta["station_full_names"]
    lead_hours: list[int] = [int(L) for L in meta["lead_hours"]]
    lead_feature_index: int = meta["lead_feature_index"]
    feature_medians: dict[str, float] = meta.get("feature_medians", {})
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

    # Build X in the SAME column order as training, then impute any NaN
    # cells using the training-time medians persisted in metadata. NaNs
    # come from outer-join long-lead rows where one of the NWPs has no
    # archive coverage at that lead (meteofrance_seamless above 72h).
    X_df = forecasts[feature_names].copy()
    if feature_medians:
        for col in feature_names:
            if X_df[col].isna().any() and col in feature_medians:
                X_df[col] = X_df[col].fillna(feature_medians[col])
    X = X_df.to_numpy(dtype="float64")
    if np.isnan(X).any():
        # Defensive — should not fire given the impute step above, but
        # without medians or with a feature missing from the bundle the
        # scaler.transform would silently produce NaN. Better to surface.
        cols_with_nan = [c for c in feature_names if X_df[c].isna().any()]
        raise RuntimeError(
            f"NaN cells remain in features after median-imputation: {cols_with_nan}. "
            f"Bundle metadata may be missing entries for these features."
        )
    X_s = scaler.transform(X).astype("float64")

    rows_written = 0
    stations_with_rows = 0
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

        # Per-station shadow metadata under data/models/.../v..._phase5a/
        # is written by extend_5a.py at TRAIN time (2026-05-11 onwards) —
        # mirrors 4a's pattern. This script only emits predictions.

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
            "LocationName":    ACTIVE_LOCATION,
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
        stations_with_rows += 1

    print()
    print(f"Phase 5a complete. Stations with predictions: {stations_with_rows}, "
          f"prediction rows: {rows_written:,}")
    if rows_written == 0:
        print("WARN: no prediction rows written across any station — "
              "live forecast window may be empty.")


if __name__ == "__main__":
    main()
