"""Phase 5 sibling of `predict_live_with_ci.py` — single lead-as-feature
posterior instead of three per-lead posteriors.

Three differences vs Phase 4 predict:
  1. NO LeadHours filter at the per-model load. Phase 4 restricted to
     {24, 48, 72}; Phase 5 takes every (cycle, lead) tuple Open-Meteo
     publishes hourly out to ~T+168h. Every valid_time covered by a
     cycle in the window gets a row.
  2. Append the standardised `lead` column to the feature matrix before
     scoring. The live_bundle's metadata.json carries the
     `lead_feature_index` (typically 7, after the 5 model-precip + 2
     hour-sincos columns) so we know which slot to fill.
  3. ONE posterior call per station. Phase 4 looped 3 leads × 3
     stations = 9 cells; Phase 5 is just per-station.

Output schema is identical to Phase 4's so the WeatherBlend renderer's
`QueryBayesianCi` reads it cleanly without changes:
    valid_time, station, lead, anchor,
    p_wet_mean, p_wet_std,
    p_wet_q0.05, p_wet_q0.1, p_wet_q0.5, p_wet_q0.9, p_wet_q0.95,
    ci80_width, ci90_width

Output path is also identical so the rclone push step in the workflow
doesn't need to change. The two predicts can co-exist (Phase 4 + Phase
5) by writing to different `model_version=` paths in a future revision;
for the initial cutover we replace Phase 4's output entirely.

Run:
    .venv/Scripts/python.exe -u scripts/predict_live_with_ci_phase5.py --anchor 2026-05-07
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
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

from src.data import (  # noqa: E402
    LOCATION,
    MODELS_NO_UKMO,
    WEATHERBLEND_DATA_ROOT,
)
from src.models.phase2_partial_pooling import (  # noqa: E402
    PartialPoolingFit,
    predict_partial_pooling_summary,
)

LIVE_BUNDLE_DIR = ROOT / "reports" / "phase5_artefacts" / "live_bundle"
POSTERIOR_DIR = ROOT / "reports" / "phase5_artefacts" / "posteriors"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
PREDICTION_OUT_ROOT = (
    WEATHERBLEND_DATA_ROOT
    / "predictions"
    / "precipitation_bayesian_ci"
    / f"location={LOCATION}"
)


def _load_metadata() -> dict:
    meta_path = LIVE_BUNDLE_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Live bundle not found at {meta_path}. "
            f"Run scripts/extend_phase5_with_ci.py first."
        )
    return json.loads(meta_path.read_text())


def _load_scaler():
    with open(LIVE_BUNDLE_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def _load_fit(feature_names: list[str], station_codes: list[str]) -> PartialPoolingFit:
    nc_path = POSTERIOR_DIR / "lead_feature.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"Phase 5 posterior not found at {nc_path}")
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--anchor",
        default=pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d"),
        help="Anchor date YYYY-MM-DD (UTC). Default = today.",
    )
    p.add_argument(
        "--out-root",
        default=str(PREDICTION_OUT_ROOT),
        help="Output root (default writes inside WeatherBlend's data tree).",
    )
    args = p.parse_args()

    anchor = pd.Timestamp(args.anchor, tz="UTC").normalize().tz_localize(None)
    out_root = Path(args.out_root)
    print(f"[{time.strftime('%H:%M:%S')}] Phase 5 live Bayesian predict — anchor={anchor.date()}")

    meta = _load_metadata()
    feature_names: list[str] = meta["feature_names"]
    station_codes: list[str] = meta["station_codes"]
    station_full_names: list[str] = meta["station_full_names"]
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

    # One posterior call per station — slice X by no condition (all rows
    # use the same posterior) but tag rows with this station's index so
    # the per-station partial-pool intercept/coefficients apply.
    rows_out: list[pd.DataFrame] = []
    for s_idx, (full_name, code) in enumerate(zip(station_full_names, station_codes)):
        station_idx = np.full(len(X_s), s_idx, dtype="int64")
        summary = predict_partial_pooling_summary(
            fit, X_s, station_idx, quantiles=QUANTILES,
        )
        out = pd.DataFrame({
            "valid_time": forecasts["ValidTimeUtc"].values,
            "station": code,
            "station_full_name": full_name,
            "lead": forecasts["LeadHours"].astype("int64").values,
            "anchor": anchor,
            "p_wet_mean": summary["mean"],
            "p_wet_std": summary["std"],
        })
        for q in QUANTILES:
            out[f"p_wet_q{q:g}"] = summary[f"q{q:g}"]
        out["ci80_width"] = summary["q0.9"] - summary["q0.1"]
        out["ci90_width"] = summary["q0.95"] - summary["q0.05"]
        rows_out.append(out)
        print(f"  {code}: scored {len(out):,} rows")

    if not rows_out:
        print("  no rows produced — exiting.")
        return

    big = pd.concat(rows_out, ignore_index=True)
    print(f"  produced {len(big):,} rows total")

    # Same hive layout as Phase 4 so the WeatherBlend renderer reads
    # without changes.
    for full_name, sub in big.groupby("station_full_name", observed=True):
        out_dir = out_root / f"station={full_name}" / f"anchor={anchor.strftime('%Y-%m-%d')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "widths.parquet"
        sub.drop(columns=["station_full_name"]).to_parquet(out_path, index=False)
        print(f"  wrote {len(sub):,} rows to {out_path}")

    print(f"[{time.strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
