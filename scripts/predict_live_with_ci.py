"""Phase 2b — live predict path for the Bayesian P(wet) confidence signal.

Loads the saved Phase 4 posteriors + scaler bundle (NO refit) and emits a
per-row prediction parquet with full CI summary (mean, std, q0.05/q0.10/
q0.50/q0.90/q0.95, ci80_width, ci90_width) for a specified anchor.

For each (station, lead) combination this script:
  1. Reads the Open-Meteo Previous Runs parquet at
     `WEATHERBLEND/data/forecasts/location=bonehill_rocks/model=<m>/
      date=<anchor-1d-or-2d-or-3d>/previous_runs.parquet`
     for each model in MODELS_NO_UKMO, where the offset_day arithmetic
     pulls the row whose RunTime maps to a 24/48/72h forecast valid on
     the target day.
  2. Builds the same feature vector the trainer used (5 model precip
     columns + hour_sin + hour_cos), applies the saved scaler.
  3. Looks up the lead-specific posterior, runs
     `predict_partial_pooling_summary`, gets posterior summary stats.
  4. Writes one parquet row per (valid_time, station, lead).

Output:
    WEATHERBLEND/data/predictions/precipitation_bayesian_ci/
        location=bonehill_rocks/station=<station_full>/
        anchor=<YYYY-MM-DD>/widths.parquet

Schema (matches Phase 4's `lead_{N}h_with_ci.parquet` structure plus an
`anchor` column for provenance):
    valid_time, station, lead, anchor,
    p_wet_mean, p_wet_std,
    p_wet_q0.05, p_wet_q0.1, p_wet_q0.5, p_wet_q0.9, p_wet_q0.95,
    ci80_width, ci90_width

Why this exists vs run_phase4_bayesian.py: Phase 4 trains + predicts on a
fixed historical test slice. This script is "trained once, applied many
times" — no retraining, just applies the saved posterior to current data.
Cheap enough to run on every cron tick (~2-3 seconds per anchor on this
box, dominated by parquet I/O not Bayesian inference).

Run with:
    .venv/Scripts/python.exe -u scripts/predict_live_with_ci.py --anchor 2026-04-25

Anchor defaults to today (UTC). The output directory under WeatherBlend's
data root is what the rclone push step in the GH workflow will sync to R2,
which is where WeatherBlend's site renderer will read it from.
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
    STATIONS,
    WEATHERBLEND_DATA_ROOT,
    WET_THRESHOLD_MM,  # noqa: F401  (re-exported for completeness)
)
from src.models.phase2_partial_pooling import (  # noqa: E402
    PartialPoolingFit,
    predict_partial_pooling_summary,
)

LIVE_BUNDLE_DIR = ROOT / "reports" / "phase4_artefacts" / "bayesian_predictions" / "live_bundle"
POSTERIOR_DIR = ROOT / "reports" / "phase4_artefacts" / "bayesian_predictions" / "posteriors"
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
            f"Run scripts/extend_phase4_with_ci.py first."
        )
    return json.loads(meta_path.read_text())


def _load_scaler():
    with open(LIVE_BUNDLE_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def _load_fit_for_lead(lead: int, feature_names: list[str], station_codes: list[str]) -> PartialPoolingFit:
    nc_path = POSTERIOR_DIR / f"lead_{lead}h.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"Posterior not found at {nc_path}")
    return PartialPoolingFit(
        idata=az.from_netcdf(nc_path),
        feature_names=feature_names,
        station_codes=station_codes,
    )


def _load_one_model_live_runs(
    model: str, window_dates: list[pd.Timestamp], lead_hours: list[int]
) -> pd.DataFrame:
    """Load all live `run=*.parquet` cycles for one model in the date window.

    Returns long-format (ValidTimeUtc, LeadHours, RunTimeUtc, precip_<model>)
    filtered to the requested leads. Multiple runs may forecast the same
    (ValidTimeUtc, LeadHours) — caller picks the most recent.

    Note: training used Phase 4's `previous_runs` (offset_day) parquets,
    which only exist in the historical R2 backfill. Live parquets are
    `RunTimeSource=reported` from the same Open-Meteo model cycles.
    Distribution shift is small but non-zero — we accept this for the
    MVP since CI WIDTH (the actual confidence signal) is robust to small
    feature distribution shifts even when the headline P(wet) calibration
    drifts slightly. Phase 2 worth checking if needed.
    """
    model_dir = WEATHERBLEND_DATA_ROOT / "forecasts" / f"location={LOCATION}" / f"model={model}"
    leads_set = set(int(l) for l in lead_hours)
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
            df = df.loc[df["LeadHours"].isin(leads_set)]
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ValidTimeUtc", "LeadHours", "RunTimeUtc", f"precip_{model}"])
    out = (
        pd.concat(frames, ignore_index=True)
        # Keep most recent RunTime for each (ValidTime, Lead).
        .sort_values(["ValidTimeUtc", "LeadHours", "RunTimeUtc"])
        .drop_duplicates(subset=["ValidTimeUtc", "LeadHours"], keep="last")
        .rename(columns={"Precipitation": f"precip_{model}"})
        .reset_index(drop=True)
    )
    return out


def build_feature_frame(anchor: pd.Timestamp, lead_hours: list[int]) -> pd.DataFrame:
    """Build a per-row feature frame for the requested anchor + leads.

    Strategy: scan live forecast cycles in a window around the anchor
    (anchor-3d to anchor+1d catches the cycles whose lead-24/48/72h
    forecasts cover the live prediction window), inner-join across all 5
    models on (ValidTimeUtc, LeadHours).

    Returned columns: ValidTimeUtc, LeadHours, precip_<model>×5,
    hour_sin, hour_cos.
    """
    # Recent cycles forecast forward by up to 168h. To catch lead-24h
    # forecasts valid in [anchor, anchor+72h+24h], the cycle that produced
    # those rows ran at most 24h before anchor. Wider window (±3-4 days)
    # catches lead-48/72 cycles too plus a margin for late landings.
    window_dates = [anchor + pd.Timedelta(days=d) for d in range(-4, 2)]

    frames: list[pd.DataFrame] = []
    missing_models: list[str] = []
    for model in MODELS_NO_UKMO:
        df = _load_one_model_live_runs(model, window_dates, lead_hours)
        if df.empty:
            print(f"  WARN: no live forecasts found for {model} in window — skipping.")
            missing_models.append(model)
            return pd.DataFrame()
        # Drop the per-model RunTimeUtc before joining — different models
        # have different cycle hours so the join would 0-row-out otherwise.
        # We keep one RunTimeUtc per row from the FIRST model joined, just
        # for provenance, attached as `provenance_run` so the caller can
        # see which cycle approximation backed this prediction.
        if "provenance_run" not in df.columns:
            df = df.rename(columns={"RunTimeUtc": "provenance_run"}) if model == MODELS_NO_UKMO[0] \
                else df.drop(columns=["RunTimeUtc"])
        frames.append(df)

    # Defence-in-depth guard against the silent "everything pulled in 0 rows"
    # mode — happened once on first CI fire (2026-05-06) when
    # WEATHERBLEND_DATA_ROOT defaulted to a Windows path on a Linux runner.
    # The glob pattern returned empty for every model, the workflow finished
    # "successfully" with zero parquet rows written, and the silent failure
    # only surfaced when the site renderer reported "Loaded 0 Bayesian CI
    # rows". Loud abort here is much easier to diagnose than a downstream
    # empty parquet.
    if missing_models:
        scanned = ", ".join(d.strftime("%Y-%m-%d") for d in window_dates)
        raise RuntimeError(
            f"No live forecasts found for {len(missing_models)}/{len(MODELS_NO_UKMO)} "
            f"models ({', '.join(missing_models)}). "
            f"Scanned date partitions {scanned} under "
            f"{WEATHERBLEND_DATA_ROOT}/forecasts/location={LOCATION}/. "
            f"Check WEATHERBLEND_DATA_ROOT env var and rclone-sync step."
        )
    forecasts = frames[0]
    for fc in frames[1:]:
        forecasts = forecasts.merge(fc, on=["ValidTimeUtc", "LeadHours"], how="inner")
    if forecasts.empty:
        return forecasts

    # Cyclical hour features — same as training.
    hours = forecasts["ValidTimeUtc"].dt.hour
    forecasts["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    forecasts["hour_cos"] = np.cos(2 * np.pi * hours / 24)

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
        help="Output root (default writes inside WeatherBlend's data tree so the GH workflow can rclone push).",
    )
    args = p.parse_args()

    anchor = pd.Timestamp(args.anchor, tz="UTC").normalize().tz_localize(None)
    out_root = Path(args.out_root)
    print(f"[{time.strftime('%H:%M:%S')}] Live Bayesian predict — anchor={anchor.date()}")

    meta = _load_metadata()
    feature_names: list[str] = meta["feature_names"]
    station_codes: list[str] = meta["station_codes"]
    station_full_names: list[str] = meta["station_full_names"]
    lead_hours: list[int] = meta["lead_hours"]
    scaler = _load_scaler()
    print(f"  loaded bundle: features={feature_names} stations={station_codes} leads={lead_hours}")

    fits = {l: _load_fit_for_lead(l, feature_names, station_codes) for l in lead_hours}
    print(f"  loaded {len(fits)} posterior NetCDFs")

    # Build the feature frame from previous_runs forecasts in the window.
    print(f"[{time.strftime('%H:%M:%S')}] Building feature frame")
    forecasts = build_feature_frame(anchor, lead_hours)
    if forecasts.empty:
        print(f"  no forecasts available for anchor {anchor.date()} — exiting.")
        return
    print(f"  {len(forecasts):,} forecast rows after inner-join across models")

    # Restrict to forecasts valid in [anchor, anchor+max_lead]. This is the
    # window the user-facing site cares about — earlier rows may exist
    # in the parquets but they refer to past valid times.
    max_lead = max(lead_hours)
    valid_lo = anchor
    valid_hi = anchor + pd.Timedelta(hours=max_lead + 24)
    forecasts = forecasts.loc[
        (forecasts["ValidTimeUtc"] >= valid_lo) & (forecasts["ValidTimeUtc"] < valid_hi)
    ].copy()
    print(f"  {len(forecasts):,} rows valid in [{valid_lo}, {valid_hi})")
    if forecasts.empty:
        print("  no forecast rows for the live window — exiting.")
        return

    # Scaling and feature matrix in the SAME column order as training.
    X = forecasts[feature_names].to_numpy(dtype="float64")
    X_s = scaler.transform(X).astype("float64")

    # Lead → posterior. Predict per-station for that lead's slice.
    lead_to_idx = {l: i for i, l in enumerate(lead_hours)}
    forecasts["lead_idx"] = forecasts["LeadHours"].map(lead_to_idx)

    # Each (station, lead) cell uses the same X_s slice but a different
    # posterior. We loop over stations × leads, not rows, so the Bayesian
    # cost stays bounded at n_stations × n_leads × O(post_size · n_rows).
    rows_out: list[pd.DataFrame] = []
    for s_idx, (full_name, code) in enumerate(zip(station_full_names, station_codes)):
        for l_idx, lead in enumerate(lead_hours):
            lead_mask = forecasts["lead_idx"].to_numpy() == l_idx
            if not lead_mask.any():
                continue
            X_lead = X_s[lead_mask]
            n_lead = X_lead.shape[0]
            station_idx = np.full(n_lead, s_idx, dtype="int64")
            summary = predict_partial_pooling_summary(
                fits[lead], X_lead, station_idx, quantiles=QUANTILES,
            )
            sub = forecasts.loc[lead_mask, ["ValidTimeUtc", "LeadHours"]].reset_index(drop=True)
            out = pd.DataFrame({
                "valid_time": sub["ValidTimeUtc"],
                "station": code,
                "station_full_name": full_name,
                "lead": int(lead),
                "anchor": anchor,
                "p_wet_mean": summary["mean"],
                "p_wet_std": summary["std"],
            })
            for q in QUANTILES:
                out[f"p_wet_q{q:g}"] = summary[f"q{q:g}"]
            out["ci80_width"] = summary["q0.9"] - summary["q0.1"]
            out["ci90_width"] = summary["q0.95"] - summary["q0.05"]
            rows_out.append(out)

    if not rows_out:
        print("  no rows produced — exiting.")
        return

    big = pd.concat(rows_out, ignore_index=True)
    print(f"  produced {len(big):,} rows total")

    # Hive layout the C# render path can join cleanly:
    # station=<full>/anchor=<YYYY-MM-DD>/widths.parquet
    for full_name, sub in big.groupby("station_full_name", observed=True):
        out_dir = out_root / f"station={full_name}" / f"anchor={anchor.strftime('%Y-%m-%d')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "widths.parquet"
        sub.drop(columns=["station_full_name"]).to_parquet(out_path, index=False)
        print(f"  wrote {len(sub):,} rows to {out_path}")

    print(f"[{time.strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
