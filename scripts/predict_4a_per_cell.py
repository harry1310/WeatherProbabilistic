"""Phase 4a per-cell — PREDICT ONLY. Loads N per-lead BART states from
the latest per-cell bundle and dispatches predictions per lead.

Mirror of predict_4a.py but for the per-cell architecture (one BART
per (station, lead) instead of one lead-pooled BART per station).
Bundle layout produced by train_4a_per_cell.py:
  data/models/precipitation/{station}/{version}_phase4a_percell/
    state_lead_24h.rds  arrays_lead_24h.npz  ... (one per lead)
    preprocess.json (per-lead blocks: median, scaler, kept_indices)

Load pattern per (station, lead):
  1. Read this lead's state_lead_NNh.rds + arrays_lead_NNh.npz + per-lead
     preprocess block.
  2. Build a tiny warm scaffold on this lead's scaled training inputs.
  3. setState(saved state) — injects the saved 500 trees × 1000 draws.
  4. predict(warm, newdata=this_lead's_scaled_live_features) →
     (ndpost, n_rows_for_this_lead) probability draws.
  5. Concatenate across leads; write predictions parquet under
     model_version=v..._phase4a_percell/date=YYYY-MM-DD/.

Output partition is distinct from deployed lead-pooled 4a's
(model_version contains `_phase4a_percell`) so both phases coexist in
the predictions tree during the challenger transition.

CLI:
    predict_4a_per_cell.py [--anchor YYYY-MM-DD] [--stations slug1 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
_r_home = os.environ.get("R_HOME", r"C:\Program Files\R\R-4.6.0")
os.environ.setdefault("R_HOME", _r_home)
_r_bin = os.path.join(_r_home, "bin", "x64")
if hasattr(os, "add_dll_directory") and os.path.isdir(_r_bin):
    os.add_dll_directory(_r_bin)
os.environ["PATH"] = _r_bin + os.pathsep + os.environ.get("PATH", "")
_user_lib = os.path.join(os.environ.get("USERPROFILE", os.environ.get("HOME", "")),
                         "R", "win-library", "4.6")
os.environ.setdefault("R_LIBS_USER", _user_lib)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import duckdb  # noqa: E402

import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import default_converter, numpy2ri, pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import LOCATION, WEATHERBLEND_DATA_ROOT, WET_THRESHOLD_MM  # noqa: E402

from _shared import (  # noqa: E402
    MODELS_LEAN,
    resolve_station,
)

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")

PHASE = "4a_percell"
STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
LEADS = [24, 48, 72, 96, 120]
HORIZON_DAYS = 7

# Same warm-scaffold rationale as predict_4a.py: NTREE is structural,
# NSKIP/NDPOST are throwaway. Read NTREE from preprocess.json's per-lead
# block (all leads share the same NTREE in current code, but pinning
# from disk avoids constant drift if hyperparams ever diverge per lead).
WARM_NSKIP = 1
WARM_NDPOST = 1

_VERSION_RE = re.compile(r"^v\d{4}-\d{2}-\d{2}_\d{6}_phase4a_percell$")


def find_latest_bundle(models_root: Path, station_slug: str) -> Path:
    """Pick the most recent v..._phase4a_percell/ dir for a station."""
    parent = models_root / "precipitation" / station_slug
    if not parent.is_dir():
        raise FileNotFoundError(
            f"no model dir for station {station_slug!r} under {parent}. "
            f"Run train_4a_per_cell.py first.")
    candidates = sorted(
        (d for d in parent.iterdir() if d.is_dir() and _VERSION_RE.match(d.name)),
        key=lambda d: d.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no *_phase4a_percell versions found under {parent}. "
            f"Run train_4a_per_cell.py to mint one.")
    latest = candidates[-1]
    required = ["preprocess.json"] + [f"state_lead_{L}h.rds" for L in LEADS] + [f"arrays_lead_{L}h.npz" for L in LEADS]
    missing = [r for r in required if not (latest / r).is_file()]
    if missing:
        raise FileNotFoundError(
            f"bundle {latest} missing required files: {missing}. "
            f"Re-run train_4a_per_cell.py.")
    return latest


def build_live_features(station_friendly: str, anchor: datetime) -> pd.DataFrame:
    """Live forecast features pooled across LEADS. Same DuckDB query as
    predict_4a.build_pooled_live_features — identical underlying NWP +
    derived features (precip stats, wind synoptic, hour/doy cyclical).
    The `lead` column comes back attached; per-cell predict filters by
    it lead-by-lead rather than passing lead as a model feature.
    """
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
    precip_pivot = ",\n        ".join(
        f"MAX(CASE WHEN Model = '{full}' THEN Precipitation END) AS precip_{short}"
        for full, short in MODELS_LEAN
    )
    horizon_end = anchor + pd.Timedelta(days=HORIZON_DAYS)
    leads_in = "(" + ",".join(str(L) for L in LEADS) + ")"
    sql = f"""
    WITH latest AS (
        SELECT ValidTimeUtc, Model, LeadHours, Precipitation,
               RelativeHumidity2m, Temperature2m, DewPoint2m,
               CloudCoverLow, CloudCoverMid, CloudCoverHigh,
               Cape, WindSpeed10m, WindDirection10m, SurfacePressure,
               ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model, LeadHours
                                  ORDER BY RunTimeUtc DESC) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND RunTimeSource = 'reported'
          AND LeadHours IN {leads_in}
          AND Model IN {model_in}
          AND ValidTimeUtc > timestamp '{anchor.isoformat()}'
          AND ValidTimeUtc <= timestamp '{horizon_end.isoformat()}'
    )
    SELECT ValidTimeUtc, LeadHours,
        {precip_pivot},
        AVG(RelativeHumidity2m)         AS rh_mean,
        AVG(Temperature2m - DewPoint2m) AS dew_depression_mean,
        AVG(CloudCoverLow)              AS cloud_low_mean,
        AVG(CloudCoverMid)              AS cloud_mid_mean,
        AVG(CloudCoverHigh)             AS cloud_high_mean,
        AVG(Cape)                       AS cape_mean,
        AVG(WindSpeed10m)               AS wind_speed_mean,
        AVG(SIN(RADIANS(WindDirection10m))) AS wind_dir_sin_mean,
        AVG(COS(RADIANS(WindDirection10m))) AS wind_dir_cos_mean,
        AVG(SurfacePressure)                AS surface_pressure_mean
    FROM latest WHERE rn = 1
    GROUP BY ValidTimeUtc, LeadHours
    ORDER BY ValidTimeUtc, LeadHours
    """
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()
    if len(df) == 0:
        return df

    precip_cols = [f"precip_{short}" for _, short in MODELS_LEAN]
    pm_arr = df[precip_cols].to_numpy(dtype="float64")
    df["precip_mean"] = np.nanmean(pm_arr, axis=1)
    present = (~np.isnan(pm_arr)).sum(axis=1)
    sumsq = np.nansum(pm_arr ** 2, axis=1)
    sumv  = np.nansum(pm_arr, axis=1)
    mean_safe = np.where(present > 0, sumv / np.maximum(present, 1), np.nan)
    var = np.maximum(0.0, sumsq / np.maximum(present, 1) - mean_safe ** 2)
    df["precip_std"] = np.where(present > 1, np.sqrt(var), 0.0)
    df["precip_max"] = np.nanmax(pm_arr, axis=1)
    wet_count = (pm_arr >= WET_THRESHOLD_MM).sum(axis=1)
    df["precip_agreement_wet_01"] = np.where(present > 0, wet_count / np.maximum(present, 1), np.nan)

    hour_angle = 2.0 * np.pi * df["ValidTimeUtc"].dt.hour / 24.0
    doy_angle  = 2.0 * np.pi * (df["ValidTimeUtc"].dt.dayofyear - 1) / 365.0
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["doy_sin"]  = np.sin(doy_angle)
    df["doy_cos"]  = np.cos(doy_angle)

    df = df.rename(columns={"LeadHours": "lead"})
    df["lead"] = df["lead"].astype(int)   # per-cell: integer lead matches dict keys
    return df


def predict_one_cell(bundle_dir: Path, lead: int, df_lead: pd.DataFrame,
                     preprocess_lead: dict) -> pd.DataFrame:
    """Score one (station, lead) cell. Loads this lead's state.rds,
    builds a warm scaffold on its arrays.npz, setStates, predicts on
    df_lead's rows. Returns the per-row predictions parquet shape."""
    arrays = np.load(bundle_dir / f"arrays_lead_{lead}h.npz")
    X_train_s = arrays["X_train_s"]
    y_train   = arrays["y_train"]

    feature_list_full = preprocess_lead["feature_list_full"]
    kept_indices      = np.array(preprocess_lead["kept_indices"], dtype=int)
    median            = np.array(preprocess_lead["median"], dtype="float64")
    scaler_mean       = np.array(preprocess_lead["scaler_mean"], dtype="float64")
    scaler_scale      = np.array(preprocess_lead["scaler_scale"], dtype="float64")
    feature_names_eff = preprocess_lead["feature_names_eff"]

    X_live_full = df_lead[feature_list_full].to_numpy(dtype="float64")
    X_live = X_live_full[:, kept_indices]
    X_live = np.where(np.isnan(X_live), median, X_live)
    X_live_s = ((X_live - scaler_mean) / scaler_scale).astype(np.float64)

    print(f"    lead {lead}h | features eff: {len(feature_names_eff)} | "
          f"live rows: {len(df_lead):,} | x_train: {X_train_s.shape}", flush=True)

    # Warm scaffold + setState — same pattern as predict_4a.py.
    warm_ntree = int(preprocess_lead.get("ntree", 500))
    t0 = time.time()
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train_s)
        y_train_r = ro.conversion.py2rpy(y_train)
    warm = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r,
        ntree=warm_ntree, nskip=WARM_NSKIP, ndpost=WARM_NDPOST,
        keeptrees=True, verbose=False, seed=preprocess_lead.get("seed", 42),
    )
    ro.globalenv["warm"] = warm

    state_path = (bundle_dir / f"state_lead_{lead}h.rds").as_posix()
    ro.r(f'bundle <- readRDS("{state_path}")')
    ro.r('warm$fit$setState(bundle$state)')

    with localconverter(_RCONVERT):
        x_live_r = ro.conversion.py2rpy(X_live_s)
    ro.globalenv["x_live"] = x_live_r
    pred_r = ro.r('predict(warm, newdata = x_live)')
    with localconverter(_RCONVERT):
        p_draws = np.array(ro.conversion.rpy2py(pred_r))
    print(f"    lead {lead}h predict done in {time.time() - t0:.1f}s — "
          f"draws shape {p_draws.shape}", flush=True)

    p_mean = p_draws.mean(axis=0)
    q05 = np.quantile(p_draws, 0.05, axis=0)
    q10 = np.quantile(p_draws, 0.10, axis=0)
    q50 = np.quantile(p_draws, 0.50, axis=0)
    q90 = np.quantile(p_draws, 0.90, axis=0)
    q95 = np.quantile(p_draws, 0.95, axis=0)

    return pd.DataFrame({
        "ValidTimeUtc": df_lead["ValidTimeUtc"].values,
        "LeadHours":    lead,
        "ProbWet":      p_mean,
        "ProbWetStd":   p_draws.std(axis=0),
        "ProbWetQ05":   q05,
        "ProbWetQ10":   q10,
        "ProbWetQ50":   q50,
        "ProbWetQ90":   q90,
        "ProbWetQ95":   q95,
        "Ci80Width":    q90 - q10,
        "Ci90Width":    q95 - q05,
    })


def predict_one_station(bundle_dir: Path, station_friendly: str,
                        anchor: datetime) -> pd.DataFrame:
    preprocess = json.loads((bundle_dir / "preprocess.json").read_text())
    if "per_lead" not in preprocess:
        raise ValueError(
            f"bundle {bundle_dir} has no `per_lead` block in preprocess.json — "
            f"likely a lead-pooled 4a bundle (use predict_4a.py instead).")

    print(f"  bundle: {bundle_dir.name}", flush=True)
    print(f"  building live features (horizon {HORIZON_DAYS}d)...", flush=True)
    df_live = build_live_features(station_friendly, anchor)
    if len(df_live) == 0:
        print(f"  no live feature rows past anchor — emitting nothing", flush=True)
        return pd.DataFrame()
    print(f"  live: {len(df_live):,} rows across leads {sorted(df_live['lead'].unique().tolist())}",
          flush=True)

    per_lead_outputs: list[pd.DataFrame] = []
    for lead in LEADS:
        if str(lead) not in preprocess["per_lead"]:
            print(f"    lead {lead}h: not in bundle's preprocess — skipping")
            continue
        df_lead = df_live[df_live["lead"] == lead].reset_index(drop=True)
        if len(df_lead) == 0:
            print(f"    lead {lead}h: no live rows for this lead — skipping")
            continue
        preprocess_lead = preprocess["per_lead"][str(lead)]
        # Carry NTREE from the top-level preprocess so we don't re-read
        # it per-cell; the train script writes one NTREE for the whole
        # bundle today.
        preprocess_lead.setdefault("ntree", preprocess.get("ntree", 500))
        preprocess_lead.setdefault("seed", preprocess.get("seed", 42))
        out = predict_one_cell(bundle_dir, lead, df_lead, preprocess_lead)
        per_lead_outputs.append(out)
        # Free R-side warm + bundle state before the next lead so peak
        # RAM stays bounded across the lead loop.
        ro.r('rm(warm, bundle, x_live); gc()')

    if not per_lead_outputs:
        return pd.DataFrame()
    return pd.concat(per_lead_outputs, ignore_index=True).sort_values(
        ["ValidTimeUtc", "LeadHours"], kind="mergesort").reset_index(drop=True)


def write_predictions_parquet(predictions_root: Path, station_slug: str,
                              bundle_name: str, anchor: datetime,
                              predictions: pd.DataFrame) -> Path:
    """Append-friendly write under model_version=<version>/date=<anchor>/.
    Same pattern as predict_4a.py / predict_5a.py: one parquet per
    (station, model_version, anchor-date) cycle."""
    date_str = anchor.date().isoformat()
    out_dir = (predictions_root / "precipitation" / station_slug
               / f"model_version={bundle_name}" / f"date={date_str}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.parquet"
    predictions = predictions.copy()
    predictions["LocationName"]        = LOCATION
    predictions["ModelVersion"]        = bundle_name
    predictions["TruthStation"]        = station_slug
    predictions["PredictionMadeAtUtc"] = datetime.now(timezone.utc)
    predictions.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", default=None,
                   help="Anchor date YYYY-MM-DD UTC (default: today).")
    p.add_argument("--stations", nargs="*", default=None,
                   help="Station subset (default: all 3 active).")
    p.add_argument("--predictions-root", default=str(WEATHERBLEND_DATA_ROOT / "predictions"),
                   help="Predictions tree root.")
    p.add_argument("--models-root", default=str(WEATHERBLEND_DATA_ROOT / "models"),
                   help="Models tree root holding the saved bundles.")
    args = p.parse_args()

    if args.anchor:
        anchor = datetime.fromisoformat(args.anchor).replace(tzinfo=timezone.utc)
    else:
        anchor = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    anchor = anchor.replace(tzinfo=None)

    stations = args.stations or STATIONS
    predictions_root = Path(args.predictions_root)
    models_root = Path(args.models_root)

    print(f"[{time.strftime('%H:%M:%S')}] Phase 4a per-cell PREDICT")
    print(f"  anchor:   {anchor.isoformat()}")
    print(f"  stations: {stations}")
    print(f"  leads:    {LEADS}")
    print(f"  horizon:  {HORIZON_DAYS} days")

    rows_written = 0
    for station_input in stations:
        station_slug, station_friendly = resolve_station(station_input)
        print(f"\n[{time.strftime('%H:%M:%S')}] {station_friendly}")
        try:
            bundle_dir = find_latest_bundle(models_root, station_slug)
        except FileNotFoundError as e:
            print(f"  WARN: {e}")
            continue
        preds = predict_one_station(bundle_dir, station_friendly, anchor)
        if preds.empty:
            print(f"  no predictions emitted for {station_slug}")
            continue
        out_path = write_predictions_parquet(predictions_root, station_slug,
                                              bundle_dir.name, anchor, preds)
        print(f"  wrote {len(preds):,} rows → {out_path}")
        rows_written += len(preds)
        # Per-station R-side cleanup so the next station starts fresh.
        ro.r('gc()')

    print()
    print(f"Phase 4a per-cell predict complete. Rows written: {rows_written:,}")
    if rows_written == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
