"""Phase 4a — PREDICT ONLY. Loads the saved BART state from the latest
versioned bundle in data/models/precipitation/{station}/{version}/,
applies it to fresh live forecast features, writes predictions parquet
under the SAME version (4× per day under different date= partitions).

Pre-split this script also did the training fit (~24 min wall, ran daily
piggybacking on the noon ERA5 tick). Now training is its own workflow
(train_4a.py + train-4a.yml) that emits a state.rds bundle, and this
script just rehydrates and predicts on each 6-hourly cycle (~30s wall).

Load pattern (verified bit-exact via scripts/smoke_dbarts_roundtrip.py):
  1. Read state.rds + arrays.npz + preprocess.json from the latest
     versioned bundle dir for each station.
  2. Build a tiny warm scaffold: bart(X_train_s, y_train, ntree=50,
     nskip=1, ndpost=1, keeptrees=TRUE, seed=SEED) on the SAME scaled
     training inputs (~1s). This reproduces the original binary
     detection + cutpoint inference that setState requires.
  3. warm$fit$setState(bundle$state) — injects the saved 500 trees ×
     1000 draws in place of the throwaway 50-tree warmup.
  4. predict(warm, newdata=x_live_scaled) returns probabilities (auto-
     pnorm for binary) of shape (ndpost, n_live).

CLI:
    predict_4a.py [--anchor YYYY-MM-DD] [--stations slug1 ...]
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

PHASE = "4a"
STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
LEADS = [12, 24, 48, 72, 96, 120]
HORIZON_DAYS = 7

# Warm-scaffold knobs — tiny by design. Real trees come from setState.
WARM_NTREE = 50
WARM_NSKIP = 1
WARM_NDPOST = 1


_VERSION_RE = re.compile(r"^v\d{4}-\d{2}-\d{2}_\d{6}_phase4a$")


def find_latest_bundle(models_root: Path, station_slug: str) -> Path:
    """Pick the most recent v<date>_<HHMMSS>_phase4a/ dir under
    {models_root}/precipitation/{station_slug}/. Lexicographic sort
    works because the version timestamp prefix is fixed-width.
    """
    parent = models_root / "precipitation" / station_slug
    if not parent.is_dir():
        raise FileNotFoundError(
            f"no model dir for station {station_slug!r} under {parent}. "
            f"Run train_4a.py first to mint an initial bundle."
        )
    candidates = sorted(
        (d for d in parent.iterdir() if d.is_dir() and _VERSION_RE.match(d.name)),
        key=lambda d: d.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no *_phase4a versions found under {parent}. "
            f"Run train_4a.py to mint one."
        )
    latest = candidates[-1]
    # Sanity-check the bundle is complete; predict-time partial bundles
    # would silently fall back to a wrong feature shape and emit garbage.
    required = ["state.rds", "arrays.npz", "preprocess.json"]
    missing = [r for r in required if not (latest / r).is_file()]
    if missing:
        raise FileNotFoundError(
            f"bundle {latest} missing required files: {missing}. "
            f"Re-run train_4a.py."
        )
    return latest


def build_pooled_live_features(station_friendly: str, anchor: datetime) -> pd.DataFrame:
    """Live (post-anchor) forecast features pooled across LEADS with a
    `lead` column. Mirrors train-time features except RunTimeSource is
    'reported' (live forecast tree, date-partitioned) instead of
    'offset_day' (training archive). Small distribution shift between
    the two is documented in the script-level comments of the original
    train+predict combo and accepted for the MVP.
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
    df["lead"] = df["lead"].astype(float)
    return df


def predict_one_station(bundle_dir: Path, station_friendly: str,
                        anchor: datetime) -> pd.DataFrame:
    preprocess = json.loads((bundle_dir / "preprocess.json").read_text())
    arrays = np.load(bundle_dir / "arrays.npz")
    X_train_s = arrays["X_train_s"]
    y_train   = arrays["y_train"]

    feature_list_full = preprocess["feature_list_full"]
    kept_indices      = np.array(preprocess["kept_indices"], dtype=int)
    median            = np.array(preprocess["median"], dtype="float64")
    scaler_mean       = np.array(preprocess["scaler_mean"], dtype="float64")
    scaler_scale      = np.array(preprocess["scaler_scale"], dtype="float64")

    print(f"  bundle: {bundle_dir.name}", flush=True)
    print(f"  features eff: {len(preprocess['feature_names_eff'])} | "
          f"x_train: {X_train_s.shape}", flush=True)

    print(f"  building pooled live features (horizon {HORIZON_DAYS}d)...", flush=True)
    df_live = build_pooled_live_features(station_friendly, anchor)
    if len(df_live) == 0:
        print(f"  no live feature rows past anchor — emitting nothing", flush=True)
        return pd.DataFrame()

    X_live_full = df_live[feature_list_full].to_numpy(dtype="float64")
    X_live = X_live_full[:, kept_indices]
    X_live = np.where(np.isnan(X_live), median, X_live)
    X_live_s = ((X_live - scaler_mean) / scaler_scale).astype(np.float64)
    print(f"  live: {len(df_live):,} rows", flush=True)

    print(f"  building warm scaffold (ntree={WARM_NTREE}, nskip={WARM_NSKIP}, "
          f"ndpost={WARM_NDPOST})...", flush=True)
    t0 = time.time()
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train_s)
        y_train_r = ro.conversion.py2rpy(y_train)
    warm = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r,
        ntree=WARM_NTREE, nskip=WARM_NSKIP, ndpost=WARM_NDPOST,
        keeptrees=True, verbose=False, seed=preprocess.get("seed", 42),
    )
    ro.globalenv["warm"] = warm

    state_path = (bundle_dir / "state.rds").as_posix()
    ro.r(f'bundle <- readRDS("{state_path}")')
    ro.r('warm$fit$setState(bundle$state)')
    print(f"  setState done in {time.time() - t0:.1f}s — predicting...", flush=True)

    t1 = time.time()
    with localconverter(_RCONVERT):
        x_live_r = ro.conversion.py2rpy(X_live_s)
    ro.globalenv["x_live"] = x_live_r
    pred_r = ro.r('predict(warm, newdata = x_live)')
    with localconverter(_RCONVERT):
        # predict.bart for binary returns probabilities directly (auto-
        # pnorm). Shape: (ndpost, n_live).
        p_draws = np.array(ro.conversion.rpy2py(pred_r))
    print(f"  predict done in {time.time() - t1:.1f}s — draws shape {p_draws.shape}",
          flush=True)

    p_mean = p_draws.mean(axis=0)
    q05 = np.quantile(p_draws, 0.05, axis=0)
    q10 = np.quantile(p_draws, 0.10, axis=0)
    q50 = np.quantile(p_draws, 0.50, axis=0)
    q90 = np.quantile(p_draws, 0.90, axis=0)
    q95 = np.quantile(p_draws, 0.95, axis=0)

    return pd.DataFrame({
        "ValidTimeUtc": df_live["ValidTimeUtc"],
        "LeadHours":    df_live["lead"].astype(int),
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", default=None,
                   help="Anchor date YYYY-MM-DD UTC (default: today). Predict "
                        "valid times in (anchor, anchor+7d).")
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

    print(f"[{time.strftime('%H:%M:%S')}] Phase 4a PREDICT (state-loading)")
    print(f"  anchor:  {anchor.isoformat()}")
    print(f"  stations: {stations}")
    print(f"  leads (pooled feature): {LEADS}")
    print(f"  horizon: {HORIZON_DAYS} days")

    rows_written = 0
    failures = []
    for station_input in stations:
        station_slug, station_friendly = resolve_station(station_input)
        print(f"\n[{time.strftime('%H:%M:%S')}] {station_friendly}")
        try:
            bundle_dir = find_latest_bundle(models_root, station_slug)
            live_preds = predict_one_station(bundle_dir, station_friendly, anchor)
        except Exception as e:
            print(f"  FAILED — {e}")
            failures.append((station_slug, str(e)))
            continue
        finally:
            # Drop the warm fit before the next station so peak RAM stays bounded.
            ro.r('if (exists("warm")) rm(warm); if (exists("bundle")) rm(bundle); '
                 'if (exists("x_live")) rm(x_live); gc()')

        if len(live_preds) == 0:
            print(f"  no live predictions emitted for {station_slug}")
            continue

        # LocationName is the C# renderer's filter target; without it
        # union_by_name fills NULL on read and the WHERE filter drops
        # every 4a row from the site.
        version = bundle_dir.name
        live_preds["LocationName"] = LOCATION
        live_preds["ModelVersion"] = version
        live_preds["TruthStation"] = station_slug
        live_preds["PredictionMadeAtUtc"] = datetime.now(timezone.utc).replace(tzinfo=None)
        date_str = anchor.strftime("%Y-%m-%d")
        out_dir = (predictions_root / "precipitation" / station_slug
                   / f"model_version={version}" / f"date={date_str}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.parquet"
        live_preds.to_parquet(out_path, index=False)
        print(f"  → {out_path}  ({len(live_preds):,} rows, "
              f"P(wet) mean {live_preds['ProbWet'].mean():.3f})")
        rows_written += len(live_preds)

    print()
    print(f"Phase 4a predict complete. Prediction rows: {rows_written:,}")
    if failures:
        print(f"  failures: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
