"""Phase 4a — production train-and-predict in a single R session,
lead-as-feature design (mirrors Phase 5's lead-pooling pattern).

Why lead-as-feature: the LightGBM 3a/3c/3d artefacts are per-lead (one
.bin per lead), but BART benefits from pooling — trees can split on
`lead` to learn a continuous lead-dependence function rather than 6
disjoint per-lead bins. Result: 3 fits per run (one per station) instead
of 18 (3 stations × 6 leads), trained on ~6× more rows per fit but
same wall-clock budget overall (~24 min for the daily run).

Why train-and-predict in one session: dbarts's BART tree state lives
in C++ pointers that don't survive saveRDS / serialize round-trips —
predict in a fresh session falls back to Y.mean() (constant ≈0.69 for
binary). Workaround: pass live forecast features as `x.test` so the
fit object's `yhat.test` carries posterior-mean predictions for them.

Schedule: daily at 12:00 UTC (piggyback on existing Cloudflare tick that
fires era5-refresh). Predict for valid times in (anchor, anchor+7d) at
all leads, write predictions parquet keyed by (station, valid_time, lead).
Predict-and-render at HH:15 reads the latest day's parquet — staleness
peaks at ~20h (acceptable for a 1000-draw × 500-tree posterior mean).

CLI:
    predict_4a.py [--anchor YYYY-MM-DD] [--stations slug1 ...]

Workflow expects WEATHERBLEND_DATA_ROOT pointing at a sibling checkout of
WeatherBlend so the R2-rcloned forecast tree is reachable.
"""
from __future__ import annotations

import argparse
import os
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
from scipy.stats import norm  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import default_converter, numpy2ri, pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import LOCATION, WEATHERBLEND_DATA_ROOT, WET_THRESHOLD_MM  # noqa: E402

from run_phase6_bart_bakeoff import (  # noqa: E402
    FEATURE_NAMES,
    MODELS_LEAN,
    build_features_via_duckdb,
    resolve_station,
    time_split,
)
from run_phase6_dbarts_richfeats import add_synoptic_features  # noqa: E402

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")

NTREE = 500
K = 3.0
NSKIP = 200
NDPOST = 1000
SEED = 42
PHASE = "4a"

STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
# Leads we train + predict at. Pooled across leads with `lead` as a
# feature column so the BART learns a continuous lead-dependence curve.
LEADS = [12, 24, 48, 72, 96, 120]
# Forecast horizon (days) — we predict for all valid times in
# (anchor, anchor + HORIZON_DAYS). Open-Meteo previous_runs typically
# carries 5-7 days of forecasts, so 7 is the practical cap.
HORIZON_DAYS = 7


def build_pooled_training_features(station_friendly: str) -> pd.DataFrame:
    """Pull training rows for ALL leads in LEADS, pool with `lead` column.
    Returns a DataFrame with the 22 base + 3 synoptic + 1 `lead` columns
    plus the wet/dry label."""
    frames = []
    for lead in LEADS:
        df = build_features_via_duckdb(station_friendly, lead)
        df, _syn_feats = add_synoptic_features(station_friendly, lead, df)
        df["lead"] = float(lead)
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled


def build_pooled_live_features(station_friendly: str, anchor: datetime) -> pd.DataFrame:
    """Pull live feature rows for upcoming valid times at all leads in
    LEADS, pool with `lead` column. Returns a DataFrame with the 22
    base + 3 synoptic + 1 `lead` columns. NO truth join."""
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
          AND RunTimeSource = 'offset_day'
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

    # Rename LeadHours → lead for parity with training-time column.
    df = df.rename(columns={"LeadHours": "lead"})
    df["lead"] = df["lead"].astype(float)
    return df


def train_and_predict_one_station(station_friendly: str, anchor: datetime) -> pd.DataFrame:
    """Single-session BART fit + predict for one station, pooled across
    all leads. Returns a DataFrame with ValidTimeUtc + LeadHours + ProbWet
    for upcoming valid times × leads."""
    syn_feats = ["wind_dir_sin_mean", "wind_dir_cos_mean", "surface_pressure_mean"]
    feats = list(FEATURE_NAMES) + syn_feats + ["lead"]

    print(f"  building pooled training features ({len(LEADS)} leads)...", flush=True)
    df_train = build_pooled_training_features(station_friendly)
    train_df, _val_df, _test_df = time_split(df_train)
    X_train_full = train_df[feats].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="int8")

    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept = np.where(~col_all_nan)[0]
    feature_names_eff = [feats[i] for i in kept]
    X_train = X_train_full[:, kept]
    print(f"  pooled train: {len(y_train):,} rows | features eff: {len(feature_names_eff)}",
          flush=True)

    print(f"  building pooled live features (horizon {HORIZON_DAYS}d)...", flush=True)
    df_live = build_pooled_live_features(station_friendly, anchor)
    if len(df_live) == 0:
        print(f"  no live features past anchor — skipping station")
        return pd.DataFrame()
    X_live_full = df_live[feats].to_numpy(dtype="float64")
    X_live = X_live_full[:, kept]
    print(f"  pooled live: {len(df_live):,} rows", flush=True)

    median = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), median, X_train)
    X_live  = np.where(np.isnan(X_live), median, X_live)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float64)
    X_live_s  = scaler.transform(X_live).astype(np.float64)

    print(f"  fitting dbarts (ntree={NTREE}, k={K}, nskip={NSKIP}, ndpost={NDPOST})...",
          flush=True)
    t0 = time.time()
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train_s)
        y_train_r = ro.conversion.py2rpy(y_train.astype(np.float64))
        x_live_r  = ro.conversion.py2rpy(X_live_s)
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_live_r,
        ntree=NTREE, k=K, nskip=NSKIP, ndpost=NDPOST,
        keeptrees=True, verbose=False, seed=SEED,
    )
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat = np.array(ro.conversion.rpy2py(yhat_test_r))
    p_wet = norm.cdf(yhat).mean(axis=0)
    print(f"  fit+predict done in {(time.time() - t0) / 60:.1f} min", flush=True)

    out = pd.DataFrame({
        "ValidTimeUtc": df_live["ValidTimeUtc"],
        "LeadHours":    df_live["lead"].astype(int),
        "ProbWet":      p_wet,
    })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", default=None,
                   help="Anchor date YYYY-MM-DD UTC (default: today). Predict only "
                        "valid times after this, up to anchor+7d.")
    p.add_argument("--stations", nargs="*", default=None,
                   help="Station subset (default: all 3 active).")
    p.add_argument("--out-root", default=str(WEATHERBLEND_DATA_ROOT / "predictions"),
                   help="Predictions tree root.")
    args = p.parse_args()

    if args.anchor:
        anchor = datetime.fromisoformat(args.anchor).replace(tzinfo=timezone.utc)
    else:
        anchor = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    anchor = anchor.replace(tzinfo=None)

    stations = args.stations or STATIONS
    out_root = Path(args.out_root)

    version = datetime.now(timezone.utc).strftime("v%Y-%m-%d_%H%M%S_phase4a")
    print(f"[{time.strftime('%H:%M:%S')}] Phase 4a lead-as-feature train-and-predict")
    print(f"  anchor:  {anchor.isoformat()}")
    print(f"  version: {version}")
    print(f"  stations: {stations}")
    print(f"  leads (pooled feature): {LEADS}")
    print(f"  horizon: {HORIZON_DAYS} days")

    rows_written = 0
    for station_input in stations:
        station_slug, station_friendly = resolve_station(station_input)
        print(f"\n[{time.strftime('%H:%M:%S')}] {station_friendly}")
        try:
            preds = train_and_predict_one_station(station_friendly, anchor)
        except Exception as e:
            print(f"  FAILED — {e}")
            raise
        if len(preds) == 0:
            print(f"  no predictions emitted for {station_slug}")
            continue
        preds["ModelVersion"] = version
        preds["TruthStation"] = station_slug
        preds["PredictionMadeAtUtc"] = datetime.now(timezone.utc).replace(tzinfo=None)
        date_str = anchor.strftime("%Y-%m-%d")
        out_dir = (out_root / "precipitation" / station_slug
                   / f"model_version={version}" / f"date={date_str}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.parquet"
        preds.to_parquet(out_path, index=False)
        print(f"  → {out_path}  ({len(preds):,} rows, P(wet) mean {preds['ProbWet'].mean():.3f})")
        rows_written += len(preds)

    print()
    print(f"Phase 4a predict complete. Total rows: {rows_written:,}")
    if rows_written == 0:
        print("WARN: no predictions written.")
        sys.exit(1)


if __name__ == "__main__":
    main()
