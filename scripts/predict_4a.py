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
import json
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

from _shared import (  # noqa: E402
    FEATURE_NAMES,
    MODELS_LEAN,
    add_synoptic_features,
    build_features_via_duckdb,
    resolve_station,
    time_split,
)

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
    base + 3 synoptic + 1 `lead` columns. NO truth join.

    RunTimeSource = 'reported' here (NOT 'offset_day') — training pulls
    from the historical offset_day archive but live forecasts land in
    the date-partitioned 'reported' tree. predict_live_with_ci.py
    documents the same split. Small distribution shift between the two
    is accepted for the MVP — calibration drifts slightly but the BART
    posterior structure is robust to it.
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

    # Rename LeadHours → lead for parity with training-time column.
    df = df.rename(columns={"LeadHours": "lead"})
    df["lead"] = df["lead"].astype(float)
    return df


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def train_and_predict_one_station(station_friendly: str, anchor: datetime) -> dict:
    """Single-session BART fit, pooled across all leads. Stacks the held-
    out test slice + live-future features as `x.test` so one fit yields
    both: honest test Brier per lead (for training_metadata.PerLead) AND
    live predictions (for predictions.parquet).

    Returns a dict with:
      live_preds  - DataFrame of live (ValidTimeUtc, LeadHours, ProbWet)
      per_lead    - list of dicts, one per lead: TestRows, BlendTestMae, BSS, ...
      train_df    - the training slice used (for date ranges)
      val_df      - val slice (for date ranges)
      test_df     - test slice (for date ranges)
      feature_names_eff - the effective feature list after all-NaN drop
    """
    syn_feats = ["wind_dir_sin_mean", "wind_dir_cos_mean", "surface_pressure_mean"]
    feats = list(FEATURE_NAMES) + syn_feats + ["lead"]

    print(f"  building pooled training features ({len(LEADS)} leads)...", flush=True)
    df_pooled = build_pooled_training_features(station_friendly)
    train_df, val_df, test_df = time_split(df_pooled)

    X_train_full = train_df[feats].to_numpy(dtype="float64")
    y_train      = train_df["wet"].to_numpy(dtype="int8")
    X_test_full  = test_df[feats].to_numpy(dtype="float64")
    y_test       = test_df["wet"].to_numpy(dtype="int8")

    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept = np.where(~col_all_nan)[0]
    feature_names_eff = [feats[i] for i in kept]
    X_train = X_train_full[:, kept]
    X_test  = X_test_full[:, kept]
    print(f"  train: {len(y_train):,} | val: {len(val_df):,} | test: {len(y_test):,} | "
          f"features eff: {len(feature_names_eff)}", flush=True)

    print(f"  building pooled live features (horizon {HORIZON_DAYS}d)...", flush=True)
    df_live = build_pooled_live_features(station_friendly, anchor)
    has_live = len(df_live) > 0
    if has_live:
        X_live_full = df_live[feats].to_numpy(dtype="float64")
        X_live = X_live_full[:, kept]
        print(f"  live: {len(df_live):,} rows", flush=True)
    else:
        X_live = np.zeros((0, X_train.shape[1]))
        print(f"  no live feature rows past anchor — emitting metadata only", flush=True)

    # Median-impute + standardise — fit on train only, apply to all slices.
    median = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), median, X_train)
    X_test  = np.where(np.isnan(X_test),  median, X_test)
    X_live  = np.where(np.isnan(X_live),  median, X_live)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float64)
    X_test_s  = scaler.transform(X_test).astype(np.float64)
    # sklearn 1.4+ rejects 0-row inputs in transform(); guard the empty-live
    # case so the script can still emit training_metadata when the live
    # forecast tree hasn't refreshed yet.
    X_live_s = (scaler.transform(X_live).astype(np.float64)
                if has_live else np.zeros((0, X_train_s.shape[1]), dtype=np.float64))

    # Stack [test, live] as x.test so one fit covers both.
    n_test = X_test_s.shape[0]
    n_live = X_live_s.shape[0]
    X_holdouts = np.vstack([X_test_s, X_live_s]) if n_live > 0 else X_test_s

    print(f"  fitting dbarts (ntree={NTREE}, k={K}, nskip={NSKIP}, ndpost={NDPOST})...",
          flush=True)
    t0 = time.time()
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train_s)
        y_train_r = ro.conversion.py2rpy(y_train.astype(np.float64))
        x_holdouts_r = ro.conversion.py2rpy(X_holdouts)
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_holdouts_r,
        ntree=NTREE, k=K, nskip=NSKIP, ndpost=NDPOST,
        keeptrees=True, verbose=False, seed=SEED,
    )
    yhat_holdouts_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat = np.array(ro.conversion.rpy2py(yhat_holdouts_r))
    # Per-draw P(wet) over (1000 draws, n_holdouts) — full posterior
    # distribution. We keep it for downstream uncertainty quantification
    # (matches Phase 5a's CI columns); the headline ProbWet is the mean
    # over draws.
    p_draws_holdouts = norm.cdf(yhat)             # (ndpost, n_holdouts)
    p_holdouts = p_draws_holdouts.mean(axis=0)    # posterior mean
    print(f"  fit done in {(time.time() - t0) / 60:.1f} min", flush=True)

    p_test = p_holdouts[:n_test]
    p_live = p_holdouts[n_test:] if n_live > 0 else np.zeros(0, dtype="float64")
    p_draws_live = (p_draws_holdouts[:, n_test:] if n_live > 0
                    else np.zeros((p_draws_holdouts.shape[0], 0), dtype="float64"))

    # Per-lead test Brier (for the Models page card).
    per_lead = []
    test_lead = test_df["lead"].astype(int).to_numpy() if "lead" in test_df.columns else None
    if test_lead is None:
        # `lead` column is set by build_pooled_training_features; if missing
        # (shouldn't happen) fall back to one bucket.
        per_lead.append({"LeadHours": -1, "TestRows": int(n_test),
                         "BlendTestMae": _brier(p_test, y_test)})
    else:
        train_clim = float(train_df["wet"].mean())  # pooled climatology
        for L in LEADS:
            mask = test_lead == L
            n = int(mask.sum())
            if n == 0:
                per_lead.append({"LeadHours": L, "TestRows": 0,
                                 "BlendTestMae": float("nan"), "BSS": float("nan")})
                continue
            p_l = p_test[mask]
            y_l = y_test[mask]
            b = _brier(p_l, y_l)
            clim_b = _brier(np.full_like(y_l, train_clim, dtype="float64"), y_l)
            bss = (clim_b - b) / clim_b if clim_b > 0 else float("nan")
            best_single_idx = None  # not meaningful for BART (per-NWP cols all blended)
            per_lead.append({
                "LeadHours":   L,
                "TestRows":    n,
                "ValRows":     int((val_df["lead"] == L).sum()) if "lead" in val_df.columns else 0,
                "TrainRows":   int((train_df["lead"] == L).sum()) if "lead" in train_df.columns else 0,
                "BlendTestMae": b,
                "BSS":          bss,
                "ClimatologyBrier": clim_b,
                "BestSingle":   "(BART blender — pooled across NWPs)",
                "BestSingleTestMae": float("nan"),
            })

    live_preds = pd.DataFrame()
    if has_live and n_live > 0:
        # CI columns mirror Phase 5a's schema so the WeatherBlend renderer
        # can read uncertainty from either phase via the same column names.
        # Quantiles use np.quantile across the 1000-draw posterior axis;
        # ci80/ci90 widths are the q10..q90 / q05..q95 spans.
        q05 = np.quantile(p_draws_live, 0.05, axis=0)
        q10 = np.quantile(p_draws_live, 0.10, axis=0)
        q50 = np.quantile(p_draws_live, 0.50, axis=0)
        q90 = np.quantile(p_draws_live, 0.90, axis=0)
        q95 = np.quantile(p_draws_live, 0.95, axis=0)
        live_preds = pd.DataFrame({
            "ValidTimeUtc": df_live["ValidTimeUtc"],
            "LeadHours":    df_live["lead"].astype(int),
            "ProbWet":      p_live,
            "ProbWetStd":   p_draws_live.std(axis=0),
            "ProbWetQ05":   q05,
            "ProbWetQ10":   q10,
            "ProbWetQ50":   q50,
            "ProbWetQ90":   q90,
            "ProbWetQ95":   q95,
            "Ci80Width":    q90 - q10,
            "Ci90Width":    q95 - q05,
        })

    return {
        "live_preds": live_preds,
        "per_lead":   per_lead,
        "train_df":   train_df,
        "val_df":     val_df,
        "test_df":    test_df,
        "feature_names_eff": feature_names_eff,
    }


def write_metadata(out_models_dir: Path, station_slug: str, station_friendly: str,
                   version: str, result: dict, anchor: datetime) -> None:
    """Emit training_metadata.json + feature_schema.json under
    data/models/precipitation/{station}/{version}/ so WeatherBlend's
    LoadModelSummaries + spec page + verify pipeline pick 4a up.
    """
    syn_feats = ["wind_dir_sin_mean", "wind_dir_cos_mean", "surface_pressure_mean"]
    base_feats = list(FEATURE_NAMES) + syn_feats + ["lead"]

    train_df = result["train_df"]
    val_df   = result["val_df"]
    test_df  = result["test_df"]
    per_lead = result["per_lead"]
    feature_names_eff = result["feature_names_eff"]

    metadata = {
        "Version":     version,
        "Target":      "precipitation",
        "Phase":       PHASE,
        "DataSource":  "open_meteo_previous_runs+ea_rainfall+dbarts_bart_lead_as_feature",
        "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
        "Hyperparameters": {
            "library":   "dbarts (R) via rpy2",
            "ntree":     NTREE,
            "k":         K,
            "nskip":     NSKIP,
            "ndpost":    NDPOST,
            "seed":      SEED,
            "objective": "binary (probit-link BART)",
            "leadAsFeature": True,
        },
        "DeviationsFromBrief": [
            "BART (Bayesian Additive Regression Trees) via R dbarts package, "
            "called from Python via rpy2. Single-session train-and-predict "
            "(dbarts trees can't round-trip via saveRDS).",
            "Lead pooled across all six leads via a `lead` feature column "
            "(Phase 5 pattern); one BART per station instead of per (station, "
            "lead).",
            "22-feature 3a base + 3 synoptic flow features (wind_dir_sin_mean, "
            "wind_dir_cos_mean, surface_pressure_mean).",
            "Predict horizon limited to 7 days from anchor (Open-Meteo "
            "previous_runs forecast extent).",
        ],
        "PerLead": {str(d["LeadHours"]): {
            **d,
            "DataRangeTrain": (
                f"{train_df['ValidTimeUtc'].min()} → {train_df['ValidTimeUtc'].max()}"
                if len(train_df) else ""),
            "DataRangeVal":   (
                f"{val_df['ValidTimeUtc'].min()} → {val_df['ValidTimeUtc'].max()}"
                if len(val_df) else ""),
            "DataRangeTest":  (
                f"{test_df['ValidTimeUtc'].min()} → {test_df['ValidTimeUtc'].max()}"
                if len(test_df) else ""),
            "TestCalendarMonths": 4,
        } for d in per_lead if d["LeadHours"] > 0},
    }

    # feature_schema.json — replicate per-lead even though BART pools (so the
    # Spec page picks up one row per (composite, phase, lead) per its
    # existing iteration model). All leads share the same FeatureNames since
    # they all use the lead-as-feature pooled BART.
    nwp_models = [m for m, _ in MODELS_LEAN]
    schema_per_lead = {
        str(L): {
            "Target": "precipitation",
            "FeatureSet": f"phase4a-bart-l{L:02}",
            "LeadHours":  L,
            "RequiredModels": [],
            "OptionalModels": nwp_models,
            "Models":         nwp_models,
            "FeatureNames":   feature_names_eff,
            "DataSource":     "open_meteo_previous_runs",
            "Tier":           "4a-bart",
            "UkvStrategy":    None,
        } for L in LEADS
    }
    schema = {"Leads": schema_per_lead}

    out_models_dir.mkdir(parents=True, exist_ok=True)
    (out_models_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str))
    (out_models_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2))
    print(f"  metadata → {out_models_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", default=None,
                   help="Anchor date YYYY-MM-DD UTC (default: today). Predict only "
                        "valid times after this, up to anchor+7d.")
    p.add_argument("--stations", nargs="*", default=None,
                   help="Station subset (default: all 3 active).")
    p.add_argument("--predictions-root", default=str(WEATHERBLEND_DATA_ROOT / "predictions"),
                   help="Predictions tree root.")
    p.add_argument("--models-root", default=str(WEATHERBLEND_DATA_ROOT / "models"),
                   help="Models tree root for metadata files.")
    args = p.parse_args()

    if args.anchor:
        anchor = datetime.fromisoformat(args.anchor).replace(tzinfo=timezone.utc)
    else:
        anchor = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    anchor = anchor.replace(tzinfo=None)

    stations = args.stations or STATIONS
    predictions_root = Path(args.predictions_root)
    models_root = Path(args.models_root)

    version = datetime.now(timezone.utc).strftime("v%Y-%m-%d_%H%M%S_phase4a")
    print(f"[{time.strftime('%H:%M:%S')}] Phase 4a lead-as-feature train-and-predict")
    print(f"  anchor:  {anchor.isoformat()}")
    print(f"  version: {version}")
    print(f"  stations: {stations}")
    print(f"  leads (pooled feature): {LEADS}")
    print(f"  horizon: {HORIZON_DAYS} days")

    rows_written = 0
    versions_emitted = 0
    for station_input in stations:
        station_slug, station_friendly = resolve_station(station_input)
        print(f"\n[{time.strftime('%H:%M:%S')}] {station_friendly}")
        try:
            result = train_and_predict_one_station(station_friendly, anchor)
        except Exception as e:
            print(f"  FAILED — {e}")
            raise

        # Emit metadata regardless of live coverage (test-set Brier is the
        # main driver for the Models card).
        models_dir = models_root / "precipitation" / station_slug / version
        write_metadata(models_dir, station_slug, station_friendly, version, result, anchor)
        versions_emitted += 1

        live_preds = result["live_preds"]
        if len(live_preds) == 0:
            print(f"  no live predictions emitted for {station_slug}")
            continue
        live_preds["ModelVersion"] = version
        live_preds["TruthStation"] = station_slug
        live_preds["PredictionMadeAtUtc"] = datetime.now(timezone.utc).replace(tzinfo=None)
        date_str = anchor.strftime("%Y-%m-%d")
        out_dir = (predictions_root / "precipitation" / station_slug
                   / f"model_version={version}" / f"date={date_str}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.parquet"
        live_preds.to_parquet(out_path, index=False)
        print(f"  → {out_path}  ({len(live_preds):,} rows, P(wet) mean {live_preds['ProbWet'].mean():.3f})")
        rows_written += len(live_preds)

    print()
    print(f"Phase 4a complete. Versions: {versions_emitted}, prediction rows: {rows_written:,}")
    if versions_emitted == 0:
        print("ERROR: no versions emitted — check forecast tree presence.")
        sys.exit(1)


if __name__ == "__main__":
    main()
