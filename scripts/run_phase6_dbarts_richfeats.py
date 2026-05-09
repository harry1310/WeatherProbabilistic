"""Phase 6 — synoptic flow + per-NWP rolling-error feature experiments.

Three dbarts fits at ntree=50 on the same Bellever 24h 5k subsample:
  v1 (synoptic):   22 base + 3 synoptic flow features (wind_dir_sin/cos, surface_pressure)
  v2 (rolling):    22 base + 14 per-NWP rolling-error features (mae_30d, bias_30d × 7 NWPs)
  v3 (both):       22 base + 3 + 14 = 39 features

Predict-time legality:
  * synoptic flow (v1): forecast-derived at the target valid time T — same
    legality as the existing 22 features, fine.
  * rolling error (v2): each NWP's |forecast − gauge truth| averaged over a
    30-day window ending at T-24h (so the most recent error in the window is
    24h before the prediction target — within the predict-time info set).

Goal: see if either (or both) breaks below the 22-feature 0.1207 baseline.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
_r_home = r"C:\Program Files\R\R-4.6.0"
os.environ.setdefault("R_HOME", _r_home)
_r_bin = os.path.join(_r_home, "bin", "x64")
if hasattr(os, "add_dll_directory") and os.path.isdir(_r_bin):
    os.add_dll_directory(_r_bin)
os.environ["PATH"] = _r_bin + os.pathsep + os.environ.get("PATH", "")
_user_lib = os.path.join(os.environ.get("USERPROFILE", ""), "R", "win-library", "4.6")
os.environ.setdefault("R_LIBS_USER", _user_lib)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scipy.stats import norm  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import default_converter, numpy2ri, pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import LOCATION, WEATHERBLEND_DATA_ROOT  # noqa: E402

from run_phase6_bart_bakeoff import (  # noqa: E402
    FEATURE_NAMES,
    MODELS_LEAN,
    OUTPUT_ROOT,
    brier,
    build_features_via_duckdb,
    reliability_table,
    resolve_station,
    time_split,
)

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")


def add_synoptic_features(station_friendly: str, lead_hours: int,
                           df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pull NWP-mean wind direction (encoded as sin/cos unit vector to avoid
    the 0°/360° circular-mean discontinuity) and NWP-mean surface pressure
    via DuckDB, then merge onto df by ValidTimeUtc.
    """
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in_clause = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
    sql = f"""
    WITH latest AS (
        SELECT
            ValidTimeUtc, Model,
            WindDirection10m, SurfacePressure,
            ROW_NUMBER() OVER (
                PARTITION BY ValidTimeUtc, Model
                ORDER BY RunTimeUtc DESC
            ) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours = {lead_hours}
          AND Model IN {model_in_clause}
    )
    SELECT
        ValidTimeUtc,
        AVG(SIN(RADIANS(WindDirection10m))) AS wind_dir_sin_mean,
        AVG(COS(RADIANS(WindDirection10m))) AS wind_dir_cos_mean,
        AVG(SurfacePressure)                AS surface_pressure_mean
    FROM latest
    WHERE rn = 1
    GROUP BY ValidTimeUtc
    ORDER BY ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    syn = con.execute(sql).fetch_df()
    con.close()
    df = df.merge(syn, on="ValidTimeUtc", how="left")
    return df, ["wind_dir_sin_mean", "wind_dir_cos_mean", "surface_pressure_mean"]


def add_rolling_error_features(station_friendly: str, lead_hours: int,
                                df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """For each NWP, compute a 30-day-trailing rolling MAE and bias of its
    precip forecast at this lead vs the EA gauge truth. Window ends 24h
    before the target valid time so it's predict-time legal.
    """
    # Long-form (valid_time, model, precip_forecast, precip_truth) — pull at
    # the SAME lead used by the prediction so trust signals stay
    # like-for-like. Forecast = latest run for each (valid_time, model);
    # truth = the strict 4-of-4 hourly EA gauge.
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    rn_glob = str((WEATHERBLEND_DATA_ROOT / "truth" / "rainfall" / "**" / "*.parquet")).replace("\\", "/")
    model_in_clause = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
    sql = f"""
    WITH hourly_truth AS (
        SELECT
            date_trunc('hour', ObservedTimeUtc) AS valid_time,
            SUM(Value15MinMm) AS precip_mm_hour
        FROM read_parquet('{rn_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND StationName  = '{station_friendly}'
          AND Value15MinMm IS NOT NULL
        GROUP BY 1
        HAVING COUNT(*) = 4
    ),
    latest AS (
        SELECT
            ValidTimeUtc, Model, Precipitation,
            ROW_NUMBER() OVER (
                PARTITION BY ValidTimeUtc, Model
                ORDER BY RunTimeUtc DESC
            ) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours = {lead_hours}
          AND Model IN {model_in_clause}
    )
    SELECT
        l.ValidTimeUtc, l.Model, l.Precipitation,
        t.precip_mm_hour AS truth
    FROM latest l
    JOIN hourly_truth t ON l.ValidTimeUtc = t.valid_time
    WHERE l.rn = 1
      AND l.Precipitation IS NOT NULL
    ORDER BY l.Model, l.ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    long_df = con.execute(sql).fetch_df()
    con.close()
    long_df["err"] = long_df["Precipitation"] - long_df["truth"]
    long_df["abs_err"] = long_df["err"].abs()

    new_features: list[str] = []
    # Per-NWP rolling stats: shift the timestamp forward by 24h so that, when
    # we evaluate the rolling mean at time T, the underlying values come from
    # times ≤ T-24h. Then 30-day window over the shifted series.
    pieces = []
    for _, short in MODELS_LEAN:
        sub = long_df[long_df["Model"] == [m for m in MODELS_LEAN if m[1] == short][0][0]]
        if sub.empty:
            continue
        sub = sub.set_index("ValidTimeUtc").sort_index()
        # Shift timestamps forward 24h so a query at T reads from values
        # originally at T-24h.
        shifted_abs = sub["abs_err"].shift(freq="24h")
        shifted_err = sub["err"].shift(freq="24h")
        mae = shifted_abs.rolling("30d", min_periods=20).mean().rename(f"mae_30d_{short}")
        bias = shifted_err.rolling("30d", min_periods=20).mean().rename(f"bias_30d_{short}")
        out = pd.concat([mae, bias], axis=1).reset_index()
        pieces.append(out)
        new_features += [f"mae_30d_{short}", f"bias_30d_{short}"]

    rolled = pieces[0]
    for piece in pieces[1:]:
        rolled = rolled.merge(piece, on="ValidTimeUtc", how="outer")
    df = df.merge(rolled, on="ValidTimeUtc", how="left")
    return df, new_features


def fit_dbarts_with_holdouts(X_train, y_train, X_val, X_test, *, n_trees, n_burn,
                              n_samples, seed):
    n_val = X_val.shape[0]
    X_holdouts = np.vstack([X_val, X_test])
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train.astype(np.float64))
        y_train_r = ro.conversion.py2rpy(y_train.astype(np.float64))
        x_holdouts_r = ro.conversion.py2rpy(X_holdouts.astype(np.float64))
    t0 = time.time()
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_holdouts_r,
        ntree=n_trees, nskip=n_burn, ndpost=n_samples,
        keeptrees=True, verbose=False, seed=seed,
    )
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat = np.array(ro.conversion.rpy2py(yhat_test_r))
    wall = time.time() - t0
    p_holdouts = norm.cdf(yhat).mean(axis=0)
    return p_holdouts[:n_val], p_holdouts[n_val:], wall


def prepare_matrices(df: pd.DataFrame, feature_list: list[str]):
    """Subsample the same 5k rows, drop all-NaN-in-train cols, median-impute,
    standardise — exactly the pipeline the sweep used."""
    train_df, val_df, test_df = time_split(df)
    rng = np.random.default_rng(42)
    wet_idx = train_df.index[train_df["wet"] == 1].to_numpy().copy()
    dry_idx = train_df.index[train_df["wet"] == 0].to_numpy().copy()
    rng.shuffle(wet_idx); rng.shuffle(dry_idx)
    wet_keep = int(round(5000 * len(wet_idx) / len(train_df)))
    keep_idx = np.sort(np.concatenate([wet_idx[:wet_keep], dry_idx[:5000 - wet_keep]]))
    train_df = train_df.loc[keep_idx].reset_index(drop=True)

    X_train_full = train_df[feature_list].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="int8")
    X_val_full = val_df[feature_list].to_numpy(dtype="float64")
    y_val = val_df["wet"].to_numpy(dtype="int8")
    X_test_full = test_df[feature_list].to_numpy(dtype="float64")
    y_test = test_df["wet"].to_numpy(dtype="int8")

    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept = np.where(~col_all_nan)[0]
    X_train = X_train_full[:, kept]
    X_val = X_val_full[:, kept]
    X_test = X_test_full[:, kept]
    median = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), median, X_train)
    X_val = np.where(np.isnan(X_val), median, X_val)
    X_test = np.where(np.isnan(X_test), median, X_test)
    scaler = StandardScaler().fit(X_train)
    return (scaler.transform(X_train), y_train,
            scaler.transform(X_val), y_val,
            scaler.transform(X_test), y_test,
            train_df, len(kept))


def main() -> None:
    station_slug, station_friendly = resolve_station("ea_bellever_dartmoor")
    out_dir = OUTPUT_ROOT / f"{station_slug}_lead24"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building base features (22)…")
    df = build_features_via_duckdb(station_friendly, 24)
    print(f"  rows: {len(df):,}")

    print(f"[{time.strftime('%H:%M:%S')}] Adding synoptic flow features…")
    df, syn_feats = add_synoptic_features(station_friendly, 24, df)
    print(f"  + {syn_feats}")
    for f in syn_feats:
        print(f"    {f:30s} NaN rate {df[f].isna().mean() * 100:5.1f}%")

    print(f"[{time.strftime('%H:%M:%S')}] Adding per-NWP rolling-error features (30d, ending T-24h)…")
    df, roll_feats = add_rolling_error_features(station_friendly, 24, df)
    print(f"  + {len(roll_feats)} features")
    for f in roll_feats:
        print(f"    {f:30s} NaN rate {df[f].isna().mean() * 100:5.1f}%")

    rows = []
    for tag, feats in [
        ("v1 synoptic only", list(FEATURE_NAMES) + syn_feats),
        ("v2 rolling only",  list(FEATURE_NAMES) + roll_feats),
        ("v3 both",          list(FEATURE_NAMES) + syn_feats + roll_feats),
    ]:
        print(f"\n[{time.strftime('%H:%M:%S')}] {tag} ({len(feats)} requested features)…")
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, train_df, eff = prepare_matrices(df, feats)
        test_clim = train_df["wet"].mean()
        clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)
        p_val, p_test, wall = fit_dbarts_with_holdouts(
            X_train_s, y_train, X_val_s, X_test_s,
            n_trees=50, n_burn=200, n_samples=1000, seed=42,
        )
        b = brier(p_test, y_test)
        bss = (clim_brier - b) / clim_brier
        delta = b - 0.1207
        print(f"  done in {wall:.1f}s | features eff={eff} | Brier {b:.4f} "
              f"(BSS {bss:+.4f}) | Δ vs baseline {delta:+.4f}")
        rows.append({
            "variant": tag,
            "n_feats_eff": eff,
            "wall_s": round(wall, 1),
            "brier": round(b, 4),
            "bss": round(bss, 4),
            "delta_vs_baseline": round(delta, 4),
        })

    summary = pd.DataFrame(rows)
    print()
    print("Summary (Bellever 24h, 5k train, 2987 test, baseline = 22 features = 0.1207):")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "dbarts_richfeats.csv", index=False)
    text = (
        "Phase 6 — rich-feature experiments\n"
        "===================================\n\n"
        "Baseline (22 features, ntree=50): Brier 0.1207\n\n"
        + summary.to_string(index=False)
        + "\n\nNegative delta_vs_baseline = improvement.\n"
    )
    (out_dir / "dbarts_richfeats.txt").write_text(text)


if __name__ == "__main__":
    main()
