"""Phase 6 — give dbarts the 3c feature set (3a 22 + 28 per-NWP + 4 EA persistence = 54).

Replicates PrecipRichFeatureBuilder.BuildSpec/BuildForLead exactly:
  * 28 per-NWP features: dew_<nwp>, rh_<nwp>, dewdep_<nwp>, pressure_<nwp>
  * 4 EA persistence (anchored at runTime = T - leadHours):
      ea_rain_prev_24h_mm   = sum of gauge mm in (runTime-24h, runTime]
      ea_rain_prev_72h_mm   = sum of gauge mm in (runTime-72h, runTime]
      ea_wet_hours_last_24h = count of hours ≥ WET_THRESHOLD in same 24h window
      ea_dry_hours_trailing = consecutive dry hours walking back from runTime
  * Strict coverage: NaN if any underlying hour is missing (matches the C# rule)

Three variants tested against the 22-feature baseline (Brier 0.1207) and the
synoptic-only winner from earlier (Brier 0.1182):
  v1 per-NWP only       22 + 28 = 50 features
  v2 EA persistence only  22 + 4 = 26 features
  v3 both (full 3c)      22 + 32 = 54 features
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

from src.data import LOCATION, WEATHERBLEND_DATA_ROOT, WET_THRESHOLD_MM  # noqa: E402

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

LEAD_HOURS = 24


def add_per_nwp_features(station_friendly: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Per-NWP dew, rh, dew_depression, pressure — mirrors 3c's SQL pivot
    block (PrecipRichFeatureBuilder.cs:262-268)."""
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
    pivot_lines = []
    for full, short in MODELS_LEAN:
        pivot_lines += [
            f"MAX(CASE WHEN Model = '{full}' THEN DewPoint2m END)                    AS dew_{short}",
            f"MAX(CASE WHEN Model = '{full}' THEN RelativeHumidity2m END)            AS rh_{short}",
            f"MAX(CASE WHEN Model = '{full}' THEN Temperature2m - DewPoint2m END)    AS dewdep_{short}",
            f"MAX(CASE WHEN Model = '{full}' THEN SurfacePressure END)               AS pressure_{short}",
        ]
    sql = f"""
    WITH latest AS (
        SELECT
            ValidTimeUtc, Model,
            DewPoint2m, RelativeHumidity2m, Temperature2m, SurfacePressure,
            ROW_NUMBER() OVER (
                PARTITION BY ValidTimeUtc, Model
                ORDER BY RunTimeUtc DESC
            ) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours = {LEAD_HOURS}
          AND Model IN {model_in}
    )
    SELECT ValidTimeUtc,
        {",\n        ".join(pivot_lines)}
    FROM latest
    WHERE rn = 1
    GROUP BY ValidTimeUtc
    ORDER BY ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    pivoted = con.execute(sql).fetch_df()
    con.close()
    df = df.merge(pivoted, on="ValidTimeUtc", how="left")
    feats: list[str] = []
    for _, short in MODELS_LEAN:
        feats += [f"dew_{short}", f"rh_{short}", f"dewdep_{short}", f"pressure_{short}"]
    return df, feats


def compute_persistence_for_row(hourly: dict, run_time, wet_threshold: float):
    """Direct port of PrecipRichFeatureBuilder.ComputePersistence (C#).
    runTime here = valid_time - LEAD_HOURS. Window: (runTime-N, runTime].
    Strict coverage — NaN if any hour is missing.
    """
    sum24 = 0.0
    sum72 = 0.0
    wet24 = 0
    cover24 = True
    cover72 = True
    for h in range(72):
        t = run_time - pd.Timedelta(hours=h)
        if t in hourly:
            mm = hourly[t]
            sum72 += mm
            if h < 24:
                sum24 += mm
                if mm >= wet_threshold:
                    wet24 += 1
        else:
            if h < 24:
                cover24 = False
            cover72 = False

    dry_run = 0
    for h in range(72):
        t = run_time - pd.Timedelta(hours=h)
        if t not in hourly:
            break
        if hourly[t] > wet_threshold:
            break
        dry_run += 1

    return (
        sum24 if cover24 else np.nan,
        sum72 if cover72 else np.nan,
        float(wet24) if cover24 else np.nan,
        float(dry_run),
    )


def add_ea_persistence_features(station_friendly: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Anchored at runTime = ValidTimeUtc - LEAD_HOURS. Mirrors C# exactly."""
    rn_glob = str((WEATHERBLEND_DATA_ROOT / "truth" / "rainfall" / "**" / "*.parquet")).replace("\\", "/")
    sql = f"""
    SELECT
        date_trunc('hour', ObservedTimeUtc) AS valid_time,
        SUM(Value15MinMm) AS mm
    FROM read_parquet('{rn_glob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{LOCATION}'
      AND StationName  = '{station_friendly}'
      AND Value15MinMm IS NOT NULL
    GROUP BY 1
    HAVING COUNT(*) = 4
    ORDER BY 1
    """
    con = duckdb.connect(":memory:")
    rn = con.execute(sql).fetch_df()
    con.close()
    hourly = {pd.Timestamp(t).to_pydatetime(): float(mm)
              for t, mm in zip(rn["valid_time"].to_numpy(), rn["mm"].to_numpy())}

    # Iterate row by row computing the 4 features. df is small (~20k rows), so
    # plain Python loop is fine — micro-optimising not worth the readability cost.
    valid_times = pd.to_datetime(df["ValidTimeUtc"]).dt.to_pydatetime()
    out = np.empty((len(df), 4), dtype="float64")
    for i, vt in enumerate(valid_times):
        run_time = vt - pd.Timedelta(hours=LEAD_HOURS)
        out[i] = compute_persistence_for_row(hourly, run_time.to_pydatetime() if hasattr(run_time, "to_pydatetime") else run_time, WET_THRESHOLD_MM)
    df = df.copy()
    df["ea_rain_prev_24h_mm"] = out[:, 0]
    df["ea_rain_prev_72h_mm"] = out[:, 1]
    df["ea_wet_hours_last_24h"] = out[:, 2]
    df["ea_dry_hours_trailing"] = out[:, 3]
    feats = ["ea_rain_prev_24h_mm", "ea_rain_prev_72h_mm",
             "ea_wet_hours_last_24h", "ea_dry_hours_trailing"]
    return df, feats


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

    print(f"[{time.strftime('%H:%M:%S')}] Building 22-feature base via 3a SQL…")
    df = build_features_via_duckdb(station_friendly, LEAD_HOURS)
    print(f"  rows: {len(df):,}")

    print(f"[{time.strftime('%H:%M:%S')}] Adding 28 per-NWP features (dew, rh, dewdep, pressure)…")
    df, per_nwp_feats = add_per_nwp_features(station_friendly, df)
    nan_summary = {f: df[f].isna().mean() * 100 for f in per_nwp_feats}
    print(f"  + {len(per_nwp_feats)} features. NaN rates: "
          f"min {min(nan_summary.values()):.1f}% / max {max(nan_summary.values()):.1f}% / "
          f"mean {np.mean(list(nan_summary.values())):.1f}%")
    high_nan = {f: r for f, r in nan_summary.items() if r > 5}
    if high_nan:
        for f, r in high_nan.items():
            print(f"    HIGH NaN: {f:30s} {r:5.1f}%")

    print(f"[{time.strftime('%H:%M:%S')}] Adding 4 EA persistence features (anchored at T-{LEAD_HOURS}h)…")
    df, ea_feats = add_ea_persistence_features(station_friendly, df)
    for f in ea_feats:
        print(f"    {f:30s} NaN rate {df[f].isna().mean() * 100:5.1f}%")

    rows = []
    for tag, feats in [
        ("v1 per-NWP only",   list(FEATURE_NAMES) + per_nwp_feats),
        ("v2 EA persistence", list(FEATURE_NAMES) + ea_feats),
        ("v3 full 3c (both)", list(FEATURE_NAMES) + per_nwp_feats + ea_feats),
    ]:
        print(f"\n[{time.strftime('%H:%M:%S')}] {tag} ({len(feats)} requested)…")
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, train_df, eff = prepare_matrices(df, feats)
        test_clim = train_df["wet"].mean()
        clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)
        p_val, p_test, wall = fit_dbarts_with_holdouts(
            X_train_s, y_train, X_val_s, X_test_s,
            n_trees=50, n_burn=200, n_samples=1000, seed=42,
        )
        b = brier(p_test, y_test)
        bss = (clim_brier - b) / clim_brier
        delta_baseline = b - 0.1207
        delta_synoptic = b - 0.1182
        print(f"  done in {wall:.1f}s | features eff={eff} | Brier {b:.4f} | "
              f"Δ vs 22-feat {delta_baseline:+.4f} | Δ vs synoptic-best {delta_synoptic:+.4f}")
        rows.append({
            "variant": tag,
            "n_feats_eff": eff,
            "wall_s": round(wall, 1),
            "brier": round(b, 4),
            "bss": round(bss, 4),
            "delta_vs_baseline_22": round(delta_baseline, 4),
            "delta_vs_synoptic_best": round(delta_synoptic, 4),
        })

    summary = pd.DataFrame(rows)
    print()
    print("Summary (Bellever 24h, 5k train, 2987 test):")
    print("  baseline 22-feat (3a):     0.1207")
    print("  synoptic-best (22 + 3):    0.1182")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "dbarts_3c_features.csv", index=False)
    text = (
        "Phase 6 — 3c feature set on dbarts (ntree=50)\n"
        "==============================================\n\n"
        "Baselines: 3a 22-feat = 0.1207, synoptic-best = 0.1182\n\n"
        + summary.to_string(index=False)
        + "\n\nNegative delta_* = improvement.\n"
    )
    (out_dir / "dbarts_3c_features.txt").write_text(text)
    print(f"\nArtefacts → {out_dir}")


if __name__ == "__main__":
    main()
