"""Phase 6 — autoregressive-gauge feature experiment.

Adds 7 lookback features built from the gauge's own historical wet/dry
state and refits dbarts at the sweep-winning config (ntree=50, 200 burn,
1000 samples). All lookback offsets ≥24h so the features are legal at
lead-24h predict time (predict-time information set is gauge truth up
to T-24h, where T is the target valid time).

Features added (all keyed off the EA gauge truth at this station):
  * gauge_wet_24h_ago    binary, was gauge wet at T-24h
  * gauge_wet_36h_ago    binary, T-36h
  * gauge_wet_48h_ago    binary, T-48h
  * gauge_wet_72h_ago    binary, T-72h
  * gauge_wet_frac_24_48h   fraction wet across T-48..T-25 (most recent 24h
                            actually available at predict time)
  * gauge_wet_frac_24_72h   fraction wet across T-72..T-25 (48h regime)
  * gauge_wet_frac_24_168h  fraction wet across T-168..T-25 (7-day regime)

Goal: see if Brier breaks below the 22-feature 0.1207 floor at ntree=50.
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


def hourly_gauge_wet(station_friendly: str) -> pd.Series:
    """Return a Series indexed by hour-truncated UTC valid_time with values
    in {0, 1, NaN} — wet=1 when total hourly precip ≥ WET_THRESHOLD_MM AND
    the strict 4-of-4 partial-hour rule is satisfied. NaN otherwise.
    Same threshold + same partial-hour rule the rest of the pipeline uses."""
    rn_glob = str((WEATHERBLEND_DATA_ROOT / "truth" / "rainfall" / "**" / "*.parquet")).replace("\\", "/")
    sql = f"""
    SELECT
        date_trunc('hour', ObservedTimeUtc) AS valid_time,
        SUM(Value15MinMm) AS precip_mm_hour
    FROM read_parquet('{rn_glob}', hive_partitioning = false, union_by_name = true)
    WHERE LocationName = '{LOCATION}'
      AND StationName  = '{station_friendly}'
      AND Value15MinMm IS NOT NULL
    GROUP BY 1
    HAVING COUNT(*) = 4
    ORDER BY 1
    """
    con = duckdb.connect(":memory:")
    out = con.execute(sql).fetch_df()
    con.close()
    s = pd.Series((out["precip_mm_hour"].to_numpy() >= WET_THRESHOLD_MM).astype(np.float64),
                  index=pd.DatetimeIndex(out["valid_time"]))
    return s


def add_autoregressive_features(df: pd.DataFrame, gauge_wet: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Add the 7 autoregressive features. Lookback offsets ≥24h so we're
    using only information available at predict time (T-24h)."""
    valid_times = pd.DatetimeIndex(df["ValidTimeUtc"])

    # Discrete-offset wet binaries — reindex picks NaN where the lookback
    # hour doesn't exist in the truth series (gauge offline, partial-hour
    # rule failed, etc.).
    for offset_h, name in [(24, "gauge_wet_24h_ago"),
                            (36, "gauge_wet_36h_ago"),
                            (48, "gauge_wet_48h_ago"),
                            (72, "gauge_wet_72h_ago")]:
        target = valid_times - pd.Timedelta(hours=offset_h)
        df[name] = gauge_wet.reindex(target).to_numpy()

    # Rolling wet fraction over windows of N hours ending at T-25 (so the
    # window [T-25-(N-1), T-25] sits entirely in the predict-time info set).
    # Easiest computation: build a precomputed rolling-mean series, then
    # reindex by (T-25) for each row.
    for window_h, name in [(24, "gauge_wet_frac_24_48h"),
                            (48, "gauge_wet_frac_24_72h"),
                            (144, "gauge_wet_frac_24_168h")]:
        # min_periods = window // 2 so we don't fill the window with too-few-
        # samples averages; if more than half the window is missing, NaN.
        roll = gauge_wet.rolling(window=window_h, min_periods=window_h // 2).mean()
        # Window ends at T-25 → we evaluate the rolling at T-25 to get the
        # mean of [T-25-(N-1), T-25].
        target = valid_times - pd.Timedelta(hours=25)
        df[name] = roll.reindex(target).to_numpy()

    new_features = [
        "gauge_wet_24h_ago", "gauge_wet_36h_ago",
        "gauge_wet_48h_ago", "gauge_wet_72h_ago",
        "gauge_wet_frac_24_48h", "gauge_wet_frac_24_72h", "gauge_wet_frac_24_168h",
    ]
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


def main() -> None:
    station_slug, station_friendly = resolve_station("ea_bellever_dartmoor")
    out_dir = OUTPUT_ROOT / f"{station_slug}_lead24"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building base features (22)…")
    df = build_features_via_duckdb(station_friendly, 24)
    print(f"  rows: {len(df):,}")

    print(f"[{time.strftime('%H:%M:%S')}] Pulling hourly gauge wet/dry truth…")
    gauge_wet = hourly_gauge_wet(station_friendly)
    print(f"  truth hours: {len(gauge_wet):,} ({gauge_wet.mean() * 100:.1f}% wet)")

    print(f"[{time.strftime('%H:%M:%S')}] Adding 7 autoregressive lookback features…")
    df, new_features = add_autoregressive_features(df, gauge_wet)
    extended_feature_names = list(FEATURE_NAMES) + new_features

    # NaN profile per new feature so we can sanity-check the join.
    print("  NaN rate per new feature:")
    for f in new_features:
        print(f"    {f:30s} {df[f].isna().mean() * 100:5.1f}%")

    train_df, val_df, test_df = time_split(df)
    rng = np.random.default_rng(42)
    wet_idx = train_df.index[train_df["wet"] == 1].to_numpy().copy()
    dry_idx = train_df.index[train_df["wet"] == 0].to_numpy().copy()
    rng.shuffle(wet_idx); rng.shuffle(dry_idx)
    wet_keep = int(round(5000 * len(wet_idx) / len(train_df)))
    keep_idx = np.sort(np.concatenate([wet_idx[:wet_keep], dry_idx[:5000 - wet_keep]]))
    train_df = train_df.loc[keep_idx].reset_index(drop=True)
    print(f"  subsampled training to {len(train_df):,} rows "
          f"({train_df['wet'].mean():.1%} wet)")

    X_train_full = train_df[extended_feature_names].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="int8")
    X_val_full = val_df[extended_feature_names].to_numpy(dtype="float64")
    y_val = val_df["wet"].to_numpy(dtype="int8")
    X_test_full = test_df[extended_feature_names].to_numpy(dtype="float64")
    y_test = test_df["wet"].to_numpy(dtype="int8")

    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept_idx = np.where(~col_all_nan)[0]
    feature_names = [extended_feature_names[i] for i in kept_idx]
    X_train = X_train_full[:, kept_idx]
    X_val = X_val_full[:, kept_idx]
    X_test = X_test_full[:, kept_idx]
    median = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), median, X_train)
    X_val = np.where(np.isnan(X_val), median, X_val)
    X_test = np.where(np.isnan(X_test), median, X_test)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    print(f"  effective feature count: {len(feature_names)} "
          f"(of {len(extended_feature_names)} requested)")

    test_clim = train_df["wet"].mean()
    clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)

    print(f"\n[{time.strftime('%H:%M:%S')}] dbarts (ntree=50, 22+7=29 features)…")
    p_val, p_test, wall = fit_dbarts_with_holdouts(
        X_train_s, y_train, X_val_s, X_test_s,
        n_trees=50, n_burn=200, n_samples=1000, seed=42,
    )
    b = brier(p_test, y_test)
    bss = (clim_brier - b) / clim_brier
    print(f"  done in {wall:.1f}s | Brier {b:.4f} (BSS {bss:+.4f})")

    # Compare to the 22-feature baseline at the same config.
    baseline_brier = 0.1207
    delta = b - baseline_brier
    pct = delta / baseline_brier * 100

    rel = reliability_table(p_test, y_test)

    lines = [
        "Phase 6 — autoregressive-gauge feature experiment",
        "==================================================",
        "",
        f"Station: {station_friendly}, lead 24h, 5,000 train rows, ntree=50",
        f"Features: {len(feature_names)} ({len(FEATURE_NAMES) - sum(1 for f in FEATURE_NAMES if f not in feature_names)} "
        f"original + {len([f for f in feature_names if f in new_features])} new autoregressive)",
        f"",
        f"Test Brier: {b:.4f}   BSS {bss:+.4f}",
        f"Baseline (22 features, ntree=50): {baseline_brier:.4f}",
        f"Δ Brier (autoreg − baseline): {delta:+.4f}  ({pct:+.2f}%)",
        f"  negative = autoregressive features improved Brier",
        f"",
        f"Wall: {wall:.1f}s",
        f"",
        f"Reliability — autoreg dbarts test predictions",
        f"---------------------------------------------",
    ]
    for _, row in rel.iterrows():
        if row["n"] == 0:
            lines.append(f"  [{row['bin_lo']:.2f},{row['bin_hi']:.2f})  n=0")
        else:
            lines.append(
                f"  [{row['bin_lo']:.2f},{row['bin_hi']:.2f})  "
                f"n={int(row['n']):>4d}  p_mean={row['p_mean']:.3f}  "
                f"y_rate={row['y_rate']:.3f}  diff={row['y_rate'] - row['p_mean']:+.3f}"
            )

    text = "\n".join(lines)
    (out_dir / "dbarts_autoreg.txt").write_text(text)
    print()
    print(text)
    print(f"\nArtefact → {out_dir / 'dbarts_autoreg.txt'}")


if __name__ == "__main__":
    main()
