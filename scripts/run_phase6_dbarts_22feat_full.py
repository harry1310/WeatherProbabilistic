"""Phase 6 — 22-feature 3a surface on the FULL 14k training set.

Same dbarts config as the 5k sweep winner (ntree=50, 200 burn, 1000 samples,
seed 42), same train/val/test time split, same 22-feature 3a surface — but
no subsample. Establishes the proper full-train baseline before any rich-
feature comparisons.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

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

    print(f"[{time.strftime('%H:%M:%S')}] Building 22-feature base via 3a SQL…")
    df = build_features_via_duckdb(station_friendly, 24)
    train_df, val_df, test_df = time_split(df)
    print(f"  total rows: {len(df):,} | "
          f"train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    feats = list(FEATURE_NAMES)
    X_train_full = train_df[feats].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="int8")
    X_val_full = val_df[feats].to_numpy(dtype="float64")
    y_val = val_df["wet"].to_numpy(dtype="int8")
    X_test_full = test_df[feats].to_numpy(dtype="float64")
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
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    print(f"  features eff: {len(kept)} (3 cloud-layer cols dropped — Open-Meteo previous_runs nulls)")

    test_clim = train_df["wet"].mean()
    clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)

    print(f"\n[{time.strftime('%H:%M:%S')}] dbarts ntree=50 on full {len(y_train):,}-row train…")
    p_val, p_test, wall = fit_dbarts_with_holdouts(
        X_train_s, y_train, X_val_s, X_test_s,
        n_trees=50, n_burn=200, n_samples=1000, seed=42,
    )
    b = brier(p_test, y_test)
    bss = (clim_brier - b) / clim_brier
    print(f"  done in {wall:.1f}s | Brier {b:.4f} | BSS {bss:+.4f}")

    delta_5k_22 = b - 0.1207
    delta_5k_synoptic = b - 0.1182
    print()
    print(f"5k 22-feat baseline (prior): 0.1207")
    print(f"5k synoptic-best (prior):    0.1182")
    print(f"14k 22-feat (this run):      {b:.4f}")
    print(f"  Δ vs 5k 22-feat:    {delta_5k_22:+.4f}  ({delta_5k_22 / 0.1207 * 100:+.2f}%)")
    print(f"  Δ vs 5k synoptic:   {delta_5k_synoptic:+.4f}  ({delta_5k_synoptic / 0.1182 * 100:+.2f}%)")

    rel = reliability_table(p_test, y_test)
    text = "\n".join([
        "Phase 6 — 22-feat dbarts on full 14k train",
        "===========================================",
        "",
        f"Train rows: {len(y_train):,} (no subsample)",
        f"Test rows:  {len(y_test):,} ({test_df['ValidTimeUtc'].min()} → "
        f"{test_df['ValidTimeUtc'].max()})",
        f"Features:   22 requested → {len(kept)} effective (3 dropped)",
        f"Config:     ntree=50, 200 burn, 1000 samples, seed 42",
        f"Wall:       {wall:.1f}s",
        "",
        f"Test Brier: {b:.4f}   BSS {bss:+.4f}",
        f"  vs 5k 22-feat baseline (0.1207):  {delta_5k_22:+.4f}",
        f"  vs 5k synoptic-best (0.1182):     {delta_5k_synoptic:+.4f}",
        "",
        "Reliability (10 equal-width bins)",
        "---------------------------------",
    ])
    for _, row in rel.iterrows():
        if row["n"] == 0:
            text += f"\n  [{row['bin_lo']:.2f},{row['bin_hi']:.2f})  n=0"
        else:
            text += (f"\n  [{row['bin_lo']:.2f},{row['bin_hi']:.2f})  "
                     f"n={int(row['n']):>4d}  p_mean={row['p_mean']:.3f}  "
                     f"y_rate={row['y_rate']:.3f}  "
                     f"diff={row['y_rate'] - row['p_mean']:+.3f}")
    (out_dir / "dbarts_22feat_full14k.txt").write_text(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
