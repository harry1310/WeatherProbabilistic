"""Phase 6 — dbarts sweep: ntree ∈ {50, 100, 200} × {raw, PAV-calibrated}.

Same 5k Bellever 24h problem we've been using. Each fit predicts on BOTH
val and test in a single dbarts call (dbarts.bart accepts x.test stacked
with the val rows, splits via index after the call). PAV is fit on
(val_pred, val_obs), applied to test, compared to raw test Brier.

Goal: pick the cheapest ntree that holds Brier, decide whether PAV is
worth it for the full-scope 3a comparison.
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
from sklearn.isotonic import IsotonicRegression  # noqa: E402
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
    """Fit once, predict on val + test in the same call (dbarts supports
    x.test as a stacked array). Returns (p_val_mean, p_test_mean, wall_s)."""
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
    p_holdouts = norm.cdf(yhat).mean(axis=0)  # (n_val + n_test,)
    p_val = p_holdouts[:n_val]
    p_test = p_holdouts[n_val:]
    return p_val, p_test, wall


def main() -> None:
    station_slug, station_friendly = resolve_station("ea_bellever_dartmoor")
    out_dir = OUTPUT_ROOT / f"{station_slug}_lead24"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building features…")
    df = build_features_via_duckdb(station_friendly, 24)
    train_df, val_df, test_df = time_split(df)

    rng = np.random.default_rng(42)
    wet_idx = train_df.index[train_df["wet"] == 1].to_numpy().copy()
    dry_idx = train_df.index[train_df["wet"] == 0].to_numpy().copy()
    rng.shuffle(wet_idx); rng.shuffle(dry_idx)
    wet_keep = int(round(5000 * len(wet_idx) / len(train_df)))
    keep_idx = np.sort(np.concatenate([wet_idx[:wet_keep], dry_idx[:5000 - wet_keep]]))
    train_df = train_df.loc[keep_idx].reset_index(drop=True)

    X_train_full = train_df[FEATURE_NAMES].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="int8")
    X_val_full = val_df[FEATURE_NAMES].to_numpy(dtype="float64")
    y_val = val_df["wet"].to_numpy(dtype="int8")
    X_test_full = test_df[FEATURE_NAMES].to_numpy(dtype="float64")
    y_test = test_df["wet"].to_numpy(dtype="int8")

    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept_idx = np.where(~col_all_nan)[0]
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
    print(f"  train {len(X_train_s):,} | val {len(X_val_s):,} | test {len(X_test_s):,} | "
          f"features {X_train_s.shape[1]}")

    test_clim = train_df["wet"].mean()
    clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)

    rows = []
    for n_trees in (50, 100, 200):
        print(f"\n[{time.strftime('%H:%M:%S')}] dbarts ntree={n_trees}…")
        p_val, p_test, wall = fit_dbarts_with_holdouts(
            X_train_s, y_train, X_val_s, X_test_s,
            n_trees=n_trees, n_burn=200, n_samples=1000, seed=42,
        )
        b_raw = brier(p_test, y_test)
        # PAV: fit isotonic on (val_pred, val_obs), apply to test.
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_val, y_val)
        p_test_pav = iso.transform(p_test)
        b_pav = brier(p_test_pav, y_test)
        bss_raw = (clim_brier - b_raw) / clim_brier
        bss_pav = (clim_brier - b_pav) / clim_brier
        print(f"  done in {wall:.1f}s | raw Brier {b_raw:.4f} (BSS {bss_raw:+.4f}) | "
              f"PAV Brier {b_pav:.4f} (BSS {bss_pav:+.4f})")
        rows.append({
            "ntree": n_trees,
            "wall_s": round(wall, 1),
            "brier_raw": round(b_raw, 4),
            "bss_raw": round(bss_raw, 4),
            "brier_pav": round(b_pav, 4),
            "bss_pav": round(bss_pav, 4),
            "delta_pav": round(b_pav - b_raw, 4),
        })

    summary = pd.DataFrame(rows)
    print()
    print("Sweep summary (Bellever 24h, 5k train, 2987 test, climatology Brier 0.2264):")
    print(summary.to_string(index=False))

    summary.to_csv(out_dir / "dbarts_sweep.csv", index=False)
    text = (
        "Phase 6 — dbarts sweep (ntree × PAV)\n"
        "====================================\n\n"
        f"Climatology Brier: {clim_brier:.4f}\n\n"
        + summary.to_string(index=False)
        + "\n\nLower brier_raw / brier_pav is better. delta_pav<0 = PAV improves.\n"
    )
    (out_dir / "dbarts_sweep.txt").write_text(text)
    print(f"\nArtefacts → {out_dir} (dbarts_sweep.csv, dbarts_sweep.txt)")


if __name__ == "__main__":
    main()
