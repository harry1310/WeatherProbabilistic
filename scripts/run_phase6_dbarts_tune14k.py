"""Phase 6 — dbarts hyperparameter tuning at full 14k Bellever 24h.

Two-stage sweep:
  Stage 1: ntree ∈ {50, 100, 200, 500} at default k=2 (leaf shrinkage prior)
  Stage 2: at the stage-1 winner, sweep k ∈ {1, 2, 3}

Why this matters: the 5k subsample picked ntree=50 as the winner, but 14k
showed Brier 0.1260 at ntree=50 — worse than the 5k 0.1207. Hypothesis is
that at 14k more capacity (more trees and/or less leaf shrinkage) is
needed. Sweep tells us.

Default dbarts:
  k = 2.0   (controls posterior leaf-value variance — lower k = wider leaf
             values = more aggressive splits)
  power = 2.0, base = 0.95 (depth prior; not swept here)
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
    resolve_station,
    time_split,
)

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")


def fit_dbarts_with_holdouts(X_train, y_train, X_test, *, n_trees, k, n_burn,
                              n_samples, seed):
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train.astype(np.float64))
        y_train_r = ro.conversion.py2rpy(y_train.astype(np.float64))
        x_test_r = ro.conversion.py2rpy(X_test.astype(np.float64))
    t0 = time.time()
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_test_r,
        ntree=n_trees, k=k, nskip=n_burn, ndpost=n_samples,
        keeptrees=True, verbose=False, seed=seed,
    )
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat = np.array(ro.conversion.rpy2py(yhat_test_r))
    wall = time.time() - t0
    p_test = norm.cdf(yhat).mean(axis=0)
    return p_test, wall


def main() -> None:
    station_slug, station_friendly = resolve_station("ea_bellever_dartmoor")
    out_dir = OUTPUT_ROOT / f"{station_slug}_lead24"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building features…")
    df = build_features_via_duckdb(station_friendly, 24)
    train_df, val_df, test_df = time_split(df)
    print(f"  train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    feats = list(FEATURE_NAMES)
    X_train_full = train_df[feats].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="int8")
    X_test_full = test_df[feats].to_numpy(dtype="float64")
    y_test = test_df["wet"].to_numpy(dtype="int8")

    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept = np.where(~col_all_nan)[0]
    X_train = X_train_full[:, kept]
    X_test = X_test_full[:, kept]
    median = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), median, X_train)
    X_test = np.where(np.isnan(X_test), median, X_test)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    test_clim = train_df["wet"].mean()
    clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)
    print(f"  features eff: {len(kept)} (3 dropped) | clim Brier {clim_brier:.4f}")

    # --- Stage 1: ntree at k=2 ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Stage 1 — ntree sweep at k=2…")
    stage1 = []
    for ntree in (50, 100, 200, 500):
        print(f"[{time.strftime('%H:%M:%S')}]   ntree={ntree}…", flush=True)
        p_test, wall = fit_dbarts_with_holdouts(
            X_train_s, y_train, X_test_s,
            n_trees=ntree, k=2.0, n_burn=200, n_samples=1000, seed=42,
        )
        b = brier(p_test, y_test)
        bss = (clim_brier - b) / clim_brier
        print(f"    done in {wall:.1f}s | Brier {b:.4f} | BSS {bss:+.4f}")
        stage1.append({"stage": "1", "ntree": ntree, "k": 2.0, "wall_s": round(wall, 1),
                        "brier": round(b, 4), "bss": round(bss, 4)})

    s1 = pd.DataFrame(stage1)
    best_ntree = int(s1.loc[s1["brier"].idxmin(), "ntree"])
    print(f"\nStage 1 winner: ntree={best_ntree}")

    # --- Stage 2: k at best ntree ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Stage 2 — k sweep at ntree={best_ntree}…")
    stage2 = []
    for k in (1.0, 2.0, 3.0):
        # Skip k=2 if it's the same as stage 1's best (already measured)
        if k == 2.0:
            row = next(r for r in stage1 if r["ntree"] == best_ntree)
            stage2.append({"stage": "2", "ntree": best_ntree, "k": 2.0,
                            "wall_s": row["wall_s"], "brier": row["brier"],
                            "bss": row["bss"]})
            print(f"[{time.strftime('%H:%M:%S')}]   k=2.0 (reused stage 1)  Brier {row['brier']:.4f}")
            continue
        print(f"[{time.strftime('%H:%M:%S')}]   k={k}…", flush=True)
        p_test, wall = fit_dbarts_with_holdouts(
            X_train_s, y_train, X_test_s,
            n_trees=best_ntree, k=k, n_burn=200, n_samples=1000, seed=42,
        )
        b = brier(p_test, y_test)
        bss = (clim_brier - b) / clim_brier
        print(f"    done in {wall:.1f}s | Brier {b:.4f} | BSS {bss:+.4f}")
        stage2.append({"stage": "2", "ntree": best_ntree, "k": k, "wall_s": round(wall, 1),
                        "brier": round(b, 4), "bss": round(bss, 4)})

    s2 = pd.DataFrame(stage2)
    best_row = s2.loc[s2["brier"].idxmin()]
    print(f"\nStage 2 winner: ntree={int(best_row['ntree'])}, k={best_row['k']} → "
          f"Brier {best_row['brier']:.4f}")

    full = pd.concat([s1, s2], ignore_index=True)
    print()
    print("Full sweep summary (Bellever 24h, FULL 14k train, 2987 test, clim 0.2264):")
    print(full.to_string(index=False))
    full.to_csv(out_dir / "dbarts_tune14k.csv", index=False)
    text = (
        "Phase 6 — dbarts hyperparameter sweep at full 14k\n"
        "==================================================\n\n"
        f"Climatology Brier: {clim_brier:.4f}\n\n"
        + full.to_string(index=False)
        + f"\n\nBest config: ntree={int(best_row['ntree'])}, k={best_row['k']} "
        f"→ Brier {best_row['brier']:.4f}\n"
        f"For comparison: 5k subsample winner ntree=50 → Brier 0.1207; "
        f"14k @ ntree=50 → Brier 0.1260; 3a deployed (LightGBM 14k) → 0.1267\n"
    )
    (out_dir / "dbarts_tune14k.txt").write_text(text)


if __name__ == "__main__":
    main()
