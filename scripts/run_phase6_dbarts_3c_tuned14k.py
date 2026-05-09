"""Phase 6 — 3c feature set at the 14k-tuned dbarts winner.

Config locked at ntree=500, k=3 (tuned via run_phase6_dbarts_tune14k.py:
3-stage Brier 0.1224 vs 0.1260 default at full 14k Bellever 24h).

Three runs at this config:
  v0 22-feat baseline (sanity check, should reproduce 0.1224)
  v1 22 + 28 per-NWP (dew/rh/dewdep/pressure × 7 NWPs)
  v2 22 + 4 EA persistence (anchored at T-24h)
  v3 22 + 32 = full 3c surface

Tells us whether the 5k "3c hurts" finding flips at 14k-tuned, or whether
3a's lean surface really is the ceiling at this dataset size.
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
from run_phase6_dbarts_3c import (  # noqa: E402
    add_per_nwp_features,
    add_ea_persistence_features,
)

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")

NTREE = 500
K = 3.0


def fit_dbarts(X_train, y_train, X_test, *, seed=42):
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train.astype(np.float64))
        y_train_r = ro.conversion.py2rpy(y_train.astype(np.float64))
        x_test_r = ro.conversion.py2rpy(X_test.astype(np.float64))
    t0 = time.time()
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_test_r,
        ntree=NTREE, k=K, nskip=200, ndpost=1000,
        keeptrees=True, verbose=False, seed=seed,
    )
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat = np.array(ro.conversion.rpy2py(yhat_test_r))
    wall = time.time() - t0
    return norm.cdf(yhat).mean(axis=0), wall


def prepare_full(df: pd.DataFrame, feature_list: list[str]):
    train_df, val_df, test_df = time_split(df)
    X_train_full = train_df[feature_list].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="int8")
    X_test_full = test_df[feature_list].to_numpy(dtype="float64")
    y_test = test_df["wet"].to_numpy(dtype="int8")

    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept = np.where(~col_all_nan)[0]
    X_train = X_train_full[:, kept]
    X_test = X_test_full[:, kept]
    median = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), median, X_train)
    X_test = np.where(np.isnan(X_test), median, X_test)
    scaler = StandardScaler().fit(X_train)
    return (scaler.transform(X_train), y_train,
            scaler.transform(X_test), y_test,
            train_df, len(kept))


def main() -> None:
    station_slug, station_friendly = resolve_station("ea_bellever_dartmoor")
    out_dir = OUTPUT_ROOT / f"{station_slug}_lead24"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building base features…")
    df = build_features_via_duckdb(station_friendly, 24)

    print(f"[{time.strftime('%H:%M:%S')}] Adding per-NWP features…")
    df, per_nwp_feats = add_per_nwp_features(station_friendly, df)
    print(f"[{time.strftime('%H:%M:%S')}] Adding EA persistence features…")
    df, ea_feats = add_ea_persistence_features(station_friendly, df)

    rows = []
    for tag, feats in [
        ("v0 22-feat (baseline)",         list(FEATURE_NAMES)),
        ("v1 22 + 28 per-NWP",            list(FEATURE_NAMES) + per_nwp_feats),
        ("v2 22 + 4 EA persistence",      list(FEATURE_NAMES) + ea_feats),
        ("v3 22 + 28 + 4 (full 3c)",      list(FEATURE_NAMES) + per_nwp_feats + ea_feats),
    ]:
        print(f"\n[{time.strftime('%H:%M:%S')}] {tag} (requested {len(feats)})…")
        X_train_s, y_train, X_test_s, y_test, td, eff = prepare_full(df, feats)
        test_clim = td["wet"].mean()
        clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)
        p_test, wall = fit_dbarts(X_train_s, y_train, X_test_s)
        b = brier(p_test, y_test)
        bss = (clim_brier - b) / clim_brier
        delta_baseline = b - 0.1224
        delta_3a = b - 0.1267
        print(f"  done in {wall:.1f}s | features eff={eff} | Brier {b:.4f} | "
              f"Δ vs 22-tuned {delta_baseline:+.4f} | Δ vs 3a-deployed {delta_3a:+.4f}")
        rows.append({
            "variant": tag,
            "n_feats_eff": eff,
            "wall_s": round(wall, 1),
            "brier": round(b, 4),
            "bss": round(bss, 4),
            "delta_vs_22tuned": round(delta_baseline, 4),
            "delta_vs_3a_deployed": round(delta_3a, 4),
        })

    summary = pd.DataFrame(rows)
    print()
    print(f"3c features × 14k tuned dbarts (ntree={NTREE}, k={K}):")
    print("  Reference: 22-tuned baseline = 0.1224 | 3a deployed (LGB 14k) = 0.1267")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "dbarts_3c_tuned14k.csv", index=False)
    text = (
        f"Phase 6 — 3c features at 14k-tuned dbarts (ntree={NTREE}, k={K})\n"
        "================================================================\n\n"
        + summary.to_string(index=False)
        + "\n\nNegative delta_* = improvement.\n"
    )
    (out_dir / "dbarts_3c_tuned14k.txt").write_text(text)


if __name__ == "__main__":
    main()
