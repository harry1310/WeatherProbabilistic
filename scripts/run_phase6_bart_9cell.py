"""Phase 6 — full 9-cell dbarts BART comparison vs deployed 3a.

3 stations × 3 leads at the champion config (dbarts ntree=500 k=3,
full 14k train, 22-feat 3a base + 3 synoptic flow features). Compare
each cell's BART Brier to the deployed 3a baseline (LightGBM, full 14k).

Stations: ea_bellever_dartmoor, ea_dartmoor_nr_hexworthy, ea_princetown
Leads:    24, 48, 72

This is the headline "BART vs 3a" number — fair training-set parity
at every cell, no 5k subsampling shortcuts.
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
    read_3a_baseline_brier,
    reliability_table,
    resolve_station,
    time_split,
)
from run_phase6_dbarts_richfeats import add_synoptic_features  # noqa: E402

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")

NTREE, K = 500, 3.0
STATIONS = ["ea_bellever_dartmoor", "ea_dartmoor_nr_hexworthy", "ea_bovey_tracey"]
LEADS = [24, 48, 72]


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
    return norm.cdf(yhat).mean(axis=0), time.time() - t0


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
            scaler.transform(X_test), y_test, train_df, test_df, len(kept))


def main() -> None:
    rows = []
    # Per-row test predictions across all 9 cells, written as a single
    # parquet at the end. Schema matches train_4a's test_predictions.parquet
    # (valid_time, station, lead, p_wet, observed_wet) so the linear-pool
    # bake-off can inner-join 3a vs per-cell BART without per-phase schema
    # branches.
    pred_frames: list[pd.DataFrame] = []
    for station_slug in STATIONS:
        station_slug, station_friendly = resolve_station(station_slug)
        for lead in LEADS:
            print(f"\n[{time.strftime('%H:%M:%S')}] {station_friendly} · lead {lead}h")
            try:
                df = build_features_via_duckdb(station_friendly, lead)
            except Exception as e:
                print(f"  WARN: feature build failed — {e}")
                continue
            if len(df) < 1000:
                print(f"  WARN: only {len(df)} rows — skipping")
                continue
            df, syn_feats = add_synoptic_features(station_friendly, lead, df)
            feats = list(FEATURE_NAMES) + syn_feats
            X_train_s, y_train, X_test_s, y_test, td, _test_df, eff = prepare_full(df, feats)
            test_clim = td["wet"].mean()
            clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)
            p_test, wall = fit_dbarts(X_train_s, y_train, X_test_s)
            b = brier(p_test, y_test)
            bss = (clim_brier - b) / clim_brier
            try:
                v_3a, brier_3a, n_test_3a = read_3a_baseline_brier(station_slug, lead)
                bss_3a = (clim_brier - brier_3a) / clim_brier
                delta_3a = b - brier_3a
                pct_3a = delta_3a / brier_3a * 100
            except FileNotFoundError:
                v_3a, brier_3a, bss_3a, delta_3a, pct_3a = "(none)", float("nan"), float("nan"), float("nan"), float("nan")
            print(f"  done in {wall:.1f}s | train {len(y_train):,} | test {len(y_test):,} | "
                  f"BART {b:.4f} (BSS {bss:+.4f}) | 3a {brier_3a:.4f} | "
                  f"Δ {delta_3a:+.4f} ({pct_3a:+.2f}%)")
            rows.append({
                "station": station_slug, "lead": lead,
                "n_train": len(y_train), "n_test": len(y_test),
                "n_feats_eff": eff, "wall_s": round(wall, 1),
                "bart_brier": round(b, 4), "bart_bss": round(bss, 4),
                "deployed_3a_brier": round(brier_3a, 4) if not np.isnan(brier_3a) else None,
                "delta_brier": round(delta_3a, 4) if not np.isnan(delta_3a) else None,
                "delta_pct": round(pct_3a, 2) if not np.isnan(pct_3a) else None,
            })
            # Per-row test predictions for the per-cell-vs-3a blend bake-off.
            pred_frames.append(pd.DataFrame({
                "valid_time":   pd.to_datetime(_test_df["ValidTimeUtc"].values),
                "station":      station_slug,
                "lead":         lead,
                "p_wet":        p_test,
                "observed_wet": y_test.astype("int8"),
            }))

    summary = pd.DataFrame(rows)
    out_dir = OUTPUT_ROOT / "_9cell_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    if pred_frames:
        all_preds = pd.concat(pred_frames, ignore_index=True)
        pred_path = out_dir / "test_predictions.parquet"
        all_preds.to_parquet(pred_path, index=False)
        print(f"  wrote {len(all_preds):,} per-row test predictions → {pred_path}")
    print()
    print(f"9-cell summary at champion config (ntree={NTREE}, k={K}):")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "bart_9cell.csv", index=False)
    text = (
        f"Phase 6 — full 9-cell BART vs deployed 3a (ntree={NTREE}, k={K})\n"
        "================================================================\n\n"
        + summary.to_string(index=False)
        + "\n\nNegative delta_brier = BART wins.\n"
    )
    (out_dir / "bart_9cell.txt").write_text(text)
    print(f"\nArtefacts → {out_dir}")


if __name__ == "__main__":
    main()
