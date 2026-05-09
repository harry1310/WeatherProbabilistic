"""Phase 6 — does the 3c surface tip positive at FULL 14k training rows?

The 5k-subsample experiment showed 3c features hurt by +0.0053. Hypothesis:
the 32 extra features (28 per-NWP + 4 EA persistence) overfit at 5k. Full
training set has ~14k Bellever rows; with 2.8× the data, the extra features
might absorb cleanly and the deployed +0.006-0.014 Brier number reproduces.

Two configs at full train (no subsample):
  baseline-14k:  22-feat 3a surface, dbarts ntree=50
  3c-14k:        54-feat 3c surface (22 + 28 per-NWP + 4 EA), dbarts ntree=50
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
    fit_dbarts_with_holdouts,
    LEAD_HOURS,
)


def prepare_full(df: pd.DataFrame, feature_list: list[str]):
    """No subsample — train on the full 70% time-ordered training slice."""
    train_df, val_df, test_df = time_split(df)

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
    print(f"  total rows: {len(df):,}")

    print(f"[{time.strftime('%H:%M:%S')}] Adding 28 per-NWP features…")
    df, per_nwp_feats = add_per_nwp_features(station_friendly, df)
    print(f"[{time.strftime('%H:%M:%S')}] Adding 4 EA persistence features…")
    df, ea_feats = add_ea_persistence_features(station_friendly, df)

    train_df, val_df, test_df = time_split(df)
    print(f"  train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    rows = []
    for tag, feats in [
        ("baseline-14k (22-feat)",       list(FEATURE_NAMES)),
        ("3c-14k (22 + 28 per-NWP + 4 EA = 54)",
         list(FEATURE_NAMES) + per_nwp_feats + ea_feats),
    ]:
        print(f"\n[{time.strftime('%H:%M:%S')}] {tag} on FULL train…")
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, td, eff = prepare_full(df, feats)
        test_clim = td["wet"].mean()
        clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)
        print(f"  train rows: {len(y_train):,} | features eff={eff}")
        p_val, p_test, wall = fit_dbarts_with_holdouts(
            X_train_s, y_train, X_val_s, X_test_s,
            n_trees=50, n_burn=200, n_samples=1000, seed=42,
        )
        b = brier(p_test, y_test)
        bss = (clim_brier - b) / clim_brier
        print(f"  done in {wall:.1f}s | Brier {b:.4f} | BSS {bss:+.4f}")
        rows.append({
            "variant": tag,
            "n_train": len(y_train),
            "n_feats_eff": eff,
            "wall_s": round(wall, 1),
            "brier": round(b, 4),
            "bss": round(bss, 4),
        })

    summary = pd.DataFrame(rows)
    print()
    print("Full-train comparison:")
    print(summary.to_string(index=False))

    base_brier = summary.iloc[0]["brier"]
    rich_brier = summary.iloc[1]["brier"]
    delta = rich_brier - base_brier
    pct = delta / base_brier * 100
    print()
    print(f"Δ Brier (3c − baseline) at 14k train: {delta:+.4f}  ({pct:+.2f}%)")
    print("Negative = 3c wins at full data → subsample hypothesis confirmed.")
    print("Positive = 3c still loses → not a subsample issue (dbarts vs LGB or station-specific).")
    print()
    print("Reference: 5k subsample numbers were baseline 0.1207, 3c full 0.1260 (Δ +0.0053).")

    summary.to_csv(out_dir / "dbarts_3c_full14k.csv", index=False)
    text = (
        "Phase 6 — full-train 14k: baseline vs 3c\n"
        "==========================================\n\n"
        + summary.to_string(index=False)
        + f"\n\nΔ Brier (3c − baseline): {delta:+.4f}  ({pct:+.2f}%)\n"
    )
    (out_dir / "dbarts_3c_full14k.txt").write_text(text)


if __name__ == "__main__":
    main()
