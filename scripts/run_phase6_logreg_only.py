"""Phase 6 — Bayesian logreg head-to-head vs matched LightGBM, on the
SAME 5,000-row subsample BART used.

This is the "as if BART never happened" path: drives directly off the
phase6 helpers (build_features_via_duckdb, time_split, fit_lightgbm_matched,
fit_bayesian_logreg_blackjax, predict_bayesian_logreg) so the data prep,
70/15/15 split, stratified subsample, all-NaN-column drop, median-impute
and StandardScaler all match the BART run exactly. The only thing that
changes is the model.

Comparison readouts:
  * test Brier / BSS for matched LightGBM (5k train) vs Bayesian logreg
    (5k train, blackjax NUTS, ~1k tune + 1k draws × 4 chains)
  * (context only) prior BART test Brier read from
    predictions_test.parquet — does NOT re-run BART
  * (context only) deployed 3a Brier from training_metadata.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import arviz as az  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.preprocessing import StandardScaler  # noqa: E402

from run_phase6_bart_bakeoff import (  # noqa: E402
    FEATURE_NAMES,
    OUTPUT_ROOT,
    brier,
    build_features_via_duckdb,
    fit_bayesian_logreg_blackjax,
    fit_lightgbm_matched,
    predict_bayesian_logreg,
    read_3a_baseline_brier,
    reliability_table,
    resolve_station,
    time_split,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--station", default="ea_bellever_dartmoor")
    p.add_argument("--lead", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample-train", type=int, default=5000,
                   help="Same default as BART run (5,000) so the comparison is "
                        "apples-to-apples.")
    p.add_argument("--lr-tune", type=int, default=1000)
    p.add_argument("--lr-draws", type=int, default=1000)
    p.add_argument("--lr-chains", type=int, default=4)
    args = p.parse_args()

    station_slug, station_friendly = resolve_station(args.station)
    out_dir = OUTPUT_ROOT / f"{station_slug}_lead{args.lead}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] Phase 6 logreg-only — {station_friendly} "
          f"({station_slug}) lead {args.lead}h")
    print(f"  output: {out_dir}")

    print(f"[{time.strftime('%H:%M:%S')}] Building features (mirror of 3a SQL)…")
    df = build_features_via_duckdb(station_friendly, args.lead)
    print(f"  rows: {len(df):,} spanning {df.ValidTimeUtc.min()} → {df.ValidTimeUtc.max()}")
    wet_rate = df["wet"].mean()
    print(f"  wet rate: {wet_rate:.1%} ({df['wet'].sum()} of {len(df)})")

    train_df, val_df, test_df = time_split(df)
    print(f"  split → train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    # Stratified subsample — match BART's exactly (same seed, same algorithm).
    if args.subsample_train > 0 and args.subsample_train < len(train_df):
        rng = np.random.default_rng(args.seed)
        wet_idx = train_df.index[train_df["wet"] == 1].to_numpy().copy()
        dry_idx = train_df.index[train_df["wet"] == 0].to_numpy().copy()
        rng.shuffle(wet_idx)
        rng.shuffle(dry_idx)
        wet_keep = int(round(args.subsample_train * len(wet_idx) / len(train_df)))
        dry_keep = args.subsample_train - wet_keep
        keep_idx = np.sort(np.concatenate([wet_idx[:wet_keep], dry_idx[:dry_keep]]))
        train_df = train_df.loc[keep_idx].reset_index(drop=True)
        print(f"  subsampled training to {len(train_df):,} rows "
              f"({train_df['wet'].mean():.1%} wet — class balance preserved).")

    X_train_full = train_df[FEATURE_NAMES].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="int8")
    X_test_full = test_df[FEATURE_NAMES].to_numpy(dtype="float64")
    y_test = test_df["wet"].to_numpy(dtype="int8")
    X_val_full = val_df[FEATURE_NAMES].to_numpy(dtype="float64")
    y_val = val_df["wet"].to_numpy(dtype="int8")

    # Drop all-NaN-in-training columns (Open-Meteo previous_runs missing).
    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept_idx = np.where(~col_all_nan)[0]
    dropped_features = [FEATURE_NAMES[i] for i in np.where(col_all_nan)[0]]
    feature_names = [FEATURE_NAMES[i] for i in kept_idx]
    X_train = X_train_full[:, kept_idx]
    X_test = X_test_full[:, kept_idx]
    X_val = X_val_full[:, kept_idx]
    if dropped_features:
        print(f"  dropped {len(dropped_features)} all-NaN-in-training feature(s): {dropped_features}")
    print(f"  effective feature count: {len(feature_names)}")

    # Median-impute + standardise (matches BART run pre-processing).
    median = np.nanmedian(X_train, axis=0)
    X_train_imp = np.where(np.isnan(X_train), median, X_train)
    X_test_imp = np.where(np.isnan(X_test), median, X_test)
    scaler = StandardScaler().fit(X_train_imp)
    X_train_s = scaler.transform(X_train_imp)
    X_test_s = scaler.transform(X_test_imp)

    # Climatology for BSS denominator.
    test_clim = train_df["wet"].mean()
    clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)

    # 1. Matched LightGBM (same 5k train, NaN-preserving features).
    X_train_lgb = train_df[feature_names].to_numpy(dtype="float64")
    X_val_lgb = X_val[:, :]  # already kept_idx-filtered
    X_test_lgb = X_test[:, :]
    print(f"[{time.strftime('%H:%M:%S')}] Training matched LightGBM ({len(train_df):,}-row train)…")
    t0 = time.time()
    booster = fit_lightgbm_matched(X_train_lgb, y_train, X_val_lgb, y_val, seed=args.seed)
    p_test_lgb = booster.predict(X_test_lgb)
    lgb_brier = brier(p_test_lgb, y_test)
    lgb_bss = (clim_brier - lgb_brier) / clim_brier
    print(f"  done in {(time.time() - t0):.1f}s, best iter {booster.best_iteration}")

    # 2. Bayesian logreg via blackjax NUTS.
    print(f"[{time.strftime('%H:%M:%S')}] Training Bayesian logreg via blackjax "
          f"(tune={args.lr_tune}, draws={args.lr_draws}, chains={args.lr_chains})…")
    t0 = time.time()
    lr_model, lr_idata = fit_bayesian_logreg_blackjax(
        X_train_s, y_train, feature_names, seed=args.seed,
        tune=args.lr_tune, draws=args.lr_draws, chains=args.lr_chains,
    )
    print(f"  done in {(time.time() - t0) / 60:.1f} min")
    p_test_lr = predict_bayesian_logreg(lr_model, lr_idata, X_test_s, seed=args.seed)
    lr_brier = brier(p_test_lr, y_test)
    lr_bss = (clim_brier - lr_brier) / clim_brier

    # 3. Context: prior BART test Brier from saved predictions, deployed 3a.
    bart_brier_ctx, bart_n = None, 0
    bart_pred_path = out_dir / "predictions_test.parquet"
    if bart_pred_path.exists():
        prev = pd.read_parquet(bart_pred_path)
        if len(prev) == len(y_test):
            bart_brier_ctx = brier(prev["p_bart"].to_numpy(), prev["y_obs"].to_numpy())
            bart_n = len(prev)

    try:
        v_3a, brier_3a, n_test_3a = read_3a_baseline_brier(station_slug, args.lead)
        bss_3a = (clim_brier - brier_3a) / clim_brier
    except FileNotFoundError as e:
        v_3a, brier_3a, n_test_3a, bss_3a = "(no 3a baseline found)", float("nan"), 0, float("nan")
        print(f"  WARN: {e}")

    # 4. Save logreg posterior (small, ~1MB).
    az.to_netcdf(lr_idata, str(out_dir / "logreg_posterior.nc"))

    # 5. Report.
    rel_lr = reliability_table(p_test_lr, y_test)
    lines = [
        f"Phase 6 — Bayesian logreg vs matched LightGBM (logreg-only re-run)",
        f"==================================================================",
        f"",
        f"Station:      {args.station}",
        f"Lead hours:   {args.lead}",
        f"Subsample:    {len(train_df):,} train rows (seed {args.seed})",
        f"Test set:     {len(test_df):,} rows ({test_df['ValidTimeUtc'].min()} → "
        f"{test_df['ValidTimeUtc'].max()})",
        f"Logreg cfg:   blackjax NUTS, tune={args.lr_tune}, draws={args.lr_draws}, "
        f"chains={args.lr_chains}",
        f"",
        f"Test Brier (lower = better)",
        f"---------------------------",
        f"Climatology (constant test_clim={test_clim:.3f}):                     {clim_brier:.4f}",
        f"LightGBM   (matched, {len(train_df):,}-row train):                    "
        f"{lgb_brier:.4f}   BSS {lgb_bss:+.4f}",
        f"Bayesian-logreg (matched, {len(train_df):,}-row train, blackjax):     "
        f"{lr_brier:.4f}   BSS {lr_bss:+.4f}",
    ]
    if bart_brier_ctx is not None:
        bart_bss_ctx = (clim_brier - bart_brier_ctx) / clim_brier
        lines.append(
            f"PyMC-BART (prior run, same {len(train_df):,}-row train, context):    "
            f"{bart_brier_ctx:.4f}   BSS {bart_bss_ctx:+.4f}")
    lines.append(
        f"3a deployed ({v_3a}, ~14k-row train, context only):                 "
        f"{brier_3a:.4f}   BSS {bss_3a:+.4f}")
    lines += [
        f"",
        f"Bayesian logreg vs matched LightGBM (same training rows)",
        f"  Δ Brier (logreg − LGB):  {lr_brier - lgb_brier:+.4f}  "
        f"({(lr_brier - lgb_brier) / lgb_brier * 100:+.2f}%)",
        f"  negative = logreg wins, positive = LightGBM wins (LGB captures "
        f"non-linearity logreg misses)",
    ]
    if bart_brier_ctx is not None:
        lines += [
            f"",
            f"DIAGNOSTIC: Bayesian logreg vs prior BART (is BART's win from non-linearity, "
            f"or just from being Bayesian?)",
            f"  Δ Brier (logreg − BART): {lr_brier - bart_brier_ctx:+.4f}  "
            f"({(lr_brier - bart_brier_ctx) / bart_brier_ctx * 100:+.2f}%)",
            f"  negative = logreg beats BART (Bayesian-ness alone explains BART's win — "
            f"trees aren't pulling weight)",
            f"  positive = BART beats logreg → trees ARE adding value beyond Bayesian "
            f"uncertainty",
        ]
    lines += [
        f"",
        f"Reliability — Bayesian logreg test predictions (10 equal-width bins)",
        f"--------------------------------------------------------------------",
    ]
    for _, row in rel_lr.iterrows():
        if row["n"] == 0:
            lines.append(f"  [{row['bin_lo']:.2f},{row['bin_hi']:.2f})  n=0")
        else:
            lines.append(
                f"  [{row['bin_lo']:.2f},{row['bin_hi']:.2f})  "
                f"n={int(row['n']):>4d}  p_mean={row['p_mean']:.3f}  "
                f"y_rate={row['y_rate']:.3f}  "
                f"diff={row['y_rate'] - row['p_mean']:+.3f}"
            )

    text = "\n".join(lines)
    (out_dir / "logreg_report.txt").write_text(text)
    print()
    print(text)
    print()
    print(f"Artefacts → {out_dir} (logreg_report.txt, logreg_posterior.nc)")


if __name__ == "__main__":
    main()
