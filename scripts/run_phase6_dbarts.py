"""Phase 6 — dbarts (R) BART head-to-head, same 5,000-row Bellever 24h problem.

Mirrors run_phase6_bart_bakeoff.py's data prep exactly (same DuckDB feature
build, same time split, same stratified subsample, same all-NaN-column drop,
same median-impute + StandardScaler) but swaps PyMC-BART for dbarts via
rpy2. dbarts uses a probit link for binary y by default; predictions come
back as latent posterior means and we apply pnorm to recover P(y=1).

Speed expectation: ~30-60s on 5k rows × 19 features × 50 trees × 1200 total
iterations (200 burn + 1000 samples), versus pymc-bart's ~3 hours on the
same data at 5 trees × 1000 iters × 4 chains. dbarts uses more trees because
it can — its default 200 trees is feasible at this scale.

Output: reports/phase6_artefacts/<station>_lead<L>/dbarts_*.{txt,parquet,RDS}.
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

import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import default_converter, numpy2ri, pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.preprocessing import StandardScaler  # noqa: E402

from run_phase6_bart_bakeoff import (  # noqa: E402
    FEATURE_NAMES,
    OUTPUT_ROOT,
    brier,
    build_features_via_duckdb,
    fit_lightgbm_matched,
    read_3a_baseline_brier,
    reliability_table,
    resolve_station,
    time_split,
)

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")


def fit_dbarts_binary(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, *, n_trees: int = 200,
                       n_burn: int = 200, n_samples: int = 1000,
                       seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Probit-link binary BART. Returns (p_test_mean, p_test_draws) where
    p_test_draws is (n_samples, n_test) — keep the full posterior so we
    can compute prediction intervals if we want them later."""
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train.astype(np.float64))
        y_train_r = ro.conversion.py2rpy(y_train.astype(np.float64))
        x_test_r = ro.conversion.py2rpy(X_test.astype(np.float64))
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_test_r,
        ntree=n_trees, nskip=n_burn, ndpost=n_samples,
        keeptrees=True, verbose=False, seed=seed,
    )
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat_test = np.array(ro.conversion.rpy2py(yhat_test_r))
    p_draws = norm.cdf(yhat_test)
    return p_draws.mean(axis=0), p_draws


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--station", default="ea_bellever_dartmoor")
    p.add_argument("--lead", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample-train", type=int, default=5000)
    p.add_argument("--n-trees", type=int, default=200,
                   help="dbarts default. PyMC-BART used 5 trees because PGBART "
                        "is slow; dbarts can afford the canonical 200.")
    p.add_argument("--n-burn", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=1000)
    args = p.parse_args()

    station_slug, station_friendly = resolve_station(args.station)
    out_dir = OUTPUT_ROOT / f"{station_slug}_lead{args.lead}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] Phase 6 dbarts — {station_friendly} "
          f"({station_slug}) lead {args.lead}h")
    print(f"  output: {out_dir}")

    print(f"[{time.strftime('%H:%M:%S')}] Building features…")
    df = build_features_via_duckdb(station_friendly, args.lead)
    train_df, val_df, test_df = time_split(df)
    print(f"  rows: {len(df):,} | train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

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
              f"({train_df['wet'].mean():.1%} wet)")

    X_train_full = train_df[FEATURE_NAMES].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="int8")
    X_test_full = test_df[FEATURE_NAMES].to_numpy(dtype="float64")
    y_test = test_df["wet"].to_numpy(dtype="int8")
    X_val_full = val_df[FEATURE_NAMES].to_numpy(dtype="float64")
    y_val = val_df["wet"].to_numpy(dtype="int8")

    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept_idx = np.where(~col_all_nan)[0]
    feature_names = [FEATURE_NAMES[i] for i in kept_idx]
    X_train = X_train_full[:, kept_idx]
    X_test = X_test_full[:, kept_idx]
    X_val = X_val_full[:, kept_idx]
    print(f"  features: {len(feature_names)} (after dropping all-NaN columns)")

    median = np.nanmedian(X_train, axis=0)
    X_train_imp = np.where(np.isnan(X_train), median, X_train)
    X_test_imp = np.where(np.isnan(X_test), median, X_test)
    scaler = StandardScaler().fit(X_train_imp)
    X_train_s = scaler.transform(X_train_imp)
    X_test_s = scaler.transform(X_test_imp)

    test_clim = train_df["wet"].mean()
    clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)

    # 1. Matched LightGBM (NaN-preserving) — same machinery as before
    X_train_lgb = train_df[feature_names].to_numpy(dtype="float64")
    print(f"[{time.strftime('%H:%M:%S')}] Matched LightGBM…")
    t0 = time.time()
    booster = fit_lightgbm_matched(X_train_lgb, y_train, X_val, y_val, seed=args.seed)
    p_test_lgb = booster.predict(X_test)
    lgb_brier = brier(p_test_lgb, y_test)
    lgb_bss = (clim_brier - lgb_brier) / clim_brier
    t_lgb = time.time() - t0
    print(f"  done in {t_lgb:.1f}s, best iter {booster.best_iteration}, Brier {lgb_brier:.4f}")

    # 2. dbarts (the headline)
    print(f"[{time.strftime('%H:%M:%S')}] dbarts BART (ntree={args.n_trees}, "
          f"nburn={args.n_burn}, nsamples={args.n_samples})…")
    t0 = time.time()
    p_test_db, p_draws = fit_dbarts_binary(
        X_train_s, y_train, X_test_s,
        n_trees=args.n_trees, n_burn=args.n_burn, n_samples=args.n_samples,
        seed=args.seed,
    )
    t_db = time.time() - t0
    db_brier = brier(p_test_db, y_test)
    db_bss = (clim_brier - db_brier) / clim_brier
    print(f"  done in {t_db:.1f}s, Brier {db_brier:.4f}")

    # 3. Context: prior pymc-bart Brier + Bayesian logreg + 3a deployed
    pmb_brier_ctx = None
    pmb_path = out_dir / "predictions_test.parquet"
    if pmb_path.exists():
        prev = pd.read_parquet(pmb_path)
        if len(prev) == len(y_test):
            pmb_brier_ctx = brier(prev["p_bart"].to_numpy(), prev["y_obs"].to_numpy())

    lr_brier_ctx = None
    lr_report = out_dir / "logreg_report.txt"
    if lr_report.exists():
        for line in lr_report.read_text().splitlines():
            if "Bayesian-logreg" in line and "blackjax" in line:
                # format: "Bayesian-logreg ...: 0.1228   BSS ..."
                try:
                    lr_brier_ctx = float(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
                break

    try:
        v_3a, brier_3a, _ = read_3a_baseline_brier(station_slug, args.lead)
        bss_3a = (clim_brier - brier_3a) / clim_brier
    except FileNotFoundError:
        v_3a, brier_3a, bss_3a = "(no 3a)", float("nan"), float("nan")

    # 4. Save dbarts test predictions + posterior draws (small enough to keep)
    pred_df = pd.DataFrame({
        "valid_time": test_df["ValidTimeUtc"].values,
        "p_dbarts": p_test_db,
        "y_obs": y_test,
    })
    pred_df.to_parquet(out_dir / "dbarts_predictions_test.parquet", index=False)
    np.savez_compressed(out_dir / "dbarts_p_draws.npz", p_draws=p_draws)

    # 5. Report
    rel_db = reliability_table(p_test_db, y_test)
    lines = [
        f"Phase 6 — dbarts (R) BART vs context",
        f"=====================================",
        f"",
        f"Station:         {args.station}",
        f"Lead hours:      {args.lead}",
        f"Train rows:      {len(train_df):,} (subsampled, seed {args.seed})",
        f"Test rows:       {len(test_df):,} ({test_df['ValidTimeUtc'].min()} → "
        f"{test_df['ValidTimeUtc'].max()})",
        f"Features:        {len(feature_names)}",
        f"dbarts cfg:      ntree={args.n_trees}, nburn={args.n_burn}, "
        f"nsamples={args.n_samples}",
        f"",
        f"Wall-clock",
        f"----------",
        f"dbarts:          {t_db:.1f}s",
        f"matched LGB:     {t_lgb:.1f}s",
        f"PyMC-BART:       ~10800s (3 hours) — saved from prior run",
        f"",
        f"Test Brier (lower = better)",
        f"---------------------------",
        f"Climatology (constant {test_clim:.3f}):                  {clim_brier:.4f}",
        f"dbarts (R, this run, {len(train_df):,}-row train, {args.n_trees} trees):  "
        f"{db_brier:.4f}   BSS {db_bss:+.4f}",
    ]
    if pmb_brier_ctx is not None:
        pmb_bss_ctx = (clim_brier - pmb_brier_ctx) / clim_brier
        lines.append(
            f"PyMC-BART (prior, same 5k train, 5 trees, context):  "
            f"{pmb_brier_ctx:.4f}   BSS {pmb_bss_ctx:+.4f}")
    if lr_brier_ctx is not None:
        lr_bss_ctx = (clim_brier - lr_brier_ctx) / clim_brier
        lines.append(
            f"Bayesian logreg (prior, same 5k train, blackjax):    "
            f"{lr_brier_ctx:.4f}   BSS {lr_bss_ctx:+.4f}")
    lines.append(
        f"LightGBM matched (this run, 5k train, NaN-preserving):  "
        f"{lgb_brier:.4f}   BSS {lgb_bss:+.4f}")
    lines.append(
        f"3a deployed ({v_3a}, ~14k train, context):  {brier_3a:.4f}   BSS {bss_3a:+.4f}")

    if pmb_brier_ctx is not None:
        lines += [
            f"",
            f"dbarts vs PyMC-BART (more trees + faster sampler — does it move Brier?)",
            f"  Δ Brier (dbarts − pmb): {db_brier - pmb_brier_ctx:+.4f}  "
            f"({(db_brier - pmb_brier_ctx) / pmb_brier_ctx * 100:+.2f}%)",
            f"  negative = dbarts wins (more trees pay off), positive = PyMC-BART wins "
            f"(despite 5 trees)",
            f"  speedup: {10800 / max(t_db, 1):.0f}× faster",
        ]

    lines += [
        f"",
        f"Reliability — dbarts test predictions (10 equal-width bins)",
        f"------------------------------------------------------------",
    ]
    for _, row in rel_db.iterrows():
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
    (out_dir / "dbarts_report.txt").write_text(text)
    print()
    print(text)
    print()
    print(f"Artefacts → {out_dir} (dbarts_report.txt, dbarts_predictions_test.parquet, "
          f"dbarts_p_draws.npz)")


if __name__ == "__main__":
    main()
