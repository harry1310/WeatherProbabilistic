"""Phase 6 — PyMC-BART vs LightGBM-3a head-to-head bake-off for P(wet ≥ 0.1 mm/h).

Trains a PyMC-BART model on the EXACT feature set + train/val/test split
that LightGBM-3a uses (offset_day forecasts, lead 24h, per-station EA gauge
truth, 70/15/15 time-ordered split), then scores Brier / BSS / reliability
on the same test slice and prints a head-to-head report.

Why BART vs 3a not vs 3d:
  3a has the bigger feature surface (22 features incl. NWP-mean covariates
  for RH / CAPE / cloud / wind alongside the 7 per-NWP precip + spread
  stats) and a 5-month test slice with ~3000 rows. That's a richer
  comparison ground than 3d's 4-month / smaller-feature setup at exact
  cycles. The 3a baseline Brier numbers are baked into the trained
  artefact's training_metadata.json — we read them directly so the
  comparison is against the exact deployed model, not a fresh re-train.

Why pure-Python (not C# parity export):
  Mirrors WeatherBlend's PrecipFeatureBuilder.BuildForLead SQL via DuckDB
  against the same parquet trees the C# trainer reads. No risk of feature
  drift (the SQL is identical line-for-line; comments inline cite the C#
  origin). Self-contained — run as `.venv/Scripts/python.exe scripts/
  run_phase6_bart_bakeoff.py [--station ea_bellever_dartmoor] [--lead 24]`.

Default knobs (matching 3a's training config + a moderate BART setup):
  * station = ea_bellever_dartmoor (most data, cleanest signal)
  * lead    = 24 (canonical comparison; 3a deploys at 24..120)
  * BART    = 50 trees, 1500 tune + 1500 draws, 4 chains, target_accept 0.95
  * Output  = reports/phase6_artefacts/<station>_lead<L>/{posterior.nc,
              report.txt, predictions_test.parquet}

Caveats:
  * BART training is slow vs LightGBM (~10× per second-per-row). One station
    one lead at our row count: roughly 30-90 minutes wall-clock on the
    Windows nutpie path. Plan accordingly.
  * Output dir is gitignored under reports/* by default; the .nc gets
    explicitly tracked via the `!reports/phase6_artefacts/**/*.nc`
    exception (mirrors the Phase 4/5 pattern).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Silence g++ warning on Windows (we use the nutpie sampler path, no C compiler
# needed — same pattern as Phase 5). Has to land before pymc imports.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

# Reconfigure stdout/stderr to utf-8 — Windows default cp1252 can't encode
# the arrow/ellipsis characters used in the report. Mirror the Phase 5 pattern.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import arviz as az  # noqa: E402
import lightgbm as lgb  # noqa: E402
import pymc as pm  # noqa: E402
import pymc_bart as pmb  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Feature-build helpers + station registry live in scripts/_shared.py so
# the production predict scripts (predict_4a / predict_5a) can use them
# without dragging the bake-off-only deps (lightgbm, pymc_bart) onto the
# import path. Re-exported here for back-compat with peer bake-off
# scripts that import these names from this module.
from _shared import (  # noqa: E402,F401
    FEATURE_NAMES,
    MODELS_LEAN,
    OUTPUT_ROOT,
    STATION_NAME_BY_SLUG,
    build_features_via_duckdb,
    resolve_station,
    time_split,
)
from src.data import LOCATION, WEATHERBLEND_DATA_ROOT, WET_THRESHOLD_MM  # noqa: E402,F401


def fit_lightgbm_matched(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> lgb.Booster:
    """Fit a LightGBM binary classifier on the SAME subsampled training
    set as BART so the comparison is apples-to-apples — not BART-on-5k vs
    deployed-3a-trained-on-14k. Hyperparameters mirror the deployed 3a's
    training_metadata exactly (commit a4f3345-ish era):
      * iter=600, lr=0.04, leaves=31, min-leaf=40
      * L1=0.1, L2=0.1, esr=40, seed=42
      * subsample=0.8 freq 1, feature_fraction=0.8, unbalanced=true
    Same NaN tolerance as the deployed model — no imputation; LightGBM
    treats NaN natively as a missingness signal."""
    train_set = lgb.Dataset(X_train, label=y_train.astype("float32"))
    val_set = lgb.Dataset(X_val, label=y_val.astype("float32"), reference=train_set)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "is_unbalance": True,
        "seed": seed,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=600,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return booster


def fit_bayesian_logreg_blackjax(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    seed: int,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
) -> tuple[pm.Model, az.InferenceData]:
    """Fit a Bayesian logistic regression on the SAME training rows as
    BART so the bake-off includes a "is the win from non-linearity (BART
    trees) or just from being Bayesian?" diagnostic. Linear-in-features
    so won't capture interactions BART can.

    Sampler: NUTS via blackjax (JAX-compiled, ~5-10× faster than PyMC's
    pure-Python NUTS on this size). For a 5k-row × 19-feature logreg
    expect ~5-15 min wall.

    Returns (model, idata). model wraps X as pm.Data so set_data works
    at predict time.
    """
    print(f"  Bayesian logreg via blackjax NUTS: tune={tune}, draws={draws}, chains={chains}")
    coords = {"feature": feature_names}
    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X_train, dims=("row", "feature"))
        y_data = pm.Data("y", y_train.astype("int8"), dims="row")
        # Standard weakly-informative priors. Coefficients on standardised
        # features (StandardScaler upstream), so unit-scale prior ≈ "doesn't
        # expect more than ±2σ swing per feature".
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.5)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")
        logits = intercept + pm.math.dot(X_data, beta)
        p = pm.Deterministic("p", pm.math.invlogit(logits), dims="row")
        pm.Bernoulli("obs", p=p, observed=y_data, dims="row")
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            random_seed=seed, target_accept=0.9,
            nuts_sampler="blackjax",
            return_inferencedata=True, progressbar=False,
        )
    return model, idata


def predict_bayesian_logreg(model: pm.Model, idata: az.InferenceData,
                            X_new: np.ndarray, seed: int = 123) -> np.ndarray:
    """Posterior-mean P(wet) per row via set_data + posterior_predictive.
    Same pattern as predict_bart — model has X + y wrapped as pm.Data."""
    n_new = X_new.shape[0]
    with model:
        pm.set_data({"X": X_new, "y": np.zeros(n_new, dtype="int8")})
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["p"], predictions=True,
            random_seed=seed, progressbar=False,
        )
    p_post = ppc.predictions["p"].values
    return p_post.mean(axis=(0, 1))


def pav_calibrate(p_val: np.ndarray, y_val: np.ndarray,
                  p_to_calibrate: np.ndarray) -> np.ndarray:
    """Post-hoc PAV (pool-adjacent-violators) isotonic calibration.
    Fits a monotonic step function on (val_pred, val_obs) pairs, then
    transforms `p_to_calibrate`. Mirrors WeatherBlend's
    DryWindowCalibrateCommand / Phase 3a-isotonic pattern.

    Why on val and not on the train predictions: train-set predictions
    are over-fit to the training labels — using them to calibrate would
    just mirror that over-fit. Val is held out from training (post the
    70/15 split, before test) so calibration on val applied to test
    is honest.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, y_val)
    return iso.transform(p_to_calibrate)


def fit_bart(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_trees: int,
    tune: int,
    draws: int,
    chains: int,
    seed: int,
    heartbeat_every: int = 50,
) -> az.InferenceData:
    """Fit a BART model on standardised features for binary classification.
    PG-BART sampler handles the BART RV; PyMC auto-selects it.

    **Per-draw heartbeat is mandatory** (per feedback memory
    `feedback_bayesian_training_must_log_progress`): PyMC's progressbar
    silently disables in non-TTY contexts (CI logs, redirected output,
    Bash run-in-background), so without an explicit callback the user
    sees the script "hang" with no visible progress for what can be
    30-90 minutes of BART sampling. The callback below prints one line
    every `heartbeat_every` draws — visible everywhere stdout flushes.
    """
    print(f"  PyMC-BART: m={n_trees} trees, tune={tune}, draws={draws}, chains={chains}, seed={seed}")
    print(f"  Heartbeat every {heartbeat_every} draws (per-chain).", flush=True)

    # Per-chain heartbeat tracking. PyMC fires the callback once per draw
    # per chain with (trace, draw); we print at heartbeat_every boundaries
    # plus at completion. Wall-clock timestamps so the user can spot
    # stalls vs slow-but-progressing runs.
    chain_starts: dict[int, float] = {}

    def heartbeat(trace, draw):  # type: ignore[no-untyped-def]
        chain = getattr(draw, "chain", 0)
        i = getattr(draw, "draw_idx", 0)
        if chain not in chain_starts:
            chain_starts[chain] = time.time()
            print(f"  [chain {chain}] starting…", flush=True)
        total = tune + draws
        if i == 0 or (i + 1) % heartbeat_every == 0 or i + 1 == total:
            phase = "tune" if i < tune else "draw"
            elapsed = time.time() - chain_starts[chain]
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [chain {chain}] {phase} {i+1}/{total} ({elapsed/60:.1f}min, {rate:.1f}/s)", flush=True)

    coords = {"feature": feature_names}
    with pm.Model(coords=coords) as model:
        # PyMC forbids RV name == dim name in v5+, so the dim is "row"
        # not "obs" — leaving the RV name "obs" for clarity downstream.
        # Both X and y wrapped as pm.Data so set_data can swap their
        # shapes at predict time (sample_posterior_predictive on a held-
        # out test set). Y= for pmb.BART takes the train values directly
        # — it's used during fit to compute leaf residuals and is
        # irrelevant at predict time (the trees already encode what
        # the prediction needs).
        X_data = pm.Data("X", X_train, dims=("row", "feature"))
        y_data = pm.Data("y", y_train.astype("int8"), dims="row")
        mu = pmb.BART("mu", X=X_data, Y=y_train.astype("float64"), m=n_trees)
        p = pm.Deterministic("p", pm.math.invlogit(mu), dims="row")
        pm.Bernoulli("obs", p=p, observed=y_data, dims="row")
        # PyMC-BART uses its own PGBART step sampler for the BART RV, not
        # NUTS — pm.sample auto-selects it. Don't pass target_accept here
        # (NUTS-only kwarg; PyMC v5 surfaces it as an "unknown step kwarg"
        # error when the only step is PGBART).
        # progressbar=False because we're providing our own heartbeat;
        # the rich-progressbar PyMC ships goes silent in non-TTY anyway.
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            random_seed=seed,
            return_inferencedata=True, progressbar=False,
            callback=heartbeat,
        )
    return model, idata


def get_bart_trees(model: pm.Model, var_name: str = "mu"):
    """Trees aren't saved in the InferenceData .nc file — they live on the
    BART RV's op (`bartrv.owner.op.all_trees`). Without them, predict on a
    new X falls back to a constant `Y.mean()` of the original training shape,
    breaking when X_new has a different row count."""
    return list(model.named_vars[var_name].owner.op.all_trees)


def set_bart_trees(model: pm.Model, trees, var_name: str = "mu") -> None:
    # `op.all_trees` is a multiprocessing ListProxy (no .clear() method).
    # Drain via pop, then extend.
    op = model.named_vars[var_name].owner.op
    while len(op.all_trees) > 0:
        op.all_trees.pop()
    op.all_trees.extend(trees)


def predict_bart(model: pm.Model, idata: az.InferenceData, X_new: np.ndarray, seed: int = 123) -> np.ndarray:
    """Posterior-mean P(wet) per row of X_new. Uses pm.set_data + posterior
    predictive sampling against the BART trees attached to the model's RV op.
    Caller must ensure trees are present on the op (fresh fit or
    set_bart_trees after rebuild)."""
    n_new = X_new.shape[0]
    with model:
        pm.set_data({"X": X_new, "y": np.zeros(n_new, dtype="int8")})
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["p"], predictions=True,
            random_seed=seed, progressbar=False,
        )
    # ppc.predictions["p"] shape: (chain, draw, row). Average over the
    # posterior axes for the headline mean P(wet) per row.
    p_post = ppc.predictions["p"].values
    return p_post.mean(axis=(0, 1))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": 0,
                         "p_mean": float("nan"), "y_rate": float("nan")})
        else:
            rows.append({
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "n": n,
                "p_mean": float(p[mask].mean()),
                "y_rate": float(y[mask].mean()),
            })
    return pd.DataFrame(rows)


def read_3a_baseline_brier(station: str, lead_hours: int) -> tuple[str, float, int]:
    """Look up the deployed 3a Brier for (station, lead) from the live
    training_metadata.json. Picks the most-recent v* whose Phase = "3a".
    Returns (version, brier, test_rows)."""
    station_dir = WEATHERBLEND_DATA_ROOT / "models" / "precipitation" / station
    if not station_dir.exists():
        raise FileNotFoundError(f"3a model dir not found: {station_dir}")
    candidates = []
    for vdir in sorted(station_dir.glob("v*")):
        meta_path = vdir / "training_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("Phase") != "3a":
            continue
        per_lead = meta.get("PerLead", {})
        if str(lead_hours) not in per_lead:
            continue
        s = per_lead[str(lead_hours)]
        candidates.append((meta.get("TrainedAtUtc", ""), vdir.name, s["BlendTestMae"], s["TestRows"]))
    if not candidates:
        raise FileNotFoundError(f"No deployed 3a artefact has Phase=3a + lead {lead_hours}h for {station}")
    candidates.sort(reverse=True)
    _, version, brier_3a, n_test = candidates[0]
    return version, float(brier_3a), int(n_test)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--station", default="ea_bellever_dartmoor",
                   help="EA gauge station slug (default Bellever — most data).")
    p.add_argument("--lead", type=int, default=24,
                   help="Forecast lead hours; must match a 3a deployed lead. Default 24.")
    p.add_argument("--n-trees", type=int, default=50,
                   help="BART m (number of trees). Default 50.")
    p.add_argument("--tune", type=int, default=1500)
    p.add_argument("--draws", type=int, default=1500)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse-bart-posterior", action="store_true",
                   help="Skip BART fit and load posterior.nc from the output "
                        "directory (saves the ~3h training when only the logreg / "
                        "PAV add-ons changed). The data prep + train/test "
                        "split must match exactly so the saved trees apply "
                        "to the same X.")
    p.add_argument("--skip-logreg", action="store_true",
                   help="Skip the Bayesian-logreg head-to-head. Default fits.")
    p.add_argument("--skip-pav", action="store_true",
                   help="Skip post-hoc PAV calibration of BART output. Default fits.")
    p.add_argument("--subsample-train", type=int, default=0,
                   help="Subsample training rows to N (stratified by wet label) "
                        "before fitting. 0 = no subsample. PyMC-BART's PGBART step "
                        "scales ~linearly with row count and runs hopelessly slowly "
                        "on the full ~14k-row 3a training set; --subsample-train 2000 "
                        "or 4000 is the practical knob for finishing in reasonable time. "
                        "Test set is always full size for an apples-to-apples comparison "
                        "to 3a's deployed Brier.")
    args = p.parse_args()

    station_slug, station_friendly = resolve_station(args.station)
    out_dir = OUTPUT_ROOT / f"{station_slug}_lead{args.lead}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] Phase 6 BART bake-off — {station_friendly} ({station_slug}) lead {args.lead}h")
    print(f"  output: {out_dir}")

    # 1. Build the feature matrix via DuckDB (3a-equivalent).
    print(f"[{time.strftime('%H:%M:%S')}] Building features (mirror of 3a SQL)…")
    df = build_features_via_duckdb(station_friendly, args.lead)
    print(f"  rows: {len(df):,} spanning {df.ValidTimeUtc.min()} → {df.ValidTimeUtc.max()}")
    wet_rate = df["wet"].mean()
    print(f"  wet rate: {wet_rate:.1%} ({df['wet'].sum()} of {len(df)})")

    # 2. Time-ordered 70/15/15 split.
    train_df, val_df, test_df = time_split(df)
    print(f"  split → train {len(train_df):,} ({train_df['wet'].mean():.1%} wet) | "
          f"val {len(val_df):,} ({val_df['wet'].mean():.1%}) | "
          f"test {len(test_df):,} ({test_df['wet'].mean():.1%})")

    # Optional stratified subsample of TRAINING rows. Wet/dry stratification
    # preserves the class balance; row order within each stratum is shuffled
    # by the chosen seed before taking the first N. Test set untouched —
    # apples-to-apples comparison vs 3a's deployed Brier requires the full
    # 4-month test slice.
    if args.subsample_train > 0 and args.subsample_train < len(train_df):
        rng = np.random.default_rng(args.seed)
        # `.to_numpy()` returns a read-only view of the pandas index — copy
        # so rng.shuffle's in-place modification doesn't ValueError.
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

    # Drop ALL-NaN-in-training columns first. Open-Meteo's offset_day API
    # doesn't include some columns (e.g. cloud_low/mid/high_mean), so the
    # AVG over the per-NWP rows is NaN for every training row. LightGBM
    # ignores all-NaN columns natively as a missingness signal; BART
    # would inherit NaN through the median-impute step and break. Drop
    # them, log which got dropped, and continue with the working subset.
    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept_idx = np.where(~col_all_nan)[0]
    dropped_features = [FEATURE_NAMES[i] for i in np.where(col_all_nan)[0]]
    feature_names = [FEATURE_NAMES[i] for i in kept_idx]
    X_train = X_train_full[:, kept_idx]
    X_test = X_test_full[:, kept_idx]
    if dropped_features:
        print(f"  dropped {len(dropped_features)} all-NaN-in-training feature(s): {dropped_features}")
    print(f"  effective feature count: {len(feature_names)} (of {len(FEATURE_NAMES)})")

    # BART tolerates NaN only via imputation — impute remaining sporadic
    # NaN (e.g. AIFS missing in the early 2024 segment, ECMWF rare misses)
    # with the PER-COLUMN training-set median. Same imputation applied
    # to test for consistency.
    median = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), median, X_train)
    X_test = np.where(np.isnan(X_test), median, X_test)

    # Standardise so BART's tree-split scores are scale-comparable across
    # features. (Strictly optional — trees are scale-invariant per-feature
    # — but doesn't hurt and makes any later linear post-hoc easier.)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 3a. Matched LightGBM baseline — same subsampled training rows as BART
    # so the comparison isn't BART-on-5k vs deployed-3a-on-14k. Uses NON-
    # imputed features (LightGBM's native NaN handling); val set is the
    # post-train slice from the original 70/15/15 split, full size, also
    # non-imputed.
    X_val_full = val_df[FEATURE_NAMES].to_numpy(dtype="float64")
    y_val = val_df["wet"].to_numpy(dtype="int8")
    X_train_lgb = train_df[feature_names].to_numpy(dtype="float64")  # NaN-preserving
    X_val_lgb = X_val_full[:, kept_idx]
    X_test_lgb = X_test_full[:, kept_idx]
    print(f"[{time.strftime('%H:%M:%S')}] Training matched LightGBM (same {len(train_df):,}-row subsample)…")
    t_lgb = time.time()
    booster = fit_lightgbm_matched(X_train_lgb, y_train, X_val_lgb, y_val, seed=args.seed)
    p_test_lgb = booster.predict(X_test_lgb)
    lgb_brier = brier(p_test_lgb, y_test)
    print(f"  done in {(time.time() - t_lgb):.1f}s, best iter {booster.best_iteration}")

    # 3b. Fit BART (on standardised, median-imputed features) — or, in
    # reuse mode, load the prior run's saved test predictions directly
    # from predictions_test.parquet. We don't try to re-predict from a
    # rebuilt model graph because the trees themselves aren't in the .nc
    # (they live on `bartrv.owner.op.all_trees`); future runs save them
    # via bart_trees.pkl, but we can't synthesise that retroactively for
    # the existing posterior. Reuse path therefore can't produce val
    # predictions, so PAV is auto-skipped.
    posterior_path = out_dir / "posterior.nc"
    trees_path = out_dir / "bart_trees.pkl"
    test_preds_path = out_dir / "predictions_test.parquet"
    test_clim = train_df["wet"].mean()
    clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)

    if args.reuse_bart_posterior and posterior_path.exists() and test_preds_path.exists():
        print(f"[{time.strftime('%H:%M:%S')}] Reusing saved BART test predictions at {test_preds_path}…")
        prev = pd.read_parquet(test_preds_path)
        if len(prev) != len(y_test):
            raise SystemExit(
                f"Saved predictions have {len(prev)} rows but current test set has {len(y_test)}. "
                f"The data window has shifted — refit instead.")
        p_test = prev["p_bart"].to_numpy()
        if not args.skip_pav:
            print(f"[{time.strftime('%H:%M:%S')}] WARN: --skip-pav implied — reuse path lacks val "
                  f"predictions (trees not pickled in the saved run), so PAV can't be fit honestly. "
                  f"Skipping PAV.")
            args.skip_pav = True
        model, idata = None, None
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Training PyMC-BART…")
        t0 = time.time()
        model, idata = fit_bart(X_train_s, y_train, feature_names,
                         n_trees=args.n_trees, tune=args.tune, draws=args.draws,
                         chains=args.chains, seed=args.seed)
        print(f"  done in {(time.time() - t0) / 60:.1f} min")
        # Trees are stored on the BART RV op, not in the .nc — pickle them
        # so future --reuse-bart-posterior runs can predict on new X.
        with open(trees_path, "wb") as f:
            pickle.dump(get_bart_trees(model), f)
        print(f"  saved trees → {trees_path}")

        print(f"[{time.strftime('%H:%M:%S')}] Scoring BART on test + val…")
        p_test = predict_bart(model, idata, X_test_s, seed=args.seed)
        if not args.skip_pav:
            # Val features through the same imputation + standardiser used for
            # train and test. Drop same dropped cols.
            X_val_full_arr = val_df[FEATURE_NAMES].to_numpy(dtype="float64")[:, kept_idx]
            X_val_imp = np.where(np.isnan(X_val_full_arr), median, X_val_full_arr)
            X_val_s = scaler.transform(X_val_imp)
            p_val = predict_bart(model, idata, X_val_s, seed=args.seed)

    bart_brier = brier(p_test, y_test)
    bart_bss = (clim_brier - bart_brier) / clim_brier
    lgb_bss = (clim_brier - lgb_brier) / clim_brier

    # 4b. PAV calibration. Fit isotonic on (val_pred, val_obs); apply to
    # test predictions. Honest: val is held out of training, so the
    # calibrator doesn't see what BART trained on. Skipped automatically
    # in reuse mode (we don't have val predictions there).
    if not args.skip_pav:
        print(f"[{time.strftime('%H:%M:%S')}] Fitting PAV calibrator on val + applying to test…")
        p_test_pav = pav_calibrate(p_val, y_val, p_test)
        bart_pav_brier = brier(p_test_pav, y_test)
        bart_pav_bss = (clim_brier - bart_pav_brier) / clim_brier
    else:
        p_test_pav, bart_pav_brier, bart_pav_bss = None, None, None

    # 4c. Bayesian logreg head-to-head (same train rows, blackjax-NUTS).
    if not args.skip_logreg:
        print(f"[{time.strftime('%H:%M:%S')}] Training Bayesian logreg via blackjax…")
        t_lr = time.time()
        lr_model, lr_idata = fit_bayesian_logreg_blackjax(
            X_train_s, y_train, feature_names, seed=args.seed,
            tune=1000, draws=1000, chains=4,
        )
        print(f"  done in {(time.time() - t_lr) / 60:.1f} min")
        p_test_lr = predict_bayesian_logreg(lr_model, lr_idata, X_test_s, seed=args.seed)
        lr_brier = brier(p_test_lr, y_test)
        lr_bss = (clim_brier - lr_brier) / clim_brier
        # Save logreg posterior for later inspection if needed.
        az.to_netcdf(lr_idata, str(out_dir / "logreg_posterior.nc"))
    else:
        lr_brier, lr_bss = None, None

    # 5. Deployed 3a baseline (for context only — DIFFERENT training set,
    # so not a fair comparison; useful as a "is the matched-LGB sane?"
    # check).
    try:
        v_3a, brier_3a, n_test_3a = read_3a_baseline_brier(station_slug, args.lead)
        bss_3a = (clim_brier - brier_3a) / clim_brier
    except FileNotFoundError as e:
        v_3a, brier_3a, n_test_3a, bss_3a = "(no 3a baseline found)", float("nan"), 0, float("nan")
        print(f"  WARN: {e}")

    # 6. Save artefacts. In reuse mode `idata` is None and we leave the
    # existing posterior.nc + predictions_test.parquet in place.
    if idata is not None:
        az.to_netcdf(idata, str(out_dir / "posterior.nc"))
        pred_df = pd.DataFrame({
            "valid_time": test_df["ValidTimeUtc"].values,
            "p_bart": p_test,
            "y_obs": y_test,
        })
        pred_df.to_parquet(out_dir / "predictions_test.parquet", index=False)

    rel = reliability_table(p_test, y_test)

    report_lines = [
        f"Phase 6 — PyMC-BART vs LightGBM-3a head-to-head",
        f"================================================",
        f"",
        f"Station:      {args.station}",
        f"Lead hours:   {args.lead}",
        f"BART config:  m={args.n_trees}, tune={args.tune}, draws={args.draws}, "
        f"chains={args.chains}, seed={args.seed}",
        f"",
        f"Data",
        f"----",
        f"Total rows:   {len(df):,} (wet rate {wet_rate:.1%})",
        f"Train:        {len(train_df):,} ({train_df['ValidTimeUtc'].min()} → {train_df['ValidTimeUtc'].max()})",
        f"Val:          {len(val_df):,} ({val_df['ValidTimeUtc'].min()} → {val_df['ValidTimeUtc'].max()})",
        f"Test:         {len(test_df):,} ({test_df['ValidTimeUtc'].min()} → {test_df['ValidTimeUtc'].max()})",
        f"",
        f"Test Brier (lower = better)",
        f"---------------------------",
        f"Climatology (constant test_clim={test_clim:.3f}):                  {clim_brier:.4f}",
        f"PyMC-BART  (this run, {len(train_df):,}-row train):                 {bart_brier:.4f}   BSS {bart_bss:+.4f}",
    ]
    if bart_pav_brier is not None:
        report_lines.append(
            f"PyMC-BART + PAV calibration (val-fit, test-applied):    {bart_pav_brier:.4f}   BSS {bart_pav_bss:+.4f}")
    report_lines += [
        f"LightGBM   (matched, same {len(train_df):,}-row train):             {lgb_brier:.4f}   BSS {lgb_bss:+.4f}",
    ]
    if lr_brier is not None:
        report_lines.append(
            f"Bayesian-logreg (matched, same {len(train_df):,}-row train, blackjax NUTS): {lr_brier:.4f}   BSS {lr_bss:+.4f}")
    report_lines += [
        f"3a deployed ({v_3a}, ~14k-row train, context only):  {brier_3a:.4f}   BSS {bss_3a:+.4f}",
        f"",
        f"PRIMARY: BART vs matched LightGBM (same training rows)",
        f"  Δ Brier:  {bart_brier - lgb_brier:+.4f}  ({(bart_brier - lgb_brier) / lgb_brier * 100:+.2f}%)",
        f"  negative = BART wins on this subsample, positive = LightGBM wins",
        f"",
        f"Sanity: matched LightGBM vs 3a deployed",
        f"  Δ Brier:  {lgb_brier - brier_3a:+.4f}  ({(lgb_brier - brier_3a) / brier_3a * 100:+.2f}%)",
        f"  expect positive (smaller training set → some loss); large gap means subsample is too small.",
    ]
    if bart_pav_brier is not None:
        report_lines += [
            f"",
            f"PAV-calibrated BART vs raw BART (does post-hoc isotonic help?)",
            f"  Δ Brier:  {bart_pav_brier - bart_brier:+.4f}  "
            f"({(bart_pav_brier - bart_brier) / bart_brier * 100:+.2f}%)",
            f"  negative = PAV improves Brier, positive = PAV hurts",
        ]
    if lr_brier is not None:
        report_lines += [
            f"",
            f"DIAGNOSTIC: Bayesian logreg vs BART (is the BART win from non-linearity, or just from being Bayesian?)",
            f"  Δ Brier (BART − logreg): {bart_brier - lr_brier:+.4f}  "
            f"({(bart_brier - lr_brier) / lr_brier * 100:+.2f}%)",
            f"  negative = BART (trees + Bayesian) beats logreg (linear + Bayesian) — non-linearity is the source",
            f"  near-zero = logreg captures most of the signal; BART's win is mostly Bayesian-uncertainty",
        ]
    report_lines += [
        f"",
        f"Reliability (10 equal-width bins)",
        f"---------------------------------",
    ]
    for _, row in rel.iterrows():
        if row["n"] == 0:
            report_lines.append(f"  [{row['bin_lo']:.2f},{row['bin_hi']:.2f})  n=0")
        else:
            report_lines.append(
                f"  [{row['bin_lo']:.2f},{row['bin_hi']:.2f})  "
                f"n={int(row['n']):>4d}  p_mean={row['p_mean']:.3f}  y_rate={row['y_rate']:.3f}  "
                f"diff={row['y_rate'] - row['p_mean']:+.3f}"
            )
    report_text = "\n".join(report_lines)
    (out_dir / "report.txt").write_text(report_text)
    print()
    print(report_text)
    print()
    print(f"Artefacts → {out_dir}")


if __name__ == "__main__":
    main()
