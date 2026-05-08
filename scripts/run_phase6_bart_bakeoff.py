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
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import LOCATION, WEATHERBLEND_DATA_ROOT, WET_THRESHOLD_MM  # noqa: E402

# Mirrors PrecipFeatureBuilder.cs's lean spec — 7 NWPs, all optional, no required.
# Order matters: matches `precip_<short>` column names in the C# pivot so any
# downstream join across artefacts (3a metadata, BART predictions) lines up.
MODELS_LEAN = [
    ("gfs_seamless", "gfs"),
    ("ecmwf_ifs025", "ecmwf"),
    ("icon_seamless", "icon"),
    ("meteofrance_seamless", "mf"),
    ("gem_seamless", "gem"),
    ("ecmwf_aifs025_single", "aifs"),
    ("jma_seamless", "jma"),
]

# Feature names in the same order PrecipFeatureBuilder.ComposeRow writes them.
# 22 columns total — pinned here so the BART model + the report use the same
# column convention as 3a's deployed feature_schema.json.
FEATURE_NAMES = (
    [f"precip_{short}" for _, short in MODELS_LEAN]
    + ["precip_mean", "precip_std", "precip_max", "precip_agreement_wet_01"]
    + ["rh_mean", "dew_depression_mean", "cloud_low_mean", "cloud_mid_mean",
       "cloud_high_mean", "cape_mean", "wind_speed_mean"]
    + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
)
assert len(FEATURE_NAMES) == 22

OUTPUT_ROOT = ROOT / "reports" / "phase6_artefacts"

# Slug ↔ friendly name. Friendly form is what's stored on the
# parquet's StationName column (filter target for the truth join);
# slug form is what model directories + predictions trees use
# (lookup target for the 3a baseline metadata). Tracked here so
# both halves of the bake-off see the same set of stations.
STATION_NAME_BY_SLUG = {
    "ea_bellever_dartmoor": "Bellever Dartmoor",
    "ea_bovey_tracey": "Bovey Tracey",
    "ea_dartmoor_nr_hexworthy": "Dartmoor nr Hexworthy",
    "ea_princetown": "Princetown",
}


def resolve_station(station_input: str) -> tuple[str, str]:
    """Accept either the slug ('ea_bellever_dartmoor') or the friendly
    name ('Bellever Dartmoor') from the user. Returns (slug, friendly)."""
    # Slug match.
    if station_input in STATION_NAME_BY_SLUG:
        return station_input, STATION_NAME_BY_SLUG[station_input]
    # Friendly-name match (case-insensitive).
    for slug, friendly in STATION_NAME_BY_SLUG.items():
        if station_input.lower() == friendly.lower():
            return slug, friendly
    raise ValueError(
        f"Unknown station '{station_input}'. Known: "
        + ", ".join(f"{s}  ({n})" for s, n in STATION_NAME_BY_SLUG.items())
    )


def build_features_via_duckdb(
    station_friendly: str,
    lead_hours: int,
) -> pd.DataFrame:
    """Mirror of WeatherBlend's PrecipFeatureBuilder.BuildForLead (C#) using
    DuckDB on the same parquet trees. Returns a DataFrame with the 22
    features in FEATURE_NAMES order plus the binary truth label.

    SQL is line-for-line equivalent to the C# pivot at
    src/WeatherBlend/Train/PrecipFeatureBuilder.cs:126-181 (with model
    names hardcoded for the lean 7-NWP set). Truth is the EA gauge with
    the strict 4-of-4 partial-hour rule the C# version uses (groups of
    15-min readings collapse to hourly only when all four are present).
    """
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    rn_glob = str((WEATHERBLEND_DATA_ROOT / "truth" / "rainfall" / "**" / "*.parquet")).replace("\\", "/")

    model_in_clause = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
    precip_pivot = ",\n        ".join(
        f"MAX(CASE WHEN Model = '{full}' THEN Precipitation END) AS precip_{short}"
        for full, short in MODELS_LEAN
    )
    precip_select = ", ".join(f"p.precip_{short}" for _, short in MODELS_LEAN)
    any_not_null = "(" + " OR ".join(f"p.precip_{short} IS NOT NULL" for _, short in MODELS_LEAN) + ")"

    sql = f"""
    WITH hourly_truth AS (
        SELECT
            date_trunc('hour', ObservedTimeUtc) AS valid_time,
            SUM(Value15MinMm) AS precip_mm_hour
        FROM read_parquet('{rn_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND StationName  = '{station_friendly}'
          AND Value15MinMm IS NOT NULL
        GROUP BY 1
        HAVING COUNT(*) = 4
    ),
    latest AS (
        SELECT
            ValidTimeUtc, Model,
            Precipitation,
            RelativeHumidity2m, Temperature2m, DewPoint2m,
            CloudCoverLow, CloudCoverMid, CloudCoverHigh,
            Cape, WindSpeed10m,
            ROW_NUMBER() OVER (
                PARTITION BY ValidTimeUtc, Model
                ORDER BY RunTimeUtc DESC
            ) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours = {lead_hours}
          AND Model IN {model_in_clause}
    ),
    pivoted AS (
        SELECT
            ValidTimeUtc,
            {precip_pivot},
            AVG(RelativeHumidity2m) AS rh_mean,
            AVG(Temperature2m - DewPoint2m) AS dew_depression_mean,
            AVG(CloudCoverLow)  AS cloud_low_mean,
            AVG(CloudCoverMid)  AS cloud_mid_mean,
            AVG(CloudCoverHigh) AS cloud_high_mean,
            AVG(Cape)           AS cape_mean,
            AVG(WindSpeed10m)   AS wind_speed_mean
        FROM latest
        WHERE rn = 1
        GROUP BY ValidTimeUtc
    )
    SELECT
        p.ValidTimeUtc,
        {precip_select},
        p.rh_mean, p.dew_depression_mean,
        p.cloud_low_mean, p.cloud_mid_mean, p.cloud_high_mean,
        p.cape_mean, p.wind_speed_mean,
        t.precip_mm_hour
    FROM pivoted p
    JOIN hourly_truth t ON p.ValidTimeUtc = t.valid_time
    WHERE {any_not_null}
    ORDER BY p.ValidTimeUtc
    """

    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()

    # Compose spread features + cyclical features in one pass — matches
    # PrecipFeatureBuilder.ComposeRow exactly. NaN-safe via numpy nanmean/
    # nanstd/nanmax; agreement is wet-count over present-count.
    precip_cols = [f"precip_{short}" for _, short in MODELS_LEAN]
    pm_arr = df[precip_cols].to_numpy(dtype="float64")
    df["precip_mean"] = np.nanmean(pm_arr, axis=1)
    # Sample std with NaN handling — match PrecipFeatureBuilder's manual
    # variance calc which uses `presentCount` not `presentCount-1`.
    present = (~np.isnan(pm_arr)).sum(axis=1)
    sumsq = np.nansum(pm_arr ** 2, axis=1)
    sumv = np.nansum(pm_arr, axis=1)
    mean_safe = np.where(present > 0, sumv / np.maximum(present, 1), np.nan)
    var = np.maximum(0.0, sumsq / np.maximum(present, 1) - mean_safe ** 2)
    df["precip_std"] = np.where(present > 1, np.sqrt(var), 0.0)
    df["precip_max"] = np.nanmax(pm_arr, axis=1)
    wet_count = (pm_arr >= WET_THRESHOLD_MM).sum(axis=1)  # NaN compares False so OK
    df["precip_agreement_wet_01"] = np.where(present > 0, wet_count / np.maximum(present, 1), np.nan)

    # Cyclical (UTC hour-of-day, day-of-year). PrecipFeatureBuilder uses
    # (DayOfYear - 1) / 365 — exact same denominator + offset here.
    hour_angle = 2.0 * np.pi * df["ValidTimeUtc"].dt.hour / 24.0
    doy_angle = 2.0 * np.pi * (df["ValidTimeUtc"].dt.dayofyear - 1) / 365.0
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["doy_sin"] = np.sin(doy_angle)
    df["doy_cos"] = np.cos(doy_angle)

    df["wet"] = (df["precip_mm_hour"] >= WET_THRESHOLD_MM).astype("int8")
    return df.reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 time-ordered split mirroring BinaryDataset.Split (C#)."""
    n = len(df)
    train_end = int(np.floor(n * train_frac))
    val_end = train_end + int(np.floor(n * val_frac))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


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


def predict_bart(model: pm.Model, idata: az.InferenceData, X_new: np.ndarray, seed: int = 123) -> np.ndarray:
    """Posterior-mean P(wet) per row of X_new. PyMC-BART 0.11 doesn't
    expose a direct `BART.predict` (that was removed somewhere along the
    API churn) so we use the canonical PyMC pattern: pm.set_data swaps
    the X (and a placeholder y of the right size) into the model, then
    sample_posterior_predictive evaluates the deterministic `p` per
    posterior draw on the new X. The BART RV's tree state already lives
    in idata; set_data on X doesn't refit, just re-evaluates."""
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

    # 3b. Fit BART (on standardised, median-imputed features).
    print(f"[{time.strftime('%H:%M:%S')}] Training PyMC-BART…")
    t0 = time.time()
    model, idata = fit_bart(X_train_s, y_train, feature_names,
                     n_trees=args.n_trees, tune=args.tune, draws=args.draws,
                     chains=args.chains, seed=args.seed)
    print(f"  done in {(time.time() - t0) / 60:.1f} min")

    # 4. Score on the FULL test set (full 4-month slice, untouched by
    # the training subsample).
    print(f"[{time.strftime('%H:%M:%S')}] Scoring on test…")
    p_test = predict_bart(model, idata, X_test_s, seed=args.seed)
    bart_brier = brier(p_test, y_test)
    test_clim = train_df["wet"].mean()
    clim_brier = brier(np.full_like(y_test, test_clim, dtype="float64"), y_test)
    bart_bss = (clim_brier - bart_brier) / clim_brier
    lgb_bss = (clim_brier - lgb_brier) / clim_brier

    # 5. Deployed 3a baseline (for context only — DIFFERENT training set,
    # so not a fair comparison; useful as a "is the matched-LGB sane?"
    # check).
    try:
        v_3a, brier_3a, n_test_3a = read_3a_baseline_brier(station_slug, args.lead)
        bss_3a = (clim_brier - brier_3a) / clim_brier
    except FileNotFoundError as e:
        v_3a, brier_3a, n_test_3a, bss_3a = "(no 3a baseline found)", float("nan"), 0, float("nan")
        print(f"  WARN: {e}")

    # 6. Save artefacts.
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
        f"LightGBM   (matched, same {len(train_df):,}-row train):             {lgb_brier:.4f}   BSS {lgb_bss:+.4f}",
        f"3a deployed ({v_3a}, ~14k-row train, context only):  {brier_3a:.4f}   BSS {bss_3a:+.4f}",
        f"",
        f"PRIMARY: BART vs matched LightGBM (same training rows)",
        f"  Δ Brier:  {bart_brier - lgb_brier:+.4f}  ({(bart_brier - lgb_brier) / lgb_brier * 100:+.2f}%)",
        f"  negative = BART wins on this subsample, positive = LightGBM wins",
        f"",
        f"Sanity: matched LightGBM vs 3a deployed",
        f"  Δ Brier:  {lgb_brier - brier_3a:+.4f}  ({(lgb_brier - brier_3a) / brier_3a * 100:+.2f}%)",
        f"  expect positive (smaller training set → some loss); large gap means subsample is too small.",
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
