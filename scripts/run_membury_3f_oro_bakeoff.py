"""Phase 3f orographic-features bake-off (Membury).

Tests whether orographic features improve the Membury rainfall_amount
(hourly intensity) NGBoost-LogNormal stage-2, layered on lean vs rich
features, per-station vs pooled. Bake-off only — no production edits.

Plan: ``WeatherBlend/docs/RAINFALL_AMOUNT_3F_ORO_BAKEOFF_PLAN.md``.

Six arms (plus an optional pooled-no-terrain attribution control G):

  A  baseline      per-station  lean 15                control (production recipe)
  B  lean+dyn-oro  per-station  lean 15 + 5 dynamic    oro-on-lean (intensity)
  C  rich          per-station  rich 59                isolate rich-feature lift
  D  rich+dyn-oro  per-station  rich 59 + 5 dynamic    the 4a-winning recipe
  E  pool lean+oro pooled (3)   lean 15 + 9 terrain    does pooling activate static terrain?
  F  pool rich+oro pooled (3)   rich 59 + 9 terrain    3o-style recipe, on intensity
  G  pool rich     pooled (3)   rich 59                control: pooling-data vs orography (F vs G)

Static terrain (elevation/relief/ruggedness/station_id) is dropped in the
per-station arms (constant within a (gauge, lead) cell ⇒ zero signal) and
only the 5 wind-flow-dependent dynamic features are kept. The pooled arms
stack all 3 gauges so the full 9-feature block (incl. station_id) carries
cross-gauge variance.

Methodology mirrors ``run_membury_two_stage_ngboost.py`` (the script that
produced the 3f baseline CRPS) and ``train_3f.py`` (the production fit):

  * Stage-1 P(wet) = π: a LightGBM binary classifier on the LEAN-15
    features, fit per (gauge, lead) on the chronological train/val split,
    predicted on the test slice. π depends ONLY on the lean features and
    the split — NOT on an arm's stage-2 features — so it is identical
    across every arm for a given cell, which is what makes the CRPS deltas
    cleanly attributable to the stage-2 feature set.
  * Stage-2 = NGBoost-LogNormal on the wet-only rows, standardised
    features, LogScore, early stopping on the wet val slice (production
    recipe from train_3f.fit_one_lead).
  * Score = test-set CRPS of the mixed distribution
    F(x) = (1-π)·δ_0 + π·LogNormal(μ,σ), via the 20-quantile crps_mixed
    estimator (same as the baseline + stage-1 bake-off).

Two hard constraints for this run (2026-05-30):
  * PREVIOUS-RUNS Open-Meteo data ONLY — RunTimeSource='offset_day'
    everywhere (build_features_via_duckdb hardcodes it; rich/aux pass it
    explicitly). NEVER 'reported'/'synthesised'.
  * 2024 ONWARDS ONLY — min_valid_time = 2024-01-01 on every builder,
    avoiding the NaN-laden 2022-2023 backfill parquets.

Ship-bar (from RICH_PER_STATION_4A_SHIP_PLAN.md): best arm vs A must clear
≥2% aggregate CRPS at 24+48 and ≥1% at 72. Otherwise record the negative
result and stop.

Usage:
    .venv/Scripts/python.exe -u scripts/run_membury_3f_oro_bakeoff.py
    .venv/Scripts/python.exe -u scripts/run_membury_3f_oro_bakeoff.py --no-arm-g
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb

# _shared resolves ACTIVE_LOCATION from WB_LOCATION at import time (defaulting
# to WB's primary site, Bonehill). This bake-off is Membury-only, so set it
# BEFORE importing _shared — otherwise every DuckDB pull filters on the wrong
# location and returns zero rows. Production sets this via the CI matrix.
os.environ.setdefault("WB_LOCATION", "membury_devon")

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import lognorm

from ngboost import NGBRegressor
from ngboost.distns import LogNormal as NGBLogNormal
from ngboost.scores import LogScore
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import WET_THRESHOLD_MM  # noqa: E402
from _shared import (  # noqa: E402
    MODELS_LEAN,
    RICH_FEATURE_NAMES,
    V1_TERRAIN_FEATURE_NAMES,
    build_features_via_duckdb,
    build_rich_features_via_duckdb,
    compose_v1_terrain_block,
    time_split,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

LOCATION = "membury_devon"

# (friendly, slug, pooled-station-index). Order fixes oro_station_id in the
# pooled arms — stable across runs so the feature is reproducible.
STATIONS = [
    ("Chards Snowdon Hill", "ea_chards_snowdon_hill", 0),
    ("Goren",               "ea_goren",               1),
    ("Raymonds Hill",       "ea_raymonds_hill",       2),
]

LEADS = (24, 48, 72)  # primary scope (matches the baseline + the ship-bar)

# Previous-runs Open-Meteo + 2024-onwards — the two run constraints.
RUN_TIME_SOURCE = "offset_day"
MIN_VALID_TIME = datetime(2024, 1, 1)

# ---- Feature sets ----------------------------------------------------------

# Lean 15 = 7 NWP precip + 4 spread (over the 7 lean NWPs) + 4 calendar. The
# lean spread cols are RENAMED with a _l7 suffix when merged into the combined
# cell table so they don't collide with the rich builder's same-named spread
# cols (which are computed over 8 NWPs incl. UKMO).
L7_PRECIP = [f"precip_{short}" for _, short in MODELS_LEAN]  # 7
LEAN_SPREAD_L7 = ["precip_mean_l7", "precip_std_l7", "precip_max_l7", "precip_agree_l7"]
CALENDAR = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
LEAN15 = L7_PRECIP + LEAN_SPREAD_L7 + CALENDAR
assert len(LEAN15) == 15

RICH59 = list(RICH_FEATURE_NAMES)            # 59, precip cols first (8 NWPs)
FULL9 = list(V1_TERRAIN_FEATURE_NAMES)       # 9 (3 static + 5 dynamic + station_id)
# Dynamic-only terrain (drop the 3 static + station_id for per-station arms).
DYN5 = ["oro_wind_sin", "oro_wind_cos", "oro_upwind_gain_per_wind_5km_m",
        "oro_uplift_m_per_s", "oro_uplift_x_q_g_per_kg"]
assert all(f in FULL9 for f in DYN5)

N_PRECIP_LEAN = len(L7_PRECIP)   # 7 — row-mean-imputed precip cols are first
N_PRECIP_RICH = 8                # rich precip cols (incl ukmo) are first

# arm -> (feature_names, n_precip, architecture). "ps" = per-station, "pool".
ARMS = {
    "A": (LEAN15,          N_PRECIP_LEAN, "ps",   "baseline lean15"),
    "B": (LEAN15 + DYN5,   N_PRECIP_LEAN, "ps",   "lean15 + dyn-oro"),
    "C": (RICH59,          N_PRECIP_RICH, "ps",   "rich59"),
    "D": (RICH59 + DYN5,   N_PRECIP_RICH, "ps",   "rich59 + dyn-oro"),
    "E": (LEAN15 + FULL9,  N_PRECIP_LEAN, "pool", "pooled lean15 + 9-terrain"),
    "F": (RICH59 + FULL9,  N_PRECIP_RICH, "pool", "pooled rich59 + 9-terrain"),
    "G": (RICH59,          N_PRECIP_RICH, "pool", "pooled rich59 (no terrain)"),
}

# ---- NGBoost + LightGBM hyperparameters (pinned, mirror the references) ----

NGB_BASE = dict(n_estimators=500, learning_rate=0.01,
                minibatch_frac=1.0, col_sample=1.0, verbose=False, random_state=42)
NGB_EARLY_STOP = 30

LGB_BASE = {"num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20,
            "lambda_l1": 0.1, "lambda_l2": 0.1, "feature_fraction": 0.9,
            "verbose": -1, "seed": 42, "num_threads": 0}
LGB_NUM_ITERS = 500
LGB_EARLY_STOP = 30

# 20-quantile grid for the CRPS estimator (same as the baseline + stage-1 bake-off).
QUANTILE_ALPHAS_EVAL = np.round(np.linspace(1 / 41, 40 / 41, 20), 4)

MIN_WET_TRAIN_ROWS = 100
MIN_WET_VAL_ROWS = 20


# ----------------------------------------------------------------------------
# Numerics (verbatim from the reference scripts so results are comparable)
# ----------------------------------------------------------------------------

def emos_impute(X: np.ndarray, n_precip: int) -> np.ndarray:
    """Row-mean impute the first ``n_precip`` (NWP precip) columns; column-median
    fill everything else. Bit-identical to train_3f.emos_impute / the bake-off
    scripts, generalised to the precip-column count of the arm's feature set."""
    X = X.copy()
    P = X[:, :n_precip]
    row_mean = np.nanmean(P, axis=1)
    row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
    idx = np.where(np.isnan(P))
    P[idx] = row_mean[idx[0]]
    X[:, :n_precip] = P
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)  # all-NaN column -> 0
    idx = np.where(np.isnan(X))
    X[idx] = med[idx[1]]
    return X


def crps_mixed(pi: np.ndarray, quantiles: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Quantile estimator of CRPS for (1-π)·δ_0 + π·LogNormal. Verbatim from
    run_membury_two_stage_ngboost.crps_mixed."""
    K = quantiles.shape[1]
    w_dry = 1.0 - pi
    w_wet = pi / K
    term1 = w_dry * y + w_wet * np.abs(quantiles - y[:, None]).sum(axis=1)
    cross_0k = 2.0 * w_dry * w_wet * quantiles.sum(axis=1)
    pairwise = np.abs(quantiles[:, :, None] - quantiles[:, None, :]).sum(axis=(1, 2))
    cross_kl = (w_wet ** 2) * pairwise
    return term1 - 0.5 * (cross_0k + cross_kl)


def fit_pwet(X_tr, y_tr, X_va, y_va):
    """Stage-1 LightGBM P(wet) — mirrors run_membury_two_stage_ngboost.fit_pwet.
    Raw (unimputed) features: LightGBM handles NaN natively."""
    params = dict(LGB_BASE, objective="binary", metric="binary_logloss")
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    return lgb.train(params, train_set, num_boost_round=LGB_NUM_ITERS,
                     valid_sets=[val_set], valid_names=["val"],
                     callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False)])


def ngb_quantiles(ngb, X_te_s) -> np.ndarray:
    """Predict the LogNormal quantile matrix on the eval grid, clipped to the
    wet threshold. Mirrors the stage-1 bake-off's predict_ngboost_lognormal."""
    dist = ngb.pred_dist(X_te_s, max_iter=ngb.best_val_loss_itr)
    s = np.asarray(dist.params["s"], dtype="float64")
    scale = np.asarray(dist.params["scale"], dtype="float64")
    qs = np.empty((X_te_s.shape[0], len(QUANTILE_ALPHAS_EVAL)), dtype="float64")
    for k, a in enumerate(QUANTILE_ALPHAS_EVAL):
        qs[:, k] = lognorm.ppf(a, s=s, scale=scale)
    return np.clip(qs, WET_THRESHOLD_MM, None)


# ----------------------------------------------------------------------------
# One-time pruned cache — the builders each scan the full forecast tree
# (all locations/models/years). For a Membury-only 2024+ bake-off that's ~18
# min/cell × 9 cells of redundant I/O. We scan the real tree ONCE here, write
# a pruned parquet (Membury / offset_day / 2024+ / rich-8 models / leads),
# then repoint _shared.WEATHERBLEND_DATA_ROOT at the tiny cache so EVERY
# builder reads the small tree with NO change to the C#-parity-tested feature
# logic. Filters mirror the builders' own WHERE clauses exactly so cached
# rows are bit-identical to what an unpruned run would see.
# ----------------------------------------------------------------------------

def build_pruned_cache(real_root: Path, cache_root: Path) -> None:
    rich_models = "(" + ",".join(f"'{full}'" for full, _ in
                                 __import__("_shared").MODELS_RICH) + ")"
    leads_in = "(" + ",".join(str(l) for l in LEADS) + ")"
    cut = f"{MIN_VALID_TIME:%Y-%m-%d %H:%M:%S}"
    stations_in = "(" + ",".join(f"'{f}'" for f, _, _ in STATIONS) + ")"

    fc_src = str((real_root / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    rn_src = str((real_root / "truth" / "rainfall" / "**" / "*.parquet")).replace("\\", "/")
    fc_out = cache_root / "forecasts" / "membury.parquet"
    rn_out = cache_root / "truth" / "rainfall" / "membury.parquet"
    fc_out.parent.mkdir(parents=True, exist_ok=True)
    rn_out.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    t0 = time.time()
    # Keep ALL runs per (valid_time, model) — the builders' ROW_NUMBER picks
    # the freshest, so we must not dedup here. SELECT * keeps every column.
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{fc_src}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='{LOCATION}' AND RunTimeSource='{RUN_TIME_SOURCE}'
              AND Model IN {rich_models} AND LeadHours IN {leads_in}
              AND ValidTimeUtc >= TIMESTAMP '{cut}'
        ) TO '{str(fc_out).replace(chr(92), '/')}' (FORMAT PARQUET)
    """)
    print(f"  [cache] forecasts pruned ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{rn_src}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='{LOCATION}' AND StationName IN {stations_in}
              AND ObservedTimeUtc >= TIMESTAMP '{cut}'
        ) TO '{str(rn_out).replace(chr(92), '/')}' (FORMAT PARQUET)
    """)
    print(f"  [cache] rainfall pruned ({time.time()-t0:.0f}s)", flush=True)
    con.close()

    # Orographic JSONs are read from {root}/static/orographic/{slug}.json.
    dst = cache_root / "static" / "orographic"
    dst.mkdir(parents=True, exist_ok=True)
    for j in (real_root / "static" / "orographic").glob("*.json"):
        shutil.copy2(j, dst / j.name)


# ----------------------------------------------------------------------------
# Cell table — one aligned frame per (station, lead)
# ----------------------------------------------------------------------------

def build_cell_table(friendly: str, slug: str, station_index: int, lead: int) -> pd.DataFrame:
    """Build a single ValidTimeUtc-aligned frame carrying lean-15, rich-59 and
    the 9 terrain features + truth/wet, so every arm scores on identical rows
    for this (station, lead). Previous-runs source + 2024 cutoff enforced."""
    lean = build_features_via_duckdb(friendly, lead, min_valid_time=MIN_VALID_TIME)
    if lean.empty:
        return lean
    lean = lean.rename(columns={
        "precip_mean": "precip_mean_l7", "precip_std": "precip_std_l7",
        "precip_max": "precip_max_l7", "precip_agreement_wet_01": "precip_agree_l7",
    })

    rich = build_rich_features_via_duckdb(
        friendly, lead, min_valid_time=MIN_VALID_TIME, run_time_source=RUN_TIME_SOURCE)
    if rich.empty:
        return rich
    richoro = compose_v1_terrain_block(
        slug, station_index, lead, rich,
        min_valid_time=MIN_VALID_TIME, run_time_source=RUN_TIME_SOURCE)
    if richoro.empty:
        return richoro

    # Merge the 4 renamed lean-spread cols onto the rich+oro frame by valid
    # time. precip_<nwp> / calendar already live in `rich` and are identical
    # to the lean builder's (same offset_day pivot), so only the 7-NWP spread
    # cols are unique to lean. Inner join => the common row universe.
    M = richoro.merge(lean[["ValidTimeUtc", *LEAN_SPREAD_L7]], on="ValidTimeUtc", how="inner")
    M = M.sort_values("ValidTimeUtc").reset_index(drop=True)
    return M


def split_and_stage1(M: pd.DataFrame):
    """70/15/15 chronological split + stage-1 π on the test slice. Returns
    (tr, va, te, pi_te, y_te) or None when the cell is too thin to fit."""
    tr, va, te = time_split(M)
    y_tr = tr["precip_mm_hour"].to_numpy(dtype="float64")
    y_va = va["precip_mm_hour"].to_numpy(dtype="float64")
    wet_tr = (y_tr >= WET_THRESHOLD_MM)
    wet_va = (y_va >= WET_THRESHOLD_MM)
    if wet_tr.sum() < MIN_WET_TRAIN_ROWS or wet_va.sum() < MIN_WET_VAL_ROWS or te.empty:
        return None

    # Stage-1 P(wet) on raw lean-15 (LightGBM tolerates NaN). Identical π for
    # every arm of this cell.
    Xtr_l = tr[LEAN15].to_numpy(dtype="float64")
    Xva_l = va[LEAN15].to_numpy(dtype="float64")
    Xte_l = te[LEAN15].to_numpy(dtype="float64")
    clf = fit_pwet(Xtr_l, wet_tr.astype("int8"), Xva_l, wet_va.astype("int8"))
    pi_te = clf.predict(Xte_l, num_iteration=clf.best_iteration)
    y_te = te["precip_mm_hour"].to_numpy(dtype="float64")
    return tr, va, te, pi_te, y_te


def fit_ngb_lognormal(X_tr_w_s, y_tr_w, X_va_w_s, y_va_w):
    ngb = NGBRegressor(Dist=NGBLogNormal, Score=LogScore, **NGB_BASE)
    ngb.fit(X_tr_w_s, np.maximum(y_tr_w, 1e-6),
            X_val=X_va_w_s, Y_val=np.maximum(y_va_w, 1e-6),
            early_stopping_rounds=NGB_EARLY_STOP)
    return ngb


# ----------------------------------------------------------------------------
# Per-station arms (A-D)
# ----------------------------------------------------------------------------

def score_per_station_arm(cell, feats, n_precip) -> tuple[float, int]:
    """Fit NGBoost-LogNormal on this cell's wet rows for the arm's feature set,
    return (mean test CRPS, n_test). Mirrors train_3f.fit_one_lead's
    impute -> StandardScaler -> NGBoost recipe."""
    tr, va, te, pi_te, y_te = cell
    y_tr = tr["precip_mm_hour"].to_numpy(dtype="float64")
    y_va = va["precip_mm_hour"].to_numpy(dtype="float64")
    wet_tr = y_tr >= WET_THRESHOLD_MM
    wet_va = y_va >= WET_THRESHOLD_MM

    X_tr_w = emos_impute(tr.loc[wet_tr, feats].to_numpy(dtype="float64"), n_precip)
    X_va_w = emos_impute(va.loc[wet_va, feats].to_numpy(dtype="float64"), n_precip)
    X_te = emos_impute(te[feats].to_numpy(dtype="float64"), n_precip)

    scaler = StandardScaler().fit(X_tr_w)
    ngb = fit_ngb_lognormal(scaler.transform(X_tr_w), y_tr[wet_tr],
                            scaler.transform(X_va_w), y_va[wet_va])
    qs = ngb_quantiles(ngb, scaler.transform(X_te))
    crps = float(crps_mixed(pi_te, qs, y_te).mean())
    return crps, len(y_te)


# ----------------------------------------------------------------------------
# Pooled arms (E-G): one NGBoost per lead over the stacked 3-gauge wet rows,
# scored on each gauge's own test slice with that gauge's own π.
# ----------------------------------------------------------------------------

def score_pooled_arm(cells_by_station, lead, feats, n_precip) -> list[dict]:
    """cells_by_station: {friendly: cell-tuple} for this lead. Returns one
    record per gauge: {station, lead, crps, n_test}."""
    Xtr_list, ytr_list, Xva_list, yva_list = [], [], [], []
    for cell in cells_by_station.values():
        tr, va, te, pi_te, y_te = cell
        y_tr = tr["precip_mm_hour"].to_numpy(dtype="float64")
        y_va = va["precip_mm_hour"].to_numpy(dtype="float64")
        wet_tr = y_tr >= WET_THRESHOLD_MM
        wet_va = y_va >= WET_THRESHOLD_MM
        Xtr_list.append(tr.loc[wet_tr, feats].to_numpy(dtype="float64"))
        ytr_list.append(y_tr[wet_tr])
        Xva_list.append(va.loc[wet_va, feats].to_numpy(dtype="float64"))
        yva_list.append(y_va[wet_va])

    # Pool raw, impute once on the pooled matrix, scale on the pooled train.
    X_tr_w = emos_impute(np.vstack(Xtr_list), n_precip)
    X_va_w = emos_impute(np.vstack(Xva_list), n_precip)
    y_tr_w = np.concatenate(ytr_list)
    y_va_w = np.concatenate(yva_list)
    scaler = StandardScaler().fit(X_tr_w)
    ngb = fit_ngb_lognormal(scaler.transform(X_tr_w), y_tr_w,
                            scaler.transform(X_va_w), y_va_w)

    records = []
    for friendly, cell in cells_by_station.items():
        tr, va, te, pi_te, y_te = cell
        X_te = emos_impute(te[feats].to_numpy(dtype="float64"), n_precip)
        qs = ngb_quantiles(ngb, scaler.transform(X_te))
        crps = float(crps_mixed(pi_te, qs, y_te).mean())
        records.append({"station": friendly, "lead": lead, "crps": crps, "n_test": len(y_te)})
    return records


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out", default=None, help="Output dir (default: reports/membury_3f_oro_bakeoff_<date>).")
    ap.add_argument("--no-arm-g", action="store_true", help="Skip the pooled-no-terrain control (arm G).")
    ap.add_argument("--no-cache", action="store_true",
                    help="Skip the pruned cache and scan the full parquet tree per cell (slow).")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(args.out) if args.out else ROOT / "reports" / f"membury_3f_oro_bakeoff_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    arms_to_run = [a for a in ARMS if not (a == "G" and args.no_arm_g)]
    ps_arms = [a for a in arms_to_run if ARMS[a][2] == "ps"]
    pool_arms = [a for a in arms_to_run if ARMS[a][2] == "pool"]

    print(f"[start] {datetime.now():%H:%M:%S}  out={out_dir}", flush=True)
    print(f"  constraints: RunTimeSource='{RUN_TIME_SOURCE}' (previous runs), "
          f"valid_time >= {MIN_VALID_TIME:%Y-%m-%d}", flush=True)
    print(f"  arms: {arms_to_run}  leads: {LEADS}  stations: {[s[0] for s in STATIONS]}", flush=True)

    # ---- One-time pruned cache, then repoint the builders at it ----
    if not args.no_cache:
        import _shared
        real_root = _shared.WEATHERBLEND_DATA_ROOT
        cache_root = out_dir / "_pruned_cache"
        print(f"  [cache] building pruned tree at {cache_root} (one scan each) …", flush=True)
        build_pruned_cache(real_root, cache_root)
        _shared.WEATHERBLEND_DATA_ROOT = cache_root
        print(f"  [cache] builders repointed at pruned tree.", flush=True)

    # ---- Build aligned cell tables + stage-1 π once per (station, lead) ----
    cells: dict[tuple[str, int], tuple] = {}
    for friendly, slug, idx in STATIONS:
        for lead in LEADS:
            t0 = time.time()
            M = build_cell_table(friendly, slug, idx, lead)
            if M.empty:
                print(f"  [skip] {friendly} {lead}h: empty cell table", flush=True)
                continue
            prepared = split_and_stage1(M)
            if prepared is None:
                print(f"  [skip] {friendly} {lead}h: too few wet rows", flush=True)
                continue
            cells[(friendly, lead)] = prepared
            tr, va, te, pi_te, y_te = prepared
            wet_rate = float((y_te >= WET_THRESHOLD_MM).mean())
            print(f"  [cell] {friendly:20s} {lead}h: rows={len(M):6d} "
                  f"te={len(te):5d} wet_te={wet_rate:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    if not cells:
        raise SystemExit("No cells built — check data / cutoff.")

    rows: list[dict] = []

    # ---- Per-station arms (A-D) ----
    for arm in ps_arms:
        feats, n_precip, _, label = ARMS[arm]
        for (friendly, lead), cell in cells.items():
            t0 = time.time()
            crps, n_te = score_per_station_arm(cell, feats, n_precip)
            rows.append({"arm": arm, "station": friendly, "lead": lead,
                         "crps": crps, "n_test": n_te})
            print(f"  [{arm}] {friendly:20s} {lead}h  CRPS={crps:.4f}  "
                  f"n={n_te}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- Pooled arms (E-G) ----
    for arm in pool_arms:
        feats, n_precip, _, label = ARMS[arm]
        for lead in LEADS:
            cells_by_station = {f: cells[(f, lead)] for f, _, _ in STATIONS if (f, lead) in cells}
            if len(cells_by_station) < 2:
                print(f"  [{arm}] lead {lead}h: <2 gauges available — skipping pool", flush=True)
                continue
            t0 = time.time()
            for rec in score_pooled_arm(cells_by_station, lead, feats, n_precip):
                rows.append({"arm": arm, **rec})
            print(f"  [{arm}] lead {lead}h  pooled fit over {len(cells_by_station)} gauges  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_cell_crps.csv", index=False)

    # ---- Aggregate: mean over stations per (arm, lead) ----
    agg = (df.groupby(["arm", "lead"])
             .agg(crps=("crps", "mean"), n_test=("n_test", "sum"))
             .reset_index())

    # Δ vs arm A per lead.
    base = agg[agg["arm"] == "A"].set_index("lead")["crps"].to_dict()
    agg["delta_vs_A_pct"] = agg.apply(
        lambda r: (r["crps"] - base[r["lead"]]) / base[r["lead"]] * 100.0
        if r["lead"] in base else float("nan"), axis=1)

    # Overall: aggregate-mean across the primary leads, weighted by n_test.
    overall = (df.groupby("arm")
                 .apply(lambda g: pd.Series({
                     "crps": float((g["crps"] * g["n_test"]).sum() / g["n_test"].sum()),
                     "n_test": int(g["n_test"].sum())}), include_groups=False)
                 .reset_index())
    base_overall = float(overall.loc[overall["arm"] == "A", "crps"].iloc[0])
    overall["delta_vs_A_pct"] = (overall["crps"] - base_overall) / base_overall * 100.0
    overall = overall.sort_values("crps").reset_index(drop=True)

    write_report(out_dir, agg, overall, arms_to_run)
    print(f"\n[done] {datetime.now():%H:%M:%S}  wrote {out_dir/'summary.md'}", flush=True)


def write_report(out_dir: Path, agg: pd.DataFrame, overall: pd.DataFrame, arms_to_run) -> None:
    lines = [
        "# Phase 3f orographic-features bake-off (Membury)",
        "",
        f"Run {datetime.now():%Y-%m-%d %H:%M} — previous-runs Open-Meteo "
        f"(`offset_day`), valid_time ≥ {MIN_VALID_TIME:%Y-%m-%d}. Stage-2 "
        "NGBoost-LogNormal; stage-1 π = LightGBM P(wet) on lean-15 (identical "
        "across arms per cell). Score = mixed-distribution test CRPS "
        "(negative Δ = variant beats baseline A).",
        "",
        "## Arms",
        "",
        "| Arm | Architecture | Features |",
        "|---|---|---|",
    ]
    for a in arms_to_run:
        feats, n_precip, arch, label = ARMS[a]
        arch_h = "per-station" if arch == "ps" else "pooled (3 gauges)"
        lines.append(f"| {a} | {arch_h} | {label} ({len(feats)} feat) |")

    lines += ["", "## Per-lead CRPS (mean across 3 gauges)", "",
              "| Arm | Lead | CRPS | Δ vs A |", "|---|---:|---:|---:|"]
    for _, r in agg.sort_values(["lead", "crps"]).iterrows():
        lines.append(f"| {r['arm']} | {int(r['lead'])} | {r['crps']:.4f} | {r['delta_vs_A_pct']:+.2f}% |")

    lines += ["", "## Overall (n_test-weighted across leads 24/48/72)", "",
              "| Arm | CRPS | Δ vs A | n_test |", "|---|---:|---:|---:|"]
    for _, r in overall.iterrows():
        lines.append(f"| {r['arm']} | {r['crps']:.4f} | {r['delta_vs_A_pct']:+.2f}% | {int(r['n_test'])} |")

    lines += [
        "", "## Attribution reads",
        "- **A → C:** rich-feature contribution (per-station).",
        "- **C → D:** dynamic-oro on top of rich (per-station).",
        "- **A → B:** dynamic-oro on lean (intensity-vs-occurrence check).",
        "- **{A,C} → {E,F}:** does pooling + static terrain help?",
        "- **F → G:** isolates pooling-data gain from orography (F≈G ⇒ gain is pooling, not oro).",
        "",
        "## Ship-bar",
        "Best arm vs A: ≥2% aggregate CRPS at 24+48, ≥1% at 72. If unmet, "
        "record the negative result and stop.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
