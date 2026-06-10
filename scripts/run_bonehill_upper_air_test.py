"""Quick upper-air signal test — do the OM Historical-Forecast pressure-level
(…hPa) fields improve the 24h-lead Phase 3a P(wet) occurrence model at Bonehill?

This is an exploratory "is there any signal" probe, NOT a production change.

Design (matches `WeatherBlend/src/WeatherBlend/Models/RunTimeSources.cs`):
  * Base model = Phase 3a recipe: LightGBM binary P(wet ≥ 0.1mm/h) on the
    lean-22 feature set, fit per (gauge) at lead 24h, on previous-runs
    (`offset_day`) forecasts, EA-gauge wet/dry truth. 70/15/15 time split.
  * Pressure-level fields live ONLY on `RunTimeSource='hist_forecast'`
    (OM Historical-Forecast API; Previous-Runs refuses pressure vars). They
    are LEAD-UNLABELLED (RunTime=ValidTime, LeadHours=0) → joined to the
    24h rows by **valid-time** as a lead-invariant upper-air field.
  * Compare test-set Brier (primary), logloss, AUC: base vs base+pressure.

CAVEAT (state it in any write-up): a hist_forecast field at valid-time is
closer to an *analysis / short-range* upper-air estimate than to a genuine
24h-lead forecast. So a positive result here is an UPPER BOUND on the
deployable benefit — it answers "is there upper-air signal for occurrence?",
not "can we get it at 24h lead in production". Both base + pressure use
2024-01-01+ data (the pressure backfill window).

Prereq: sync the Bonehill hist_forecast files from R2 first:
  rclone copy r2:weatherblend/data/forecasts/location=bonehill_rocks/ \
    data/forecasts/location=bonehill_rocks/ --include "**/hist_forecast.parquet"

Usage:
  .venv/Scripts/python.exe -u scripts/run_bonehill_upper_air_test.py
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# UTF-8 stdout so non-ASCII (→) doesn't crash on CP1252 hosts (mirror train_3f).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# 3a/Bonehill is the default location, but set it explicitly so the _shared
# builders resolve the right tree regardless of ambient env.
os.environ.setdefault("WB_LOCATION", "bonehill_rocks")

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import WET_THRESHOLD_MM  # noqa: E402
import _shared  # noqa: E402
from _shared import (  # noqa: E402
    FEATURE_NAMES as LEAN_FEATURES,
    build_features_via_duckdb,
    resolve_station,
    time_split,
)

LOCATION = "bonehill_rocks"
LEAD = 24
MIN_VALID_TIME = datetime(2024, 1, 1)

# Bonehill EA rainfall gauges (slugs → resolved to friendly names below).
STATION_SLUGS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]

# Raw NWP-mean pressure-level fields (averaged across all hist_forecast models
# per valid-time). Wind direction carried as sin/cos to avoid the 0/360 wrap.
PRESSURE_RAW = ["ua_t850", "ua_t700", "ua_t500", "ua_z850", "ua_z500",
                "ua_ws850", "ua_ws500", "ua_wd850_sin", "ua_wd850_cos",
                "ua_wd500_sin", "ua_wd500_cos", "ua_rh850"]
PRESSURE_DERIVED = ["ua_thickness_500_850", "ua_lapse_850_500",
                    "ua_lapse_850_700", "ua_lapse_700_500", "ua_shear_850_500"]
PRESSURE_FEATURES = PRESSURE_RAW + PRESSURE_DERIVED

LGB_BASE = {"num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20,
            "lambda_l1": 0.1, "lambda_l2": 0.1, "feature_fraction": 0.9,
            "verbose": -1, "seed": 42, "num_threads": 0,
            "objective": "binary", "metric": "binary_logloss"}
NUM_ITERS = 500
EARLY_STOP = 30


def build_pruned_cache(real_root: Path, cache_root: Path) -> None:
    """One scan of the large Bonehill tree → a tiny pruned parquet, then the
    builders read that. Keeps: offset_day lead-24 rows (base 3a features) +
    ALL hist_forecast rows (pressure), Bonehill, valid_time >= 2024-01-01;
    plus the 3 gauges' rainfall truth. Mirrors run_membury_3f_oro_bakeoff."""
    fc_src = str((real_root / "forecasts" / "location=bonehill_rocks" / "**" / "*.parquet")).replace("\\", "/")
    rn_src = str((real_root / "truth" / "rainfall" / "**" / "*.parquet")).replace("\\", "/")
    fc_out = cache_root / "forecasts" / "location=bonehill_rocks" / "all.parquet"
    rn_out = cache_root / "truth" / "rainfall" / "bonehill.parquet"
    fc_out.parent.mkdir(parents=True, exist_ok=True)
    rn_out.parent.mkdir(parents=True, exist_ok=True)
    cut = f"{MIN_VALID_TIME:%Y-%m-%d %H:%M:%S}"
    con = duckdb.connect(":memory:")
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{fc_src}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='{LOCATION}' AND ValidTimeUtc >= TIMESTAMP '{cut}'
              AND ( (RunTimeSource='offset_day' AND LeadHours={LEAD})
                    OR RunTimeSource='hist_forecast' )
        ) TO '{str(fc_out).replace(chr(92), '/')}' (FORMAT PARQUET)
    """)
    print(f"  [cache] forecasts pruned ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{rn_src}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='{LOCATION}' AND ObservedTimeUtc >= TIMESTAMP '{cut}'
        ) TO '{str(rn_out).replace(chr(92), '/')}' (FORMAT PARQUET)
    """)
    print(f"  [cache] rainfall pruned ({time.time()-t0:.0f}s)", flush=True)
    con.close()

    # Orographic JSONs (read by compose_v1_terrain_block for the 3o variant).
    src_oro = real_root / "static" / "orographic"
    if src_oro.exists():
        dst_oro = cache_root / "static" / "orographic"
        dst_oro.mkdir(parents=True, exist_ok=True)
        for j in src_oro.glob("*.json"):
            shutil.copy2(j, dst_oro / j.name)


def load_upper_air() -> pd.DataFrame:
    """NWP-mean pressure-level fields per valid-time from hist_forecast rows.
    Returns ValidTimeUtc + the 12 raw + 5 derived upper-air features."""
    fc_glob = str((_shared.WEATHERBLEND_DATA_ROOT / "forecasts" / "location=bonehill_rocks"
                   / "**" / "*.parquet")).replace("\\", "/")
    cut = f"{MIN_VALID_TIME:%Y-%m-%d %H:%M:%S}"
    sql = f"""
        WITH latest AS (
            SELECT ValidTimeUtc, Model,
                   Temperature850hPa, Temperature700hPa, Temperature500hPa,
                   GeopotentialHeight850hPa, GeopotentialHeight500hPa,
                   WindSpeed850hPa, WindSpeed500hPa,
                   WindDirection850hPa, WindDirection500hPa,
                   RelativeHumidity850hPa,
                   ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model
                                      ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
            WHERE LocationName='{LOCATION}' AND RunTimeSource='hist_forecast'
              AND ValidTimeUtc >= TIMESTAMP '{cut}'
              AND Temperature500hPa IS NOT NULL
        )
        SELECT ValidTimeUtc,
            AVG(Temperature850hPa) ua_t850, AVG(Temperature700hPa) ua_t700,
            AVG(Temperature500hPa) ua_t500,
            AVG(GeopotentialHeight850hPa) ua_z850, AVG(GeopotentialHeight500hPa) ua_z500,
            AVG(WindSpeed850hPa) ua_ws850, AVG(WindSpeed500hPa) ua_ws500,
            AVG(SIN(WindDirection850hPa*pi()/180.0)) ua_wd850_sin,
            AVG(COS(WindDirection850hPa*pi()/180.0)) ua_wd850_cos,
            AVG(SIN(WindDirection500hPa*pi()/180.0)) ua_wd500_sin,
            AVG(COS(WindDirection500hPa*pi()/180.0)) ua_wd500_cos,
            AVG(RelativeHumidity850hPa) ua_rh850,
            count(DISTINCT Model) n_models
        FROM latest WHERE rn = 1
        GROUP BY ValidTimeUtc ORDER BY ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()
    if df.empty:
        return df
    df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"])

    # Derived synoptic quantities.
    df["ua_thickness_500_850"] = df["ua_z500"] - df["ua_z850"]       # 1000-500 layer warmth proxy
    df["ua_lapse_850_500"] = df["ua_t850"] - df["ua_t500"]           # mid-tropo instability
    df["ua_lapse_850_700"] = df["ua_t850"] - df["ua_t700"]
    df["ua_lapse_700_500"] = df["ua_t700"] - df["ua_t500"]
    # Vector shear magnitude 850→500 (u=-ws·sin, v=-ws·cos).
    u850 = -df["ua_ws850"] * df["ua_wd850_sin"]; v850 = -df["ua_ws850"] * df["ua_wd850_cos"]
    u500 = -df["ua_ws500"] * df["ua_wd500_sin"]; v500 = -df["ua_ws500"] * df["ua_wd500_cos"]
    df["ua_shear_850_500"] = np.sqrt((u500 - u850) ** 2 + (v500 - v850) ** 2)
    return df


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_pwet(X_tr, y_tr, X_va, y_va):
    ds_tr = lgb.Dataset(X_tr, label=y_tr)
    ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)
    return lgb.train(LGB_BASE, ds_tr, num_boost_round=NUM_ITERS,
                     valid_sets=[ds_va], valid_names=["val"],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def evaluate(df, feats, label="wet"):
    tr, va, te = time_split(df)
    Xtr, ytr = tr[feats].to_numpy("float64"), tr[label].to_numpy("int8")
    Xva, yva = va[feats].to_numpy("float64"), va[label].to_numpy("int8")
    Xte, yte = te[feats].to_numpy("float64"), te[label].to_numpy("int8")
    booster = fit_pwet(Xtr, ytr, Xva, yva)
    p = booster.predict(Xte, num_iteration=booster.best_iteration)
    auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan")
    return {"brier": brier(p, yte), "logloss": logloss(p, yte), "auc": auc,
            "n_test": len(yte), "wet_rate": float(yte.mean()),
            "best_iter": booster.best_iteration}


def main() -> None:
    print(f"[start] {datetime.now():%H:%M:%S}  Bonehill upper-air signal test, lead {LEAD}h", flush=True)
    print("  base = 3a lean-22 P(wet); +UA adds 17 hist_forecast pressure features (join by valid-time)", flush=True)

    # One-time pruned cache then repoint the builders at it (the Bonehill tree
    # is ~43k files; un-cached this is ~4 full scans).
    out_dir = ROOT / "reports" / f"bonehill_upper_air_test_{datetime.now():%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "_pruned_cache"
    print(f"  [cache] building pruned tree at {cache_root} …", flush=True)
    build_pruned_cache(_shared.WEATHERBLEND_DATA_ROOT, cache_root)
    _shared.WEATHERBLEND_DATA_ROOT = cache_root
    print("  [cache] builders repointed at pruned tree.", flush=True)

    ua = load_upper_air()
    if ua.empty:
        raise SystemExit("No hist_forecast pressure rows found — sync from R2 first "
                         "(see module docstring).")
    print(f"  upper-air rows: {len(ua):,}  valid_time {ua['ValidTimeUtc'].min()} -> "
          f"{ua['ValidTimeUtc'].max()}  (models/row median {int(ua['n_models'].median())})", flush=True)

    rows = []
    for slug in STATION_SLUGS:
        s_slug, friendly = resolve_station(slug)
        t0 = time.time()
        base = build_features_via_duckdb(friendly, LEAD, min_valid_time=MIN_VALID_TIME)
        if base.empty:
            print(f"  [skip] {friendly}: no base rows", flush=True)
            continue
        base["ValidTimeUtc"] = pd.to_datetime(base["ValidTimeUtc"])
        merged = base.merge(ua.drop(columns=["n_models"]), on="ValidTimeUtc", how="left")
        cov = merged[PRESSURE_FEATURES[0]].notna().mean()

        res_base = evaluate(merged, LEAN_FEATURES)
        res_ua = evaluate(merged, LEAN_FEATURES + PRESSURE_FEATURES)
        d_brier = (res_ua["brier"] - res_base["brier"]) / res_base["brier"] * 100.0
        rows.append({"station": friendly, "n_test": res_base["n_test"],
                     "wet_rate": res_base["wet_rate"], "ua_coverage": cov,
                     "brier_base": res_base["brier"], "brier_ua": res_ua["brier"],
                     "d_brier_pct": d_brier,
                     "auc_base": res_base["auc"], "auc_ua": res_ua["auc"],
                     "logloss_base": res_base["logloss"], "logloss_ua": res_ua["logloss"]})
        print(f"  [{friendly:22s}] base Brier={res_base['brier']:.4f}  +UA={res_ua['brier']:.4f}  "
              f"Δ={d_brier:+.2f}%  AUC {res_base['auc']:.3f}→{res_ua['auc']:.3f}  "
              f"cov={cov:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("No stations evaluated.")
    df = pd.DataFrame(rows)
    out = out_dir
    df.to_csv(out / "results.csv", index=False)

    # n_test-weighted aggregate.
    w = df["n_test"].to_numpy()
    agg_base = float((df["brier_base"] * w).sum() / w.sum())
    agg_ua = float((df["brier_ua"] * w).sum() / w.sum())
    print("\n=== AGGREGATE (n_test-weighted) ===", flush=True)
    print(f"  Brier base={agg_base:.4f}  +UA={agg_ua:.4f}  Δ={ (agg_ua-agg_base)/agg_base*100:+.2f}%", flush=True)
    print(f"  wrote {out/'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
