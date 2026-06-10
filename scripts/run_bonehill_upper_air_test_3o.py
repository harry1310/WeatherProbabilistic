"""Upper-air signal test — 3o variant. Same probe as
``run_bonehill_upper_air_test.py`` (3a/lean) but the base model is the 3o
recipe: rich-59 precip features + 9 v1 terrain features (= 3c-oro), the
deployed Bonehill occurrence champion. LightGBM P(wet) at lead 24h per gauge.

Compare test Brier / logloss / AUC: base (68 feat) vs base + 17 upper-air
pressure features (joined by valid-time from hist_forecast). Same CAVEAT as
the 3a test: hist_forecast pressure at valid-time is closer to analysis than
a true 24h-lead forecast — a positive result is an upper bound on deployable
benefit. Bake-off only, no production change.

Reuses the 3a test's load_upper_air / build_pruned_cache / evaluate so the
two reads are methodologically identical apart from the base feature set.
Reuses an existing pruned cache from a sibling 3a run if present (skips the
~10-min re-prune); else builds its own.

Usage:
  .venv/Scripts/python.exe -u scripts/run_bonehill_upper_air_test_3o.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("WB_LOCATION", "bonehill_rocks")

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _shared  # noqa: E402
from _shared import (  # noqa: E402
    RICH_FEATURE_NAMES,
    V1_TERRAIN_FEATURE_NAMES,
    build_rich_features_via_duckdb,
    compose_v1_terrain_block,
    resolve_station,
)
# Reuse the 3a test's machinery verbatim — identical methodology.
from run_bonehill_upper_air_test import (  # noqa: E402
    LEAD, LOCATION, MIN_VALID_TIME, PRESSURE_FEATURES, STATION_SLUGS,
    build_pruned_cache, evaluate, load_upper_air,
)

BASE_3O = list(RICH_FEATURE_NAMES) + list(V1_TERRAIN_FEATURE_NAMES)  # 68


def find_existing_cache() -> Path | None:
    """Reuse a sibling run's pruned cache if it has the Bonehill forecasts."""
    for d in sorted((ROOT / "reports").glob("bonehill_upper_air_test*/_pruned_cache")):
        if (d / "forecasts" / "location=bonehill_rocks" / "all.parquet").exists():
            return d
    return None


def main() -> None:
    print(f"[start] {datetime.now():%H:%M:%S}  Bonehill upper-air test — 3o base, lead {LEAD}h", flush=True)
    print(f"  base = 3o rich-59 + oro-9 ({len(BASE_3O)} feat) P(wet); "
          f"+UA adds {len(PRESSURE_FEATURES)} pressure features (join by valid-time)", flush=True)

    out_dir = ROOT / "reports" / f"bonehill_upper_air_test_3o_{datetime.now():%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = find_existing_cache()
    if cache_root is not None:
        print(f"  [cache] reusing existing pruned tree {cache_root}", flush=True)
    else:
        cache_root = out_dir / "_pruned_cache"
        print(f"  [cache] building pruned tree at {cache_root} …", flush=True)
        build_pruned_cache(_shared.WEATHERBLEND_DATA_ROOT, cache_root)
    _shared.WEATHERBLEND_DATA_ROOT = cache_root
    print("  [cache] builders repointed at pruned tree.", flush=True)

    ua = load_upper_air()
    if ua.empty:
        raise SystemExit("No hist_forecast pressure rows — sync from R2 first.")
    print(f"  upper-air rows: {len(ua):,}  valid_time {ua['ValidTimeUtc'].min()} -> "
          f"{ua['ValidTimeUtc'].max()}", flush=True)

    rows = []
    for idx, slug in enumerate(STATION_SLUGS):
        s_slug, friendly = resolve_station(slug)
        t0 = time.time()
        rich = build_rich_features_via_duckdb(
            friendly, LEAD, min_valid_time=MIN_VALID_TIME, run_time_source="offset_day")
        if rich.empty:
            print(f"  [skip] {friendly}: no rich rows", flush=True)
            continue
        richoro = compose_v1_terrain_block(
            s_slug, idx, LEAD, rich, min_valid_time=MIN_VALID_TIME, run_time_source="offset_day")
        richoro["ValidTimeUtc"] = pd.to_datetime(richoro["ValidTimeUtc"])
        merged = richoro.merge(ua.drop(columns=["n_models"]), on="ValidTimeUtc", how="left")
        cov = merged[PRESSURE_FEATURES[0]].notna().mean()

        res_base = evaluate(merged, BASE_3O)
        res_ua = evaluate(merged, BASE_3O + PRESSURE_FEATURES)
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
    df.to_csv(out_dir / "results.csv", index=False)
    w = df["n_test"].to_numpy()
    agg_base = float((df["brier_base"] * w).sum() / w.sum())
    agg_ua = float((df["brier_ua"] * w).sum() / w.sum())
    print("\n=== AGGREGATE (n_test-weighted) ===", flush=True)
    print(f"  Brier base={agg_base:.4f}  +UA={agg_ua:.4f}  Δ={(agg_ua-agg_base)/agg_base*100:+.2f}%", flush=True)
    print(f"  wrote {out_dir/'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
