"""Phase 4a upper-air A/B (exploratory, 2026-06-02).

The same honest lead-resolved multi-level pressure block tested on 3a/3c/3o
(WeatherBlend `precip-ua-bakeoff`), now on 4a — per-cell dbarts BART on the
rich+oro (68) feature set.

Why 4a is the interesting case: unlike 3o, 4a is NOT pooled — one BART per
(station, lead). The UA pressure block is a *location-level* point forecast
(identical across stations for a given valid-time), so in pooled 3o it could
only shift the overall P(wet) level, not separate stations, which diluted its
apparent value to -0.7%. In a per-cell model UA gets the full per-valid signal
to itself within each cell. This tests whether that recovers the value pooling
hid (3o), or whether the oro terrain block is genuinely redundant with UA.

Baseline = rich+oro (68, RICH_ORO_FEATURE_NAMES — the exact production 4a spec).
+UA = rich+oro + per-exact-model pressure block (UpperAirModels x UaPressureCols)
+ ensemble t850_mean/rh850_mean, attached by a leak-free backward ASOF
(merge_asof direction='backward'): for each offset_day valid-time V, the freshest
EXACT lead-L pressure row with valid_time <= V (issued >= L h before V — in hand
at the L-lead decision point). Mirrors PrecipFeatureBuilder's exact_ua CTE.

Reuses train_4a's `_prepare_cell` + `_fit_and_store` verbatim, so the base arm is
bit-identical to production 4a. Persists nothing. Reports per-(station, lead)
test Brier base vs +UA + delta, plus an aggregate.

Usage:
  .venv/Scripts/python.exe scripts/run_4a_upper_air_test.py --leads 24
  .venv/Scripts/python.exe scripts/run_4a_upper_air_test.py --leads 24 48 72 --stations "Bellever Dartmoor"
"""
from __future__ import annotations

import argparse
import time

import duckdb
import numpy as np
import pandas as pd

# Importing train_4a sets up R_HOME / dbarts / sys.path and exposes the
# per-cell prep + BART fit helpers, the production LEADS/STATIONS, and the
# rich+oro feature builders. Its main() is __main__-guarded, so import is safe.
import train_4a as t4
from train_4a import (
    ACTIVE_LOCATION,
    LEADS as PROD_LEADS,
    STATIONS,
    WEATHERBLEND_DATA_ROOT,
    _fit_and_store,
    _prepare_cell,
    brier,
    build_rich_features_via_duckdb,
    compose_v1_terrain_block,
    resolve_station,
    stations_for_location,
)

# ---------------------------------------------------------------------------
# Upper-air block definition — MUST mirror WeatherBlend
# PrecipFeatureBuilder.{UpperAirModels, UaPressureCols} so this A/B measures
# the same feature set the .NET 3a/3c/3o tests did. met_office_global has no
# pressure backfill, so it is not a UA model.
# ---------------------------------------------------------------------------
UPPER_AIR_MODELS = [
    ("gfs_ncep", "gfs"),
    ("ecmwf_ifs_oper", "ifs"),
    ("ecmwf_aifs_oper", "aifs"),
    ("gefs_ncep_mean", "gefsm"),
]
UA_PRESSURE_COLS = [
    ("Temperature850hPa", "t850"),
    ("Temperature700hPa", "t700"),
    ("Temperature500hPa", "t500"),
    ("GeopotentialHeight850hPa", "gh850"),
    ("GeopotentialHeight500hPa", "gh500"),
    ("WindSpeed850hPa", "ws850"),
    ("WindSpeed500hPa", "ws500"),
    ("WindDirection850hPa", "wd850"),
    ("WindDirection500hPa", "wd500"),
    ("RelativeHumidity850hPa", "rh850"),
]


def _ua_pivot_for_lead(lead: int, min_valid_time) -> tuple[pd.DataFrame, list[str]]:
    """DuckDB pivot of the freshest EXACT lead-L pressure row per valid_time,
    one column per (model, pressure-col). Returns (ua_df sorted by
    valid_time_ua, ua_feature_names). UA is location-level, so this is
    computed ONCE per lead and reused across stations."""
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in = "(" + ",".join(f"'{m}'" for m, _ in UPPER_AIR_MODELS) + ")"
    inner_cols = ", ".join(col for col, _ in UA_PRESSURE_COLS)
    pivots = ",\n           ".join(
        f"MAX(CASE WHEN Model = '{m}' THEN {col} END) AS {sh}_{msh}"
        for m, msh in UPPER_AIR_MODELS
        for col, sh in UA_PRESSURE_COLS
    )
    # No min_valid_time filter on the UA side: the backward ASOF needs the
    # freshest exact row <= V even when that row predates the cutoff, and the
    # base df is already clipped, so the merge result is clipped anyway.
    sql = f"""
    WITH exact_ua AS (
        SELECT valid_time_ua,
               {pivots}
        FROM (
            SELECT ValidTimeUtc AS valid_time_ua, Model, {inner_cols},
                   ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model
                                      ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
            WHERE LocationName = '{ACTIVE_LOCATION}'
              AND RunTimeSource = 'exact'
              AND LeadHours = {lead}
              AND Model IN {model_in}
        )
        WHERE rn = 1
        GROUP BY valid_time_ua
    )
    SELECT * FROM exact_ua ORDER BY valid_time_ua
    """
    con = duckdb.connect(":memory:")
    ua = con.execute(sql).fetch_df()
    con.close()

    ua["valid_time_ua"] = pd.to_datetime(ua["valid_time_ua"])
    per_model_cols = [f"{sh}_{msh}" for _, msh in UPPER_AIR_MODELS for _, sh in UA_PRESSURE_COLS]
    # Ensemble means across the (available) models — mirrors C# t850_mean /
    # rh850_mean (NaN-skipping). AIFS carries no rh850, so rh850_mean is the
    # mean of the 3 models that do.
    t850_cols = [f"t850_{msh}" for _, msh in UPPER_AIR_MODELS]
    rh850_cols = [f"rh850_{msh}" for _, msh in UPPER_AIR_MODELS]
    ua["t850_mean"] = ua[t850_cols].mean(axis=1, skipna=True)
    ua["rh850_mean"] = ua[rh850_cols].mean(axis=1, skipna=True)
    ua_feature_names = per_model_cols + ["t850_mean", "rh850_mean"]
    return ua, ua_feature_names


def _attach_ua(df: pd.DataFrame, ua: pd.DataFrame) -> pd.DataFrame:
    """Leak-free backward ASOF attach: each df row at valid V gets the freshest
    UA row with valid_time_ua <= V (LEFT — pre-pressure rows get NaN, BART
    tolerates). The Python equivalent of the .NET `ASOF LEFT JOIN exact_ua x
    ON p.ValidTimeUtc >= x.valid_time_ua`."""
    df = df.copy()
    df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"])
    df = df.sort_values("ValidTimeUtc").reset_index(drop=True)
    merged = pd.merge_asof(
        df, ua,
        left_on="ValidTimeUtc", right_on="valid_time_ua",
        direction="backward",
    )
    return merged.drop(columns=["valid_time_ua"])


def _score(df: pd.DataFrame, feature_list: list[str], lead: int) -> tuple[float, float, int]:
    """Prep + BART-fit one cell, return (test Brier, BSS, n_test)."""
    cell = _prepare_cell(df, feature_list)
    fit = _fit_and_store(cell, lead)
    b = brier(fit["p_test"], fit["y_test"].astype(np.float64))
    clim_b = brier(
        np.full_like(fit["y_test"], cell["train_df"]["wet"].mean(), dtype="float64"),
        fit["y_test"].astype(np.float64),
    )
    bss = (clim_b - b) / clim_b if clim_b > 0 else float("nan")
    # Free the R-side fit so peak RAM stays bounded across the loop.
    t4.ro.r(f'if (exists("fit_lead_{lead}h")) rm(fit_lead_{lead}h)')
    t4.ro.r("gc()")
    return b, bss, int(len(fit["y_test"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--leads", nargs="*", type=int, default=[24],
                   help="Leads to test (default: 24 — where the 3a/3c/3o UA signal lived).")
    p.add_argument("--stations", nargs="*", default=None,
                   help="Station subset (friendly or slug). Default: all active 4a stations.")
    args = p.parse_args()

    stations = args.stations or STATIONS
    leads = [L for L in args.leads if L in PROD_LEADS] or args.leads
    all_slugs = list(stations_for_location(ACTIVE_LOCATION))

    from src.phase_registry import min_valid_time_for
    mvt = min_valid_time_for("precipitation", "4a")

    print(f"=== 4a (per-cell BART, rich+oro) upper-air A/B ===")
    print(f"  location: {ACTIVE_LOCATION}   minValidTime: {mvt.date().isoformat() if mvt else 'none'}")
    print(f"  leads:    {leads}")
    print(f"  stations: {stations}")
    print(f"  fits:     {len(stations)} stations x {len(leads)} leads x 2 arms = "
          f"{len(stations) * len(leads) * 2} BART fits", flush=True)

    # rows: (station, lead, base_brier, ua_brier, base_bss, ua_bss, n)
    results: list[tuple] = []

    for lead in leads:
        print(f"\n[{time.strftime('%H:%M:%S')}] lead {lead}h — building UA pivot (once, location-level)", flush=True)
        ua, ua_names = _ua_pivot_for_lead(lead, mvt)
        print(f"    UA rows: {len(ua):,}  ({ua['valid_time_ua'].min()} -> {ua['valid_time_ua'].max()})  "
              f"+{len(ua_names)} UA feats", flush=True)

        for station_input in stations:
            station_slug, station_friendly = resolve_station(station_input)
            try:
                station_index = all_slugs.index(station_slug)
            except ValueError:
                print(f"    !! {station_slug} not in stations_for_location — skipping", flush=True)
                continue

            print(f"  [{time.strftime('%H:%M:%S')}] {station_friendly} lead {lead}h — building rich+oro base", flush=True)
            df_rich = build_rich_features_via_duckdb(station_friendly, lead, min_valid_time=mvt)
            df_base = compose_v1_terrain_block(station_slug, station_index, lead, df_rich, min_valid_time=mvt)

            b_base, bss_base, n = _score(df_base, t4.RICH_ORO_FEATURE_NAMES, lead)
            print(f"      [base] feats={len(t4.RICH_ORO_FEATURE_NAMES)}  test={n}  Brier={b_base:.4f}  BSS={bss_base:+.3f}", flush=True)

            df_ua = _attach_ua(df_base, ua)
            b_ua, bss_ua, n_ua = _score(df_ua, t4.RICH_ORO_FEATURE_NAMES + ua_names, lead)
            d = b_ua - b_base
            pct = d / b_base * 100.0 if b_base > 0 else 0.0
            print(f"      [+UA ] feats={len(t4.RICH_ORO_FEATURE_NAMES) + len(ua_names)}  test={n_ua}  "
                  f"Brier={b_ua:.4f}  BSS={bss_ua:+.3f}   Δ {d:+.4f} ({pct:+.1f}%)", flush=True)

            results.append((station_slug, lead, b_base, b_ua, bss_base, bss_ua, n))

    # Summary table + aggregate.
    print("\n=== SUMMARY: 4a upper-air A/B (Δ negative = UA better) ===")
    print(f"{'station':<28} {'lead':>4} {'base':>8} {'+UA':>8} {'Δ':>9} {'Δ%':>7}  {'n':>5}")
    for slug, lead, b_base, b_ua, _, _, n in results:
        d = b_ua - b_base
        pct = d / b_base * 100.0 if b_base > 0 else 0.0
        print(f"{slug:<28} {lead:>4} {b_base:>8.4f} {b_ua:>8.4f} {d:>+9.4f} {pct:>+6.1f}%  {n:>5}")

    for lead in leads:
        cells = [r for r in results if r[1] == lead]
        if not cells:
            continue
        # Sample-weighted aggregate Brier across stations at this lead.
        tot = sum(r[6] for r in cells)
        agg_base = sum(r[2] * r[6] for r in cells) / tot
        agg_ua = sum(r[3] * r[6] for r in cells) / tot
        d = agg_ua - agg_base
        pct = d / agg_base * 100.0 if agg_base > 0 else 0.0
        verdict = "UA BETTER" if d < 0 else "UA WORSE" if d > 0 else "flat"
        print(f"{'AGGREGATE lead ' + str(lead) + 'h':<28} {lead:>4} {agg_base:>8.4f} {agg_ua:>8.4f} "
              f"{d:>+9.4f} {pct:>+6.1f}%  {tot:>5}   -> {verdict}")


if __name__ == "__main__":
    main()
