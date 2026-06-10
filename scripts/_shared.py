"""Shared helpers used by both production predict scripts and bake-off
scripts. Lives here so production code (predict_4a) doesn't have to drag
bake-off-only deps (lightgbm, pymc_bart, rpy2.dbarts) onto its critical
path.

Strict policy: imports in this module MUST stay light — pure stdlib +
duckdb / numpy / pandas / src.data. No lightgbm, no pymc, no pymc_bart,
no rpy2. If you find yourself wanting to add one of those here, it
belongs in a bake-off-specific module instead.

Contains:
  * MODELS_LEAN, FEATURE_NAMES — the 7-NWP / 22-feature spec the precip
    blender is trained on (mirrors PrecipFeatureBuilder.cs).
  * OUTPUT_ROOT, STATION_NAME_BY_SLUG — paths + station registry used
    across phase 6 artefact dirs.
  * resolve_station, build_features_via_duckdb, time_split — the
    feature-build pipeline shared by production + bake-off paths.
  * add_synoptic_features — synoptic flow features used by 4a + the
    rich-feats bake-off.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import LOCATION, WEATHERBLEND_DATA_ROOT, WET_THRESHOLD_MM  # noqa: E402
from src.weatherblend_config import STATION_NAME_BY_SLUG  # noqa: E402, F401

# WB_LOCATION env var is the per-matrix-job location selector that
# retrain-python.yml sets. Phase B (2026-05-12) made train_4a aware
# of it for stations + metadata stamping, but build_features_via_duckdb
# still queried the legacy module-level LOCATION constant — silently
# producing 0 rows for any Membury matrix job. Resolved at module load
# (not query time) since every script invocation gets a fresh process
# with WB_LOCATION already set by the matrix step's env block.
ACTIVE_LOCATION = os.environ.get("WB_LOCATION", LOCATION)

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

# STATION_NAME_BY_SLUG is now imported from src.weatherblend_config
# (derived from WeatherBlend's config.yaml, the single source of truth
# for station slug ↔ friendly name across both repos). Until 2026-05-20
# this lived as a hand-curated dict here — Princetown was retired from
# WeatherBlend's rainfall config 2026-05-04 but its row sat here until
# the loader rewrite caught the drift, and the 3 Membury stations
# added 2026-05-11 never landed here at all. The import above replaces
# this block; the resolve_station helper below uses the imported dict
# unchanged.


def resolve_station(station_input: str) -> tuple[str, str]:
    """Accept either the slug ('ea_bellever_dartmoor') or the friendly
    name ('Bellever Dartmoor') from the user. Returns (slug, friendly)."""
    if station_input in STATION_NAME_BY_SLUG:
        return station_input, STATION_NAME_BY_SLUG[station_input]
    for slug, friendly in STATION_NAME_BY_SLUG.items():
        if station_input.lower() == friendly.lower():
            return slug, friendly
    raise ValueError(
        f"Unknown station '{station_input}'. Known: "
        + ", ".join(f"{s}  ({n})" for s, n in STATION_NAME_BY_SLUG.items())
    )


def lead_day_bucket(valid_time, anchor) -> int:
    """Trained-lead bucket (24 / 48 / 72 / 96 / 120 …) that ``valid_time``
    falls into, for the day-bucketed hourly predict convention — bucket L
    covers the whole calendar day ``anchor_date + L/24 days``. Mirrors
    WeatherBlend's ``PrecipPredictCommand.BuildHourlyTargets``.

    Returns ``days_after_anchor * 24``; the caller keeps only the buckets in
    its trained ``LEADS`` set, so a same-day valid time resolves to 0 and a
    far-horizon one past the longest trained lead is dropped. Accepts
    anything ``pandas.Timestamp`` can parse for either argument.
    """
    v = pd.Timestamp(valid_time).normalize()
    a = pd.Timestamp(anchor).normalize()
    return (v - a).days * 24


def build_features_via_duckdb(
    station_friendly: str,
    lead_hours: int,
    min_valid_time=None,
) -> pd.DataFrame:
    """Mirror of WeatherBlend's PrecipFeatureBuilder.BuildForLead (C#) using
    DuckDB on the same parquet trees. Returns a DataFrame with the 22
    features in FEATURE_NAMES order plus the binary truth label.

    SQL is line-for-line equivalent to the C# pivot at
    src/WeatherBlend/Train/PrecipFeatureBuilder.cs:126-181 (with model
    names hardcoded for the lean 7-NWP set). Truth is the EA gauge with
    the strict 4-of-4 partial-hour rule the C# version uses (groups of
    15-min readings collapse to hourly only when all four are present).

    ``min_valid_time`` (optional ``datetime``) is the training-data
    cutoff (2026-05-26 — see ``src.phase_registry``). When set, both
    SQL CTEs (hourly_truth + latest) clip on the same time boundary so
    the inner-join post-clip cannot silently drop rows past the cutoff.
    Default ``None`` = no cutoff (back-compat for the bake-off scripts).
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

    # Optional minValidTime cutoff. ISO format keeps the DuckDB literal
    # unambiguous regardless of locale; applied to BOTH CTEs (truth +
    # forecasts) so the join doesn't silently drop one side's rows.
    if min_valid_time is not None:
        ts = min_valid_time.strftime("%Y-%m-%d %H:%M:%S")
        observed_filter = f"      AND ObservedTimeUtc >= TIMESTAMP '{ts}'\n"
        valid_filter    = f"      AND ValidTimeUtc >= TIMESTAMP '{ts}'\n"
    else:
        observed_filter = ""
        valid_filter = ""

    sql = f"""
    WITH hourly_truth AS (
        SELECT
            date_trunc('hour', ObservedTimeUtc) AS valid_time,
            SUM(Value15MinMm) AS precip_mm_hour
        FROM read_parquet('{rn_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{ACTIVE_LOCATION}'
          AND StationName  = '{station_friendly}'
          AND Value15MinMm IS NOT NULL
{observed_filter}        GROUP BY 1
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
        WHERE LocationName = '{ACTIVE_LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours = {lead_hours}
          AND Model IN {model_in_clause}
{valid_filter}    ),
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


# =============================================================================
# Rich-oro spec — Python port of WeatherBlend's
# PrecipRichFeatureBuilder.cs + PrecipRichOroFeatureBuilder.cs.
#
# Two-builder design mirrors the C# split:
#   build_rich_features_via_duckdb  — 59 rich features (5 per-NWP × 8 NWPs +
#                                     mean/spread + cross-aggregates + calendar
#                                     + 4 EA persistence)
#   compose_v1_terrain_block        — 9 v1 terrain features (3 static +
#                                     5 wind-rotated dynamic + 1 station_id)
#
# Bit-equivalence with the C# builders is asserted by
# tests/test_rich_oro_python_vs_csharp.py — DO NOT diverge without updating
# both implementations + that test.
# =============================================================================

# 8-NWP rich model list — adds UKMO to MODELS_LEAN's 7. Order matches the C#
# TempFeatureBuilder.CanonicalModelOrder used by PrecipRichFeatureBuilder.BuildSpec.
MODELS_RICH = [
    ("gfs_seamless",         "gfs"),
    ("ecmwf_ifs025",         "ecmwf"),
    ("icon_seamless",        "icon"),
    ("meteofrance_seamless", "mf"),
    ("ukmo_seamless",        "ukmo"),
    ("gem_seamless",         "gem"),
    ("ecmwf_aifs025_single", "aifs"),
    ("jma_seamless",         "jma"),
]

# 59 rich feature names. Order MUST match PrecipRichFeatureBuilder.BuildSpec's
# FeatureNames list exactly — downstream BART preprocess.json + bundle metadata
# keys on this order, so a reorder silently breaks predict-time scaling.
RICH_FEATURE_NAMES = (
    [f"precip_{s}" for _, s in MODELS_RICH]
    + ["precip_mean", "precip_std", "precip_max", "precip_agreement_wet_01"]
    + ["rh_mean", "dew_depression_mean",
       "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean",
       "cape_mean", "wind_speed_mean"]
    + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    + [f"dew_{s}" for _, s in MODELS_RICH]
    + [f"rh_{s}" for _, s in MODELS_RICH]
    + [f"dew_depression_{s}" for _, s in MODELS_RICH]
    + [f"pressure_{s}" for _, s in MODELS_RICH]
    + ["ea_rain_prev_24h_mm", "ea_rain_prev_72h_mm",
       "ea_wet_hours_last_24h", "ea_dry_hours_trailing"]
)
assert len(RICH_FEATURE_NAMES) == 59

# 9 v1 terrain feature names — matches PrecipRichOroFeatureBuilder.TerrainFeatureNames.
V1_TERRAIN_FEATURE_NAMES = [
    "oro_elevation_vs_cell_m",
    "oro_relief_5km_m",
    "oro_ruggedness_5km_m",
    "oro_wind_sin",
    "oro_wind_cos",
    "oro_upwind_gain_per_wind_5km_m",
    "oro_uplift_m_per_s",
    "oro_uplift_x_q_g_per_kg",
    "oro_station_id",
]

RICH_ORO_FEATURE_NAMES = RICH_FEATURE_NAMES + V1_TERRAIN_FEATURE_NAMES
assert len(RICH_ORO_FEATURE_NAMES) == 68

# 8-sector lookup used by the v1 terrain block's wind-direction-picked
# upwind_gain feature. Matches OroStaticFeatures.Sectors.
_ORO_SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _load_hourly_rain(station_friendly: str, min_valid_time=None) -> dict:
    """Mirror PrecipRichFeatureBuilder.LoadHourlyRain — hourly EA rainfall (mm)
    keyed by hour-of-observation UTC, with the strict 4-of-4 partial-hour rule
    (a row is emitted only when all four 15-min observations are present)."""
    rn_glob = str((WEATHERBLEND_DATA_ROOT / "truth" / "rainfall" / "**" / "*.parquet")).replace("\\", "/")
    filt = (f"  AND ObservedTimeUtc >= TIMESTAMP '{min_valid_time:%Y-%m-%d %H:%M:%S}'\n"
            if min_valid_time is not None else "")
    sql = f"""
        SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
               SUM(Value15MinMm) AS mm
        FROM read_parquet('{rn_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{ACTIVE_LOCATION}'
          AND StationName  = '{station_friendly}'
          AND Value15MinMm IS NOT NULL
        {filt}GROUP BY 1
        HAVING COUNT(*) = 4
    """
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()
    return {pd.Timestamp(v, tz=None): float(m) for v, m in zip(df["valid_time"], df["mm"])}


def _compute_persistence(hourly_rain: dict, run_time):
    """Port of PrecipRichFeatureBuilder.ComputePersistence — 4 trailing-rainfall
    persistence values anchored at run_time = valid_time - lead. Returns
    (prev24, prev72, wet24, dry_run); NaN where coverage is partial."""
    run_time = pd.Timestamp(run_time)
    sum24 = 0.0
    sum72 = 0.0
    wet24 = 0
    cover24 = True
    cover72 = True
    for h in range(72):
        t = run_time - pd.Timedelta(hours=h)
        mm = hourly_rain.get(t)
        if mm is None:
            if h < 24: cover24 = False
            cover72 = False
            continue
        sum72 += mm
        if h < 24:
            sum24 += mm
            if mm >= WET_THRESHOLD_MM:
                wet24 += 1
    dry_run = 0
    for h in range(72):
        t = run_time - pd.Timedelta(hours=h)
        mm = hourly_rain.get(t)
        if mm is None: break
        if mm > WET_THRESHOLD_MM: break
        dry_run += 1
    return (
        sum24 if cover24 else float("nan"),
        sum72 if cover72 else float("nan"),
        float(wet24) if cover24 else float("nan"),
        float(dry_run),
    )


def build_rich_features_via_duckdb(
    station_friendly: str,
    lead_hours: int,
    *,
    min_valid_time=None,
    run_time_source: str = "offset_day",
) -> pd.DataFrame:
    """Python equivalent of WeatherBlend's PrecipRichOroFeatureBuilder.BuildForLead's
    SQL-pivot stage (the 59 rich features WITHOUT the 9 terrain block — that's
    compose_v1_terrain_block's job). Mirrors PrecipRichFeatureBuilder.BuildForLead.

    Returns a DataFrame with ValidTimeUtc + the 59 features in
    RICH_FEATURE_NAMES order + 'precip_mm_hour' (truth) + 'wet' (binary label).

    Caveat: row drop policy mirrors the C# 'lenient with at-least-one' rule
    for precip (target/featureSet='precipitation/rich' has no required models).
    Rows are dropped only when the inner-join against hourly truth has no
    matching valid_time. Per-NWP NaNs are preserved (BART/LightGBM tolerate).

    `run_time_source` switches between training data ('offset_day' = Open-Meteo
    Previous Runs) and live-cycle predict data ('reported' = freshest cycle).
    """
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in = "(" + ",".join(f"'{full}'" for full, _ in MODELS_RICH) + ")"
    n = len(MODELS_RICH)

    valid_filter = (f"          AND ValidTimeUtc >= TIMESTAMP '{min_valid_time:%Y-%m-%d %H:%M:%S}'\n"
                    if min_valid_time is not None else "")

    # Per-NWP pivots: 5 columns per NWP (precip, dew, rh, dewdep, pressure).
    per_nwp_pivots_lines = []
    for full, short in MODELS_RICH:
        per_nwp_pivots_lines.append(f"        MAX(CASE WHEN Model = '{full}' THEN Precipitation END) AS precip_{short},")
    for full, short in MODELS_RICH:
        per_nwp_pivots_lines.append(f"        MAX(CASE WHEN Model = '{full}' THEN DewPoint2m END) AS dew_{short},")
    for full, short in MODELS_RICH:
        per_nwp_pivots_lines.append(f"        MAX(CASE WHEN Model = '{full}' THEN RelativeHumidity2m END) AS rh_{short},")
    for full, short in MODELS_RICH:
        per_nwp_pivots_lines.append(f"        MAX(CASE WHEN Model = '{full}' THEN Temperature2m - DewPoint2m END) AS dew_depression_{short},")
    for full, short in MODELS_RICH:
        per_nwp_pivots_lines.append(f"        MAX(CASE WHEN Model = '{full}' THEN SurfacePressure END) AS pressure_{short},")
    per_nwp_pivots = "\n".join(per_nwp_pivots_lines)

    sql = f"""
        WITH latest AS (
            SELECT ValidTimeUtc, Model,
                   Precipitation, RelativeHumidity2m, Temperature2m, DewPoint2m,
                   CloudCoverLow, CloudCoverMid, CloudCoverHigh,
                   Cape, WindSpeed10m, SurfacePressure,
                   ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model
                                      ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
            WHERE LocationName = '{ACTIVE_LOCATION}'
              AND RunTimeSource = '{run_time_source}'
              AND LeadHours = {lead_hours}
              AND Model IN {model_in}
{valid_filter}        )
        SELECT ValidTimeUtc,
{per_nwp_pivots}
            AVG(RelativeHumidity2m)         AS rh_mean,
            AVG(Temperature2m - DewPoint2m) AS dew_depression_mean,
            AVG(CloudCoverLow)  AS cloud_low_mean,
            AVG(CloudCoverMid)  AS cloud_mid_mean,
            AVG(CloudCoverHigh) AS cloud_high_mean,
            AVG(Cape)           AS cape_mean,
            AVG(WindSpeed10m)   AS wind_speed_mean
        FROM latest WHERE rn = 1
        GROUP BY ValidTimeUtc
        ORDER BY ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()

    # Drop empty result early so downstream calls return cleanly.
    if len(df) == 0:
        return df

    # Drop rows where ALL precip models are NULL — mirrors C# BuildForLead's
    # `anyNotNull` filter on `(precip_a IS NOT NULL OR precip_b IS NOT NULL ...)`.
    # Without this we silently retain valid-time rows with no usable precip
    # signal (only ~2% of total but they trip np.nanmax warnings and break
    # bit-equivalence vs the C# dump).
    precip_cols = [f"precip_{s}" for _, s in MODELS_RICH]
    pm = df[precip_cols].to_numpy(dtype="float64")
    any_present = (~np.isnan(pm)).any(axis=1)
    df = df[any_present].copy()
    pm = pm[any_present]

    # Inner-join against EA hourly truth. Mirrors C# BuildForLead's
    # "if (!hourlyRain.TryGetValue(valid, out var truth)) continue;" drop.
    hourly = _load_hourly_rain(station_friendly, min_valid_time=min_valid_time)
    df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"])
    truth_series = df["ValidTimeUtc"].map(lambda v: hourly.get(pd.Timestamp(v)))
    keep = truth_series.notna()
    df = df[keep].copy()
    df["precip_mm_hour"] = truth_series[keep].values
    pm = pm[keep.to_numpy()]

    # Cross-NWP precip aggregates (same NaN-safe logic as the lean builder).
    present = (~np.isnan(pm)).sum(axis=1)
    sumv = np.nansum(pm, axis=1)
    sumsq = np.nansum(pm ** 2, axis=1)
    mean_safe = np.where(present > 0, sumv / np.maximum(present, 1), np.nan)
    var = np.maximum(0.0, sumsq / np.maximum(present, 1) - mean_safe ** 2)
    df["precip_mean"] = np.where(present > 0, mean_safe, np.nan)
    df["precip_std"]  = np.where(present > 1, np.sqrt(var), 0.0)
    df["precip_max"]  = np.where(present > 0, np.nanmax(pm, axis=1), np.nan)
    wet_count = (pm >= WET_THRESHOLD_MM).sum(axis=1)
    df["precip_agreement_wet_01"] = np.where(present > 0, wet_count / np.maximum(present, 1), np.nan)

    # Cyclical calendar encodings — exact same denominator + offset as C#.
    hour_angle = 2.0 * np.pi * df["ValidTimeUtc"].dt.hour / 24.0
    doy_angle  = 2.0 * np.pi * (df["ValidTimeUtc"].dt.dayofyear - 1) / 365.0
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["doy_sin"]  = np.sin(doy_angle)
    df["doy_cos"]  = np.cos(doy_angle)

    # 4 EA persistence features per row, anchored at run_time = valid_time - lead.
    run_times = df["ValidTimeUtc"] - pd.Timedelta(hours=lead_hours)
    persistence = [_compute_persistence(hourly, rt) for rt in run_times]
    df["ea_rain_prev_24h_mm"]    = [p[0] for p in persistence]
    df["ea_rain_prev_72h_mm"]    = [p[1] for p in persistence]
    df["ea_wet_hours_last_24h"]  = [p[2] for p in persistence]
    df["ea_dry_hours_trailing"]  = [p[3] for p in persistence]

    df["wet"] = (df["precip_mm_hour"] >= WET_THRESHOLD_MM).astype("int8")
    return df.reset_index(drop=True)


def _load_aux_nwp_means(lead_hours: int, *, min_valid_time=None,
                        run_time_source: str = "offset_day") -> pd.DataFrame:
    """Port of PrecipRichOroFeatureBuilder.LoadAuxNwpMeans — NWP-ensemble-mean
    covariates per valid_time, used to compose the dynamic terrain features.

    Returns wind_sin, wind_cos, wind_speed, temp_c, dew_c, pres_hpa, all
    NaN-safe (DuckDB AVG with NaNs in inputs gracefully ignores them per row,
    but if all inputs for a column are NaN the AVG returns NULL → NaN).
    """
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in = "(" + ",".join(f"'{full}'" for full, _ in MODELS_RICH) + ")"
    valid_filter = (f"          AND ValidTimeUtc >= TIMESTAMP '{min_valid_time:%Y-%m-%d %H:%M:%S}'\n"
                    if min_valid_time is not None else "")
    sql = f"""
        WITH latest AS (
            SELECT ValidTimeUtc, Model,
                   WindDirection10m, WindSpeed10m, Temperature2m, DewPoint2m, SurfacePressure,
                   ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model
                                      ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
            WHERE LocationName = '{ACTIVE_LOCATION}'
              AND RunTimeSource = '{run_time_source}'
              AND LeadHours = {lead_hours}
              AND Model IN {model_in}
{valid_filter}        )
        SELECT ValidTimeUtc,
               AVG(SIN(WindDirection10m * pi() / 180.0)) AS wind_sin,
               AVG(COS(WindDirection10m * pi() / 180.0)) AS wind_cos,
               AVG(WindSpeed10m)    AS wind_speed,
               AVG(Temperature2m)   AS temp_c,
               AVG(DewPoint2m)      AS dew_c,
               AVG(SurfacePressure) AS pres_hpa
        FROM latest WHERE rn = 1
        GROUP BY ValidTimeUtc
        ORDER BY ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    aux = con.execute(sql).fetch_df()
    con.close()
    aux["ValidTimeUtc"] = pd.to_datetime(aux["ValidTimeUtc"])
    return aux


def _load_oro_static(station_slug: str) -> dict:
    """Read data/static/orographic/{slug}.json. Returns the raw dict; callers
    pull the fields they need with .get(key, default)."""
    import json
    path = WEATHERBLEND_DATA_ROOT / "static" / "orographic" / f"{station_slug}.json"
    return json.loads(path.read_text())


def _upwind_gain_at(oro_rec: dict, wind_sin: float, wind_cos: float) -> float:
    """Mirror OroStaticFeatures.UpwindGainAt — pick the 1 sector from the 8
    stored upwind_gain_5km values matching this row's NWP-mean wind direction.
    Returns 0.0 if wind direction is NaN ('no orographic context')."""
    if np.isnan(wind_sin) or np.isnan(wind_cos):
        return 0.0
    deg = (np.degrees(np.arctan2(wind_sin, wind_cos)) + 360.0) % 360.0
    idx = int(((deg + 22.5) // 45.0)) % 8
    sector = _ORO_SECTORS[idx]
    return float(oro_rec.get("upwind_gain_5km", {}).get(sector, 0.0))


def compose_v1_terrain_block(
    station_slug: str,
    station_index: int,
    lead_hours: int,
    rich_df: pd.DataFrame,
    *,
    min_valid_time=None,
    run_time_source: str = "offset_day",
) -> pd.DataFrame:
    """Append the 9 v1 terrain features to a rich-features DataFrame. Returns
    a NEW DataFrame with V1_TERRAIN_FEATURE_NAMES added; rows where the aux
    NWP-mean pull has no matching valid_time are dropped (mirrors the C#
    BuildForLead's `if (!aux.TryGetValue...) continue;` drop).

    Magnus formula for specific humidity and the wind-vector uplift calc
    mirror PrecipRichOroFeatureBuilder.ComposeTerrainBlock exactly.
    """
    oro = _load_oro_static(station_slug)
    aux = _load_aux_nwp_means(lead_hours, min_valid_time=min_valid_time,
                              run_time_source=run_time_source)
    df = rich_df.merge(aux, on="ValidTimeUtc", how="inner")
    if len(df) == 0:
        return df

    elev_vs_cell = float(oro.get("elevation_vs_cell_m", 0.0))
    relief_5km   = float(oro.get("relief_5km_m", 0.0))
    rugged_5km   = float(oro.get("terrain_ruggedness_5km_m", 0.0))
    grad_dx      = float(oro.get("terrain_gradient_dx", 0.0))
    grad_dy      = float(oro.get("terrain_gradient_dy", 0.0))

    ws  = df["wind_speed"].to_numpy(dtype="float64")
    wsn = df["wind_sin"].to_numpy(dtype="float64")
    wcs = df["wind_cos"].to_numpy(dtype="float64")
    td  = df["dew_c"].to_numpy(dtype="float64")
    p   = df["pres_hpa"].to_numpy(dtype="float64")

    # Wind-vector uplift = max(0, u·dx + v·dy). u = -ws·sin, v = -ws·cos
    # (convention: WindDirection10m is direction wind is FROM).
    valid_wind = ~(np.isnan(ws) | np.isnan(wsn) | np.isnan(wcs))
    u_east  = np.where(valid_wind, -ws * wsn, 0.0)
    v_north = np.where(valid_wind, -ws * wcs, 0.0)
    w = u_east * grad_dx + v_north * grad_dy
    uplift = np.where(valid_wind, np.maximum(0.0, w), 0.0)

    # Magnus saturation vapour pressure → specific humidity (g/kg).
    valid_q = ~(np.isnan(td) | np.isnan(p)) & (p > 0)
    e_hpa = 6.112 * np.exp(17.62 * td / (td + 243.12))
    q_kgkg = 0.622 * e_hpa / (p - 0.378 * e_hpa)
    q_gkg = np.where(valid_q, np.maximum(0.0, q_kgkg * 1000.0), 0.0)

    # Upwind gain per row — sector picked by NWP-mean wind direction.
    upwind_gain = np.array([_upwind_gain_at(oro, s, c) for s, c in zip(wsn, wcs)],
                           dtype="float64")

    df["oro_elevation_vs_cell_m"]        = elev_vs_cell
    df["oro_relief_5km_m"]               = relief_5km
    df["oro_ruggedness_5km_m"]           = rugged_5km
    df["oro_wind_sin"]                   = np.where(np.isnan(wsn), 0.0, wsn)
    df["oro_wind_cos"]                   = np.where(np.isnan(wcs), 0.0, wcs)
    df["oro_upwind_gain_per_wind_5km_m"] = upwind_gain
    df["oro_uplift_m_per_s"]             = uplift
    df["oro_uplift_x_q_g_per_kg"]        = uplift * q_gkg
    df["oro_station_id"]                 = float(station_index)

    return df.reset_index(drop=True)


def build_rich_oro_features_live(
    station_friendly: str,
    station_slug: str,
    station_index: int,
    anchor,
    horizon_days: int,
    leads: list,
) -> pd.DataFrame:
    """Predict-time builder for the 68-feature rich+oro spec. Produces the
    SAME column shape as the training-time pair
    (build_rich_features_via_duckdb + compose_v1_terrain_block) but pulls
    from the live forecast tree:

      * RunTimeSource='reported' (live cycle, not the offset_day backfill)
      * No `LeadHours = N` filter — live cycle picks freshest run regardless
        of resulting lead. Mirrors PrecipRichOroFeatureBuilder.LoadAuxNwpMeansLive
        which dropped the lead filter for the same reason (see its XML doc
        for the C# rationale).
      * Time-window filter `(anchor, anchor + horizon_days]` on ValidTimeUtc.
      * Per-row `lead` column derived from valid_time vs anchor via
        lead_day_bucket — matches predict_4a.py's existing
        day-bucketed convention, so the per-cell predictor can filter
        df['lead']==N for each cell.
      * EA persistence features computed per-row anchored at
        run_time = valid_time - lead (lead being the per-row bucket).

    Caller still has to inner-join against the bundle's expected feature
    list via predict_one_cell's existing scaler path.
    """
    anchor = pd.Timestamp(anchor)
    horizon_end = anchor + pd.Timedelta(days=horizon_days)

    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in = "(" + ",".join(f"'{full}'" for full, _ in MODELS_RICH) + ")"
    n = len(MODELS_RICH)

    per_nwp_pivots_lines = []
    for full, short in MODELS_RICH:
        per_nwp_pivots_lines.append(f"        MAX(CASE WHEN Model = '{full}' THEN Precipitation END) AS precip_{short},")
    for full, short in MODELS_RICH:
        per_nwp_pivots_lines.append(f"        MAX(CASE WHEN Model = '{full}' THEN DewPoint2m END) AS dew_{short},")
    for full, short in MODELS_RICH:
        per_nwp_pivots_lines.append(f"        MAX(CASE WHEN Model = '{full}' THEN RelativeHumidity2m END) AS rh_{short},")
    for full, short in MODELS_RICH:
        per_nwp_pivots_lines.append(f"        MAX(CASE WHEN Model = '{full}' THEN Temperature2m - DewPoint2m END) AS dew_depression_{short},")
    for full, short in MODELS_RICH:
        per_nwp_pivots_lines.append(f"        MAX(CASE WHEN Model = '{full}' THEN SurfacePressure END) AS pressure_{short},")
    per_nwp_pivots = "\n".join(per_nwp_pivots_lines)

    # Rich pivot + aux NWP means in one CTE chain so we share the freshest-cycle
    # row-numbering pass. anchor + horizon define the live window.
    sql = f"""
        WITH latest AS (
            SELECT ValidTimeUtc, Model,
                   Precipitation, RelativeHumidity2m, Temperature2m, DewPoint2m,
                   CloudCoverLow, CloudCoverMid, CloudCoverHigh,
                   Cape, WindSpeed10m, WindDirection10m, SurfacePressure,
                   ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model
                                      ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
            WHERE LocationName = '{ACTIVE_LOCATION}'
              AND (RunTimeSource IS NULL OR RunTimeSource <> 'offset_day')
              AND Model IN {model_in}
              AND ValidTimeUtc >  TIMESTAMP '{anchor:%Y-%m-%d %H:%M:%S}'
              AND ValidTimeUtc <= TIMESTAMP '{horizon_end:%Y-%m-%d %H:%M:%S}'
        )
        SELECT ValidTimeUtc,
{per_nwp_pivots}
            AVG(RelativeHumidity2m)         AS rh_mean,
            AVG(Temperature2m - DewPoint2m) AS dew_depression_mean,
            AVG(CloudCoverLow)  AS cloud_low_mean,
            AVG(CloudCoverMid)  AS cloud_mid_mean,
            AVG(CloudCoverHigh) AS cloud_high_mean,
            AVG(Cape)           AS cape_mean,
            AVG(WindSpeed10m)   AS wind_speed_mean,
            -- aux for terrain block — same window-shaped query so one pass suffices
            AVG(SIN(WindDirection10m * pi() / 180.0)) AS _aux_wind_sin,
            AVG(COS(WindDirection10m * pi() / 180.0)) AS _aux_wind_cos,
            AVG(WindSpeed10m)                          AS _aux_wind_speed,
            AVG(Temperature2m)                         AS _aux_temp_c,
            AVG(DewPoint2m)                            AS _aux_dew_c,
            AVG(SurfacePressure)                       AS _aux_pres_hpa
        FROM latest WHERE rn = 1
        GROUP BY ValidTimeUtc
        ORDER BY ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()
    if len(df) == 0:
        return df
    df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"])

    # Drop rows with no usable precip (mirrors anyNotNull filter from training).
    precip_cols = [f"precip_{s}" for _, s in MODELS_RICH]
    pm = df[precip_cols].to_numpy(dtype="float64")
    any_present = (~np.isnan(pm)).any(axis=1)
    df = df[any_present].copy()
    pm = pm[any_present]

    # Cross-NWP precip aggregates.
    present = (~np.isnan(pm)).sum(axis=1)
    sumv = np.nansum(pm, axis=1)
    sumsq = np.nansum(pm ** 2, axis=1)
    mean_safe = np.where(present > 0, sumv / np.maximum(present, 1), np.nan)
    var = np.maximum(0.0, sumsq / np.maximum(present, 1) - mean_safe ** 2)
    df["precip_mean"] = np.where(present > 0, mean_safe, np.nan)
    df["precip_std"]  = np.where(present > 1, np.sqrt(var), 0.0)
    df["precip_max"]  = np.where(present > 0, np.nanmax(pm, axis=1), np.nan)
    wet_count = (pm >= WET_THRESHOLD_MM).sum(axis=1)
    df["precip_agreement_wet_01"] = np.where(present > 0, wet_count / np.maximum(present, 1), np.nan)

    # Calendar.
    hour_angle = 2.0 * np.pi * df["ValidTimeUtc"].dt.hour / 24.0
    doy_angle  = 2.0 * np.pi * (df["ValidTimeUtc"].dt.dayofyear - 1) / 365.0
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["doy_sin"]  = np.sin(doy_angle)
    df["doy_cos"]  = np.cos(doy_angle)

    # Per-row trained-lead bucket (mirrors predict_4a.build_live_features).
    # Same-day rows bucket to 0 — relabel them as the lead-12 cell (today's
    # remaining hours; 2026-06-10, the per-lead policy plan's short-lead
    # extension). No lead-12 model exists — the caller routes the 12-bucket
    # to the m24 state, which the 4a cross-lead study found to be the best
    # candidate at 12-18h on both SELECT and SCORE slices. Whether the rows
    # survive is still the caller's `leads` filter, so 3f/other callers that
    # don't pass 12 are unchanged.
    df["lead"] = df["ValidTimeUtc"].apply(lambda v: lead_day_bucket(v, anchor))
    df.loc[df["lead"] == 0, "lead"] = 12
    df = df[df["lead"].isin(leads)].reset_index(drop=True)
    df["lead"] = df["lead"].astype(int)

    # EA persistence per row, anchored at run_time = valid_time - per-row lead.
    hourly = _load_hourly_rain(station_friendly, min_valid_time=None)
    run_times = df["ValidTimeUtc"] - pd.to_timedelta(df["lead"], unit="h")
    persistence = [_compute_persistence(hourly, rt) for rt in run_times]
    df["ea_rain_prev_24h_mm"]   = [p[0] for p in persistence]
    df["ea_rain_prev_72h_mm"]   = [p[1] for p in persistence]
    df["ea_wet_hours_last_24h"] = [p[2] for p in persistence]
    df["ea_dry_hours_trailing"] = [p[3] for p in persistence]

    # ---- v1 terrain block from the pre-fetched aux columns ----
    oro = _load_oro_static(station_slug)
    elev_vs_cell = float(oro.get("elevation_vs_cell_m", 0.0))
    relief_5km   = float(oro.get("relief_5km_m", 0.0))
    rugged_5km   = float(oro.get("terrain_ruggedness_5km_m", 0.0))
    grad_dx      = float(oro.get("terrain_gradient_dx", 0.0))
    grad_dy      = float(oro.get("terrain_gradient_dy", 0.0))

    ws  = df["_aux_wind_speed"].to_numpy(dtype="float64")
    wsn = df["_aux_wind_sin"].to_numpy(dtype="float64")
    wcs = df["_aux_wind_cos"].to_numpy(dtype="float64")
    td  = df["_aux_dew_c"].to_numpy(dtype="float64")
    p   = df["_aux_pres_hpa"].to_numpy(dtype="float64")

    valid_wind = ~(np.isnan(ws) | np.isnan(wsn) | np.isnan(wcs))
    u_east  = np.where(valid_wind, -ws * wsn, 0.0)
    v_north = np.where(valid_wind, -ws * wcs, 0.0)
    w = u_east * grad_dx + v_north * grad_dy
    uplift = np.where(valid_wind, np.maximum(0.0, w), 0.0)

    valid_q = ~(np.isnan(td) | np.isnan(p)) & (p > 0)
    e_hpa = 6.112 * np.exp(17.62 * td / (td + 243.12))
    q_kgkg = 0.622 * e_hpa / (p - 0.378 * e_hpa)
    q_gkg = np.where(valid_q, np.maximum(0.0, q_kgkg * 1000.0), 0.0)

    upwind_gain = np.array([_upwind_gain_at(oro, s, c) for s, c in zip(wsn, wcs)],
                           dtype="float64")

    df["oro_elevation_vs_cell_m"]        = elev_vs_cell
    df["oro_relief_5km_m"]               = relief_5km
    df["oro_ruggedness_5km_m"]           = rugged_5km
    df["oro_wind_sin"]                   = np.where(np.isnan(wsn), 0.0, wsn)
    df["oro_wind_cos"]                   = np.where(np.isnan(wcs), 0.0, wcs)
    df["oro_upwind_gain_per_wind_5km_m"] = upwind_gain
    df["oro_uplift_m_per_s"]             = uplift
    df["oro_uplift_x_q_g_per_kg"]        = uplift * q_gkg
    df["oro_station_id"]                 = float(station_index)

    # Drop the temporary _aux_* columns now that the terrain block is packed.
    df = df.drop(columns=[c for c in df.columns if c.startswith("_aux_")])
    return df.reset_index(drop=True)


def add_synoptic_features(station_friendly: str, lead_hours: int,
                          df: pd.DataFrame,
                          min_valid_time=None) -> tuple[pd.DataFrame, list[str]]:
    """Pull NWP-mean wind direction (encoded as sin/cos unit vector to avoid
    the 0°/360° circular-mean discontinuity) and NWP-mean surface pressure
    via DuckDB, then merge onto df by ValidTimeUtc.

    ``min_valid_time`` mirrors ``build_features_via_duckdb`` — clipped at
    the same boundary so the synoptic SQL doesn't pull rows that the
    later merge would discard anyway. Default ``None`` = no cutoff.
    """
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in_clause = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
    valid_filter = (
        f"          AND ValidTimeUtc >= TIMESTAMP '{min_valid_time.strftime('%Y-%m-%d %H:%M:%S')}'\n"
        if min_valid_time is not None else ""
    )
    sql = f"""
    WITH latest AS (
        SELECT
            ValidTimeUtc, Model,
            WindDirection10m, SurfacePressure,
            ROW_NUMBER() OVER (
                PARTITION BY ValidTimeUtc, Model
                ORDER BY RunTimeUtc DESC
            ) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{ACTIVE_LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours = {lead_hours}
          AND Model IN {model_in_clause}
{valid_filter}    )
    SELECT
        ValidTimeUtc,
        AVG(SIN(RADIANS(WindDirection10m))) AS wind_dir_sin_mean,
        AVG(COS(RADIANS(WindDirection10m))) AS wind_dir_cos_mean,
        AVG(SurfacePressure)                AS surface_pressure_mean
    FROM latest
    WHERE rn = 1
    GROUP BY ValidTimeUtc
    ORDER BY ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    syn = con.execute(sql).fetch_df()
    con.close()
    df = df.merge(syn, on="ValidTimeUtc", how="left")
    return df, ["wind_dir_sin_mean", "wind_dir_cos_mean", "surface_pressure_mean"]
