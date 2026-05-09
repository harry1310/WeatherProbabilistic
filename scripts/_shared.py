"""Shared helpers used by both production predict scripts and bake-off
scripts. Lives here so production code (predict_4a, predict_5a) doesn't
have to drag bake-off-only deps (lightgbm, pymc_bart, rpy2.dbarts) onto
its critical path.

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

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

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
    if station_input in STATION_NAME_BY_SLUG:
        return station_input, STATION_NAME_BY_SLUG[station_input]
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


def add_synoptic_features(station_friendly: str, lead_hours: int,
                          df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pull NWP-mean wind direction (encoded as sin/cos unit vector to avoid
    the 0°/360° circular-mean discontinuity) and NWP-mean surface pressure
    via DuckDB, then merge onto df by ValidTimeUtc.
    """
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in_clause = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
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
        WHERE LocationName = '{LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours = {lead_hours}
          AND Model IN {model_in_clause}
    )
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
