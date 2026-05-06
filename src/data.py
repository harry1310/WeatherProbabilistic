"""Phase 1 + Phase 2 + Phase 3 data loaders.

Phase 1 loader: single station (Bellever Dartmoor), lead 24h.
Phase 2 loader: three stations at the same grid point, lead 24h.
Phase 3 loader: three stations × three leads (24/48/72h).

All reuse the WeatherBlend parquet tree at
`C:/Projects/Weather/WeatherBlend/data/`. We deliberately do not set up
a new ingestion pipeline.

Per-station chronological 80/20 split
-------------------------------------
The brief says "earliest 80% / latest 20%" - with multiple stations a
single global chronological cut could under-represent a station in the
test set if its data starts later. We split each station chronologically
then concatenate, which keeps station representation balanced in train
and test and makes per-station test metrics directly comparable.

For Phase 3, the split is on *unique valid times within each station*:
all three lead-times for a given (station, valid_time) go to the same
side of the split, so the test set is "what does each model predict for
the held-out future at each lead?" rather than mixing leads across the
boundary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Local default points at the WeatherBlend sibling checkout on the dev
# machine. CI / cron / non-Windows callers override via the
# WEATHERBLEND_DATA_ROOT environment variable — the predict-bayesian.yml
# workflow sets this to the runner-relative path it rclone-copies into.
WEATHERBLEND_DATA_ROOT = Path(
    os.environ.get("WEATHERBLEND_DATA_ROOT")
    or r"C:/Projects/Weather/WeatherBlend/data"
)

# These six models have ~2 years of backfill in the WeatherBlend tree.
# Other models (ecmwf_hres_wb2, gfs_ncep, met_office_spot) have only a
# handful of files and are excluded from Phase 1 / 2.
MODELS: tuple[str, ...] = (
    "ecmwf_ifs025",
    "gem_seamless",
    "gfs_seamless",
    "icon_seamless",
    "meteofrance_seamless",
    "ukmo_seamless",
)

# Phase 4 5-model variant — UKMO removed. UKMO has ~26% null in WeatherBlend's
# Previous Runs tree at every (lead, variable). Phase 1-3 silently row-dropped
# those, training on a non-random 74% subset. Phase 4 fits without UKMO so
# the comparison against WeatherBlend's pattern-1 LightGBM (which also drops
# UKMO entirely) is on identical inputs. See reports/phase4_audit.md.
MODELS_NO_UKMO: tuple[str, ...] = tuple(m for m in MODELS if m != "ukmo_seamless")

LOCATION = "bonehill_rocks"

# Station identifiers as they appear in the WeatherBlend `station=` parquet
# partitions, paired with short codes used in plots and tables.
STATIONS: tuple[tuple[str, str], ...] = (
    ("Bellever Dartmoor", "Bellever"),
    ("Princetown", "Princetown"),
    ("Dartmoor nr Hexworthy", "Hexworthy"),
)

LEAD_HOURS = 24
PHASE3_LEAD_HOURS: tuple[int, ...] = (24, 48, 72)
WET_THRESHOLD_MM = 0.1
MAX_NULL_FRACTION = 0.5  # drop any feature column that is more than half null


# ---------------------------------------------------------------------------
# Phase 1 dataset (single station)
# ---------------------------------------------------------------------------

@dataclass
class Phase1Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    valid_time_train: pd.Series
    valid_time_test: pd.Series
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Phase 2 dataset (three stations) and Phase 3 dataset (stations × leads)
# ---------------------------------------------------------------------------

@dataclass
class Phase3Dataset:
    """Multi-station, multi-lead dataset with pooled standardisation.

    One row per (valid_time, station, lead). Standardisation is fit on the
    *combined* training set across all (station, lead) cells — per-cell
    standardisation would leak information across the hierarchy and is
    inappropriate for a model whose hyperparameters are interpreted in
    standardised-feature units.
    """

    # Unstandardised features (handy for inspection)
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    # Standardised feature matrices (numpy float64) - what models actually use
    X_train_s: np.ndarray
    X_test_s: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    # Integer station code per row (0..n_stations-1)
    station_idx_train: np.ndarray
    station_idx_test: np.ndarray
    # Integer lead index per row (0..n_leads-1, ordered by lead_hours)
    lead_idx_train: np.ndarray
    lead_idx_test: np.ndarray
    valid_time_train: pd.Series
    valid_time_test: pd.Series
    feature_names: list[str]
    station_codes: list[str]
    station_full_names: list[str]
    lead_hours: list[int]
    scaler: StandardScaler


@dataclass
class Phase2Dataset:
    """Multi-station dataset with pooled standardisation already applied.

    `X_train_s` and `X_test_s` are standardised on the *combined* training
    set across all three stations (per-station standardisation would leak
    information and isn't appropriate for hierarchical fitting).
    """

    # Unstandardised features (handy for inspection / re-scaling)
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    # Standardised feature matrices (numpy float64) - what models actually use
    X_train_s: np.ndarray
    X_test_s: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    # Integer station code per row (0..n_stations-1, aligned with `station_codes`)
    station_idx_train: np.ndarray
    station_idx_test: np.ndarray
    valid_time_train: pd.Series
    valid_time_test: pd.Series
    feature_names: list[str]
    station_codes: list[str]  # short codes, e.g. ["Bellever", "Princetown", "Hexworthy"]
    station_full_names: list[str]
    scaler: StandardScaler


# ---------------------------------------------------------------------------
# Forecast / truth loaders
# ---------------------------------------------------------------------------

def _load_model_forecasts(model: str) -> pd.DataFrame:
    """Load all lead-24h forecasts for one model across the full archive."""
    model_dir = WEATHERBLEND_DATA_ROOT / "forecasts" / f"location={LOCATION}" / f"model={model}"
    files = sorted(model_dir.glob("date=*/previous_runs.parquet"))
    if not files:
        raise FileNotFoundError(f"No forecast files found for {model} under {model_dir}")

    frames = []
    for f in files:
        df = pd.read_parquet(
            f,
            columns=["ValidTimeUtc", "LeadHours", "Precipitation", "PrecipitationProbability"],
        )
        df = df.loc[df["LeadHours"] == LEAD_HOURS].drop(columns=["LeadHours"])
        if not df.empty:
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["ValidTimeUtc"], keep="last").sort_values("ValidTimeUtc")
    out = out.rename(
        columns={
            "Precipitation": f"precip_{model}",
            "PrecipitationProbability": f"prob_{model}",
        }
    )
    return out.reset_index(drop=True)


def _load_model_forecasts_multi_lead(model: str, leads: tuple[int, ...]) -> pd.DataFrame:
    """Load forecasts for one model at multiple leads.

    Returns a long-format DataFrame: one row per (valid_time, lead) with
    columns ValidTimeUtc, LeadHours, precip_<model>, prob_<model>.
    """
    model_dir = WEATHERBLEND_DATA_ROOT / "forecasts" / f"location={LOCATION}" / f"model={model}"
    files = sorted(model_dir.glob("date=*/previous_runs.parquet"))
    if not files:
        raise FileNotFoundError(f"No forecast files found for {model} under {model_dir}")

    leads_set = set(leads)
    frames = []
    for f in files:
        df = pd.read_parquet(
            f,
            columns=["ValidTimeUtc", "LeadHours", "Precipitation", "PrecipitationProbability"],
        )
        df = df.loc[df["LeadHours"].isin(leads_set)]
        if not df.empty:
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = (
        out.drop_duplicates(subset=["ValidTimeUtc", "LeadHours"], keep="last")
        .sort_values(["ValidTimeUtc", "LeadHours"])
    )
    out = out.rename(
        columns={
            "Precipitation": f"precip_{model}",
            "PrecipitationProbability": f"prob_{model}",
        }
    )
    return out.reset_index(drop=True)


def _load_rainfall_truth(station: str) -> pd.DataFrame:
    """Load all EA 15-min rainfall for a station and aggregate to hourly totals."""
    rain_dir = (
        WEATHERBLEND_DATA_ROOT
        / "truth"
        / "rainfall"
        / f"location={LOCATION}"
        / f"station={station}"
    )
    files = sorted(rain_dir.glob("date=*/rainfall.parquet"))
    if not files:
        raise FileNotFoundError(f"No rainfall truth files under {rain_dir}")

    frames = [
        pd.read_parquet(f, columns=["ObservedTimeUtc", "Value15MinMm", "Quality"]) for f in files
    ]
    raw = pd.concat(frames, ignore_index=True)

    raw = raw.loc[raw["Quality"] == "Good"].copy()
    raw["hour_start"] = raw["ObservedTimeUtc"].dt.floor("h")
    hourly = raw.groupby("hour_start").agg(
        mm=("Value15MinMm", "sum"),
        n_obs=("Value15MinMm", "size"),
    )
    hourly = hourly.loc[hourly["n_obs"] == 4].reset_index()
    hourly = hourly.rename(columns={"hour_start": "ValidTimeUtc"})
    hourly["observed_wet"] = (hourly["mm"] >= WET_THRESHOLD_MM).astype(int)
    return hourly[["ValidTimeUtc", "observed_wet", "mm"]]


def _load_all_forecasts(verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Load all six models' lead-24h forecasts and inner-join them.

    Also adds cyclical hour-of-day features. Returns the joined forecast
    frame plus the precip column name list.
    """
    fc_frames: list[pd.DataFrame] = []
    for model in MODELS:
        if verbose:
            print(f"Loading forecasts: {model}...")
        fc = _load_model_forecasts(model)
        fc_frames.append(fc)
    forecasts = fc_frames[0]
    for fc in fc_frames[1:]:
        forecasts = forecasts.merge(fc, on="ValidTimeUtc", how="inner")

    hours = forecasts["ValidTimeUtc"].dt.hour
    forecasts["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    forecasts["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    precip_cols = [f"precip_{m}" for m in MODELS]
    return forecasts, precip_cols


def _load_model_forecasts_multi_lead_rich(model: str, leads: tuple[int, ...]) -> pd.DataFrame:
    """Phase 4 (a) supporting context: load precip + prob + meteo covariates per model.

    Mirrors WeatherBlend's PrecipFeatureBuilder column set so we can train a
    27-feature LightGBM in Python on the same Phase3Dataset rows, byte-identical
    with the Bayesian/stripped comparison. Returns long-format with
    ValidTimeUtc + LeadHours + precip_<m> + prob_<m> + 7 covariate columns
    (rh, t, td, cloud_low, cloud_mid, cloud_high, cape, wind), each suffixed `_<model>`.
    """
    model_dir = WEATHERBLEND_DATA_ROOT / "forecasts" / f"location={LOCATION}" / f"model={model}"
    files = sorted(model_dir.glob("date=*/previous_runs.parquet"))
    if not files:
        raise FileNotFoundError(f"No forecast files found for {model} under {model_dir}")
    leads_set = set(leads)
    cols = [
        "ValidTimeUtc", "LeadHours",
        "Precipitation", "PrecipitationProbability",
        "RelativeHumidity2m", "Temperature2m", "DewPoint2m",
        "CloudCoverLow", "CloudCoverMid", "CloudCoverHigh",
        "Cape", "WindSpeed10m",
    ]
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        df = df.loc[df["LeadHours"].isin(leads_set)]
        if not df.empty:
            frames.append(df)
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["ValidTimeUtc", "LeadHours"], keep="last")
        .sort_values(["ValidTimeUtc", "LeadHours"])
    )
    out = out.rename(columns={
        "Precipitation": f"precip_{model}",
        "PrecipitationProbability": f"prob_{model}",
        "RelativeHumidity2m": f"rh_{model}",
        "Temperature2m": f"t_{model}",
        "DewPoint2m": f"td_{model}",
        "CloudCoverLow": f"cloud_low_{model}",
        "CloudCoverMid": f"cloud_mid_{model}",
        "CloudCoverHigh": f"cloud_high_{model}",
        "Cape": f"cape_{model}",
        "WindSpeed10m": f"wind_{model}",
    })
    return out.reset_index(drop=True)


def _load_all_forecasts_multi_lead_rich(
    leads: tuple[int, ...],
    verbose: bool = True,
    models: tuple[str, ...] = MODELS,
) -> tuple[pd.DataFrame, list[str]]:
    """Inner-join all configured models' rich forecasts. Returns (df, precip_cols)
    matching `_load_all_forecasts_multi_lead`'s shape but with all rich columns
    plus the WeatherBlend-style derived features (ensemble spread + covariate means).
    """
    fc_frames = []
    for model in models:
        if verbose:
            print(f"Loading rich forecasts: {model}...")
        fc = _load_model_forecasts_multi_lead_rich(model, leads)
        fc_frames.append(fc)
    forecasts = fc_frames[0]
    for fc in fc_frames[1:]:
        forecasts = forecasts.merge(fc, on=["ValidTimeUtc", "LeadHours"], how="inner")

    # Cyclical calendar — hour AND day-of-year (WeatherBlend's 27-feature set has both)
    hours = forecasts["ValidTimeUtc"].dt.hour
    forecasts["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    forecasts["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    doy = forecasts["ValidTimeUtc"].dt.dayofyear
    forecasts["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.0)
    forecasts["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.0)

    # Ensemble spread on precip — mean / std / max / agreement_wet01.
    # Skip-NaN aware so the 5-model variant works the same way.
    precip_cols = [f"precip_{m}" for m in models]
    pmat = forecasts[precip_cols].to_numpy(dtype="float64")
    forecasts["precip_mean"] = np.nanmean(pmat, axis=1)
    forecasts["precip_std"]  = np.nanstd(pmat, axis=1)
    forecasts["precip_max"]  = np.nanmax(pmat, axis=1)
    wet = (pmat >= WET_THRESHOLD_MM).astype("float64")
    present = (~np.isnan(pmat)).astype("float64")
    forecasts["precip_agreement_wet01"] = np.where(
        present.sum(axis=1) > 0,
        wet.sum(axis=1) / present.sum(axis=1),
        np.nan,
    )

    # Ensemble-mean covariates — match WeatherBlend's AVG(...) logic.
    # NB: with inner-join + 5 models, all covariate columns should be non-null.
    for short, src in [("rh", "rh"), ("cloud_low", "cloud_low"),
                       ("cloud_mid", "cloud_mid"), ("cloud_high", "cloud_high"),
                       ("cape", "cape"), ("wind_speed", "wind")]:
        cols = [f"{src}_{m}" for m in models]
        forecasts[f"{short}_mean"] = np.nanmean(forecasts[cols].to_numpy(dtype="float64"), axis=1)
    # Dew depression mean = mean across models of (T - Td)
    dewdep_cols = []
    for m in models:
        forecasts[f"_dewdep_{m}"] = forecasts[f"t_{m}"] - forecasts[f"td_{m}"]
        dewdep_cols.append(f"_dewdep_{m}")
    forecasts["dew_depression_mean"] = np.nanmean(forecasts[dewdep_cols].to_numpy(dtype="float64"), axis=1)
    forecasts = forecasts.drop(columns=dewdep_cols)

    return forecasts, precip_cols


def _load_all_forecasts_multi_lead(
    leads: tuple[int, ...],
    verbose: bool = True,
    models: tuple[str, ...] = MODELS,
) -> tuple[pd.DataFrame, list[str]]:
    """Load all configured models' forecasts at multiple leads, inner-joined.

    Returns a long-format DataFrame (one row per (valid_time, lead)) with
    one precip and (where present) one prob column per model plus cyclical
    hour features. The inner join on (ValidTimeUtc, LeadHours) keeps only
    cells where every configured model has a value at that lead. Phase 4's
    5-model variant passes ``MODELS_NO_UKMO`` to drop UKMO from the join,
    so the inner join no longer requires UKMO presence.
    """
    fc_frames: list[pd.DataFrame] = []
    for model in models:
        if verbose:
            print(f"Loading forecasts: {model}...")
        fc = _load_model_forecasts_multi_lead(model, leads)
        fc_frames.append(fc)
    forecasts = fc_frames[0]
    for fc in fc_frames[1:]:
        forecasts = forecasts.merge(fc, on=["ValidTimeUtc", "LeadHours"], how="inner")

    hours = forecasts["ValidTimeUtc"].dt.hour
    forecasts["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    forecasts["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    precip_cols = [f"precip_{m}" for m in models]
    return forecasts, precip_cols


def _select_features(
    df: pd.DataFrame,
    precip_cols: list[str],
    verbose: bool = True,
    models: tuple[str, ...] = MODELS,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop rows with any null precip, decide which prob_* cols to keep, fill residual nulls.

    `models` controls which model suffixes are checked for prob_* columns. For Phase 4's
    5-model variant, pass ``MODELS_NO_UKMO`` so prob_ukmo isn't even examined (it's
    100% null anyway).
    """
    before = len(df)
    df = df.dropna(subset=precip_cols).copy()
    if verbose:
        print(f"  drop rows with any null precip_*: {before:,} -> {len(df):,}")

    prob_cols = [f"prob_{m}" for m in models if f"prob_{m}" in df.columns]
    kept_prob_cols: list[str] = []
    for c in prob_cols:
        null_frac = df[c].isna().mean()
        if null_frac <= MAX_NULL_FRACTION:
            kept_prob_cols.append(c)
        if verbose:
            status = "keep" if c in kept_prob_cols else "drop"
            print(f"  {c}: null_frac={null_frac:.3f}  [{status}]")
    for c in kept_prob_cols:
        df[c] = df[c].fillna(df[c].mean())

    feature_names = precip_cols + kept_prob_cols + ["hour_sin", "hour_cos"]
    return df, feature_names


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def prepare_phase1_dataset(test_fraction: float = 0.2, verbose: bool = True) -> Phase1Dataset:
    """Single-station (Bellever) loader, kept identical to Phase 1."""
    if verbose:
        print("Loading rainfall truth (Bellever Dartmoor)...")
    truth = _load_rainfall_truth("Bellever Dartmoor")
    if verbose:
        print(f"  truth rows: {len(truth):,}  wet fraction: {truth['observed_wet'].mean():.3f}")

    forecasts, precip_cols = _load_all_forecasts(verbose=verbose)
    df = truth[["ValidTimeUtc", "observed_wet"]].merge(forecasts, on="ValidTimeUtc", how="inner")
    df, feature_names = _select_features(df, precip_cols, verbose=verbose)
    df = df.sort_values("ValidTimeUtc").reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_fraction))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    if verbose:
        print(
            f"Final rows: {len(df):,}  features: {len(feature_names)}  "
            f"train: {len(train):,} ({train['ValidTimeUtc'].min()} -> {train['ValidTimeUtc'].max()})  "
            f"test: {len(test):,} ({test['ValidTimeUtc'].min()} -> {test['ValidTimeUtc'].max()})"
        )

    return Phase1Dataset(
        X_train=train[feature_names].reset_index(drop=True),
        X_test=test[feature_names].reset_index(drop=True),
        y_train=train["observed_wet"].reset_index(drop=True),
        y_test=test["observed_wet"].reset_index(drop=True),
        valid_time_train=train["ValidTimeUtc"].reset_index(drop=True),
        valid_time_test=test["ValidTimeUtc"].reset_index(drop=True),
        feature_names=feature_names,
    )


def prepare_phase2_dataset(test_fraction: float = 0.2, verbose: bool = True) -> Phase2Dataset:
    """Three-station loader for Phase 2.

    Builds one row per (valid_time, station). Splits each station's data
    chronologically then concatenates, so station representation in train/test
    is balanced. Standardises features on the *combined* training set.
    """
    forecasts, precip_cols = _load_all_forecasts(verbose=verbose)

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    station_codes = [code for _, code in STATIONS]
    station_full_names = [full for full, _ in STATIONS]

    for code_idx, (full_name, code) in enumerate(STATIONS):
        if verbose:
            print(f"\nLoading rainfall truth: {full_name}")
        truth = _load_rainfall_truth(full_name)
        if verbose:
            print(f"  truth rows: {len(truth):,}  wet fraction: {truth['observed_wet'].mean():.3f}")

        df = truth[["ValidTimeUtc", "observed_wet"]].merge(forecasts, on="ValidTimeUtc", how="inner")
        df, feature_names = _select_features(df, precip_cols, verbose=verbose)
        df = df.sort_values("ValidTimeUtc").reset_index(drop=True)
        df["station_idx"] = code_idx
        df["station"] = code

        split_idx = int(len(df) * (1 - test_fraction))
        train_frames.append(df.iloc[:split_idx])
        test_frames.append(df.iloc[split_idx:])

        if verbose:
            tr, te = df.iloc[:split_idx], df.iloc[split_idx:]
            print(
                f"  {code}: train {len(tr):,} ({tr['ValidTimeUtc'].min()} -> {tr['ValidTimeUtc'].max()})"
                f"  test {len(te):,} ({te['ValidTimeUtc'].min()} -> {te['ValidTimeUtc'].max()})"
                f"  wet train={tr['observed_wet'].mean():.3f} test={te['observed_wet'].mean():.3f}"
            )

    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)

    # `feature_names` was set in the last loop iteration; identical across stations
    # because the same forecast frame and selection logic is used.
    X_train = train[feature_names]
    X_test = test[feature_names]

    # Pooled standardisation on the combined training set.
    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values).astype("float64")
    X_test_s = scaler.transform(X_test.values).astype("float64")

    if verbose:
        print(f"\nCombined: train {len(train):,}  test {len(test):,}  features {len(feature_names)}")
        for code_idx, code in enumerate(station_codes):
            n_tr = int((train["station_idx"] == code_idx).sum())
            n_te = int((test["station_idx"] == code_idx).sum())
            print(f"  {code}: train={n_tr:,}  test={n_te:,}")

    return Phase2Dataset(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        X_train_s=X_train_s,
        X_test_s=X_test_s,
        y_train=train["observed_wet"].reset_index(drop=True),
        y_test=test["observed_wet"].reset_index(drop=True),
        station_idx_train=train["station_idx"].to_numpy(dtype="int64"),
        station_idx_test=test["station_idx"].to_numpy(dtype="int64"),
        valid_time_train=train["ValidTimeUtc"].reset_index(drop=True),
        valid_time_test=test["ValidTimeUtc"].reset_index(drop=True),
        feature_names=feature_names,
        station_codes=station_codes,
        station_full_names=station_full_names,
        scaler=scaler,
    )


def prepare_phase3_dataset(
    leads: tuple[int, ...] = PHASE3_LEAD_HOURS,
    test_fraction: float = 0.2,
    verbose: bool = True,
    models: tuple[str, ...] = MODELS,
) -> Phase3Dataset:
    """Three-station × multi-lead loader for Phase 3.

    Builds one row per (valid_time, station, lead). The chronological
    train/test cut is per-station on **unique valid times**: all three
    leads for a given (station, valid_time) go to the same side of the
    split, so a single valid_time isn't half in train and half in test
    via different leads.

    Phase 4 passes ``models=MODELS_NO_UKMO`` to fit a 5-model variant with
    UKMO removed entirely from the join (so we no longer row-drop on UKMO
    null). Default is the original 6-model behaviour.
    """
    forecasts, precip_cols = _load_all_forecasts_multi_lead(leads, verbose=verbose, models=models)

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    station_codes = [code for _, code in STATIONS]
    station_full_names = [full for full, _ in STATIONS]
    lead_to_idx = {l: i for i, l in enumerate(leads)}

    feature_names: list[str] | None = None

    for s_idx, (full_name, code) in enumerate(STATIONS):
        if verbose:
            print(f"\nLoading rainfall truth: {full_name}")
        truth = _load_rainfall_truth(full_name)
        if verbose:
            print(f"  truth rows: {len(truth):,}  wet fraction: {truth['observed_wet'].mean():.3f}")

        df = truth[["ValidTimeUtc", "observed_wet"]].merge(forecasts, on="ValidTimeUtc", how="inner")
        df, fn = _select_features(df, precip_cols, verbose=verbose, models=models)
        feature_names = fn  # identical across stations
        df = df.sort_values(["ValidTimeUtc", "LeadHours"]).reset_index(drop=True)
        df["station_idx"] = s_idx
        df["station"] = code
        df["lead_idx"] = df["LeadHours"].map(lead_to_idx).astype("int64")

        # Per-station chronological split on unique valid_times so all
        # leads for a given valid_time go to the same side.
        unique_vt = np.sort(df["ValidTimeUtc"].unique())
        if len(unique_vt) == 0:
            raise RuntimeError(f"No rows after join for station {code} at leads {leads}")
        cut_pos = int(len(unique_vt) * (1 - test_fraction))
        cut_vt = unique_vt[cut_pos]
        train_mask = df["ValidTimeUtc"] < cut_vt
        train_frames.append(df[train_mask].copy())
        test_frames.append(df[~train_mask].copy())

        if verbose:
            tr, te = df[train_mask], df[~train_mask]
            print(
                f"  {code}: train {len(tr):,}  test {len(te):,}  "
                f"({len(unique_vt):,} unique valid_times, cut at {pd.Timestamp(cut_vt)})"
            )
            for li, lh in enumerate(leads):
                trl = tr[tr["lead_idx"] == li]
                tel = te[te["lead_idx"] == li]
                wt_tr = trl["observed_wet"].mean() if len(trl) else float("nan")
                wt_te = tel["observed_wet"].mean() if len(tel) else float("nan")
                print(
                    f"    lead {lh}h: train={len(trl):,} test={len(tel):,}  "
                    f"wet train={wt_tr:.3f} test={wt_te:.3f}"
                )

    assert feature_names is not None  # guaranteed by the loop above

    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)

    X_train = train[feature_names]
    X_test = test[feature_names]

    # Pooled standardisation on the combined training set (across all cells).
    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values).astype("float64")
    X_test_s = scaler.transform(X_test.values).astype("float64")

    if verbose:
        print(
            f"\nCombined: train {len(train):,}  test {len(test):,}  "
            f"features {len(feature_names)}  cells {len(station_codes)}×{len(leads)}"
        )
        for s_idx, code in enumerate(station_codes):
            for l_idx, lh in enumerate(leads):
                n_tr = int(((train["station_idx"] == s_idx) & (train["lead_idx"] == l_idx)).sum())
                n_te = int(((test["station_idx"] == s_idx) & (test["lead_idx"] == l_idx)).sum())
                print(f"  {code} @ {lh}h: train={n_tr:,}  test={n_te:,}")

    return Phase3Dataset(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        X_train_s=X_train_s,
        X_test_s=X_test_s,
        y_train=train["observed_wet"].reset_index(drop=True),
        y_test=test["observed_wet"].reset_index(drop=True),
        station_idx_train=train["station_idx"].to_numpy(dtype="int64"),
        station_idx_test=test["station_idx"].to_numpy(dtype="int64"),
        lead_idx_train=train["lead_idx"].to_numpy(dtype="int64"),
        lead_idx_test=test["lead_idx"].to_numpy(dtype="int64"),
        valid_time_train=train["ValidTimeUtc"].reset_index(drop=True),
        valid_time_test=test["ValidTimeUtc"].reset_index(drop=True),
        feature_names=feature_names,
        station_codes=station_codes,
        station_full_names=station_full_names,
        lead_hours=list(leads),
        scaler=scaler,
    )


if __name__ == "__main__":
    ds = prepare_phase3_dataset()
    print("\nFeature sample (train head):")
    print(ds.X_train.head().to_string())
