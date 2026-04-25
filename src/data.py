"""Phase 1 + Phase 2 data loaders.

Phase 1 loader: single station (Bellever Dartmoor), lead 24h.
Phase 2 loader: three stations (Bellever, Princetown, Dartmoor nr
Hexworthy) at the same forecast grid point, lead 24h.

Both reuse the WeatherBlend parquet tree at
`C:/Projects/Weather/WeatherBlend/data/`. We deliberately do not set up
a new ingestion pipeline.

Per-station chronological 80/20 split for Phase 2
-------------------------------------------------
The brief says "earliest 80% / latest 20%" - with three stations a
single global chronological cut could under-represent a station in the
test set if its data starts later (which doesn't happen here, but the
principle holds). We split each station chronologically then concatenate,
which keeps station representation balanced in train and test and makes
per-station test metrics directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


WEATHERBLEND_DATA_ROOT = Path(r"C:/Projects/Weather/WeatherBlend/data")

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

LOCATION = "bonehill_rocks"

# Station identifiers as they appear in the WeatherBlend `station=` parquet
# partitions, paired with short codes used in plots and tables.
STATIONS: tuple[tuple[str, str], ...] = (
    ("Bellever Dartmoor", "Bellever"),
    ("Princetown", "Princetown"),
    ("Dartmoor nr Hexworthy", "Hexworthy"),
)

LEAD_HOURS = 24
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
# Phase 2 dataset (three stations)
# ---------------------------------------------------------------------------

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


def _select_features(df: pd.DataFrame, precip_cols: list[str], verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Drop rows with any null precip, decide which prob_* cols to keep, fill residual nulls."""
    before = len(df)
    df = df.dropna(subset=precip_cols).copy()
    if verbose:
        print(f"  drop rows with any null precip_*: {before:,} -> {len(df):,}")

    prob_cols = [f"prob_{m}" for m in MODELS]
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


if __name__ == "__main__":
    ds = prepare_phase2_dataset()
    print("\nFeature sample (train head):")
    print(ds.X_train.head().to_string())
