"""Phase 1 data loader.

Assembles a single pandas DataFrame for Bellever Dartmoor at lead 24h:

- One row per ValidTimeUtc where all six NWP models have a 24h forecast
  and EA rainfall truth is available.
- Target `observed_wet`: 1 if the observed Bellever hourly rainfall
  (aggregated from 15-min EA readings) is >= 0.1 mm, else 0.
- Features: per-model Precipitation (mm/h) and PrecipitationProbability
  (%) at lead 24h, plus cyclical hour-of-day encoding.

Data source is the existing WeatherBlend parquet tree. We deliberately
reuse it rather than setting up a new ingestion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


WEATHERBLEND_DATA_ROOT = Path(r"C:/Projects/Weather/WeatherBlend/data")

# These six models have ~2 years of backfill in the WeatherBlend tree.
# Other models (ecmwf_hres_wb2, gfs_ncep, met_office_spot) have only a
# handful of files and are excluded from Phase 1.
MODELS: tuple[str, ...] = (
    "ecmwf_ifs025",
    "gem_seamless",
    "gfs_seamless",
    "icon_seamless",
    "meteofrance_seamless",
    "ukmo_seamless",
)

LOCATION = "bonehill_rocks"
STATION = "Bellever Dartmoor"
LEAD_HOURS = 24
WET_THRESHOLD_MM = 0.1
MAX_NULL_FRACTION = 0.5  # drop any feature column that is more than half null


@dataclass
class Phase1Dataset:
    """Container for the prepared train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    valid_time_train: pd.Series
    valid_time_test: pd.Series
    feature_names: list[str]


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
    # Occasional duplicates can appear across adjacent date partitions;
    # keep the last occurrence (assumed freshest write).
    out = out.drop_duplicates(subset=["ValidTimeUtc"], keep="last").sort_values("ValidTimeUtc")
    out = out.rename(
        columns={
            "Precipitation": f"precip_{model}",
            "PrecipitationProbability": f"prob_{model}",
        }
    )
    return out.reset_index(drop=True)


def _load_rainfall_truth() -> pd.DataFrame:
    """Load all EA 15-min rainfall for Bellever and aggregate to hourly totals."""
    rain_dir = (
        WEATHERBLEND_DATA_ROOT
        / "truth"
        / "rainfall"
        / f"location={LOCATION}"
        / f"station={STATION}"
    )
    files = sorted(rain_dir.glob("date=*/rainfall.parquet"))
    if not files:
        raise FileNotFoundError(f"No rainfall truth files under {rain_dir}")

    frames = [
        pd.read_parquet(f, columns=["ObservedTimeUtc", "Value15MinMm", "Quality"]) for f in files
    ]
    raw = pd.concat(frames, ignore_index=True)

    # Only trust EA-flagged "Good" quality readings. Unknown/missing Quality
    # flags get dropped conservatively.
    raw = raw.loc[raw["Quality"] == "Good"].copy()

    # ObservedTimeUtc is the *start* of each 15-min interval. Bucket to the
    # containing hour and require all four 15-min slots to be present before
    # we consider the hour complete.
    raw["hour_start"] = raw["ObservedTimeUtc"].dt.floor("h")
    hourly = raw.groupby("hour_start").agg(
        mm=("Value15MinMm", "sum"),
        n_obs=("Value15MinMm", "size"),
    )
    hourly = hourly.loc[hourly["n_obs"] == 4].reset_index()
    hourly = hourly.rename(columns={"hour_start": "ValidTimeUtc"})
    hourly["observed_wet"] = (hourly["mm"] >= WET_THRESHOLD_MM).astype(int)
    return hourly[["ValidTimeUtc", "observed_wet", "mm"]]


def build_phase1_frame(verbose: bool = True) -> pd.DataFrame:
    """Build the joined features + target DataFrame (not yet train/test split)."""
    if verbose:
        print("Loading rainfall truth (Bellever Dartmoor)...")
    truth = _load_rainfall_truth()
    if verbose:
        print(f"  truth rows: {len(truth):,}  wet fraction: {truth['observed_wet'].mean():.3f}")

    df = truth[["ValidTimeUtc", "observed_wet"]].copy()

    for model in MODELS:
        if verbose:
            print(f"Loading forecasts: {model}...")
        fc = _load_model_forecasts(model)
        if verbose:
            print(f"  rows: {len(fc):,}  null precip: {fc[f'precip_{model}'].isna().mean():.3f}")
        df = df.merge(fc, on="ValidTimeUtc", how="inner")

    # Calendar features: hour-of-day encoded cyclically so that 23:00 and
    # 00:00 sit next to each other in feature space.
    hours = df["ValidTimeUtc"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    df = df.sort_values("ValidTimeUtc").reset_index(drop=True)
    return df


def prepare_phase1_dataset(
    test_fraction: float = 0.2, verbose: bool = True
) -> Phase1Dataset:
    """Load, clean, feature-select and chronologically split the data."""
    df = build_phase1_frame(verbose=verbose)

    # Drop rows where any precip_* feature is null (hard requirement: all
    # six models must have a 24h forecast for this valid time).
    precip_cols = [f"precip_{m}" for m in MODELS]
    before = len(df)
    df = df.dropna(subset=precip_cols)
    if verbose:
        print(f"After dropping rows with any null precip_*: {before:,} -> {len(df):,}")

    # Candidate probability features: drop any column >50% null, keep the rest.
    prob_cols = [f"prob_{m}" for m in MODELS]
    kept_prob_cols: list[str] = []
    for c in prob_cols:
        null_frac = df[c].isna().mean()
        if null_frac <= MAX_NULL_FRACTION:
            kept_prob_cols.append(c)
        if verbose:
            status = "keep" if c in kept_prob_cols else "drop"
            print(f"  {c}: null_frac={null_frac:.3f}  [{status}]")
    # For probability features we keep, fill residual NaNs with the column
    # mean so we do not have to drop whole rows; weakly informative stand-in.
    for c in kept_prob_cols:
        df[c] = df[c].fillna(df[c].mean())

    feature_names = precip_cols + kept_prob_cols + ["hour_sin", "hour_cos"]

    # Chronological 80/20 split.
    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    if verbose:
        print(
            f"Final rows: {len(df):,}  features: {len(feature_names)}  "
            f"train: {len(train):,} ({train['ValidTimeUtc'].min()} -> {train['ValidTimeUtc'].max()})  "
            f"test: {len(test):,} ({test['ValidTimeUtc'].min()} -> {test['ValidTimeUtc'].max()})"
        )
        print(
            f"  wet fraction  train={train['observed_wet'].mean():.3f}  "
            f"test={test['observed_wet'].mean():.3f}"
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


if __name__ == "__main__":
    ds = prepare_phase1_dataset()
    print("\nFeature sample (train head):")
    print(ds.X_train.head().to_string())
