"""Phase 5 — LightGBM dry-window baselines for comparison.

Two baselines are produced from the existing Phase 4 LightGBM predictions
(no retraining, no shell-out to WeatherBlend):

  * `lightgbm_independent_bernoulli` — for each (station, date, lead)
    cell, take the per-hour LightGBM-stripped predictions on the test
    set, treat them as 24 independent Bernoullis, and compute the
    decision-relevant aggregations exactly the same way the Bayesian
    pipeline does. This is the "matched-methodology" baseline — both
    methods produce per-hour P(wet), both use independent-Bernoulli
    rollup, only the per-hour probability source differs (LightGBM vs
    Bayesian Monte Carlo).

WeatherBlend's separately-trained 3b / 3d-shape classifiers (which
predict the day-level binary "is there an N-hour dry window today?"
target directly, not via per-hour rollup) would be the canonical
reference, but accessing them requires shelling out to WeatherBlend's
predict pipeline for each historical demo date — out of scope for the
in-repo Phase 5 work. Documented as a follow-up in the report.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date as DateType
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.aggregations import (
    LongestDryRunSummary, WindowExistsResult,
    longest_dry_run_distribution, p_window_exists,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent.parent
LGB_STRIPPED_DIR = ROOT / "reports" / "phase4_artefacts" / "lightgbm_predictions" / "stripped_7feature"
LGB_NATIVE_DIR = ROOT / "reports" / "phase4_artefacts" / "lightgbm_predictions" / "native_25feature"

_lgb_cache: dict[tuple[str, int], pd.DataFrame] = {}


def _load_lightgbm_predictions(variant: str, lead: int) -> pd.DataFrame:
    """Load per-row Phase 4 LightGBM test-set predictions. Cached per
    (variant, lead). Schema: valid_time, station, lead, p_wet, observed_wet."""
    key = (variant, lead)
    if key in _lgb_cache:
        return _lgb_cache[key]
    base = LGB_STRIPPED_DIR if variant == "stripped" else LGB_NATIVE_DIR
    fp = base / f"lead_{lead}h.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            f"LightGBM {variant} predictions not found at {fp}. "
            "Re-run scripts/run_phase4_lightgbm.py / run_phase4_lightgbm_native.py.")
    df = pd.read_parquet(fp)
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    _lgb_cache[key] = df
    return df


def get_lightgbm_day(
    variant: str, station: str, date: DateType, lead: int,
) -> pd.DataFrame:
    """Return the 24 hourly LightGBM predictions for one (station, date,
    lead) cell. Filters cached predictions; raises if cell not present
    or has != 24 hours."""
    df = _load_lightgbm_predictions(variant, lead)
    sub = df[
        (df["station"] == station)
        & (df["lead"] == lead)
        & (df["valid_time"].dt.date == date)
    ].sort_values("valid_time").reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(
            f"No LightGBM-{variant} predictions for {station} lead={lead}h date={date}")
    if len(sub) != 24:
        raise ValueError(
            f"Partial day: only {len(sub)}/24 hours present for "
            f"LightGBM-{variant} {station} lead={lead}h date={date}")
    return sub


# ---------------------------------------------------------------------------
# Independent-Bernoulli rollup of LightGBM hourly P(wet)
# ---------------------------------------------------------------------------

@dataclass
class LightGBMDayBaseline:
    """LightGBM-via-independent-Bernoulli baseline for one (station, date, lead).
    Provides aggregations directly comparable to the Bayesian Monte Carlo,
    plus a Bernoulli-sampled (N × 24) array if the consumer wants the same
    distributional output shape as `simulate_day`."""
    variant: str
    station: str
    date: DateType
    lead: int
    p_wet: np.ndarray              # (24,) hourly P(wet) from LightGBM
    samples: np.ndarray            # (N, 24) Bernoulli samples
    longest_dry_run: LongestDryRunSummary
    p_window_exists: dict[int, WindowExistsResult]


def lightgbm_independent_bernoulli(
    variant: str, station: str, date: DateType, lead: int,
    n_samples: int = 1000, seed: int = 42,
) -> LightGBMDayBaseline:
    """LightGBM hourly P(wet) → 24 independent Bernoulli rollup.

    The per-hour P(wet) values come from a single LightGBM fit (no
    parameter uncertainty), so the resulting (N × 24) array reflects
    only the within-day Bernoulli noise, not posterior uncertainty.
    Compared to the Bayesian's (N × 24) — which adds posterior parameter
    variation — the LightGBM samples will have NARROWER dispersion
    around their mean, even when the per-hour P(wet) values are similar.
    That's expected: it's the gap that the Bayesian fills.
    """
    sub = get_lightgbm_day(variant, station, date, lead)
    p = sub["p_wet"].to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    samples = (rng.random((n_samples, 24)) < p[None, :]).astype("int8")
    return LightGBMDayBaseline(
        variant=variant, station=station, date=date, lead=lead,
        p_wet=p, samples=samples,
        longest_dry_run=longest_dry_run_distribution(samples),
        p_window_exists=p_window_exists(samples, lengths=(2, 3, 4, 6)),
    )
