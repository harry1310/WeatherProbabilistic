"""Phase 5 — Monte Carlo dry-window simulation core.

Given a (station, date, lead) cell, draw N samples from Phase 3 Model A's
posterior, apply the Phase 4.5 isotonic calibrator, and Bernoulli-sample 24
hourly wet/dry outcomes per posterior draw. The resulting (N × 24) array is
the substrate for every decision-relevant aggregation in
`src.simulation.aggregations` — longest dry run, P(window of length L exists),
window-start-time distributions, user-specified time-range probability.

Independence assumption (deliberate, documented):
  Each hour is sampled independently, conditional on its calibrated P(wet).
  Real weather has within-day correlation (wet hours cluster, dry hours
  cluster); independent Bernoulli sampling will under-represent both very-dry
  and very-wet days. Phase 5's Step 6 calibration check on simulated outputs
  is the place to detect whether this matters in practice; if it does, a
  future phase could add a within-day Markov component.

Determinism:
  Given the same (station, date, lead, n_samples, seed), the entire
  pipeline is reproducible. Posterior draw indices are sampled with a
  numpy Generator seeded from the user-supplied seed; Bernoulli draws
  use a separate seed-derived stream. Calibrator + posterior + features
  are all on-disk artefacts loaded at call time.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date as DateType
from pathlib import Path
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd

from src.calibration.isotonic import IsotonicCalibratorBundle, load_bundle
from src.data import (
    MODELS_NO_UKMO, PHASE3_LEAD_HOURS, STATIONS, prepare_phase3_dataset,
)

# Suppress arviz/xarray warmup-iter warnings that are noisy + benign here.
warnings.filterwarnings("ignore", category=UserWarning, module="arviz")

ROOT = Path(__file__).resolve().parent.parent.parent
POSTERIOR_DIR = ROOT / "reports" / "phase4_artefacts" / "bayesian_predictions" / "posteriors"
ISOTONIC_DIR = ROOT / "reports" / "phase4_artefacts" / "bayesian_isotonic"

_STATION_INDEX = {code: i for i, (_, code) in enumerate(STATIONS)}
_LEAD_INDEX = {lead: i for i, lead in enumerate(PHASE3_LEAD_HOURS)}


# ---------------------------------------------------------------------------
# Cached loaders — posteriors + features are heavyweight; we only want to
# pay the I/O cost once per process even if simulate_day is called many times.
# ---------------------------------------------------------------------------

@dataclass
class PosteriorParams:
    """Per-station composed parameters extracted from Phase 3 Model A's
    saved posterior. `intercept` shape (chain, draw); `beta` shape
    (chain, draw, feature). Reshape-flat shapes (`intercept_flat`,
    `beta_flat`) are pre-computed for fast index sampling."""
    lead: int
    station: str
    intercept_flat: np.ndarray  # (chain*draw,)
    beta_flat: np.ndarray       # (chain*draw, feature)
    feature_names: list[str]

    @property
    def n_total_samples(self) -> int:
        return self.intercept_flat.shape[0]


_posterior_cache: dict[tuple[int, str], PosteriorParams] = {}
_calibrator_cache: dict[int, IsotonicCalibratorBundle] = {}
_dataset_cache: Optional[object] = None


def _load_posterior(lead: int, station: str) -> PosteriorParams:
    key = (lead, station)
    if key in _posterior_cache:
        return _posterior_cache[key]
    if station not in _STATION_INDEX:
        raise ValueError(f"Unknown station {station!r}; expected one of {list(_STATION_INDEX)}")
    s_idx = _STATION_INDEX[station]
    nc_path = POSTERIOR_DIR / f"lead_{lead}h.nc"
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Phase 3 Model A posterior not found at {nc_path}. "
            "Re-run scripts/run_phase4_bayesian.py to regenerate.")
    idata = az.from_netcdf(str(nc_path))
    intercept = idata.posterior["intercept_s"].values[..., s_idx]   # (chain, draw)
    beta = idata.posterior["beta_s"].values[..., s_idx, :]           # (chain, draw, feat)
    n_chain, n_draw = intercept.shape
    feat_dim = beta.shape[-1]
    pp = PosteriorParams(
        lead=lead, station=station,
        intercept_flat=intercept.reshape(-1).astype("float64"),
        beta_flat=beta.reshape(n_chain * n_draw, feat_dim).astype("float64"),
        feature_names=list(idata.posterior["feature"].values) if "feature" in idata.posterior.coords else [],
    )
    _posterior_cache[key] = pp
    return pp


def _load_calibrators(lead: int) -> IsotonicCalibratorBundle:
    if lead in _calibrator_cache:
        return _calibrator_cache[lead]
    bundle = load_bundle(ISOTONIC_DIR)
    _calibrator_cache[lead] = bundle
    return bundle


def _load_dataset(verbose: bool = False):
    """Returns the cached Phase3Dataset (5-model variant matching Phase 3
    Model A's training). Loaded once per process — heavyweight DuckDB +
    standardisation pass takes ~30s."""
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = prepare_phase3_dataset(models=MODELS_NO_UKMO, verbose=verbose)
    return _dataset_cache


# ---------------------------------------------------------------------------
# Per-day feature lookup
# ---------------------------------------------------------------------------

@dataclass
class DayFeatures:
    """The 24 hourly standardised feature vectors for one (station, date,
    lead) cell, plus the matching observed binary labels for verification."""
    station: str
    date: DateType
    lead: int
    valid_times: np.ndarray              # (n_hours,) datetime64
    X_s: np.ndarray                      # (n_hours, n_feat) standardised
    observed_wet: np.ndarray             # (n_hours,) int8
    n_hours: int


def get_day_features(station: str, date: DateType, lead: int) -> DayFeatures:
    """Pull the 24 hourly standardised feature vectors for the requested
    (station, date, lead) cell from the test set. Raises if the cell isn't
    in the test set or has incomplete coverage (≠24 hours).

    The test set is the chronological 20% tail of the post-2024-09 dataset
    that Phase 3 Model A held back from training, so any (station, lead)
    cell + valid_time within that window is fair game."""
    if station not in _STATION_INDEX:
        raise ValueError(f"Unknown station {station!r}")
    if lead not in _LEAD_INDEX:
        raise ValueError(f"Lead {lead}h not in {PHASE3_LEAD_HOURS}")
    s_idx = _STATION_INDEX[station]
    l_idx = _LEAD_INDEX[lead]
    ds = _load_dataset()
    vt = pd.to_datetime(ds.valid_time_test.values)
    mask = (
        (ds.station_idx_test == s_idx)
        & (ds.lead_idx_test == l_idx)
        & (pd.Series(vt).dt.date == date).to_numpy()
    )
    n = int(mask.sum())
    if n == 0:
        raise ValueError(
            f"No test rows for station={station} lead={lead}h date={date}. "
            "Check date is in the test window.")
    if n != 24:
        raise ValueError(
            f"Partial day: only {n}/24 hours present for station={station} lead={lead}h date={date}. "
            "Pick a different demo date or accept the gap downstream.")
    hours_order = np.argsort(vt[mask])
    return DayFeatures(
        station=station, date=date, lead=lead,
        valid_times=vt[mask][hours_order],
        X_s=ds.X_test_s[mask][hours_order],
        observed_wet=ds.y_test.values[mask][hours_order].astype("int8"),
        n_hours=n,
    )


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------

def simulate_day(
    station: str,
    date: DateType,
    lead: int,
    n_samples: int = 1000,
    seed: int = 42,
    apply_calibration: bool = True,
) -> dict:
    """Generate N posterior-driven samples of the 24-hour wet/dry sequence
    for one (station, date, lead) cell.

    Returns a dict with:
        ``samples``        — (N, 24) int8 array of simulated wet (1) / dry (0)
        ``calibrated_p``   — (N, 24) float64 array of per-hour calibrated P(wet)
                             used to draw each row's Bernoullis
        ``raw_p``          — (N, 24) float64 array of pre-calibration P(wet)
        ``observed_wet``   — (24,) int8 actual observations for verification
        ``valid_times``    — (24,) datetime64 hours of the target day
        ``feature_names``  — list of standardised feature column names
        ``params``         — PosteriorParams used (for audit)
        ``seed``           — the seed actually used
        ``n_samples``      — N

    Pipeline (deterministic given seed):
        1. Load Phase 3 Model A's posterior for (station, lead). Returns
           composed `intercept_s[s]` shape (8000,) and `beta_s[s]` shape
           (8000, 7) — 4 chains × 2000 draws per fit.
        2. Load Phase 4.5 isotonic calibrator for (station, lead).
        3. Pull the 24 hourly feature vectors for (station, date, lead)
           from the cached test set.
        4. Sample N indices into the 8000-row posterior pool via
           `rng.choice(...)` — sample WITH replacement so N can exceed
           8000 if needed.
        5. For each posterior sample i and hour h:
             logit = intercept[i] + beta[i, :] @ X_s[h, :]
             raw_p[i, h] = sigmoid(logit)
        6. If apply_calibration: piecewise-monotonic isotonic on raw_p[i, h].
        7. Bernoulli sample: samples[i, h] ~ Bernoulli(calibrated_p[i, h])
           using a seed-derived second RNG stream so calibration and
           sampling rng-streams don't entangle.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if station not in _STATION_INDEX:
        raise ValueError(f"Unknown station {station!r}; expected one of {list(_STATION_INDEX)}")
    if lead not in _LEAD_INDEX:
        raise ValueError(f"Lead {lead}h not in {PHASE3_LEAD_HOURS}")

    pp = _load_posterior(lead, station)
    bundle = _load_calibrators(lead) if apply_calibration else None
    day = get_day_features(station, date, lead)

    n_h, feat_dim = day.X_s.shape
    if pp.beta_flat.shape[1] != feat_dim:
        raise RuntimeError(
            f"Posterior feature dim {pp.beta_flat.shape[1]} != dataset feature dim {feat_dim}; "
            "posterior and dataset are out of sync (re-fit Phase 3 Model A).")

    rng_idx = np.random.default_rng(seed)
    rng_bern = np.random.default_rng(seed ^ 0xA5A5A5A5)
    sample_idx = rng_idx.integers(low=0, high=pp.n_total_samples, size=n_samples)

    intercepts = pp.intercept_flat[sample_idx]                # (N,)
    betas = pp.beta_flat[sample_idx]                          # (N, feat)
    logits = intercepts[:, None] + betas @ day.X_s.T          # (N, 24)
    raw_p = 1.0 / (1.0 + np.exp(-logits))                     # (N, 24)

    if apply_calibration:
        cal = bundle.calibrators[(lead, station)]
        flat = raw_p.reshape(-1)
        calibrated = cal.predict(flat).reshape(raw_p.shape)
    else:
        calibrated = raw_p

    samples = (rng_bern.random(calibrated.shape) < calibrated).astype("int8")

    return dict(
        samples=samples,
        calibrated_p=calibrated,
        raw_p=raw_p,
        observed_wet=day.observed_wet,
        valid_times=day.valid_times,
        feature_names=pp.feature_names,
        params=pp,
        seed=seed,
        n_samples=n_samples,
    )


def list_test_days(
    station: str,
    lead: int,
    full_24h_only: bool = True,
) -> list[DateType]:
    """Convenience: enumerate all dates in the test set for which
    `get_day_features(station, date, lead)` would succeed. Used by the
    demo + sensitivity-check scripts to pick representative days."""
    if station not in _STATION_INDEX:
        raise ValueError(f"Unknown station {station!r}")
    if lead not in _LEAD_INDEX:
        raise ValueError(f"Unknown lead {lead}")
    s_idx = _STATION_INDEX[station]
    l_idx = _LEAD_INDEX[lead]
    ds = _load_dataset()
    vt = pd.to_datetime(ds.valid_time_test.values)
    mask = (ds.station_idx_test == s_idx) & (ds.lead_idx_test == l_idx)
    sub = pd.Series(vt[mask]).dt.date
    counts = sub.value_counts()
    if full_24h_only:
        counts = counts[counts == 24]
    return sorted(counts.index.tolist())
