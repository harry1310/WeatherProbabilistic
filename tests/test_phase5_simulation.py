"""Phase 5 — simulation core determinism + edge cases.

These tests load real artefacts (Phase 3 Model A posterior NetCDFs +
Phase 4.5 calibrators + Phase 3 dataset) — slower than the aggregation
tests but verifies the end-to-end pipeline against fixed test rows.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation.core import (  # noqa: E402
    POSTERIOR_DIR, get_day_features, list_test_days, simulate_day,
)


# Skip the whole module if the Phase 4 artefacts aren't on disk yet
# (e.g. on a fresh clone before scripts/run_phase4_bayesian.py runs).
pytestmark = pytest.mark.skipif(
    not (POSTERIOR_DIR / "lead_24h.nc").exists(),
    reason="Phase 4 Bayesian posteriors not on disk; run scripts/run_phase4_bayesian.py first",
)


# ---------------------------------------------------------------------------
# Pick a known-good demo day at module scope so tests don't all re-enumerate
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_day():
    """A day from the test set that we know has all 24 hours present at
    Bellever lead 24h. Picked deterministically as the first such day."""
    days = list_test_days("Bellever", 24, full_24h_only=True)
    if not days:
        pytest.skip("No full-24h test days available for Bellever lead 24h")
    return days[0]


# ---------------------------------------------------------------------------
# Determinism — same inputs + same seed → identical outputs
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_produces_identical_samples(self, demo_day):
        out1 = simulate_day("Bellever", demo_day, 24, n_samples=100, seed=42)
        out2 = simulate_day("Bellever", demo_day, 24, n_samples=100, seed=42)
        np.testing.assert_array_equal(out1["samples"], out2["samples"])
        np.testing.assert_array_equal(out1["calibrated_p"], out2["calibrated_p"])
        np.testing.assert_array_equal(out1["raw_p"], out2["raw_p"])

    def test_different_seeds_produce_different_samples(self, demo_day):
        out1 = simulate_day("Bellever", demo_day, 24, n_samples=100, seed=42)
        out2 = simulate_day("Bellever", demo_day, 24, n_samples=100, seed=99)
        # Same posterior, same features, different seed → different posterior
        # samples chosen → different raw_p / calibrated_p arrays.
        assert not np.array_equal(out1["raw_p"], out2["raw_p"])
        assert not np.array_equal(out1["samples"], out2["samples"])


# ---------------------------------------------------------------------------
# Output-shape + value-range invariants
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_samples_are_2d_int8_in_zero_one(self, demo_day):
        out = simulate_day("Bellever", demo_day, 24, n_samples=100, seed=0)
        assert out["samples"].shape == (100, 24)
        assert out["samples"].dtype == np.int8
        assert set(np.unique(out["samples"])).issubset({0, 1})

    def test_calibrated_p_in_unit_interval(self, demo_day):
        out = simulate_day("Bellever", demo_day, 24, n_samples=200, seed=0)
        assert out["calibrated_p"].shape == (200, 24)
        assert (out["calibrated_p"] >= 0).all()
        assert (out["calibrated_p"] <= 1).all()

    def test_raw_p_in_unit_interval(self, demo_day):
        out = simulate_day("Bellever", demo_day, 24, n_samples=200, seed=0)
        assert (out["raw_p"] > 0).all() and (out["raw_p"] < 1).all()

    def test_observed_wet_is_24_int8(self, demo_day):
        out = simulate_day("Bellever", demo_day, 24, n_samples=10, seed=0)
        assert out["observed_wet"].shape == (24,)
        assert out["observed_wet"].dtype == np.int8

    def test_n_samples_one_works(self, demo_day):
        # Edge case from the brief: N=1 must not crash
        out = simulate_day("Bellever", demo_day, 24, n_samples=1, seed=0)
        assert out["samples"].shape == (1, 24)


# ---------------------------------------------------------------------------
# Calibration application
# ---------------------------------------------------------------------------

class TestCalibrationApplication:
    def test_calibration_changes_probabilities(self, demo_day):
        # The calibrator is non-trivial (~30 knots per cell); it should
        # measurably move probabilities away from raw values for at least
        # some inputs in the population.
        out = simulate_day("Bellever", demo_day, 24, n_samples=500, seed=0)
        assert not np.allclose(out["calibrated_p"], out["raw_p"], atol=1e-6)

    def test_apply_calibration_false_skips_calibration(self, demo_day):
        # When apply_calibration=False, calibrated_p == raw_p exactly
        out = simulate_day("Bellever", demo_day, 24, n_samples=100, seed=0,
                          apply_calibration=False)
        np.testing.assert_array_equal(out["calibrated_p"], out["raw_p"])

    def test_calibration_is_monotonic_in_raw_p(self, demo_day):
        # For one fixed posterior sample evaluated at all 24 hours,
        # rank-order of raw_p should match rank-order of calibrated_p
        # (isotonic preserves order).
        out = simulate_day("Bellever", demo_day, 24, n_samples=10, seed=0)
        for i in range(out["raw_p"].shape[0]):
            raw_order = np.argsort(out["raw_p"][i])
            cal_order = np.argsort(out["calibrated_p"][i])
            # Allow ties (PAV groups violators into plateaus); the sorted
            # raw values and the calibrated values they map to should be
            # non-decreasing in the same direction.
            sorted_cal = out["calibrated_p"][i][raw_order]
            assert (np.diff(sorted_cal) >= -1e-12).all(), \
                f"Sample {i}: isotonic broke monotonicity, diffs={np.diff(sorted_cal)}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_station_raises(self, demo_day):
        with pytest.raises(ValueError, match="Unknown station"):
            simulate_day("NotARealStation", demo_day, 24, n_samples=10, seed=0)

    def test_unknown_lead_raises(self, demo_day):
        with pytest.raises(ValueError, match="Lead"):
            simulate_day("Bellever", demo_day, 36, n_samples=10, seed=0)

    def test_date_outside_test_window_raises(self):
        # Pre-test-window date — Phase 3's test set starts ~Jun 2025.
        with pytest.raises(ValueError, match="No test rows"):
            simulate_day("Bellever", date(2024, 1, 1), 24, n_samples=10, seed=0)

    def test_n_samples_zero_raises(self, demo_day):
        with pytest.raises(ValueError, match="n_samples"):
            simulate_day("Bellever", demo_day, 24, n_samples=0, seed=0)


# ---------------------------------------------------------------------------
# get_day_features sanity
# ---------------------------------------------------------------------------

class TestGetDayFeatures:
    def test_returns_24_hours_in_chronological_order(self, demo_day):
        import pandas as pd
        feat = get_day_features("Bellever", demo_day, 24)
        assert feat.n_hours == 24
        assert feat.X_s.shape == (24, 7)   # 5 precip + hour_sin + hour_cos
        # Hours are chronological
        ts = pd.DatetimeIndex(feat.valid_times)
        assert (np.diff(ts.values.astype("int64")) > 0).all()
        # Hours-of-day cover 0..23
        assert sorted(ts.hour.tolist()) == list(range(24))

    def test_observed_wet_is_binary(self, demo_day):
        feat = get_day_features("Bellever", demo_day, 24)
        assert set(np.unique(feat.observed_wet)).issubset({0, 1})


# ---------------------------------------------------------------------------
# list_test_days returns sensible output
# ---------------------------------------------------------------------------

def test_list_test_days_returns_dates():
    days = list_test_days("Bellever", 24, full_24h_only=True)
    assert len(days) > 50  # we expect ~120 full-24h days per cell
    assert all(isinstance(d, date) for d in days)
    assert days == sorted(days)
