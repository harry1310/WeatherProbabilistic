"""Phase 5 — aggregation correctness on hand-checkable fixtures.

Pure-numpy aggregations, easy to pin. The simulation core itself is
tested separately in test_phase5_simulation.py — it requires loading
~14 MB of posterior NetCDFs which is slower.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation.aggregations import (  # noqa: E402
    _all_dry_run_starts, _longest_dry_run, _longest_dry_run_with_start, _wilson_ci,
    longest_dry_run_distribution, p_window_exists, p_window_in_range,
    window_start_time_distribution,
)


# ---------------------------------------------------------------------------
# Per-row primitives
# ---------------------------------------------------------------------------

class TestLongestDryRun:
    def test_all_dry(self):
        assert _longest_dry_run(np.zeros(24, dtype="int8")) == 24

    def test_all_wet(self):
        assert _longest_dry_run(np.ones(24, dtype="int8")) == 0

    def test_alternating(self):
        # 010101... — every dry run is length 1
        row = np.array([0, 1] * 12, dtype="int8")
        assert _longest_dry_run(row) == 1

    def test_single_run_in_middle(self):
        # 5 wet, 6 dry, 13 wet → longest = 6
        row = np.concatenate([np.ones(5), np.zeros(6), np.ones(13)]).astype("int8")
        assert _longest_dry_run(row) == 6

    def test_two_runs_different_lengths(self):
        # 3 dry, 1 wet, 5 dry, 15 wet → longest = 5
        row = np.concatenate([np.zeros(3), np.ones(1), np.zeros(5), np.ones(15)]).astype("int8")
        assert _longest_dry_run(row) == 5

    def test_run_at_end(self):
        # 18 wet, 6 dry → longest = 6 (boundary case — termination)
        row = np.concatenate([np.ones(18), np.zeros(6)]).astype("int8")
        assert _longest_dry_run(row) == 6

    def test_run_at_start(self):
        row = np.concatenate([np.zeros(6), np.ones(18)]).astype("int8")
        assert _longest_dry_run(row) == 6


class TestLongestDryRunWithStart:
    def test_all_dry_starts_at_zero(self):
        length, start = _longest_dry_run_with_start(np.zeros(24, dtype="int8"))
        assert (length, start) == (24, 0)

    def test_run_in_middle(self):
        # 5 wet, 6 dry starting at hour 5
        row = np.concatenate([np.ones(5), np.zeros(6), np.ones(13)]).astype("int8")
        length, start = _longest_dry_run_with_start(row)
        assert (length, start) == (6, 5)

    def test_ties_pick_earliest(self):
        # Two equal 4h dry runs at hours 0-3 and 12-15
        row = np.array([0,0,0,0, 1,1,1,1, 1,1,1,1, 0,0,0,0, 1,1,1,1, 1,1,1,1], dtype="int8")
        length, start = _longest_dry_run_with_start(row)
        assert length == 4
        assert start == 0

    def test_all_wet_returns_zero_zero(self):
        length, start = _longest_dry_run_with_start(np.ones(24, dtype="int8"))
        assert (length, start) == (0, 0)


class TestAllDryRunStarts:
    def test_no_runs(self):
        assert _all_dry_run_starts(np.ones(24, dtype="int8"), min_length=1) == []

    def test_min_length_filter(self):
        # 2 dry, 1 wet, 4 dry, 1 wet, 1 dry — runs of length [2, 4, 1]
        row = np.array([0,0, 1, 0,0,0,0, 1, 0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype="int8")
        # All runs (min_length=1): starts at 0, 3, 8
        assert _all_dry_run_starts(row, min_length=1) == [0, 3, 8]
        # >= 3: starts at 3 only
        assert _all_dry_run_starts(row, min_length=3) == [3]
        # >= 5: nothing
        assert _all_dry_run_starts(row, min_length=5) == []


class TestWilsonCI:
    def test_zero_count_ci_starts_at_zero(self):
        lo, hi = _wilson_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.05

    def test_full_count_ci_ends_at_one(self):
        lo, hi = _wilson_ci(100, 100)
        assert hi == 1.0
        assert lo > 0.95

    def test_half_count_ci_centred(self):
        lo, hi = _wilson_ci(50, 100)
        assert 0.40 < lo < 0.50
        assert 0.50 < hi < 0.60

    def test_zero_n_returns_nan(self):
        lo, hi = _wilson_ci(0, 0)
        assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# Aggregation: longest dry run distribution
# ---------------------------------------------------------------------------

class TestLongestDryRunDistribution:
    def test_all_dry_all_samples(self):
        samples = np.zeros((100, 24), dtype="int8")
        s = longest_dry_run_distribution(samples)
        assert s.n_samples == 100
        assert s.mean == 24.0
        assert s.median == 24.0
        assert s.histogram == {24: 100}

    def test_all_wet_all_samples(self):
        samples = np.ones((50, 24), dtype="int8")
        s = longest_dry_run_distribution(samples)
        assert s.mean == 0.0
        assert s.histogram == {0: 50}

    def test_mixed_distribution(self):
        # 30 sims with 24h dry, 70 sims with 6h dry max
        samples = np.zeros((100, 24), dtype="int8")
        samples[30:, :] = 1   # all wet for 70 sims
        # Carve a 6-hour dry run into the wet ones
        samples[30:, 12:18] = 0
        s = longest_dry_run_distribution(samples)
        assert s.histogram == {6: 70, 24: 30}
        # Median: 100 sorted values → midpoint of 50th and 51st
        # First 70 are 6, last 30 are 24 → both at index 49 and 50 are 6
        assert s.median == 6.0

    def test_rejects_non_2d(self):
        import pytest
        with pytest.raises(ValueError, match="2D"):
            longest_dry_run_distribution(np.zeros(24, dtype="int8"))


# ---------------------------------------------------------------------------
# Aggregation: P(window of length L exists)
# ---------------------------------------------------------------------------

class TestPWindowExists:
    def test_all_dry_window_always_exists(self):
        samples = np.zeros((100, 24), dtype="int8")
        out = p_window_exists(samples, lengths=(2, 3, 4, 6))
        for L in (2, 3, 4, 6):
            assert out[L].probability == 1.0
            assert out[L].n_have_window == 100

    def test_all_wet_no_window(self):
        samples = np.ones((50, 24), dtype="int8")
        out = p_window_exists(samples, lengths=(2,))
        assert out[2].probability == 0.0

    def test_partial_population_with_known_p(self):
        # 60 sims with 4h dry block, 40 sims all-wet
        samples = np.ones((100, 24), dtype="int8")
        samples[:60, 10:14] = 0   # 4h dry block in first 60
        out = p_window_exists(samples, lengths=(2, 4, 6))
        # 2h: present in first 60 (since 4h ⊃ 2h)
        assert out[2].n_have_window == 60
        assert out[2].probability == 0.6
        # 4h: same 60
        assert out[4].n_have_window == 60
        # 6h: zero
        assert out[6].n_have_window == 0
        assert out[6].probability == 0.0

    def test_ci_brackets_point_estimate(self):
        samples = np.ones((100, 24), dtype="int8")
        samples[:50, 10:13] = 0
        out = p_window_exists(samples, lengths=(2,))
        assert out[2].ci_low <= out[2].probability <= out[2].ci_high


# ---------------------------------------------------------------------------
# Aggregation: window-start-time distribution
# ---------------------------------------------------------------------------

class TestWindowStartTimeDistribution:
    def test_all_starts_at_same_hour(self):
        # All sims have a 4h dry window starting at hour 10
        samples = np.ones((100, 24), dtype="int8")
        samples[:, 10:14] = 0
        s = window_start_time_distribution(samples, length=4)
        assert s.n_samples_with_window == 100
        assert s.start_hour_histogram == {10: 100}
        assert s.mode_start_hour == 10
        assert s.median_start_hour == 10.0

    def test_split_starts(self):
        # 30 sims: window at hour 5, 70 sims: window at hour 15
        samples = np.ones((100, 24), dtype="int8")
        samples[:30, 5:9] = 0
        samples[30:, 15:19] = 0
        s = window_start_time_distribution(samples, length=4)
        assert s.n_samples_with_window == 100
        assert s.start_hour_histogram == {5: 30, 15: 70}
        assert s.mode_start_hour == 15

    def test_no_window_returns_empty(self):
        samples = np.ones((50, 24), dtype="int8")
        s = window_start_time_distribution(samples, length=4)
        assert s.n_samples_with_window == 0
        assert s.start_hour_histogram == {}
        assert s.mode_start_hour == -1


# ---------------------------------------------------------------------------
# Aggregation: P(window in user time range)
# ---------------------------------------------------------------------------

class TestPWindowInRange:
    def test_window_inside_range(self):
        # 100 sims, all have 2h window at hour 10
        samples = np.ones((100, 24), dtype="int8")
        samples[:, 10:12] = 0
        # Range [9, 13) covers start hour 10
        out = p_window_in_range(samples, length=2, start_hour=9, end_hour=13)
        assert out.probability == 1.0
        assert out.n_have_window_in_range == 100

    def test_window_outside_range(self):
        # 100 sims, all have 2h window at hour 20
        samples = np.ones((100, 24), dtype="int8")
        samples[:, 20:22] = 0
        # Range [9, 13) does not include 20
        out = p_window_in_range(samples, length=2, start_hour=9, end_hour=13)
        assert out.probability == 0.0

    def test_range_boundary(self):
        # half-open [9, 13): start at 12 in, start at 13 out
        samples = np.ones((100, 24), dtype="int8")
        samples[:50, 12:15] = 0   # starts at 12 (in range)
        samples[50:, 13:16] = 0   # starts at 13 (out of range)
        out = p_window_in_range(samples, length=2, start_hour=9, end_hour=13)
        assert out.n_have_window_in_range == 50
        assert out.probability == 0.5

    def test_invalid_range_throws(self):
        import pytest
        samples = np.ones((10, 24), dtype="int8")
        with pytest.raises(ValueError):
            p_window_in_range(samples, length=2, start_hour=15, end_hour=10)
        with pytest.raises(ValueError):
            p_window_in_range(samples, length=2, start_hour=-1, end_hour=10)
        with pytest.raises(ValueError):
            p_window_in_range(samples, length=2, start_hour=0, end_hour=25)
