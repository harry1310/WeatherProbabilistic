"""Phase 5 — decision-relevant aggregations over (N × 24) Monte Carlo samples.

All functions take an `(N, 24)` int8 array of simulated wet/dry sequences
(0 = dry, 1 = wet) and return summary statistics. Pure numpy, no I/O,
fully unit-testable on hand-checkable fixtures.

The four headline aggregations:

  * `longest_dry_run_distribution` — what's the distribution over "longest
    consecutive dry block in this day"? Median, percentiles, full histogram.
  * `p_window_exists` — for each window length L ∈ {2, 3, 4, 6}, what
    fraction of simulations contain at least one dry run of length ≥ L?
    Returns point estimates + Wilson CIs.
  * `window_start_time_distribution` — for simulations that DO contain a
    dry window of length ≥ L, when does the longest one start?
  * `p_window_in_range` — the climbing-decision-layer question: P(at least
    one dry window of length L starts within hour range [a, b])?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Helpers — per-row primitives. Vectorise over the N axis where possible
# but the longest-run scan is intrinsically sequential per row.
# ---------------------------------------------------------------------------

def _longest_dry_run(row: np.ndarray) -> int:
    """Length of the longest consecutive run of 0s in a 1-D wet/dry row."""
    longest = 0
    current = 0
    for v in row:
        if v == 0:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def _all_dry_run_starts(row: np.ndarray, min_length: int) -> list[int]:
    """All start hours of dry runs of at least min_length consecutive zeros.
    Each run contributes only its start; runs longer than min_length count
    once (at the start). Used for window-start distributions."""
    starts: list[int] = []
    n = len(row)
    i = 0
    while i < n:
        if row[i] == 0:
            j = i
            while j < n and row[j] == 0:
                j += 1
            run_len = j - i
            if run_len >= min_length:
                starts.append(i)
            i = j
        else:
            i += 1
    return starts


def _longest_dry_run_with_start(row: np.ndarray) -> tuple[int, int]:
    """Return (length, start_hour) of THE longest dry run. If multiple runs
    tie for longest, the EARLIEST is returned (deterministic)."""
    longest = 0
    longest_start = 0
    n = len(row)
    i = 0
    while i < n:
        if row[i] == 0:
            j = i
            while j < n and row[j] == 0:
                j += 1
            run_len = j - i
            if run_len > longest:
                longest = run_len
                longest_start = i
            i = j
        else:
            i += 1
    return longest, longest_start


def _wilson_ci(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion. Default z=1.645 for 90%
    (matches the credible-interval convention used elsewhere in the report)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# 1. Longest dry run distribution
# ---------------------------------------------------------------------------

@dataclass
class LongestDryRunSummary:
    n_samples: int
    mean: float
    median: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    histogram: dict[int, int]      # length -> count


def longest_dry_run_distribution(samples: np.ndarray) -> LongestDryRunSummary:
    """For each of the N rows in `samples`, find the longest consecutive run
    of dry (0) hours. Return summary statistics + full histogram."""
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (N, 24), got shape {samples.shape}")
    lengths = np.array([_longest_dry_run(row) for row in samples], dtype="int32")
    hist = {int(k): int(v) for k, v in zip(*np.unique(lengths, return_counts=True))}
    return LongestDryRunSummary(
        n_samples=int(len(lengths)),
        mean=float(lengths.mean()),
        median=float(np.median(lengths)),
        p05=float(np.percentile(lengths,  5)),
        p25=float(np.percentile(lengths, 25)),
        p50=float(np.percentile(lengths, 50)),
        p75=float(np.percentile(lengths, 75)),
        p95=float(np.percentile(lengths, 95)),
        histogram=hist,
    )


# ---------------------------------------------------------------------------
# 2. P(window of length L exists)
# ---------------------------------------------------------------------------

@dataclass
class WindowExistsResult:
    length: int
    n_samples: int
    n_have_window: int
    probability: float
    ci_low: float
    ci_high: float


def p_window_exists(samples: np.ndarray, lengths: tuple[int, ...] = (2, 3, 4, 6)) -> dict[int, WindowExistsResult]:
    """For each window length L, fraction of simulations that contain at
    least one dry run of length >= L. Wilson 90% CI from the N samples
    (the "uncertainty" here is just Monte Carlo sample-size; doesn't capture
    posterior parameter uncertainty — that's already baked into the
    samples themselves)."""
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (N, 24), got shape {samples.shape}")
    longest = np.array([_longest_dry_run(row) for row in samples], dtype="int32")
    n = len(longest)
    out: dict[int, WindowExistsResult] = {}
    for L in lengths:
        k = int((longest >= L).sum())
        lo, hi = _wilson_ci(k, n)
        out[L] = WindowExistsResult(
            length=L, n_samples=n, n_have_window=k,
            probability=k / n if n > 0 else float("nan"),
            ci_low=lo, ci_high=hi,
        )
    return out


# ---------------------------------------------------------------------------
# 3. Window start time distribution (conditional on existence)
# ---------------------------------------------------------------------------

@dataclass
class WindowStartTimeSummary:
    length: int
    n_samples_with_window: int
    n_samples_total: int
    start_hour_histogram: dict[int, int]    # 0..23 → count of "longest run started here"
    median_start_hour: float
    mode_start_hour: int                    # peak of the histogram
    p_conditional_on_existence: float       # = n_samples_with_window / n_samples_total


def window_start_time_distribution(samples: np.ndarray, length: int) -> WindowStartTimeSummary:
    """For each simulation that contains a dry window of length >= L,
    record the start hour of the LONGEST dry run (ties broken by earliest).
    Returns the histogram + summary stats. Only sims where the longest
    run is >= L contribute; the rest are excluded from the start-time
    histogram but counted in `n_samples_total`."""
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (N, 24), got shape {samples.shape}")
    starts: list[int] = []
    for row in samples:
        run_len, run_start = _longest_dry_run_with_start(row)
        if run_len >= length:
            starts.append(int(run_start))
    n_total = len(samples)
    n_with = len(starts)
    if n_with == 0:
        return WindowStartTimeSummary(
            length=length, n_samples_with_window=0, n_samples_total=n_total,
            start_hour_histogram={}, median_start_hour=float("nan"),
            mode_start_hour=-1, p_conditional_on_existence=0.0,
        )
    arr = np.array(starts, dtype="int32")
    hist = {int(k): int(v) for k, v in zip(*np.unique(arr, return_counts=True))}
    mode_hour = int(max(hist, key=hist.get))
    return WindowStartTimeSummary(
        length=length, n_samples_with_window=n_with, n_samples_total=n_total,
        start_hour_histogram=hist,
        median_start_hour=float(np.median(arr)),
        mode_start_hour=mode_hour,
        p_conditional_on_existence=n_with / n_total,
    )


# ---------------------------------------------------------------------------
# 4. P(window of length L starts within [start_hour, end_hour))
# ---------------------------------------------------------------------------

@dataclass
class WindowInRangeResult:
    length: int
    start_hour: int
    end_hour: int
    n_samples: int
    n_have_window_in_range: int
    probability: float
    ci_low: float
    ci_high: float


def p_window_in_range(
    samples: np.ndarray,
    length: int,
    start_hour: int,
    end_hour: int,
) -> WindowInRangeResult:
    """Probability that AT LEAST ONE dry run of length >= L STARTS within
    the half-open interval [start_hour, end_hour). The window itself can
    extend past `end_hour` — the range constraint is on the start.

    Climbing-decision-layer question. E.g. p_window_in_range(samples, length=2,
    start_hour=9, end_hour=13) = "P(2h dry window starts between 09:00 and
    13:00)" — i.e. "I want to be out climbing 09:00-13:00, what's the
    chance there's a 2-hour dry window starting in that morning slot?"
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (N, 24), got shape {samples.shape}")
    if not (0 <= start_hour < 24 and 0 < end_hour <= 24 and start_hour < end_hour):
        raise ValueError(f"Invalid hour range [{start_hour}, {end_hour})")
    n = len(samples)
    n_have = 0
    for row in samples:
        starts = _all_dry_run_starts(row, min_length=length)
        if any(start_hour <= s < end_hour for s in starts):
            n_have += 1
    lo, hi = _wilson_ci(n_have, n)
    return WindowInRangeResult(
        length=length, start_hour=start_hour, end_hour=end_hour,
        n_samples=n, n_have_window_in_range=n_have,
        probability=n_have / n if n > 0 else float("nan"),
        ci_low=lo, ci_high=hi,
    )
