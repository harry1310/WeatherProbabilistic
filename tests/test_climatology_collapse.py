"""Unit tests for the 4a climatology-collapse guard (_shared.climatology_collapse_reason).

Pure numpy — no R/rpy2 — so it runs in any pytest session. Guards the
2026-05-10 failure mode: a dbarts state/ntree round-trip mismatch makes
predict() return the training base rate y_train.mean() for every row (a flat
line == climatology), which predict_4a must now refuse to write.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _shared import (  # noqa: E402
    COLLAPSE_MIN_ROWS,
    climatology_collapse_reason,
)


def test_flags_flat_vector_equal_to_base_rate():
    base = 0.234
    p = np.full(48, base)  # exact Y.mean() collapse
    reason = climatology_collapse_reason(p, base)
    assert reason is not None
    assert "base rate" in reason


def test_does_not_flag_a_varying_forecast():
    rng = np.random.default_rng(0)
    p = np.clip(0.3 + 0.2 * rng.standard_normal(48), 0.01, 0.99)
    # Even if the mean happens near the base rate, the spread says "real".
    assert climatology_collapse_reason(p, float(p.mean())) is None


def test_does_not_flag_flat_but_off_base_rate():
    # Flat at 0.10 but the training base rate is 0.40 — not a climatology
    # collapse (a flat-but-real forecast that doesn't sit on the base rate).
    p = np.full(48, 0.10)
    assert climatology_collapse_reason(p, 0.40) is None


def test_too_few_rows_never_flagged():
    base = 0.5
    p = np.full(COLLAPSE_MIN_ROWS - 1, base)
    assert climatology_collapse_reason(p, base) is None


def test_just_enough_rows_is_judged():
    base = 0.5
    p = np.full(COLLAPSE_MIN_ROWS, base)
    assert climatology_collapse_reason(p, base) is not None
