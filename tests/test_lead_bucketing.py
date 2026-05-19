"""Phase 4a hourly predict — lead_day_bucket() day-bucketing.

predict_4a was made hourly (2026-05-18): instead of keeping only exact-lead
{24,48,72,96,120} forecast rows (a 6-hourly grid), build_live_features now
takes every hourly valid time and tags it with the trained-lead bucket for
the calendar day it falls in — bucket L covers the whole day anchor+L/24,
mirroring WeatherBlend's PrecipPredictCommand.BuildHourlyTargets.

lead_day_bucket is the pure rule behind that. These tests lock the day
boundaries so the bucketing can't silently drift (a wrong bucket would
mis-route a valid time to the wrong lead's BART, or break the 4b join).
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _shared import lead_day_bucket  # noqa: E402

# predict_4a's anchor is UTC midnight of the run day.
ANCHOR = datetime(2026, 5, 18, 0, 0, 0)


def test_first_day_is_bucket_24():
    # The whole calendar day anchor+1 → lead-24 bucket, every hour of it.
    assert lead_day_bucket(datetime(2026, 5, 19, 0, 0), ANCHOR) == 24
    assert lead_day_bucket(datetime(2026, 5, 19, 14, 0), ANCHOR) == 24
    assert lead_day_bucket(datetime(2026, 5, 19, 23, 0), ANCHOR) == 24


def test_each_day_maps_to_its_lead_bucket():
    assert lead_day_bucket(datetime(2026, 5, 20, 9, 0), ANCHOR) == 48
    assert lead_day_bucket(datetime(2026, 5, 21, 9, 0), ANCHOR) == 72
    assert lead_day_bucket(datetime(2026, 5, 22, 9, 0), ANCHOR) == 96
    assert lead_day_bucket(datetime(2026, 5, 23, 9, 0), ANCHOR) == 120


def test_same_day_resolves_to_bucket_zero():
    # A valid time on the anchor's own date → 0; not in LEADS, so the caller
    # (build_live_features' isin(LEADS) filter) drops it — 4a has no <24h model.
    assert lead_day_bucket(datetime(2026, 5, 18, 22, 0), ANCHOR) == 0


def test_far_horizon_past_longest_trained_lead():
    # Day +6 → 144, past the trained {24..120}; the caller's isin(LEADS) drops it.
    assert lead_day_bucket(datetime(2026, 5, 24, 6, 0), ANCHOR) == 144


def test_accepts_pandas_timestamp_for_either_arg():
    assert lead_day_bucket(pd.Timestamp("2026-05-19 06:00"), ANCHOR) == 24
    assert lead_day_bucket(
        pd.Timestamp("2026-05-21 06:00"), pd.Timestamp("2026-05-18 00:00")
    ) == 72
