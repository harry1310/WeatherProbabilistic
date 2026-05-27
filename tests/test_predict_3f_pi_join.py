"""Regression tests for the 2026-05-26 broadcast-shape bug in
``scripts/predict_3f.load_bound_3a_pi``.

The bug: ``predict.yml`` fires every 6 hours, so a single anchor's 3a
predictions parquet typically holds 4-5 intra-day cycles. Each cycle
emits one row per (ValidTimeUtc, LeadHours) — so the parquet has 4-5
rows per key, NOT one. ``predict_3f`` was reading the parquet without
deduping, then left-merging it against ``df_lead`` (24 rows per lead).
The merge multiplied 24 → 120, and the next numpy op
(``mixed_quantile``, etc.) raised
``operands could not be broadcast together with shapes (120,) (24,)``.
Surfaced by predict-3f run 26447110992 — three Membury stations, all
silently skipped, 0 rows written, R2 push failed for
``directory not found``.

Fix: ``load_bound_3a_pi`` now uses a DuckDB ROW_NUMBER ranking on
``ORDER BY PredictionMadeAtUtc DESC`` to keep only the freshest cycle's
row per (ValidTimeUtc, LeadHours). Same pattern Phase4bPredictCommand's
3o read uses (``WeatherBlend/src/WeatherBlend/Commands/Phase4bPredictCommand.cs``
line ~239).

Belt-and-braces in predict_one_station: a post-merge length assertion
fails loud + early if a future ``load_bound_3a_pi`` refactor ever ships
un-deduped rows again. The asserts and these tests are intentional
duplicates of the invariant.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from predict_3f import load_bound_3a_pi  # noqa: E402


def _write_3a_parquet(
    tmp_path: Path,
    station_slug: str,
    precip_3a_version: str,
    anchor: datetime,
    rows: list[dict],
) -> Path:
    """Build the parquet at the exact path load_bound_3a_pi expects."""
    target_dir = (
        tmp_path
        / "precipitation"
        / station_slug
        / f"model_version={precip_3a_version}"
        / f"date={anchor.date().isoformat()}"
    )
    target_dir.mkdir(parents=True)
    parquet = target_dir / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    return parquet


def test_load_bound_3a_pi_dedupes_duplicate_keys(tmp_path):
    """The exact scenario the regression triggered on: 5 intra-day
    cycles in the parquet, each with one row per (V, L). Result must
    be exactly one row per (V, L)."""
    anchor = datetime(2026, 5, 26)
    valid_time = datetime(2026, 5, 27, 12, 0, 0)
    cycles = [
        datetime(2026, 5, 26, 3, 12, 0),
        datetime(2026, 5, 26, 9, 12, 0),
        datetime(2026, 5, 26, 15, 12, 0),
        datetime(2026, 5, 26, 21, 12, 0),
        datetime(2026, 5, 26, 23, 30, 0),  # freshest
    ]
    # Same (V, L) key written 5 times, one per cycle.
    rows = [
        {
            "ValidTimeUtc": valid_time,
            "LeadHours": 24,
            "ProbWet": 0.1 * (i + 1),  # 0.1 / 0.2 / 0.3 / 0.4 / 0.5
            "PredictionMadeAtUtc": pmt,
        }
        for i, pmt in enumerate(cycles)
    ]
    _write_3a_parquet(tmp_path, "ea_test_station", "v2026-05-24_141845", anchor, rows)

    out, resolved = load_bound_3a_pi(tmp_path, "ea_test_station", "v2026-05-24_141845", anchor)
    assert resolved == "v2026-05-24_141845", "stamped version is on disk — no fallback expected"

    assert len(out) == 1, (
        f"load_bound_3a_pi must dedupe (V, L) keys before returning; "
        f"got {len(out)} rows for one key (5 cycles in the parquet)."
    )
    # Freshest cycle was the last one with ProbWet=0.5.
    assert out["ProbWet"].iloc[0] == pytest.approx(0.5), (
        "ROW_NUMBER ORDER BY PredictionMadeAtUtc DESC must keep the "
        "freshest cycle's value."
    )


def test_load_bound_3a_pi_preserves_distinct_keys(tmp_path):
    """Sanity inverse: keys that are genuinely distinct must all
    survive the dedup. 5 leads × 24 hours = 120 distinct (V, L)
    keys, plus one duplicate cycle for one of them. Expect 120
    rows out, with the duplicate having been collapsed."""
    anchor = datetime(2026, 5, 26)
    leads = [24, 48, 72, 96, 120]
    rows = []
    for lead in leads:
        for hour in range(24):
            rows.append({
                "ValidTimeUtc": datetime(2026, 5, 27, hour, 0, 0),
                "LeadHours": lead,
                "ProbWet": 0.42,
                "PredictionMadeAtUtc": datetime(2026, 5, 26, 9, 12, 0),
            })
    # Add one duplicate for the first (V, L) — must be collapsed.
    rows.append({
        "ValidTimeUtc": datetime(2026, 5, 27, 0, 0, 0),
        "LeadHours": 24,
        "ProbWet": 0.99,  # stale (older PMT)
        "PredictionMadeAtUtc": datetime(2026, 5, 26, 3, 12, 0),
    })
    _write_3a_parquet(tmp_path, "ea_test_station", "v2026-05-24_141845", anchor, rows)

    out, resolved = load_bound_3a_pi(tmp_path, "ea_test_station", "v2026-05-24_141845", anchor)
    assert resolved == "v2026-05-24_141845", "stamped version is on disk — no fallback expected"

    # 5 leads × 24 hours = 120 distinct keys (the +1 duplicate collapses).
    assert len(out) == 120, f"expected 120 rows after dedup, got {len(out)}"
    # No (V, L) appears more than once.
    grouped = out.groupby(["ValidTimeUtc", "LeadHours"]).size()
    assert grouped.max() == 1, "every (V, L) must be unique post-dedup"
    # Freshest (0.42 from the 9:12 cycle) wins over stale (0.99 from 3:12).
    first_row = out[(out["LeadHours"] == 24)
                    & (out["ValidTimeUtc"] == datetime(2026, 5, 27, 0, 0, 0))]
    assert first_row["ProbWet"].iloc[0] == pytest.approx(0.42), (
        "duplicate (V, L) must resolve to the freshest PMT's ProbWet."
    )


def test_load_bound_3a_pi_raises_when_parquet_missing(tmp_path):
    """3f without a stage-1 source is meaningless — must fail loud."""
    anchor = datetime(2026, 5, 26)
    with pytest.raises(FileNotFoundError, match="bound Phase 3a predictions parquet missing"):
        load_bound_3a_pi(tmp_path, "ea_nonexistent_station", "v2026-05-24_141845", anchor)


def test_load_bound_3a_pi_falls_back_when_stamp_is_stale(tmp_path):
    """Regression for the 2026-05-27 predict-3f failure: 3f's bundle
    stamped 3a v2026-05-24_141845 at train time but Sunday's auto-
    retrain produced a newer v2026-05-26 3a, and only the newer
    version has today's parquet on disk. Predict must fall back to
    the freshest unsuffixed sibling rather than black-holing every
    cycle until 3f next rebinds."""
    anchor = datetime(2026, 5, 27)
    valid_time = datetime(2026, 5, 28, 12, 0, 0)
    rows_newer = [{
        "ValidTimeUtc": valid_time,
        "LeadHours": 24,
        "ProbWet": 0.42,
        "PredictionMadeAtUtc": datetime(2026, 5, 27, 6, 12, 0),
    }]
    # ONLY the newer version has today's parquet; the stamped older
    # version has nothing at this anchor.
    _write_3a_parquet(tmp_path, "ea_test_station", "v2026-05-26_102758", anchor, rows_newer)

    out, resolved = load_bound_3a_pi(
        tmp_path, "ea_test_station", "v2026-05-24_141845", anchor)

    assert resolved == "v2026-05-26_102758", (
        "fallback must pick the freshest unsuffixed sibling with today's parquet"
    )
    assert len(out) == 1
    assert out["ProbWet"].iloc[0] == pytest.approx(0.42)


def test_load_bound_3a_pi_picks_latest_3a_even_when_stamp_is_on_disk(tmp_path):
    """The resolver ALWAYS picks the latest 3a-phase champion on disk,
    not the stamped one — the stamp is informational only after the
    2026-05-27 redesign. So if both the stamped version AND a newer
    3a have today's parquet, the newer wins; if 3f wanted the stamped
    version specifically it'd be tying itself to a frozen calibration."""
    anchor = datetime(2026, 5, 27)
    valid_time = datetime(2026, 5, 28, 12, 0, 0)
    older_stamp = "v2026-05-24_141845"
    newer_3a    = "v2026-05-26_102758"
    _write_3a_parquet(tmp_path, "ea_test_station", older_stamp, anchor,
        [{"ValidTimeUtc": valid_time, "LeadHours": 24, "ProbWet": 0.10,
          "PredictionMadeAtUtc": datetime(2026, 5, 27, 3, 0, 0)}])
    _write_3a_parquet(tmp_path, "ea_test_station", newer_3a, anchor,
        [{"ValidTimeUtc": valid_time, "LeadHours": 24, "ProbWet": 0.90,
          "PredictionMadeAtUtc": datetime(2026, 5, 27, 9, 0, 0)}])

    out, resolved = load_bound_3a_pi(tmp_path, "ea_test_station", older_stamp, anchor)
    assert resolved == newer_3a, "latest 3a on disk must win over the stamped version"
    assert out["ProbWet"].iloc[0] == pytest.approx(0.90), (
        "must read from the latest 3a's parquet, not the stamped one"
    )


def test_load_bound_3a_pi_rejects_3a_outside_age_window(tmp_path):
    """A 3a parquet sitting on disk from months ago shouldn't be
    silently picked up if today's 3a pipeline is broken — that would
    mask a real problem under stale data. The resolver enforces a
    MAX_3A_AGE_DAYS window on the version timestamp."""
    anchor = datetime(2026, 5, 27)
    # Version is from 2026-01 — well outside the 30-day window.
    too_old = "v2026-01-15_120000"
    valid_time = datetime(2026, 5, 28, 12, 0, 0)
    _write_3a_parquet(tmp_path, "ea_test_station", too_old, anchor,
        [{"ValidTimeUtc": valid_time, "LeadHours": 24, "ProbWet": 0.50,
          "PredictionMadeAtUtc": datetime(2026, 5, 27, 9, 0, 0)}])

    with pytest.raises(FileNotFoundError, match="within the last .* days"):
        load_bound_3a_pi(tmp_path, "ea_test_station", too_old, anchor)


def test_load_bound_3a_pi_skips_phase_suffixed_siblings_in_fallback(tmp_path):
    """Unsuffixed = 3a champion; _phase3c / _phase3d / _phase3o / _phase4a /
    _phase4b carry their own siblings under the same station tree but are
    NOT the 3a marginal predict_3f wants. The fallback must filter them
    out — picking a 3c parquet by accident would feed 3f the wrong π."""
    anchor = datetime(2026, 5, 27)
    valid_time = datetime(2026, 5, 28, 12, 0, 0)
    # Only a _phase3c sibling has today's parquet; no plain 3a entry exists.
    _write_3a_parquet(
        tmp_path, "ea_test_station", "v2026-05-26_102922_phase3c", anchor,
        [{"ValidTimeUtc": valid_time, "LeadHours": 24, "ProbWet": 0.99,
          "PredictionMadeAtUtc": datetime(2026, 5, 27, 6, 12, 0)}])
    with pytest.raises(FileNotFoundError, match="bound Phase 3a predictions parquet missing"):
        load_bound_3a_pi(tmp_path, "ea_test_station", "v2026-05-24_141845", anchor)
