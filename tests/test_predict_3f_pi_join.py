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

    out = load_bound_3a_pi(tmp_path, "ea_test_station", "v2026-05-24_141845", anchor)

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

    out = load_bound_3a_pi(tmp_path, "ea_test_station", "v2026-05-24_141845", anchor)

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
