"""Bit-equivalence check: the Python rich-oro feature builders in
`scripts/_shared.py` must produce features identical to the C# dump from
WeatherBlend's `dump-oro-features` command (which itself wraps
PrecipRichFeatureBuilder + PrecipRichOroFeatureBuilder).

Why this exists: Phase A of the rich-per-station 4a productionisation kept the
canonical rich-oro feature spec in C# (used by 3o) while adding a Python
twin (used by 4a). Without a contract test, the two implementations would
drift silently — a future C# spec change would silently produce different 4a
training data than what 3o sees.

What this checks: load an EXISTING C# dump from
`WB/data/scratch/oro_dump/{slug}_lead{N}h_rich-oro.parquet` (produced by
yesterday's `dump-oro-features` runs), run the Python builders for the same
(station, lead, min_valid_time), inner-join on `ValidTimeUtc`, assert all
68 feature columns match within float epsilon.

The dumps used here were produced 2026-05-29 by the bake-off; if they ever
move or get pruned, the test will skip (we don't want to fail just because
the snapshot is gone, since this is a local-only test for the moment).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402

DUMP_ROOT = WEATHERBLEND_DATA_ROOT / "scratch" / "oro_dump"

# Tight tolerance — both implementations use float64 throughout. The dump
# packs as float32 at the end, so we expect agreement to ~1e-5.
ATOL = 1e-4
RTOL = 1e-5

# Bellever Dartmoor lead 24h is the canonical fixture — dumped 2026-05-29,
# 20388 rows × 68 features, station_index = 0.
FIXTURE_STATION_SLUG = "ea_bellever_dartmoor"
FIXTURE_STATION_FRIENDLY = "Bellever Dartmoor"
FIXTURE_LEAD = 24
FIXTURE_MIN_VALID_TIME = datetime(2024, 1, 1)
FIXTURE_STATION_INDEX = 0  # matches the C# dump's --stations order in the bake-off


def _dump_exists() -> bool:
    return (DUMP_ROOT / f"{FIXTURE_STATION_SLUG}_lead{FIXTURE_LEAD}h_rich-oro.parquet").exists()


@pytest.mark.skipif(not _dump_exists(),
                    reason="C# rich-oro dump fixture not present; re-run "
                           "`dotnet run -- dump-oro-features ...` to regenerate")
def test_rich_oro_python_matches_csharp_bellever_24h(monkeypatch):
    """Per-cell + per-feature comparison of Python builder vs C# dump on the
    Bellever 24h fixture. Diagnostic on failure lists the worst-mismatched
    columns + sample row indices so debugging targets the real disagreement.

    Explicitly pins WEATHERBLEND_DATA_ROOT + WB_LOCATION to production
    values + reloads src.data and _shared. Other smoke tests in the same
    pytest invocation monkeypatch these to tmpdir values and reload —
    after their teardown the env is clean but our module-level constants
    are still bound to the (now-deleted) tmpdir. Pinning + reloading here
    makes this test order-independent.
    """
    monkeypatch.setenv("WEATHERBLEND_DATA_ROOT", r"C:/Projects/Weather/WeatherBlend/data")
    monkeypatch.setenv("WB_LOCATION", "bonehill_rocks")
    import importlib
    import src.data
    importlib.reload(src.data)
    import _shared
    importlib.reload(_shared)
    from _shared import (  # noqa: E402
        RICH_ORO_FEATURE_NAMES,
        build_rich_features_via_duckdb,
        compose_v1_terrain_block,
    )

    # ---- Load the C# dump as the reference -------------------------------
    parquet_path = DUMP_ROOT / f"{FIXTURE_STATION_SLUG}_lead{FIXTURE_LEAD}h_rich-oro.parquet"
    schema_path  = DUMP_ROOT / f"{FIXTURE_STATION_SLUG}_lead{FIXTURE_LEAD}h_rich-oro.schema.json"
    sidecar = json.loads(schema_path.read_text())
    csharp_feat_names = list(sidecar["FeatureNames"])
    assert csharp_feat_names == RICH_ORO_FEATURE_NAMES, (
        "C# dump's FeatureNames disagrees with Python RICH_ORO_FEATURE_NAMES order; "
        "spec drift before we even started comparing values."
    )
    csharp_narrow = pd.read_parquet(parquet_path)
    csharp_features = np.asarray(csharp_narrow["Features"].tolist(), dtype="float64")
    csharp_wide = pd.DataFrame(csharp_features, columns=csharp_feat_names)
    csharp_wide["ValidTimeUtc"] = pd.to_datetime(csharp_narrow["ValidTimeUtc"].values)

    # ---- Build the Python equivalent ------------------------------------
    py_rich = build_rich_features_via_duckdb(
        FIXTURE_STATION_FRIENDLY, FIXTURE_LEAD,
        min_valid_time=FIXTURE_MIN_VALID_TIME,
    )
    assert len(py_rich) > 0, "Python rich builder returned empty df — check fixture data"
    py_full = compose_v1_terrain_block(
        FIXTURE_STATION_SLUG, FIXTURE_STATION_INDEX, FIXTURE_LEAD,
        py_rich, min_valid_time=FIXTURE_MIN_VALID_TIME,
    )
    assert len(py_full) > 0, "Python oro merge returned empty df — check aux NWP availability"

    # ---- Inner-join on ValidTimeUtc -------------------------------------
    merged = csharp_wide.merge(
        py_full[["ValidTimeUtc"] + RICH_ORO_FEATURE_NAMES],
        on="ValidTimeUtc",
        suffixes=("_cs", "_py"),
        how="inner",
    )
    assert len(merged) > 0, "no overlapping ValidTimeUtc between C# dump and Python build"
    assert len(merged) >= 0.95 * min(len(csharp_wide), len(py_full)), (
        f"low join coverage: cs={len(csharp_wide)} py={len(py_full)} merged={len(merged)} — "
        "row filter divergence?"
    )

    # ---- Per-feature comparison -----------------------------------------
    failures = []
    for col in RICH_ORO_FEATURE_NAMES:
        cs = merged[f"{col}_cs"].to_numpy(dtype="float64")
        py = merged[f"{col}_py"].to_numpy(dtype="float64")
        # Treat NaN-aligned as equal: count rows where one is NaN and the other isn't.
        both_nan = np.isnan(cs) & np.isnan(py)
        only_one_nan = np.isnan(cs) ^ np.isnan(py)
        if only_one_nan.any():
            n = int(only_one_nan.sum())
            failures.append((col, "nan-mismatch", n, None, None))
            continue
        finite_mask = ~both_nan
        if not finite_mask.any():
            continue
        diff = np.abs(cs[finite_mask] - py[finite_mask])
        # rtol-based for non-zero, atol for near-zero — same pattern as np.allclose
        thresh = ATOL + RTOL * np.abs(cs[finite_mask])
        bad = diff > thresh
        if bad.any():
            n_bad = int(bad.sum())
            max_diff = float(diff[bad].max())
            worst_idx = int(np.argmax(diff))
            failures.append((col, "value-mismatch", n_bad, max_diff, worst_idx))

    if failures:
        msg = (f"\nC# vs Python rich-oro feature mismatch over {len(merged)} aligned rows:\n"
               + "\n".join(f"  {col}: kind={kind} n_bad={n} max_diff={d} idx={i}"
                           for col, kind, n, d, i in failures))
        raise AssertionError(msg)
