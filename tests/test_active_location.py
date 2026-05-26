"""Regression for the 2026-05-26 silent-zero-rows bug in
``scripts/_shared.build_features_via_duckdb``.

Phase B (2026-05-12) made ``train_4a`` / ``train_5a`` aware of
``WB_LOCATION`` via an ``ACTIVE_LOCATION = os.environ.get("WB_LOCATION",
LOCATION)`` hoist at the top of each script. But the underlying
``_shared`` helper ``build_features_via_duckdb`` kept querying the
module-level ``LOCATION`` constant in all three of its SQL
``LocationName`` filters (hourly_truth, latest forecasts in the main
builder, and the synoptic add-on). The CI matrix's
``WB_LOCATION=membury_devon`` job therefore filtered to
``LocationName='bonehill_rocks'`` and silently returned 0 rows from
DuckDB for every (Membury station, lead) cell.

Surfaced by the first live Membury 3f training attempt (retrain-python
run 26444751915 on 2026-05-26) — train_3f reported "loaded 0 rows
(wet 0.0%)" for every (station, lead) pair and the workflow exited
with "0 bundles written". Would have hit Membury 4a/5a the same way
if either had ever been dispatched in production before this date.

Fix (commit 4586e34): hoist ``ACTIVE_LOCATION`` at module load of
``_shared.py`` so it picks up ``WB_LOCATION`` from the matrix step's
env block; substitute it into the three SQL filters.

These tests pin both the env-aware override path and the legacy
fallback so the next refactor that touches that resolution can't
silently flip Membury back to a no-rows state.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))


def _reload_shared(wb_location: str | None) -> object:
    """Reload scripts/_shared.py under a defined WB_LOCATION state.

    ACTIVE_LOCATION is resolved at module-load time (build_features_via_duckdb
    closes over it via an f-string substitution) so testing env-aware
    behaviour requires a fresh import after the env var is set/cleared.
    """
    if wb_location is None:
        os.environ.pop("WB_LOCATION", None)
    else:
        os.environ["WB_LOCATION"] = wb_location
    sys.modules.pop("_shared", None)
    return importlib.import_module("_shared")


def test_active_location_picks_up_wb_location_membury():
    """The exact scenario the CI matrix sets for the Membury job —
    must resolve away from the legacy bonehill_rocks default."""
    shared = _reload_shared("membury_devon")
    assert shared.ACTIVE_LOCATION == "membury_devon"


def test_active_location_picks_up_wb_location_arbitrary_slug():
    """Future-proofing: if a third location is added to phases.yaml,
    the resolution must reflect whatever WB_LOCATION carries — no
    hardcoded allow-list."""
    shared = _reload_shared("hypothetical_new_location")
    assert shared.ACTIVE_LOCATION == "hypothetical_new_location"


def test_active_location_falls_back_to_legacy_when_env_unset():
    """Local-dev runs (no WB_LOCATION in the shell) must still land
    on the legacy bonehill_rocks default — that's what every test
    fixture and existing notebook assumes."""
    shared = _reload_shared(None)
    # Don't hardcode "bonehill_rocks" — use the src.data constant so
    # this test survives a future rename of the legacy default.
    from src.data import LOCATION
    assert shared.ACTIVE_LOCATION == LOCATION


def test_sql_filter_strings_substitute_active_location_not_legacy():
    """End-to-end defense: even if a future refactor accidentally reverts
    one of the three SQL sites to '{LOCATION}', the SQL string would
    contain 'bonehill_rocks' under a Membury WB_LOCATION. Inspect the
    actual SQL produced via DuckDB string interpolation.

    We can't easily snapshot the SQL without executing the function,
    but we can assert the substitution variable bound into the
    f-string is ACTIVE_LOCATION (since string-templated SQL is the
    bug surface)."""
    shared = _reload_shared("membury_devon")
    import inspect
    source = inspect.getsource(shared.build_features_via_duckdb)
    assert "{LOCATION}" not in source, (
        "build_features_via_duckdb still uses the module-level LOCATION "
        "constant in its SQL filter — this is the exact substitution that "
        "silently filtered Membury data to zero rows on 2026-05-26. "
        "Use ACTIVE_LOCATION instead."
    )
    assert "{ACTIVE_LOCATION}" in source, (
        "build_features_via_duckdb must substitute ACTIVE_LOCATION (env-aware) "
        "into its SQL LocationName filter."
    )

    # And the synoptic add-on shares the same trap.
    syn_source = inspect.getsource(shared.add_synoptic_features)
    assert "{LOCATION}" not in syn_source
    assert "{ACTIVE_LOCATION}" in syn_source
