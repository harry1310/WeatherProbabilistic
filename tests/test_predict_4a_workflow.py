"""Workflow/feature-dependency guard for predict-4a.

The 2026-05-31 predict-4a failure: commit 3641d44 swapped 4a to per-station
rich+oro (68-feature) BART, whose LIVE feature builder
(``_shared.build_rich_oro_features_live`` -> ``_load_hourly_rain``) reads EA
rainfall as INPUT features (``ea_rain_prev_24h_mm`` etc.). But ``predict-4a.yml``
still only pulled forecasts + bundles ("NO truth pull -- predict has no
labels"), so the first predict after the first rich+oro bundle was minted
crashed on an empty ``data/truth/rainfall/**/*.parquet`` glob.

``test_smoke_4a`` didn't catch it because it FABRICATES rainfall
(``make_rainfall_truth``) -- the code path passes with data present; nothing
asserted the production workflow actually PULLS that tree. This guard closes
that gap: if predict_4a's rich-oro path reads rainfall, predict-4a.yml must
pull it.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICT_4A_YML = REPO_ROOT / ".github" / "workflows" / "predict-4a.yml"
SHARED_PY = REPO_ROOT / "scripts" / "_shared.py"


def test_rich_oro_live_builder_reads_rainfall():
    """Sanity-anchor: the live rich-oro builder really does read the rainfall
    tree (so the workflow assertion below is meaningful, not vacuous).
    _shared.py builds the glob from path components ("truth" / "rainfall")."""
    shared = SHARED_PY.read_text(encoding="utf-8")
    assert "_load_hourly_rain" in shared
    assert '"rainfall"' in shared, (
        "_shared.py no longer references the rainfall tree -- if 4a stopped "
        "using antecedent-rain features, relax the workflow guard below too."
    )


def test_predict_4a_workflow_pulls_all_rich_oro_inputs():
    """predict-4a.yml must pull every tree the rich+oro live builder reads.
    The 2026-05-31 failure hit these one at a time (rainfall, then orographic),
    so assert the FULL set here. The rich-oro dependency set mirrors phase 3o:
    forecasts + EA rainfall + static orographic + saved bundles."""
    text = PREDICT_4A_YML.read_text(encoding="utf-8")
    required = {
        "data/truth/rainfall": "antecedent-rain features (_load_hourly_rain)",
        "data/static/orographic": "oro_* terrain features (per-station {slug}.json)",
        "data/forecasts/location=": "NWP forecast features",
        "data/models/precipitation": "the saved BART bundles to load",
    }
    missing = [f"{tree}  ({why})" for tree, why in required.items() if tree not in text]
    assert not missing, (
        "predict-4a.yml is missing rclone pulls the rich+oro predict path needs "
        "(build_rich_oro_features_live). Each missing tree crashes predict_4a "
        "live. Missing:\n  " + "\n  ".join(missing)
    )
