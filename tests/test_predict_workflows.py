"""Guard: every predict-*.yml pulls its data via sync_train_data.sh MODE=predict.

Background: the 2026-05-31 predict-4a failures were caused by predict workflows
hand-rolling their own rclone pulls, which silently drifted from what the
feature builders actually read (missing rainfall, then orographic). The fix is
structural: predict pulls come from the SAME single source of truth as train
(scripts/sync_train_data.sh), which the smoke validates. This guard stops any
predict workflow regressing back to a hand-rolled pull list.

It is intentionally cheap (static YAML/text inspection). The deeper, run-it-
and-fail guarantee is the predict-mode smoke (materialise only what MODE=predict
declares, then run the predict entrypoint)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WF_DIR = REPO_ROOT / ".github" / "workflows"
SHARED_PY = REPO_ROOT / "scripts" / "_shared.py"

# Every predict workflow -> the phase id it must hand sync_train_data.sh.
PREDICT_WORKFLOWS = {
    "predict-4a.yml": "4a",
    "predict-3f.yml": "3f",
    "predict-wind-direction.yml": "wind_mvn",
}


def test_predict_workflows_route_through_sync_train_data():
    problems = []
    for fname, phase in PREDICT_WORKFLOWS.items():
        path = WF_DIR / fname
        if not path.exists():
            problems.append(f"{fname}: workflow not found")
            continue
        text = path.read_text(encoding="utf-8")
        if "sync_train_data.sh" not in text:
            problems.append(
                f"{fname}: does not call scripts/sync_train_data.sh "
                f"(predict pulls must use the single source of truth, not hand-rolled rclone)"
            )
        if "MODE=predict" not in text:
            problems.append(f"{fname}: missing MODE=predict on the sync_train_data.sh call")
        # The phase id must be the last arg of the call, under any of the
        # location forms the workflows use (literal slug or $LOC).
        if not any(
            f"sync_train_data.sh {loc} {phase}" in text
            for loc in ("bonehill_rocks", "membury_devon", '"$LOC"')
        ):
            problems.append(f"{fname}: sync_train_data.sh is not invoked with phase '{phase}'")
    assert not problems, "predict workflow data-pull regressions:\n  " + "\n  ".join(problems)


def test_no_predict_workflow_hand_rolls_a_feature_tree():
    """Belt-and-braces: a predict workflow that re-introduces a raw rclone of a
    feature tree (rainfall/orographic/forecasts) is drifting from the single
    source again. config.yaml + 3f's runtime-pinned 3a predictions are the only
    inline data rclones a predict workflow is allowed."""
    problems = []
    for fname in PREDICT_WORKFLOWS:
        text = (WF_DIR / fname).read_text(encoding="utf-8")
        for tree in ("data/truth/rainfall", "data/static/orographic", "data/forecasts/location="):
            if f"r2:${{R2_BUCKET}}/{tree}" in text:
                problems.append(
                    f"{fname}: hand-rolls an rclone for '{tree}' — route it through "
                    f"sync_train_data.sh MODE=predict instead"
                )
    assert not problems, "\n  ".join(problems)


def test_shared_rich_oro_builder_still_reads_rainfall_and_orographic():
    """Anchor: if the rich-oro builder stops reading rainfall/orographic, the
    sync-script predict declarations (and these guards) should be revisited."""
    shared = SHARED_PY.read_text(encoding="utf-8")
    assert "_load_hourly_rain" in shared
    assert '"rainfall"' in shared
    assert "orographic" in shared
