#!/usr/bin/env bash
# Quick smoke test that the new progress wiring actually emits something
# parseable into reports/phase3_progress.log. Use after editing model files.
set -e
cd "$(dirname "$0")/.."
PYTHONUNBUFFERED=1 PYTHONUTF8=1 .venv/Scripts/python.exe -u \
    scripts/run_phase3.py --subset 500 \
    > reports/phase3_subset_smoke.log \
    2> reports/phase3_subset_smoke_progress.log
echo "Done. Tail both logs:"
tail -5 reports/phase3_subset_smoke.log
echo "---progress---"
tail -10 reports/phase3_subset_smoke_progress.log
