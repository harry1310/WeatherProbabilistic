"""Run-it-and-fail guard for the predict data-pull path.

scripts/sync_train_data.sh MODE=predict must pull the feature trees + the
latest model bundle a phase reads at predict time, and skip the train-only
promote MANIFEST. This drives a fake-R2 through the REAL script (same one the
predict-*.yml workflows now call) and asserts the right trees land — so a
predict-mode under-declaration fails at PR time, not in production (the
2026-05-31 predict-4a failure: rainfall, then orographic, missing).

Complements test_predict_workflows.py (static: workflows route through the
script) — this one is dynamic: the script actually pulls what predict needs.
Cheap: rclone copies from a local fake-R2, no BART, no module reloads.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ — for _smoke_fixtures

from _smoke_fixtures import run_sync_train_data  # noqa: E402


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")


def _fake_r2_precip(root: Path) -> Path:
    """fake-R2 with the trees a precip phase reads + two phase4a bundle
    versions (to prove 'latest' selection) + a promote MANIFEST."""
    d = root / "data"
    _touch(d / "forecasts/location=bonehill_rocks/model=gfs_seamless/date=2026-01-01/run=00.parquet")
    _touch(d / "truth/rainfall/ea_bellever_dartmoor/r.parquet")
    _touch(d / "static/orographic/ea_bellever_dartmoor.json")
    _touch(d / "models/precipitation/ea_bellever_dartmoor/v2025-01-01_phase4a/state.rds")
    _touch(d / "models/precipitation/ea_bellever_dartmoor/v2026-05-01_phase4a/state.rds")  # newest
    _touch(d / "models/precipitation/MANIFEST.json")
    return root


def _files(dest: Path) -> set[str]:
    return {str(p.relative_to(dest)).replace("\\", "/") for p in dest.rglob("*") if p.is_file()}


def test_predict_mode_pulls_features_and_latest_bundle_no_manifest(tmp_path):
    fake_r2 = _fake_r2_precip(tmp_path / "r2")
    dest = tmp_path / "dest"
    run_sync_train_data(location="bonehill_rocks", phases="4a",
                        r2_source=fake_r2, local_root=dest, mode="predict")
    f = _files(dest)
    # The rich-oro builder's feature trees — all three were the 2026-05-31 gaps.
    assert any(s.startswith("forecasts/") for s in f), f
    assert any(s.startswith("truth/rainfall/") for s in f), f
    assert any(s.startswith("static/orographic/") for s in f), f
    # Predict needs the LATEST bundle, and must not need the older one.
    assert any("v2026-05-01_phase4a/state.rds" in s for s in f), f
    # Predict must NOT pull the promote-target MANIFEST (train-only).
    assert not any(s.endswith("precipitation/MANIFEST.json") for s in f), f


def test_train_mode_pulls_manifest_not_bundle(tmp_path):
    """Back-compat / inverse: train still pulls the MANIFEST and pulls no
    model bundle (train mints fresh)."""
    fake_r2 = _fake_r2_precip(tmp_path / "r2")
    dest = tmp_path / "dest"
    run_sync_train_data(location="bonehill_rocks", phases="4a",
                        r2_source=fake_r2, local_root=dest, mode="train")
    f = _files(dest)
    assert any(s.endswith("precipitation/MANIFEST.json") for s in f), f
    assert not any("_phase4a/state.rds" in s for s in f), f


def test_predict_mode_wind_mvn_pulls_bundle_and_orographic(tmp_path):
    """Generalises beyond 4a: wind_mvn predict pulls its wind_direction bundle
    + orographic + forecasts, and not the MIDAS truth label."""
    root = tmp_path / "r2"; d = root / "data"
    _touch(d / "forecasts/location=bonehill_rocks/model=gfs_seamless/date=2026-01-01/run=00.parquet")
    _touch(d / "static/orographic/bonehill_rocks.json")
    _touch(d / "models/wind_direction/bonehill_rocks/v2026-05-01_wind_mvn/model.pt")
    _touch(d / "truth/midas/raw/midas-open_x_01383_y.csv")
    dest = tmp_path / "dest"
    run_sync_train_data(location="bonehill_rocks", phases="wind_mvn",
                        r2_source=root, local_root=dest, mode="predict")
    f = _files(dest)
    assert any("v2026-05-01_wind_mvn/model.pt" in s for s in f), f
    assert any(s.startswith("static/orographic/") for s in f), f
    assert any(s.startswith("forecasts/") for s in f), f
    assert not any(s.startswith("truth/midas/") for s in f), f  # MIDAS is train-only label
