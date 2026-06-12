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


# Anchor every predict-mode call pins its pull window to. Fixture
# partitions are dated relative to this (the production default is
# "today", which would exclude these absolute-dated fixtures).
ANCHOR = "2026-01-02"


def _fake_r2_precip(root: Path) -> Path:
    """fake-R2 with the trees a precip phase reads + two phase4a bundle
    versions (to prove 'latest' selection) + a promote MANIFEST. Rainfall
    uses the REAL production tree shape (location=/station=/date= — the
    flat fixture pre-2026-06-11 couldn't catch window/scoping bugs)."""
    d = root / "data"
    _touch(d / "forecasts/location=bonehill_rocks/model=gfs_seamless/date=2026-01-01/run=00.parquet")
    _touch(d / "truth/rainfall/location=bonehill_rocks/station=Bellever Dartmoor/date=2026-01-01/rainfall.parquet")
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
                        r2_source=fake_r2, local_root=dest, mode="predict",
                        predict_pull_anchor=ANCHOR)
    f = _files(dest)
    # The rich-oro builder's feature trees — all three were the 2026-05-31 gaps.
    assert any(s.startswith("forecasts/") for s in f), f
    assert any(s.startswith("truth/rainfall/") for s in f), f
    assert any(s.startswith("static/orographic/") for s in f), f
    # Predict needs the LATEST bundle, and must not need the older one.
    assert any("v2026-05-01_phase4a/state.rds" in s for s in f), f
    # Predict must NOT pull the promote-target MANIFEST (train-only).
    assert not any(s.endswith("precipitation/MANIFEST.json") for s in f), f


def test_predict_mode_window_excludes_old_partitions_train_keeps_them(tmp_path):
    """Regression for the 2026-06-11 over-pull fix: predict-mode pulls only
    [anchor-14d, anchor+6d] of date= partitions (the full-history pull was
    3-5 min of the 41k-file forecast tree per predict workflow per cycle);
    train mode still pulls everything. Out-of-window forecast AND rainfall
    partitions pin both trees' filters; an in-window forward (valid-dated)
    partition pins the +6d headroom."""
    root = tmp_path / "r2"; d = root / "data"
    # In-window (anchor 2026-01-02): -1d, +5d. Out-of-window: -60d.
    _touch(d / "forecasts/location=bonehill_rocks/model=gfs_seamless/date=2026-01-01/run=00.parquet")
    _touch(d / "forecasts/location=bonehill_rocks/model=gfs_seamless/date=2026-01-07/run=00.parquet")
    _touch(d / "forecasts/location=bonehill_rocks/model=gfs_seamless/date=2025-11-03/run=00.parquet")
    _touch(d / "truth/rainfall/location=bonehill_rocks/station=Bellever Dartmoor/date=2026-01-01/rainfall.parquet")
    _touch(d / "truth/rainfall/location=bonehill_rocks/station=Bellever Dartmoor/date=2025-11-03/rainfall.parquet")
    _touch(d / "static/orographic/ea_bellever_dartmoor.json")
    _touch(d / "models/precipitation/ea_bellever_dartmoor/v2026-05-01_phase4a/state.rds")
    dest = tmp_path / "dest"
    run_sync_train_data(location="bonehill_rocks", phases="4a",
                        r2_source=root, local_root=dest, mode="predict",
                        predict_pull_anchor=ANCHOR)
    f = _files(dest)
    assert any("date=2026-01-01" in s and s.startswith("forecasts/") for s in f), f
    assert any("date=2026-01-07" in s for s in f), f                  # +5d headroom
    assert not any("date=2025-11-03" in s for s in f), f              # out of window, BOTH trees
    assert any("date=2026-01-01" in s and "rainfall" in s for s in f), f

    # Train mode: full history, both trees.
    dest_train = tmp_path / "dest_train"
    _touch(d / "models/precipitation/MANIFEST.json")
    run_sync_train_data(location="bonehill_rocks", phases="4a",
                        r2_source=root, local_root=dest_train, mode="train")
    ft = _files(dest_train)
    assert any("date=2025-11-03" in s and s.startswith("forecasts/") for s in ft), ft
    assert any("date=2025-11-03" in s and "rainfall" in s for s in ft), ft


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
                        r2_source=root, local_root=dest, mode="predict", predict_pull_anchor=ANCHOR)
    f = _files(dest)
    assert any("v2026-05-01_wind_mvn/model.pt" in s for s in f), f
    assert any(s.startswith("static/orographic/") for s in f), f
    assert any(s.startswith("forecasts/") for s in f), f
    assert not any(s.startswith("truth/midas/") for s in f), f  # MIDAS is train-only label


def test_predict_mode_wind_speed_lgb_pulls_latest_lgb_bundle_only(tmp_path):
    """wind_speed_lgb predict (the predict-wind.yml csv) pulls the
    LATEST *_wind_speed_lgb bundle from the shared models/wind tree, skipping
    both an older lgb bundle and the .NET champion `wind` bundles (plain
    v{ts}, no suffix) that live in the same per-location dir — plus the
    MIDAS-is-train-only + no-MANIFEST predict contracts."""
    root = tmp_path / "r2"; d = root / "data"
    _touch(d / "forecasts/location=bonehill_rocks/model=gfs_seamless/date=2026-01-01/run=00.parquet")
    _touch(d / "static/orographic/bonehill_rocks.json")
    _touch(d / "models/wind/bonehill_rocks/v2026-05-01_120000_wind_speed_lgb/q_med_lead_24h.txt")
    _touch(d / "models/wind/bonehill_rocks/v2026-06-01_120000_wind_speed_lgb/q_med_lead_24h.txt")  # newest lgb
    _touch(d / "models/wind/bonehill_rocks/v2026-06-05_120000/lead_24h.zip")  # .NET champion `wind` — not ours
    _touch(d / "models/wind/MANIFEST.json")
    _touch(d / "truth/midas/raw/midas-open_x_01383_y.csv")
    dest = tmp_path / "dest"
    run_sync_train_data(location="bonehill_rocks", phases="wind_mvn,wind_speed_lgb",
                        r2_source=root, local_root=dest, mode="predict", predict_pull_anchor=ANCHOR)
    f = _files(dest)
    assert any("v2026-06-01_120000_wind_speed_lgb/q_med_lead_24h.txt" in s for s in f), f
    assert not any("v2026-05-01_120000_wind_speed_lgb" in s for s in f), f
    assert not any("v2026-06-05_120000/" in s for s in f), f
    assert not any(s.endswith("models/wind/MANIFEST.json") for s in f), f
    assert not any(s.startswith("truth/midas/") for s in f), f


def test_predict_mode_survives_cells_without_the_phase(tmp_path):
    """Regression for the 2026-06-10 predict-4a failure (run 27264913136):
    models/precipitation holds EVERY location's gauges, and a cell with no
    phase4a bundle (each Membury/Sennen gauge) made pull_latest_bundle's
    grep exit 1 — under `set -euo pipefail` that killed the whole sync. A
    phase-less cell must be SKIPPED, and cells after it still pulled. Cells
    placed both before and after the real one pin skip-and-continue."""
    root = tmp_path / "r2"; d = root / "data"
    _touch(d / "forecasts/location=bonehill_rocks/model=gfs_seamless/date=2026-01-01/run=00.parquet")
    _touch(d / "truth/rainfall/location=bonehill_rocks/station=Bellever Dartmoor/date=2026-01-01/rainfall.parquet")
    _touch(d / "static/orographic/ea_bellever_dartmoor.json")
    _touch(d / "models/precipitation/ea_aaa_membury_no_4a/v2026-05-01_phase3c/model.zip")
    _touch(d / "models/precipitation/ea_bellever_dartmoor/v2026-05-01_phase4a/state.rds")
    _touch(d / "models/precipitation/ea_zzz_sennen_no_4a/v2026-05-01_phase3c/model.zip")
    dest = tmp_path / "dest"
    run_sync_train_data(location="bonehill_rocks", phases="4a",
                        r2_source=root, local_root=dest, mode="predict", predict_pull_anchor=ANCHOR)
    f = _files(dest)
    assert any("ea_bellever_dartmoor/v2026-05-01_phase4a/state.rds" in s for s in f), f
    assert not any("ea_aaa_membury_no_4a" in s for s in f), f
    assert not any("ea_zzz_sennen_no_4a" in s for s in f), f


def test_train_mode_wind_mvn_truth_is_per_location(tmp_path):
    """wind_mvn's train label is per-location (the 2026-06-12 Sennen ERA5
    decision, SENNEN_SEA_STATE_PLAN.md Phase 4): sennen_cove pulls the ERA5
    cell tree and NOT MIDAS; bonehill_rocks keeps MIDAS and pulls no ERA5.
    The declaration must mirror train_wind_mvn.TRUTH_SOURCE_BY_LOCATION."""
    root = tmp_path / "r2"; d = root / "data"
    for loc in ("bonehill_rocks", "sennen_cove"):
        _touch(d / f"forecasts/location={loc}/model=gfs_seamless/date=2026-01-01/run=00.parquet")
        _touch(d / f"static/orographic/{loc}.json")
        _touch(d / f"truth/era5/location={loc}/date=2026-01-01/data.parquet")
    _touch(d / "truth/midas/raw/midas-open_x_01383_y.csv")
    _touch(d / "models/wind_direction/MANIFEST.json")

    dest_sennen = tmp_path / "dest_sennen"
    run_sync_train_data(location="sennen_cove", phases="wind_mvn",
                        r2_source=root, local_root=dest_sennen, mode="train")
    f = _files(dest_sennen)
    assert any(s.startswith("truth/era5/location=sennen_cove/") for s in f), f
    assert not any(s.startswith("truth/midas/") for s in f), f
    assert any(s.endswith("wind_direction/MANIFEST.json") for s in f), f

    dest_bonehill = tmp_path / "dest_bonehill"
    run_sync_train_data(location="bonehill_rocks", phases="wind_mvn",
                        r2_source=root, local_root=dest_bonehill, mode="train")
    fb = _files(dest_bonehill)
    assert any(s.startswith("truth/midas/") for s in fb), fb
    assert not any(s.startswith("truth/era5/") for s in fb), fb


def test_train_mode_wind_speed_lgb_pulls_wind_manifest_and_midas(tmp_path):
    """Train-mode inverse: wind_speed_lgb pulls the wind MANIFEST (promote
    target) + MIDAS truth, and no model bundle (train mints fresh)."""
    root = tmp_path / "r2"; d = root / "data"
    _touch(d / "forecasts/location=bonehill_rocks/model=gfs_seamless/date=2026-01-01/run=00.parquet")
    _touch(d / "static/orographic/bonehill_rocks.json")
    _touch(d / "models/wind/bonehill_rocks/v2026-06-01_120000_wind_speed_lgb/q_med_lead_24h.txt")
    _touch(d / "models/wind/MANIFEST.json")
    _touch(d / "truth/midas/raw/midas-open_x_01383_y.csv")
    dest = tmp_path / "dest"
    run_sync_train_data(location="bonehill_rocks", phases="wind_speed_lgb",
                        r2_source=root, local_root=dest, mode="train")
    f = _files(dest)
    assert any(s.endswith("models/wind/MANIFEST.json") for s in f), f
    assert any(s.startswith("truth/midas/") for s in f), f
    assert not any("_wind_speed_lgb/q_med_lead_24h.txt" in s for s in f), f

