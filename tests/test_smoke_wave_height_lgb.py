"""End-to-end smoke tests for wave_height_lgb (train + predict) — the Sennen
significant-wave-height blender (quantile-LGB + split-CQR, any-lead v1,
Sennen sea-state Phase 2, 2026-06-12).

Goal: catch the wiring / typing / env-var / SQL-shape class of bugs locally
before a real retrain — the sync_train_data pull declarations (marine tree +
era5_ocean truth + wave_height manifest), bundle layout, conformal scalar,
RetrainGuard summary, champion manifest promotion, and the predict output's
schema including the Sea state tab's pass-through columns (tide / swell /
SST), the anchored-replay reference-time semantics, and the band invariants.

Runtime budget: ~20-40s for train (3 boosters on a 220-day fixture — the
trainer refuses <5000 joined rows, so the fixture is deliberately longer
than the wind smokes' 30 days), ~5s for predict.

Non-goal: skill testing. The fixture is a shared sinusoid + noise.
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from _smoke_fixtures import (  # noqa: E402
    add_decoy_bundles,
    make_marine_tree,
    make_wave_truth_tree,
    run_sync_train_data,
)

LOCATION = "sennen_cove"
TRAIN_START = datetime(2024, 1, 1)
TRAIN_DAYS = 220            # 220×24 = 5,280 rows ≥ the trainer's 5,000 floor
PREDICT_ANCHOR = TRAIN_START + timedelta(days=TRAIN_DAYS)


@pytest.fixture
def smoke_root(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHERBLEND_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("WB_LOCATION", LOCATION)
    import src.data
    importlib.reload(src.data)
    return tmp_path


@pytest.fixture
def trained_bundle(smoke_root):
    """Fixtures land in a fake-R2 dir; run_sync_train_data (phase csv
    ``wave_height_lgb`` — the SAME declaration retrain-python.yml resolves)
    copies them into the data root, so a missing pull declaration fails here
    rather than in a Sunday retrain. Then train_wave_pi.main() end-to-end."""
    fake_r2 = smoke_root / "fake-r2"
    fake_r2_data = fake_r2 / "data"
    make_marine_tree(fake_r2_data, LOCATION, TRAIN_START, n_days=TRAIN_DAYS,
                     kind="hist_forecast")
    make_wave_truth_tree(fake_r2_data, LOCATION, TRAIN_START, n_days=TRAIN_DAYS)
    run_sync_train_data(
        location=LOCATION, phases="wave_height_lgb",
        r2_source=fake_r2, local_root=smoke_root,
    )

    import train_wave_pi
    importlib.reload(train_wave_pi)
    sys_argv = sys.argv
    try:
        sys.argv = [
            "train_wave_pi",
            "--location", LOCATION,
            "--models-root", str(smoke_root / "models"),
        ]
        try:
            train_wave_pi.main()
        except SystemExit as e:
            if e.code not in (None, 0):
                raise AssertionError(
                    f"train_wave_pi.main exited with code {e.code} on the smoke fixture"
                ) from e
    finally:
        sys.argv = sys_argv

    bundle_parent = smoke_root / "models" / "wave_height" / LOCATION
    versions = sorted([p for p in bundle_parent.iterdir() if p.is_dir()])
    assert versions, f"no wave_height_lgb bundle written under {bundle_parent}"
    real = versions[-1]
    add_decoy_bundles(bundle_parent, "wave_height_lgb")
    return real


class TestTrainWaveHeightLgbSmoke:
    def test_bundle_has_boosters_and_sidecars(self, trained_bundle):
        for name in ("point_model.txt", "q_lo.txt", "q_hi.txt",
                     "feature_schema.json", "calibration.json",
                     "training_metadata.json", "training_summary.json"):
            assert (trained_bundle / name).exists(), f"bundle missing {name}"

    def test_metadata_stamps_phase_location_and_truth(self, trained_bundle):
        metadata = json.loads((trained_bundle / "training_metadata.json").read_text())
        assert metadata["Phase"] == "wave_height_lgb"
        assert metadata["LocationName"] == LOCATION
        assert metadata["TruthSource"] == "era5_ocean"
        assert metadata["FeatureCount"] == 51   # the RICH set — drift breaks predict's X

    def test_calibration_carries_split_cqr_scalar(self, trained_bundle):
        calibration = json.loads((trained_bundle / "calibration.json").read_text())
        assert calibration["coverage"] == 0.90
        assert calibration["conformal_q"] is not None
        assert calibration["lead_mode"] == "any-lead-v1"
        # On a well-behaved fixture the band must roughly cover — a wildly
        # low number means the quantile heads or the CQR step are broken.
        assert calibration["cal_band_coverage"] > 0.7

    def test_manifest_promoted_as_champion(self, trained_bundle):
        smoke_root = trained_bundle.parents[3]
        manifest = json.loads(
            (smoke_root / "models" / "wave_height" / "MANIFEST.json").read_text())
        entry = manifest["Stations"][LOCATION]
        # First (only) model of its target — champion owns Current, unlike
        # the wind_speed_lgb challenger contract.
        assert "_wave_height_lgb" in (entry.get("Current") or ""), entry


class TestPredictWaveHeightLgbSmoke:
    def test_predict_writes_rows_with_band_and_passthrough(self, trained_bundle):
        smoke_root = trained_bundle.parents[3]
        bundle_name = trained_bundle.name

        # Live runs covering the anchor — predict must pick the newest run
        # and, because --anchor is passed, judge staleness/LeadHours against
        # the ANCHOR not the wall clock (the anchored-replay bug this smoke
        # was written to pin down).
        make_marine_tree(smoke_root, LOCATION,
                         PREDICT_ANCHOR - timedelta(days=1), n_days=2, kind="live")

        import predict_wave_pi
        importlib.reload(predict_wave_pi)
        sys_argv = sys.argv
        try:
            sys.argv = [
                "predict_wave_pi",
                "--location", LOCATION,
                "--anchor",   PREDICT_ANCHOR.date().isoformat(),
                "--models-root",      str(smoke_root / "models"),
                "--predictions-root", str(smoke_root / "predictions"),
            ]
            try:
                predict_wave_pi.main()
            except SystemExit as e:
                if e.code not in (None, 0):
                    raise AssertionError(
                        f"predict_wave_pi.main exited with code {e.code} on the smoke fixture"
                    ) from e
        finally:
            sys.argv = sys_argv

        out_path = (
            smoke_root / "predictions" / "wave_height"
            / f"model_version={bundle_name}"
            / f"date={PREDICT_ANCHOR.date().isoformat()}"
            / "predictions.parquet"
        )
        assert out_path.exists(), f"no predictions.parquet at {out_path}"

        out = pd.read_parquet(out_path)
        assert len(out) > 0
        # The Sea state tab's full contract: blend + band + the pass-through
        # site extras (tide is the one Harry asked for by name).
        for col in (
            "LocationName", "Element", "ModelVersion", "PredictionMadeAtUtc",
            "ValidTimeUtc", "LeadHours", "BlendValue",
            "BandLoM", "BandHiM", "ConformalQ",
            "WavePeriodS", "WaveDirectionDeg",
            "SwellHeightM", "SwellPeriodS", "SwellDirectionDeg",
            "SecondarySwellHeightM", "SecondarySwellPeriodS",
            "TideHeightMsl", "SeaSurfaceTempC",
        ):
            assert col in out.columns, f"predictions parquet missing column {col}"

        assert (out["Element"] == "wave_height").all()
        assert (out["ModelVersion"] == bundle_name).all()
        assert (out["LeadHours"] >= 0).all()
        # Pass-through must actually carry values (the 2026-06-11 mapping
        # collision shipped all-NaN tide on the first cut).
        assert out["TideHeightMsl"].notna().any(), "tide pass-through is all-NaN"
        assert out["SwellPeriodS"].notna().any(), "swell-period pass-through is all-NaN"
        # Band invariants: lo >= 0 and lo <= hi.
        assert (out["BandLoM"] >= 0).all()
        assert (out["BandLoM"] <= out["BandHiM"] + 1e-9).all()
