"""End-to-end smoke tests for wind_speed_lgb (train + predict) — the Python
quantile-LGB + cross-conformal CQR port of the .NET WindSpeedLgbBlender
(Option B cutover 2026-06-10).

Goal: catch the wiring / typing / env-var / SQL-shape class of bugs locally
before a real retrain — bundle layout, conformal calibration scalars,
RetrainGuard summary, challenger manifest promotion, and the predict output's
ElementPredictionRow schema + CQR band sidecar.

Runtime budget: ~30-60s for train (3 leads × [3 early-stopped quantile fits +
K=10 cross-conformal folds × 2 sides + the held-out diagnostic re-run] on a
30-day fixture), ~5s for predict.

Non-goal: skill testing. The fixture's NWP winds are noise around a shared
signal, so q50 MAE / band width are meaningless — but every contract predict
and the .NET consumers rely on still gets exercised.

wind_speed_lgb is bonehill-only per ``phases.yaml``; truth is Dunkeswell
SYNOP via a fake MIDAS BADC-CSV file (same fixture as wind_mvn — the two
models share the train_wind_mvn feature recipe by construction).
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
    make_dunkeswell_midas_truth,
    make_forecast_tree,
    make_orographic_static,
    run_sync_train_data,
)

LOCATION = "bonehill_rocks"
LEADS = (24, 48, 72)

# Same window arithmetic as test_smoke_wind_mvn: 30 days × 24 h gives ~720
# joined rows per lead — comfortably above train_wind_speed_pi's 500-row
# per-lead floor after the required-models filter (the fixture emits every
# REQUIRED_MODELS NWP dense, so the filter drops nothing).
TRAIN_START = datetime(2024, 2, 1)
TRAIN_DAYS = 30
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
    """Fixtures land in a fake-R2 dir; ``run_sync_train_data`` (phase csv
    ``wind_speed_lgb``, the SAME declaration retrain-python.yml resolves)
    copies them into the data root. A missing pull declaration fails here
    rather than in a Sunday retrain. Then train_wind_speed_pi.main() runs
    end-to-end in production mode (--conformal crossconf)."""
    fake_r2 = smoke_root / "fake-r2"
    fake_r2_data = fake_r2 / "data"
    make_forecast_tree(
        fake_r2_data, LOCATION, TRAIN_START, n_days=TRAIN_DAYS,
        run_time_source="offset_day",
    )
    # The MIDAS fixture generates days from Jan 1; the forecast window starts
    # Feb 1 (day 32). train_wind_speed_pi has a 500-row per-lead floor after
    # the required-models filter, so the truth must cover the FULL forecast
    # window (through day 62) — wind_mvn's smoke gets away with a 9-day
    # overlap (216 rows/lead) but this one would skip every lead.
    make_dunkeswell_midas_truth(
        fake_r2_data, years=(TRAIN_START.year,), days_per_year=75,
    )
    make_orographic_static(fake_r2_data, LOCATION)
    run_sync_train_data(
        location=LOCATION, phases="wind_speed_lgb",
        r2_source=fake_r2, local_root=smoke_root,
    )

    import train_wind_mvn
    importlib.reload(train_wind_mvn)
    import train_wind_speed_pi
    importlib.reload(train_wind_speed_pi)
    sys_argv = sys.argv
    try:
        sys.argv = [
            "train_wind_speed_pi",
            "--location", LOCATION,
            "--models-root", str(smoke_root / "models"),
            "--conformal", "crossconf",
        ]
        try:
            train_wind_speed_pi.main()
        except SystemExit as e:
            if e.code not in (None, 0):
                raise AssertionError(
                    f"train_wind_speed_pi.main exited with code {e.code} on the smoke fixture"
                ) from e
    finally:
        sys.argv = sys_argv

    bundle_parent = smoke_root / "models" / "wind" / LOCATION
    versions = sorted([p for p in bundle_parent.iterdir() if p.is_dir()])
    assert versions, f"no wind_speed_lgb bundle written under {bundle_parent}"
    return versions[-1]


class TestTrainWindSpeedLgbSmoke:
    def test_bundle_has_per_lead_boosters_and_sidecars(self, trained_bundle):
        bundle = trained_bundle
        for lead in LEADS:
            for tag in ("lo", "med", "hi"):
                assert (bundle / f"q_{tag}_lead_{lead}h.txt").exists(), \
                    f"bundle missing q_{tag}_lead_{lead}h.txt"
        # training_summary.json is the RetrainGuard baseline — its absence
        # means the guard never ran (and the R2 push semantics regress).
        for name in ("feature_schema.json", "calibration.json",
                     "training_metadata.json", "training_summary.json"):
            assert (bundle / name).exists(), f"bundle missing {name}"

    def test_training_metadata_stamps_phase_and_location(self, trained_bundle):
        metadata = json.loads((trained_bundle / "training_metadata.json").read_text())
        assert metadata["Phase"] == "wind_speed_lgb"
        assert metadata["LocationName"] == LOCATION
        assert "cross-conformal" in metadata["Hyperparameters"]["conformal"]
        for lead in LEADS:
            entry = metadata["PerLead"][str(lead)]
            assert entry["TrainRows"] > 0
            assert entry["TestRows"] > 0
            # BlendTestMae carries q50 MAE — the Models page card reads it.
            assert entry["BlendTestMae"] is not None

    def test_feature_schema_locks_29_production_features(self, trained_bundle):
        schema = json.loads((trained_bundle / "feature_schema.json").read_text())
        # Same 29-feature recipe as wind_mvn (shared builder) — drift here
        # breaks predict's positional X matrix silently.
        assert len(schema["FeatureNames"]) == 29

    def test_calibration_has_crossconf_per_lead(self, trained_bundle):
        calibration = json.loads((trained_bundle / "calibration.json").read_text())
        assert calibration["coverage"] == 0.90
        for lead in LEADS:
            cell = calibration["per_lead"][str(lead)]
            assert cell["conformal_mode"] == "crossconf"
            assert cell["k_folds"] == 10
            # Q can be negative (band tightens) but must be present + finite.
            assert cell["conformal_q"] is not None
            assert cell["pooled_oof_coverage"] is not None

    def test_manifest_promoted_as_challenger_not_champion(self, trained_bundle):
        smoke_root = trained_bundle.parents[3]
        manifest_path = smoke_root / "models" / "wind" / "MANIFEST.json"
        assert manifest_path.exists(), \
            "train_wind_speed_pi did not promote into wind/MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["Stations"][LOCATION]
        assert any("_wind_speed_lgb" in v for v in entry["Active"]), (
            f"manifest Active list lacks a wind_speed_lgb entry; got {entry['Active']}"
        )
        # Challenger promote must NOT claim Current — the ERA5-truth .NET
        # `wind` champion owns that slot.
        assert "_wind_speed_lgb" not in (entry.get("Current") or ""), (
            f"wind_speed_lgb stole the Current slot: {entry.get('Current')}"
        )


class TestPredictWindSpeedLgbSmoke:
    def test_predict_writes_element_rows_with_cqr_band(self, trained_bundle):
        smoke_root = trained_bundle.parents[3]
        bundle_name = trained_bundle.name

        live_start = PREDICT_ANCHOR + timedelta(hours=1)
        make_forecast_tree(
            smoke_root, LOCATION, live_start, n_days=5,
            run_time_source="reported",
        )
        make_orographic_static(smoke_root, LOCATION)

        import predict_wind_mvn
        importlib.reload(predict_wind_mvn)
        import predict_wind_speed_pi
        importlib.reload(predict_wind_speed_pi)
        sys_argv = sys.argv
        try:
            sys.argv = [
                "predict_wind_speed_pi",
                "--location", LOCATION,
                "--anchor",   PREDICT_ANCHOR.date().isoformat(),
                "--models-root",      str(smoke_root / "models"),
                "--predictions-root", str(smoke_root / "predictions"),
            ]
            try:
                predict_wind_speed_pi.main()
            except SystemExit as e:
                if e.code not in (None, 0):
                    raise AssertionError(
                        f"predict_wind_speed_pi.main exited with code {e.code} "
                        f"on the smoke fixture"
                    ) from e
        finally:
            sys.argv = sys_argv

        # Element tree: NO location subdir (LocationName is a column).
        out_path = (
            smoke_root / "predictions" / "wind"
            / f"model_version={bundle_name}"
            / f"date={PREDICT_ANCHOR.date().isoformat()}"
            / "predictions.parquet"
        )
        assert out_path.exists(), \
            f"predict_wind_speed_pi wrote no predictions.parquet at {out_path}"

        out = pd.read_parquet(out_path)
        assert len(out) > 0
        assert set(out["LeadHours"].unique().tolist()) == set(LEADS)

        # The exact columns WindBlendPredictCommand + the chart consume
        # (union_by_name means extras are fine, absences are silent NULLs —
        # so pin the full set).
        for col in (
            "LocationName", "Element", "ModelVersion", "PredictionMadeAtUtc",
            "ValidTimeUtc", "LeadHours", "BlendValue",
            "ModelGfs", "ModelEcmwf", "ModelIcon", "ModelMf", "ModelUkmo",
            "ModelGem", "ModelAifs",
            "RunTimeGfs", "RunTimeEcmwf", "RunTimeIcon", "RunTimeMf",
            "RunTimeUkmo", "RunTimeGem", "RunTimeAifs",
            "Mean", "Std", "Range", "FeatureVectorHash",
            "BandLoMs", "BandHiMs", "ConformalQ",
        ):
            assert col in out.columns, f"predictions parquet missing column {col}"

        assert (out["Element"] == "wind").all()
        assert (out["ModelVersion"] == bundle_name).all()
        # Band invariants: lo <= point <= hi, lo clamped at 0.
        assert (out["BandLoMs"] >= 0).all()
        assert (out["BandLoMs"] <= out["BlendValue"] + 1e-6).all(), \
            "BandLoMs > BlendValue on at least one row"
        assert (out["BandHiMs"] >= out["BlendValue"] - 1e-6).all(), \
            "BandHiMs < BlendValue on at least one row"
