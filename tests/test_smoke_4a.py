"""End-to-end smoke tests for Phase 4a (per-cell BART, train + predict).

Goal: catch the wiring / typing / env-var / SQL-shape class of bugs locally
before kicking off a real retrain. See
``WeatherBlend/docs/SMOKE_TEST_PLAN.md``.

Runtime budget: ~10s for train (env-var-shrunk BART hyperparams + 1 station
× 5 leads on a 30-day fixture), a few seconds for predict.

Production hyperparams (NTREE=500, NSKIP=200, NDPOST=1000) take ~64s on
this fixture — too slow for an interactive smoke. The smoke fixture sets
``WB_BART_NTREE/NSKIP/NDPOST`` env vars to tiny values; ``train_4a.py``
reads them at module import time. Live retrains never set those env
vars so production BART hyperparams are untouched.

4a is Bonehill-only per ``phases.yaml`` (``precipitation.4a.locations =
[bonehill_rocks]``), so the fixture trains a Bellever bundle, not a
Membury one.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from _smoke_fixtures import (  # noqa: E402
    make_forecast_tree,
    make_orographic_static,
    make_rainfall_truth,
    run_sync_train_data,
)


# --------------------------------------------------------------------------
# Fixture constants
# --------------------------------------------------------------------------

STATION_FRIENDLY = "Bellever Dartmoor"
STATION_SLUG = "ea_bellever_dartmoor"
LOCATION = "bonehill_rocks"
TRAIN_START = datetime(2026, 2, 1)
TRAIN_DAYS = 30
PREDICT_ANCHOR = TRAIN_START + timedelta(days=TRAIN_DAYS)

# Tiny hyperparams via env var. The defaults (500/200/1000) take ~64s for
# 5 leads on this fixture; these cut it to roughly 1s per lead.
SMOKE_BART_ENV = {
    "WB_BART_NTREE": "50",
    "WB_BART_NSKIP": "20",
    "WB_BART_NDPOST": "50",
}


@pytest.fixture
def smoke_root(tmp_path, monkeypatch):
    """Tempdir-scoped data root + bonehill location + tiny BART
    hyperparams. The env vars must be set BEFORE train_4a is imported
    because NTREE/NSKIP/NDPOST are read at module load time."""
    monkeypatch.setenv("WEATHERBLEND_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("WB_LOCATION", LOCATION)
    for k, v in SMOKE_BART_ENV.items():
        monkeypatch.setenv(k, v)
    # Reload so the env vars are honoured. data → _shared → train_4a.
    import src.data
    importlib.reload(src.data)
    import _shared
    importlib.reload(_shared)
    return tmp_path


@pytest.fixture
def trained_bundle(smoke_root):
    """Write the synthetic forecast + truth tree AND run train_4a.main()
    end-to-end. Returns the bundle directory.

    Train-side fixtures land in a fake-R2 dir; ``run_sync_train_data``
    invokes the production sync script (the SAME script
    ``retrain-python.yml`` calls) to copy them into the data root
    train_4a will read. A missing pull declaration for 4a then fails
    here rather than in a Sunday retrain."""
    fake_r2 = smoke_root / "fake-r2"
    fake_r2_data = fake_r2 / "data"
    make_forecast_tree(
        fake_r2_data, LOCATION, TRAIN_START, n_days=TRAIN_DAYS,
        run_time_source="offset_day",
    )
    make_rainfall_truth(
        fake_r2_data, LOCATION, STATION_FRIENDLY, TRAIN_START, n_days=TRAIN_DAYS,
    )
    # train_4a's rich+oro builder reads data/static/orographic/{slug}.json
    # for the per-station terrain block (3 static + 5 dynamic + station_id).
    # Synth one for Bellever inside the fake-R2 root so sync_train_data
    # pulls it alongside forecasts + rainfall.
    make_orographic_static(fake_r2_data, STATION_SLUG)
    run_sync_train_data(
        location=LOCATION, phases="4a",
        r2_source=fake_r2, local_root=smoke_root,
    )

    import train_4a
    importlib.reload(train_4a)
    # Sanity check the env override took effect — guards against future
    # refactors that move the constants somewhere ``importlib.reload``
    # doesn't reach.
    assert train_4a.NTREE == int(SMOKE_BART_ENV["WB_BART_NTREE"]), (
        f"WB_BART_NTREE override not picked up by train_4a — got "
        f"NTREE={train_4a.NTREE}, expected {SMOKE_BART_ENV['WB_BART_NTREE']}"
    )

    sys_argv = sys.argv
    try:
        sys.argv = [
            "train_4a",
            "--anchor", PREDICT_ANCHOR.date().isoformat(),
            "--stations", STATION_FRIENDLY,
            "--models-root", str(smoke_root / "models"),
        ]
        try:
            train_4a.main()
        except SystemExit as e:
            if e.code not in (None, 0):
                raise AssertionError(
                    f"train_4a.main exited with code {e.code} on the smoke fixture"
                ) from e
    finally:
        sys.argv = sys_argv

    station_dir = smoke_root / "models" / "precipitation" / STATION_SLUG
    versions = sorted([p for p in station_dir.iterdir() if p.is_dir()])
    assert versions, f"no 4a bundle written under {station_dir}"
    return versions[-1]


# --------------------------------------------------------------------------
# Train smoke
# --------------------------------------------------------------------------

class TestTrain4aSmoke:
    """The smoke contract: train_4a.main() against a synthetic fixture
    produces a per-cell bundle with state_lead_NNh.rds + arrays + per-lead
    preprocess that predict_4a can warm-scaffold + setState from."""

    def test_bundle_has_per_lead_state_and_arrays_and_sidecars(self, trained_bundle):
        bundle = trained_bundle
        # 5 leads → 5 state RDS + 5 arrays NPZ. predict_4a.find_latest_bundle
        # requires both per-lead artefacts; missing either silently
        # filters out the bundle as "lead-pooled legacy".
        for lead in (24, 48, 72, 96, 120):
            assert (bundle / f"state_lead_{lead}h.rds").exists(), \
                f"bundle missing state_lead_{lead}h.rds"
            assert (bundle / f"arrays_lead_{lead}h.npz").exists(), \
                f"bundle missing arrays_lead_{lead}h.npz"
        for name in ("preprocess.json", "training_metadata.json",
                     "feature_schema.json", "training_summary.json"):
            assert (bundle / name).exists(), f"bundle missing {name}"

    def test_training_metadata_stamps_phase_and_location(self, trained_bundle):
        metadata = json.loads((trained_bundle / "training_metadata.json").read_text())
        assert metadata["Phase"] == "4a"
        assert metadata["Target"] == "precipitation"
        assert metadata["LocationName"] == LOCATION
        # Tiny-hyperparam stamp persists into the bundle so a smoke-trained
        # bundle is identifiable vs a real retrain at audit time.
        hyper = metadata["Hyperparameters"]
        assert hyper["ntree"] == int(SMOKE_BART_ENV["WB_BART_NTREE"])
        assert hyper["nskip"] == int(SMOKE_BART_ENV["WB_BART_NSKIP"])
        assert hyper["ndpost"] == int(SMOKE_BART_ENV["WB_BART_NDPOST"])

    def test_preprocess_json_has_per_lead_kept_indices_and_scaler(self, trained_bundle):
        pre = json.loads((trained_bundle / "preprocess.json").read_text())
        assert pre["ntree"] == int(SMOKE_BART_ENV["WB_BART_NTREE"])
        for lead in ("24", "48", "72", "96", "120"):
            entry = pre["per_lead"][lead]
            # 68 features = rich (59 — _shared.RICH_FEATURE_NAMES) + 9 v1 oro
            # terrain (_shared.V1_TERRAIN_FEATURE_NAMES). Bumped from 25 (lean
            # + 3 syn) on 2026-05-30 when train_4a swapped to the rich+oro
            # spec — the per-station BART winner from the 2026-05-29 bake-off.
            assert len(entry["feature_list_full"]) == 68
            # oro_* features must be present — confirms compose_v1_terrain_block
            # ran and produced terrain columns the dispatcher in predict_4a
            # will route through the rich+oro live builder.
            assert any(f.startswith("oro_") for f in entry["feature_list_full"]), \
                "expected oro_* features in feature_list_full; rich+oro swap may have regressed"
            assert len(entry["scaler_mean"]) == len(entry["feature_names_eff"])
            assert len(entry["scaler_scale"]) == len(entry["feature_names_eff"])
            assert "kept_indices" in entry

    def test_arrays_npz_round_trips_with_correct_shape(self, trained_bundle):
        for lead in (24, 48, 72, 96, 120):
            path = trained_bundle / f"arrays_lead_{lead}h.npz"
            with np.load(path) as z:
                X = z["X_train_s"]
                y = z["y_train"]
                assert X.ndim == 2 and X.shape[0] > 0
                assert y.ndim == 1 and y.shape[0] == X.shape[0]
                # predict_4a warm-scaffold consumes float64 arrays.
                assert X.dtype == np.float64
                assert y.dtype == np.float64

    def test_manifest_promoted_with_phase4a_tag(self, trained_bundle):
        smoke_root = trained_bundle.parents[3]
        manifest_path = smoke_root / "models" / "precipitation" / "MANIFEST.json"
        assert manifest_path.exists(), "train_4a did not promote into MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        active = manifest["Stations"][STATION_SLUG]["Active"]
        assert any("_phase4a" in v for v in active), (
            f"manifest Active list lacks a phase4a entry; got {active}"
        )


# --------------------------------------------------------------------------
# Predict smoke
# --------------------------------------------------------------------------

class TestPredict4aSmoke:
    """Predict-side: given a trained 4a bundle + 'reported' live forecasts,
    predict_4a.main() must write a non-empty hourly per-lead parquet."""

    def test_predict_writes_non_empty_predictions_parquet(self, trained_bundle):
        smoke_root = trained_bundle.parents[3]
        bundle_name = trained_bundle.name

        # Live 'reported' forecasts spanning the 7-day horizon predict_4a
        # uses (build_live_features keeps ValidTime > anchor and <=
        # anchor + 7 days). 6 days of valid hours covers leads 24..120.
        live_start = PREDICT_ANCHOR + timedelta(hours=1)
        make_forecast_tree(
            smoke_root, LOCATION, live_start, n_days=6,
            run_time_source="reported",
        )

        import predict_4a
        importlib.reload(predict_4a)
        sys_argv = sys.argv
        try:
            sys.argv = [
                "predict_4a",
                "--anchor", PREDICT_ANCHOR.date().isoformat(),
                "--stations", STATION_FRIENDLY,
                "--models-root", str(smoke_root / "models"),
                "--predictions-root", str(smoke_root / "predictions"),
            ]
            try:
                predict_4a.main()
            except SystemExit as e:
                if e.code not in (None, 0):
                    raise AssertionError(
                        f"predict_4a.main exited with code {e.code} on the smoke fixture"
                    ) from e
        finally:
            sys.argv = sys_argv

        out_path = (
            smoke_root
            / "predictions"
            / "precipitation"
            / STATION_SLUG
            / f"model_version={bundle_name}"
            / f"date={PREDICT_ANCHOR.date().isoformat()}"
            / "predictions.parquet"
        )
        assert out_path.exists(), f"predict_4a wrote no predictions.parquet at {out_path}"

        import pandas as pd
        out = pd.read_parquet(out_path)
        assert len(out) > 0, "predictions parquet is empty"
        emitted_leads = set(out["LeadHours"].unique().tolist())
        assert emitted_leads == {24, 48, 72, 96, 120}, (
            f"expected predictions for every trained lead; got {emitted_leads}"
        )
        # Required output columns the 4b mint + verify_4a both read.
        for col in ("ValidTimeUtc", "LeadHours", "ProbWet",
                    "LocationName", "ModelVersion", "TruthStation",
                    "PredictionMadeAtUtc"):
            assert col in out.columns, f"predictions parquet missing column {col}"
