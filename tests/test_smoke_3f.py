"""End-to-end smoke tests for Phase 3f (train + predict).

Goal: catch the wiring / typing / env-var / SQL-shape class of bugs locally
before pushing to GH Actions. See ``WeatherBlend/docs/SMOKE_TEST_PLAN.md``
for the motivating incidents (2026-05-26 ship-day burned ~6 of these).

Non-goal: data-quality / skill testing. Realistic synthetic data, but the
assertions only check "the code path works end-to-end and writes the
right shape," not "the model is accurate."

Runtime budget: ~25s for the train test (NGBoost 500 estimators × 5 leads
× 1 station on a 30-day fixture), ~2s for predict.
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
    make_manifest,
    make_rainfall_truth,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

STATION_FRIENDLY = "Chards Snowdon Hill"
STATION_SLUG = "ea_chards_snowdon_hill"
LOCATION = "membury_devon"
TRAIN_START = datetime(2026, 2, 1)
TRAIN_DAYS = 30
PREDICT_ANCHOR = TRAIN_START + timedelta(days=TRAIN_DAYS)


@pytest.fixture
def smoke_root(tmp_path, monkeypatch):
    """Tempdir-scoped WEATHERBLEND_DATA_ROOT + WB_LOCATION env override
    + module reload so _shared / src.data / train_3f pick the new env.
    Per-test isolation: each test gets its own tmp_path, so we can leave
    train artefacts on disk for the predict test to read."""
    monkeypatch.setenv("WEATHERBLEND_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("WB_LOCATION", LOCATION)
    # Reload data + _shared so the env var is honoured at module load.
    import src.data
    importlib.reload(src.data)
    import _shared
    importlib.reload(_shared)
    return tmp_path


@pytest.fixture
def trained_bundle(smoke_root):
    """Fixture that writes the synthetic forecast + truth tree AND runs
    train_3f end-to-end. Returns the bundle directory path. Reused by
    both the train assertions and the predict test (so we don't pay the
    24s NGBoost cost twice)."""
    # Train-side fixture: 30 days of offset_day forecasts + matching truth.
    make_forecast_tree(
        smoke_root, LOCATION, TRAIN_START, n_days=TRAIN_DAYS,
        run_time_source="offset_day",
    )
    make_rainfall_truth(
        smoke_root, LOCATION, STATION_FRIENDLY, TRAIN_START, n_days=TRAIN_DAYS,
    )

    # Invoke train_3f in-process so monkeypatched env is preserved (a
    # subprocess would lose the pytest tmp_path + WB_LOCATION pair unless
    # we re-export them).
    import train_3f
    importlib.reload(train_3f)
    sys_argv = sys.argv
    try:
        sys.argv = [
            "train_3f",
            "--anchor", PREDICT_ANCHOR.date().isoformat(),
            "--stations", STATION_FRIENDLY,
            "--models-root", str(smoke_root / "models"),
        ]
        # train_3f.main() sys.exits non-zero on guard failure / no bundles.
        # Catch SystemExit so the assertion error is informative.
        try:
            train_3f.main()
        except SystemExit as e:
            if e.code not in (None, 0):
                raise AssertionError(
                    f"train_3f.main exited with code {e.code} on the smoke fixture"
                ) from e
    finally:
        sys.argv = sys_argv

    # Locate the bundle that was just written. Per-station-versioned dir.
    station_dir = smoke_root / "models" / "rainfall_amount" / STATION_SLUG
    versions = sorted([p for p in station_dir.iterdir() if p.is_dir()])
    assert versions, f"no 3f bundle written under {station_dir}"
    return versions[-1]


# --------------------------------------------------------------------------
# Train smoke
# --------------------------------------------------------------------------

class TestTrain3fSmoke:
    """The smoke contract: train_3f.main() against a synthetic fixture
    produces a bundle with the per-lead pickles + sidecar JSONs the
    predict path reads."""

    def test_bundle_directory_contains_per_lead_pickles_and_sidecars(self, trained_bundle):
        bundle = trained_bundle
        # 5 leads → 5 pickled NGBRegressors.
        pickles = sorted(bundle.glob("stage2_lognormal_lead*h.pkl"))
        assert len(pickles) == 5, (
            f"expected 5 per-lead pickles in {bundle}, got {[p.name for p in pickles]}"
        )
        for name in ("preprocess.json", "training_metadata.json",
                     "feature_schema.json", "training_summary.json"):
            assert (bundle / name).exists(), f"bundle missing {name}"

    def test_training_metadata_stamps_phase_and_target(self, trained_bundle):
        metadata = json.loads((trained_bundle / "training_metadata.json").read_text())
        assert metadata["Phase"] == "3f"
        assert metadata["Target"] == "rainfall_amount"
        assert metadata["LocationName"] == LOCATION
        # Per-lead block has TrainRows / ValRows populated for every lead.
        for lead in (24, 48, 72, 96, 120):
            entry = metadata["PerLead"][str(lead)]
            assert entry["TrainRows"] > 0, f"lead {lead}h has zero TrainRows"
            assert entry["ValRows"] > 0, f"lead {lead}h has zero ValRows"

    def test_preprocess_json_has_per_lead_scaler_params(self, trained_bundle):
        pre = json.loads((trained_bundle / "preprocess.json").read_text())
        assert pre["imputation"] == "row_mean_precip+col_median_else"
        # 15 features = 7 NWP precip + 4 spread + 4 calendar.
        assert len(pre["feature_names"]) == 15
        for lead in ("24", "48", "72", "96", "120"):
            entry = pre["per_lead"][lead]
            assert len(entry["scaler_mean"]) == 15
            assert len(entry["scaler_scale"]) == 15
            assert entry["n_train_wet"] >= 100

    def test_train_one_station_returns_ndarray_train_features(self, smoke_root):
        """Regression for the 4th-attempt list-vs-ndarray bug on 2026-05-26.

        train_one_station's RetrainGuard-feeding ``train_features`` field
        must be a 2D numpy ndarray (shape (n_rows, n_features), dtype
        float32). Earlier iterations returned a Python list; the guard's
        compute_feature_stats calls X.shape and X.mean(axis=0) and silently
        breaks if you hand it a list.
        """
        # Write the fixture but skip train_3f.main()'s sys.exit path —
        # call train_one_station directly so we can introspect its return.
        make_forecast_tree(
            smoke_root, LOCATION, TRAIN_START, n_days=TRAIN_DAYS,
            run_time_source="offset_day",
        )
        make_rainfall_truth(
            smoke_root, LOCATION, STATION_FRIENDLY, TRAIN_START, n_days=TRAIN_DAYS,
        )
        import train_3f
        importlib.reload(train_3f)

        result = train_3f.train_one_station(STATION_FRIENDLY, STATION_SLUG)

        tf = result["train_features"]
        assert isinstance(tf, np.ndarray), (
            f"train_features must be a numpy ndarray for the RetrainGuard "
            f"to consume; got {type(tf).__name__}"
        )
        assert tf.ndim == 2, f"train_features must be 2D; got {tf.ndim}D"
        assert tf.dtype == np.float32, (
            f"train_features must be float32 (matches train_4a's canonical "
            f"shape); got {tf.dtype}"
        )
        assert tf.shape[1] == 15, (
            f"train_features must have 15 columns (FEATURE_NAMES); "
            f"got {tf.shape[1]}"
        )
        assert tf.shape[0] > 0, "train_features must have at least one row"


# --------------------------------------------------------------------------
# Predict smoke
# --------------------------------------------------------------------------

class TestPredict3fSmoke:
    """Predict-side smoke: given a trained 3f bundle + a 3a-style stage-1
    parquet + 'reported'-source live forecasts, predict_3f.main() writes
    a non-empty predictions parquet under
    ``data/predictions/rainfall_amount/...``."""

    def test_predict_writes_non_empty_predictions_parquet(self, trained_bundle):
        smoke_root = trained_bundle.parents[3]  # bundle=<root>/models/rainfall_amount/{slug}/{version}
        bundle_name = trained_bundle.name

        # Live ('reported') forecasts covering anchor → anchor + 7 days.
        # 5 days at 24h granularity covers all 5 trained leads via
        # lead_day_bucket(v, anchor).
        live_start = PREDICT_ANCHOR + timedelta(hours=1)
        make_forecast_tree(
            smoke_root, LOCATION, live_start, n_days=6,
            run_time_source="reported",
        )

        # Bound 3a stage-1 parquet at the anchor. Read precip_3a_version
        # stamp out of the bundle metadata so the test stays in sync if
        # train_3f's stamping logic changes.
        metadata = json.loads((trained_bundle / "training_metadata.json").read_text())
        precip_3a_version = metadata["Hyperparameters"]["precip_3a_version"]
        if not precip_3a_version:
            # train_3f stamps "" when there's no manifest yet. Synthesise
            # a plausible 3a champion version so predict_3f can resolve.
            precip_3a_version = "v2026-02-28_120000"
            metadata["Hyperparameters"]["precip_3a_version"] = precip_3a_version
            (trained_bundle / "training_metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str, allow_nan=False)
            )

        from _smoke_fixtures import make_3a_predictions
        make_3a_predictions(
            smoke_root, STATION_FRIENDLY, precip_3a_version, PREDICT_ANCHOR,
        )

        # Manifest: 3a champion entry so resolve_bound_3a_version finds it.
        make_manifest(
            smoke_root, "precipitation", STATION_SLUG,
            active_versions=[precip_3a_version],
        )

        # Invoke predict_3f.main() in-process.
        import predict_3f
        importlib.reload(predict_3f)
        sys_argv = sys.argv
        try:
            sys.argv = [
                "predict_3f",
                "--anchor", PREDICT_ANCHOR.date().isoformat(),
                "--stations", STATION_FRIENDLY,
                "--models-root", str(smoke_root / "models"),
                "--predictions-root", str(smoke_root / "predictions"),
            ]
            try:
                predict_3f.main()
            except SystemExit as e:
                if e.code not in (None, 0):
                    raise AssertionError(
                        f"predict_3f.main exited with code {e.code} on the smoke fixture"
                    ) from e
        finally:
            sys.argv = sys_argv

        # Output parquet under the predictions root.
        out_dir = (
            smoke_root
            / "predictions"
            / "rainfall_amount"
            / STATION_SLUG
            / f"model_version={bundle_name}"
            / f"date={PREDICT_ANCHOR.date().isoformat()}"
        )
        out_path = out_dir / "predictions.parquet"
        assert out_path.exists(), f"predict_3f wrote no predictions.parquet at {out_path}"

        import pandas as pd
        out = pd.read_parquet(out_path)
        assert len(out) > 0, "predictions parquet is empty"
        # Must have rows for every trained lead (the live fixture spans
        # 6 days which covers buckets 24..120 inclusive).
        emitted_leads = set(out["LeadHours"].unique().tolist())
        assert emitted_leads == {24, 48, 72, 96, 120}, (
            f"expected predictions for every trained lead; got {emitted_leads}"
        )

        # Required output columns — write_predictions_parquet's
        # column_order list. Names pinned because workers + verify_3f.py
        # read them directly.
        for col in (
            "ValidTimeUtc", "LeadHours", "Pi", "MuLog", "SigmaLog",
            "MeanMmPerHr", "MedianMmPerHr",
            "P2_5MmPerHr", "P10MmPerHr", "P50MmPerHr", "P90MmPerHr", "P97_5MmPerHr",
            "PExceed0_1", "PExceed1", "PExceed5", "PExceed10",
            "Precip3aVersion",
        ):
            assert col in out.columns, f"predictions parquet missing column {col}"

    def test_predict_dedupes_intra_day_3a_cycles_at_main_level(self, trained_bundle):
        """End-to-end version of the 2026-05-26 broadcast-shape bug
        regression — the helper-level test in
        ``test_predict_3f_pi_join.py`` covers load_bound_3a_pi; this one
        asserts the main() driver also produces non-empty output when
        the 3a parquet holds multiple intra-day cycles."""
        smoke_root = trained_bundle.parents[3]
        live_start = PREDICT_ANCHOR + timedelta(hours=1)
        make_forecast_tree(
            smoke_root, LOCATION, live_start, n_days=6,
            run_time_source="reported",
        )

        metadata = json.loads((trained_bundle / "training_metadata.json").read_text())
        precip_3a_version = metadata["Hyperparameters"]["precip_3a_version"] or "v2026-02-28_120000"
        if metadata["Hyperparameters"]["precip_3a_version"] != precip_3a_version:
            metadata["Hyperparameters"]["precip_3a_version"] = precip_3a_version
            (trained_bundle / "training_metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str, allow_nan=False)
            )

        # 8 intra-day cycles — heavier than production's typical 4-5 —
        # so dedup must keep ONE row per (V, L) or the broadcast shape
        # mismatch raises during predict_one_station.
        from _smoke_fixtures import make_3a_predictions
        make_3a_predictions(
            smoke_root, STATION_FRIENDLY, precip_3a_version, PREDICT_ANCHOR,
            n_cycles=8,
        )
        make_manifest(
            smoke_root, "precipitation", STATION_SLUG,
            active_versions=[precip_3a_version],
        )

        import predict_3f
        importlib.reload(predict_3f)
        sys_argv = sys.argv
        try:
            sys.argv = [
                "predict_3f",
                "--anchor", PREDICT_ANCHOR.date().isoformat(),
                "--stations", STATION_FRIENDLY,
                "--models-root", str(smoke_root / "models"),
                "--predictions-root", str(smoke_root / "predictions"),
            ]
            try:
                predict_3f.main()
            except SystemExit as e:
                if e.code not in (None, 0):
                    raise AssertionError(
                        f"predict_3f.main exited with {e.code} on 8-cycle 3a fixture — "
                        f"likely the dedup regressed and the broadcast shape mismatch fired."
                    ) from e
        finally:
            sys.argv = sys_argv

        out_path = (
            smoke_root
            / "predictions"
            / "rainfall_amount"
            / STATION_SLUG
            / f"model_version={trained_bundle.name}"
            / f"date={PREDICT_ANCHOR.date().isoformat()}"
            / "predictions.parquet"
        )
        assert out_path.exists() and out_path.stat().st_size > 0, (
            f"predict_3f.main produced no predictions.parquet on the 8-cycle 3a fixture"
        )
