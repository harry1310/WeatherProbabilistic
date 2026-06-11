"""End-to-end smoke tests for Phase 2 wind_mvn (train + predict).

Goal: catch the wiring / typing / env-var / SQL-shape class of bugs locally
before kicking off a real retrain. See
``WeatherBlend/docs/SMOKE_TEST_PLAN.md``.

Runtime budget: ~45s for train (3 leads × ~150 effective epochs on a
30-day fixture + α' grid-search), ~5s for predict.

Non-goal: skill testing. The forecast fixture's wind directions are
mutually uncorrelated across NWPs, so the MLP cannot converge to a
meaningful NLL — but the bundle layout, calibration scalars, manifest
promotion, and predict output schema all still get exercised.

wind_mvn is bonehill-only per ``phases.yaml``
(``wind_direction.wind_mvn.locations = [bonehill_rocks]``), so the
fixture trains a bonehill bundle. Truth is Dunkeswell SYNOP via a
fake MIDAS BADC-CSV file at ``data/truth/midas/raw/``.
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from _smoke_fixtures import (  # noqa: E402
    add_decoy_bundles,
    make_dunkeswell_midas_truth,
    make_forecast_tree,
    make_orographic_static,
    run_sync_train_data,
)


# --------------------------------------------------------------------------
# Fixture constants
# --------------------------------------------------------------------------

LOCATION = "bonehill_rocks"
LEADS = (24, 48, 72)

# train_wind_mvn pulls all forecast rows with RunTimeSource='offset_day' +
# LeadHours in {24,48,72} and inner-joins against the (hourly) MIDAS truth.
# 30 fixture days × 24 hours × 3 leads = up to ~2160 paired rows per lead;
# the 70/15/15 split leaves ~108 val rows + ~108 test rows per lead, well
# above the BATCH_SIZE=256 floor (one full + one partial batch per epoch).
#
# TRAIN_START lives in 2024 because train_wind_mvn.DUNKESWELL_YEARS is
# hard-coded to (2022, 2023, 2024) — the smoke fixture has to land inside
# that window or load_dunkeswell raises FileNotFoundError.
TRAIN_START = datetime(2024, 2, 1)
TRAIN_DAYS = 30
PREDICT_ANCHOR = TRAIN_START + timedelta(days=TRAIN_DAYS)


@pytest.fixture
def smoke_root(tmp_path, monkeypatch):
    """Tempdir-scoped WEATHERBLEND_DATA_ROOT + bonehill_rocks location.

    Reload ``src.data`` so WEATHERBLEND_DATA_ROOT is re-resolved at module
    load time. (No ``_shared`` reload — train_wind_mvn doesn't import the
    _shared MODELS_LEAN list.)"""
    monkeypatch.setenv("WEATHERBLEND_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("WB_LOCATION", LOCATION)
    import src.data
    importlib.reload(src.data)
    return tmp_path


@pytest.fixture
def trained_bundle(smoke_root):
    """Write the synthetic forecast + Dunkeswell SYNOP + orography fixtures
    AND run train_wind_mvn.main() end-to-end. Returns the bundle directory.
    Per-test isolation: each test gets its own tmp_path, so the predict
    test reads from this fixture's output, not a sibling test's.

    Train-side fixtures land in a fake-R2 dir; ``run_sync_train_data``
    invokes the production sync script (the SAME script
    ``retrain-python.yml`` calls) to copy them into the data root
    train_wind_mvn will read. A missing pull declaration for wind_mvn
    then fails here rather than in a Sunday retrain."""
    fake_r2 = smoke_root / "fake-r2"
    fake_r2_data = fake_r2 / "data"
    make_forecast_tree(
        fake_r2_data, LOCATION, TRAIN_START, n_days=TRAIN_DAYS,
        run_time_source="offset_day",
    )
    # Dunkeswell SYNOP fixture spans Jan-Feb of the same year as the
    # forecast window so the inner join lands non-empty per lead.
    make_dunkeswell_midas_truth(
        fake_r2_data, years=(TRAIN_START.year,), days_per_year=TRAIN_DAYS + 10,
    )
    make_orographic_static(fake_r2_data, LOCATION)
    run_sync_train_data(
        location=LOCATION, phases="wind_mvn",
        r2_source=fake_r2, local_root=smoke_root,
    )

    import train_wind_mvn
    importlib.reload(train_wind_mvn)
    sys_argv = sys.argv
    try:
        sys.argv = [
            "train_wind_mvn",
            "--location", LOCATION,
            "--leads", *(str(l) for l in LEADS),
            "--models-root", str(smoke_root / "models"),
        ]
        try:
            train_wind_mvn.main()
        except SystemExit as e:
            if e.code not in (None, 0):
                raise AssertionError(
                    f"train_wind_mvn.main exited with code {e.code} on the smoke fixture"
                ) from e
    finally:
        sys.argv = sys_argv

    bundle_parent = smoke_root / "models" / "wind_direction" / LOCATION
    versions = sorted([p for p in bundle_parent.iterdir() if p.is_dir()])
    assert versions, f"no wind_mvn bundle written under {bundle_parent}"
    real = versions[-1]
    add_decoy_bundles(bundle_parent, "wind_mvn")
    return real


# --------------------------------------------------------------------------
# Train smoke
# --------------------------------------------------------------------------

class TestTrainWindMvnSmoke:
    """The smoke contract: train_wind_mvn.main() against a synthetic
    fixture produces a per-lead bundle with state_lead_NNh.pt + scaler +
    calibration + metadata that predict_wind_mvn can load."""

    def test_bundle_has_per_lead_state_and_sidecars(self, trained_bundle):
        bundle = trained_bundle
        # Every requested lead must have a state_lead_{L}h.pt — predict
        # silently skips leads whose state file is missing, which would
        # silently shrink the output without failing the workflow.
        for lead in LEADS:
            assert (bundle / f"state_lead_{lead}h.pt").exists(), \
                f"bundle missing state_lead_{lead}h.pt"
        for name in ("feature_scaler.json", "feature_schema.json",
                     "calibration.json", "training_metadata.json",
                     "training_summary.json", "test_predictions.parquet"):
            assert (bundle / name).exists(), f"bundle missing {name}"

    def test_training_metadata_stamps_phase_and_location(self, trained_bundle):
        metadata = json.loads((trained_bundle / "training_metadata.json").read_text())
        assert metadata["Phase"] == "wind_mvn"
        assert metadata["Target"] == "wind_direction"
        assert metadata["LocationName"] == LOCATION
        nwps = {"gfs_seamless", "ecmwf_ifs025", "icon_seamless",
                "gem_seamless", "ukmo_seamless", "jma_seamless"}
        # Per-lead block populated for every trained lead.
        for lead in LEADS:
            entry = metadata["PerLead"][str(lead)]
            assert entry["TrainRows"] > 0, f"lead {lead}h has zero TrainRows"
            assert entry["ValRows"] > 0,   f"lead {lead}h has zero ValRows"
            assert entry["TestRows"] > 0,  f"lead {lead}h has zero TestRows"
            # The Models card's Δ-vs-best-single baseline must be a REAL
            # per-NWP direction MAE (pre-2026-06-10 it was the blend's own
            # MAE → Δ 0% everywhere, and the val slot carried an NLL).
            assert entry["BestSingle"] in nwps, entry["BestSingle"]
            assert entry["BestSingleTestMae"] > 0
            assert entry["BestSingleValMae"] > 0

    def test_feature_schema_locks_29_production_features(self, trained_bundle):
        schema = json.loads((trained_bundle / "feature_schema.json").read_text())
        # 6 wsp + 6 wdir + 5 ORO + 12 SPREAD = 29 features. If this count
        # drifts, the scaler positional indexing in predict breaks
        # silently.
        assert len(schema["FeatureNames"]) == 29, (
            f"expected 29 production-scope features; got "
            f"{len(schema['FeatureNames'])}"
        )
        # 6 production NWPs (no HARMONIE, no MF, no AIFS).
        assert len(schema["NwpModels"]) == 6
        for nwp in ("gfs_seamless", "ecmwf_ifs025", "icon_seamless",
                    "gem_seamless", "ukmo_seamless", "jma_seamless"):
            assert nwp in schema["NwpModels"], f"missing production NWP {nwp}"

    def test_feature_scaler_per_lead_dims_match_feature_count(self, trained_bundle):
        scaler = json.loads((trained_bundle / "feature_scaler.json").read_text())
        n_features = len(scaler["feature_names"])
        for lead in LEADS:
            cell = scaler["per_lead"][str(lead)]
            # `kept_mask` is positional over the full feature list; the
            # actual scaler vectors only span the kept columns. predict's
            # standardise step indexes X by `kept_mask` before applying
            # mu/scale so the lengths must match accordingly.
            assert len(cell["kept_mask"]) == n_features
            kept_count = int(sum(cell["kept_mask"]))
            assert len(cell["mean"])    == kept_count
            assert len(cell["scale"])   == kept_count
            assert len(cell["medians"]) == kept_count

    def test_calibration_per_lead_has_alpha_and_blend(self, trained_bundle):
        calibration = json.loads((trained_bundle / "calibration.json").read_text())
        for lead in LEADS:
            cell = calibration["per_lead"][str(lead)]
            # α' grid in train_wind_mvn is [0.5, 3.0]; outside bounds
            # means the search collapsed silently.
            assert 0.5 <= cell["alpha_prime_dir"] <= 3.0
            assert 0.5 <= cell["alpha_prime_spd"] <= 3.0
            # Plan defaults are 2.50/3.00 when no LGB val predictions are
            # bound — train smoke runs without LGB so we expect defaults.
            assert cell["blend_center"] == 2.50
            assert cell["blend_scale"]  == 3.00

    def test_test_predictions_parquet_has_per_lead_rows(self, trained_bundle):
        out = pd.read_parquet(trained_bundle / "test_predictions.parquet")
        emitted_leads = set(out["Lead"].unique().tolist())
        assert emitted_leads == set(LEADS), (
            f"test_predictions.parquet missing rows for trained leads "
            f"{set(LEADS) - emitted_leads}"
        )
        for col in ("ValidTimeUtc", "Lead",
                    "mu_u", "mu_v", "sigma_u", "sigma_v", "rho",
                    "truth_u", "truth_v", "truth_speed", "truth_dir"):
            assert col in out.columns, f"test_predictions missing {col}"

    def test_manifest_promoted_with_wind_mvn_tag(self, trained_bundle):
        smoke_root = trained_bundle.parents[3]
        manifest_path = smoke_root / "models" / "wind_direction" / "MANIFEST.json"
        assert manifest_path.exists(), \
            "train_wind_mvn did not promote into wind_direction/MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        # Stations-keyed under the LOCATION (wind_mvn doesn't have a
        # separate truth-station concept — Dunkeswell SYNOP is the only
        # truth source, but the manifest is location-keyed for parity
        # with the temperature/element trees).
        active = manifest["Stations"][LOCATION]["Active"]
        assert any("_wind_mvn" in v for v in active), (
            f"manifest Active list lacks a wind_mvn entry; got {active}"
        )


# --------------------------------------------------------------------------
# Predict smoke
# --------------------------------------------------------------------------

class TestPredictWindMvnSmoke:
    """Predict-side smoke: given a trained wind_mvn bundle + 'reported'
    live forecasts + the static orography JSON, predict_wind_mvn.main()
    writes a non-empty predictions parquet whose columns match the C#
    WindDirectionPredictionRow consumer schema."""

    def test_predict_writes_non_empty_predictions_parquet(self, trained_bundle):
        smoke_root = trained_bundle.parents[3]
        bundle_name = trained_bundle.name

        # build_live_for_lead anchors at PREDICT_ANCHOR, slices
        # [anchor + lead//24 days, anchor + lead//24 + 1 days). For
        # lead 72h that's day +3..+4, so 5 days of live forecasts from
        # PREDICT_ANCHOR+1h cover every requested lead.
        live_start = PREDICT_ANCHOR + timedelta(hours=1)
        make_forecast_tree(
            smoke_root, LOCATION, live_start, n_days=5,
            run_time_source="reported",
        )
        # Static orography must be present at predict time too — fresh
        # smoke_root, so re-emit.
        make_orographic_static(smoke_root, LOCATION)

        import predict_wind_mvn
        importlib.reload(predict_wind_mvn)
        sys_argv = sys.argv
        try:
            sys.argv = [
                "predict_wind_mvn",
                "--location", LOCATION,
                "--anchor",   PREDICT_ANCHOR.date().isoformat(),
                "--models-root",     str(smoke_root / "models"),
                "--predictions-root", str(smoke_root / "predictions"),
            ]
            try:
                predict_wind_mvn.main()
            except SystemExit as e:
                if e.code not in (None, 0):
                    raise AssertionError(
                        f"predict_wind_mvn.main exited with code {e.code} "
                        f"on the smoke fixture"
                    ) from e
        finally:
            sys.argv = sys_argv

        out_path = (
            smoke_root
            / "predictions"
            / "wind_direction"
            / LOCATION
            / f"model_version={bundle_name}"
            / f"date={PREDICT_ANCHOR.date().isoformat()}"
            / "predictions.parquet"
        )
        assert out_path.exists(), \
            f"predict_wind_mvn wrote no predictions.parquet at {out_path}"

        out = pd.read_parquet(out_path)
        assert len(out) > 0, "predictions parquet is empty"
        emitted_leads = set(out["LeadHours"].unique().tolist())
        assert emitted_leads == set(LEADS), (
            f"expected predictions for every trained lead; got {emitted_leads}"
        )

        # Required output columns the C# WindDirectionPredictionRow
        # consumer reads (Models/WindDirectionPredictionRow.cs). Drift
        # here breaks the wind_blend mint + future site rendering.
        for col in (
            "ValidTimeUtc", "LeadHours", "PredictionMadeAtUtc",
            "ModelVersion", "LocationName",
            "MuU", "MuV", "SigmaU", "SigmaV", "Rho",
            "BlendDirection", "BlendDirectionCi95Lo", "BlendDirectionCi95Hi",
            "BlendDirectionCi80Lo", "BlendDirectionCi80Hi",
            "BlendSpeedMagnitude", "BlendSpeedCi95Lo", "BlendSpeedCi95Hi",
            "BlendSpeedCi80Lo", "BlendSpeedCi80Hi",
        ):
            assert col in out.columns, f"predictions parquet missing column {col}"

        # CI bounds must respect the lo<=hi invariant on the speed
        # axis (direction is circular so we don't check that ordering).
        assert (out["BlendSpeedCi95Lo"] <= out["BlendSpeedCi95Hi"]).all(), (
            "BlendSpeedCi95Lo > BlendSpeedCi95Hi on at least one row"
        )
        assert (out["BlendSpeedCi80Lo"] <= out["BlendSpeedCi80Hi"]).all(), (
            "BlendSpeedCi80Lo > BlendSpeedCi80Hi on at least one row"
        )
        # CI80 must be NARROWER than CI95 — same MC samples, lower
        # confidence level. Any row that violates means the percentile
        # call was wired wrong.
        assert ((out["BlendSpeedCi80Hi"] - out["BlendSpeedCi80Lo"])
                <= (out["BlendSpeedCi95Hi"] - out["BlendSpeedCi95Lo"]) + 1e-6).all(), (
            "BlendSpeedCi80 width should be ≤ BlendSpeedCi95 width on every row"
        )
        # Point estimate must lie INSIDE both CI bands. Regression for the
        # 2026-05-28 bug where pt_speed = ||μ|| could fall below the MC
        # 10th-percentile CI80 low bound because ||(u,v)|| with
        # (u,v) ~ N(μ,Σ) has E[||·||] ≥ ||E[·]|| (Jensen) — the MC speed
        # distribution sits above the location of the latent.
        assert (out["BlendSpeedCi80Lo"] - 1e-6 <= out["BlendSpeedMagnitude"]).all(), (
            "BlendSpeedMagnitude must be ≥ BlendSpeedCi80Lo"
        )
        assert (out["BlendSpeedMagnitude"] <= out["BlendSpeedCi80Hi"] + 1e-6).all(), (
            "BlendSpeedMagnitude must be ≤ BlendSpeedCi80Hi"
        )
        # Direction must wrap into [0, 360).
        assert ((out["BlendDirection"] >= 0) & (out["BlendDirection"] < 360)).all()
