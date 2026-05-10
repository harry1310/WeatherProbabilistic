"""Tests for the Python RetrainGuard mirror of WeatherBlend's .NET guard.

The .NET equivalent (RetrainGuardTests.cs) covers the same scenarios;
keeping this suite in lockstep is what guarantees a 4a / 5a guard
fires on the same upstream-data shifts that trip 2b / 3a.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from src.retrain_guard import (
    DEFAULTS,
    GuardTolerances,
    build_check_and_save_singleton,
    build_check_and_save_versioned,
    check,
    try_load_previous_summary_singleton,
    try_load_previous_summary_versioned,
)
from src.training_summary import (
    TrainingSummary,
    FeatureStats,
    build_summary,
    compute_feature_stats,
    from_json,
    to_json,
)


def _make_features(rows: int = 100, cols: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, cols))


def _make_summary(
    rows_train: int = 1000,
    rows_val: int = 200,
    rows_test: int = 200,
    features: int = 3,
    per_feature: dict | None = None,
    label_rates: dict | None = None,
    phase: str = "4a",
    composite: str = "precipitation/test",
) -> TrainingSummary:
    return TrainingSummary(
        SchemaVersion="1",
        Composite=composite,
        Phase=phase,
        Version="v-test",
        ComputedAtUtc="2026-05-10T12:00:00+00:00",
        RowsTrain=rows_train,
        RowsVal=rows_val,
        RowsTest=rows_test,
        FeaturesEffective=features,
        PerFeature=per_feature or {},
        LabelRates=label_rates or {},
    )


# ----- compute_feature_stats -------------------------------------------


def test_compute_feature_stats_handles_partial_nans():
    X = np.array([[1.0, 2.0, np.nan], [1.1, 2.1, 3.0], [0.9, 1.9, 2.9]])
    stats = compute_feature_stats(X, ["a", "b", "c"])
    assert stats["a"].NanPct == pytest.approx(0.0)
    assert stats["a"].Mean == pytest.approx(1.0, abs=1e-6)
    # 1 of 3 rows NaN on column c
    assert stats["c"].NanPct == pytest.approx(1 / 3, abs=1e-6)


def test_compute_feature_stats_emits_zero_sentinel_for_all_nan_column():
    X = np.full((10, 2), np.nan)
    X[:, 0] = 1.0  # only second column is fully NaN
    stats = compute_feature_stats(X, ["a", "b"])
    assert stats["b"].NanPct == 1.0
    assert stats["b"].Mean == 0.0
    assert stats["b"].Std == 0.0


def test_compute_feature_stats_rejects_shape_mismatch():
    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="feature_names"):
        compute_feature_stats(X, ["a", "b"])  # 2 names for 3 columns


# ----- check (the core guard) ------------------------------------------


def test_check_returns_passing_with_note_when_previous_is_null():
    result = check(_make_summary(), previous=None, tolerances=DEFAULTS)
    assert result.passed
    assert result.breaches == []
    assert "First-ever train" in result.note


def test_check_passes_when_two_summaries_are_within_all_tolerances():
    prev = _make_summary(rows_train=10_000, rows_val=2_000, rows_test=2_000, features=3)
    curr = _make_summary(rows_train=10_500, rows_val=1_950, rows_test=2_050, features=3)
    result = check(curr, prev, DEFAULTS)
    assert result.passed
    assert result.breaches == []


def test_check_fails_when_row_count_drops_more_than_tolerance():
    prev = _make_summary(rows_train=10_000)
    curr = _make_summary(rows_train=4_000)
    result = check(curr, prev, DEFAULTS)
    assert not result.passed
    assert any(b.field == "rowsTrain" for b in result.breaches)


def test_check_fails_when_features_effective_changes_at_all():
    prev = _make_summary(features=22)
    curr = _make_summary(features=21)
    result = check(curr, prev, DEFAULTS)
    assert not result.passed
    assert any(b.field == "featuresEffective" for b in result.breaches)


def test_check_fails_when_per_feature_nan_pct_jumps_more_than_tolerance():
    prev = _make_summary(per_feature={
        "precip_gfs":   FeatureStats(NanPct=0.02),
        "precip_ecmwf": FeatureStats(NanPct=0.01),
    })
    curr = _make_summary(per_feature={
        "precip_gfs":   FeatureStats(NanPct=0.02),
        "precip_ecmwf": FeatureStats(NanPct=0.55),
    })
    result = check(curr, prev, DEFAULTS)
    assert not result.passed
    assert any(b.field == "perFeature.precip_ecmwf.nanPct" for b in result.breaches)


def test_check_fails_when_label_rate_shifts_more_than_tolerance():
    prev = _make_summary(label_rates={"ea_bellever_dartmoor": 0.35})
    curr = _make_summary(label_rates={"ea_bellever_dartmoor": 0.20})
    result = check(curr, prev, DEFAULTS)
    assert not result.passed
    assert any(b.field == "labelRates.ea_bellever_dartmoor" for b in result.breaches)


def test_check_aggregates_multiple_breaches_into_one_result():
    prev = _make_summary(rows_train=10_000, features=22)
    curr = _make_summary(rows_train=4_000, features=20)
    result = check(curr, prev, DEFAULTS)
    assert not result.passed
    assert {b.field for b in result.breaches} == {"rowsTrain", "featuresEffective"}


def test_check_skips_relative_row_check_when_previous_count_is_zero():
    prev = _make_summary(rows_train=0)
    curr = _make_summary(rows_train=10_000)
    result = check(curr, prev, DEFAULTS)
    # No rowsTrain breach despite the wild jump — undefined relative on zero.
    assert not any(b.field == "rowsTrain" for b in result.breaches)


def test_check_only_compares_features_present_in_both_summaries():
    # 'c' is new in current; per-feature check skips it (FeaturesEffective
    # already covers set differences).
    prev = _make_summary(features=2, per_feature={
        "a": FeatureStats(NanPct=0.1),
        "b": FeatureStats(NanPct=0.1),
    })
    curr = _make_summary(features=2, per_feature={
        "a": FeatureStats(NanPct=0.1),
        "c": FeatureStats(NanPct=0.95),  # would breach if compared
    })
    result = check(curr, prev, DEFAULTS)
    assert result.passed


# ----- BuildCheckAndSave end-to-end -------------------------------------


def test_build_check_and_save_versioned_writes_summary_when_no_previous(tmp_path):
    log = logging.getLogger("test")
    parent = tmp_path / "precipitation" / "ea_test"
    version_dir = parent / "v-current"
    version_dir.mkdir(parents=True)

    X = _make_features(100, 3)
    result = build_check_and_save_versioned(
        log, version_dir,
        composite="precipitation/ea_test", phase="4a", version="v-current",
        rows_train=100, rows_val=20, rows_test=20,
        train_features=X, feature_names=["a", "b", "c"],
    )
    assert result.passed
    assert "First-ever train" in result.note
    assert (version_dir / "training_summary.json").exists()


def test_build_check_and_save_versioned_finds_previous_in_sibling_dir(tmp_path):
    log = logging.getLogger("test")
    parent = tmp_path / "precipitation" / "ea_test"
    parent.mkdir(parents=True)

    # Previous version dir with phase-tagged metadata + summary.
    prev_dir = parent / "v-old_phase4a"
    prev_dir.mkdir()
    (prev_dir / "training_metadata.json").write_text(
        json.dumps({"Phase": "4a", "Version": "v-old"})
    )
    prev_summary = build_summary(
        composite="precipitation/ea_test", phase="4a", version="v-old",
        rows_train=10_000, rows_val=2_000, rows_test=2_000,
        train_features=_make_features(100, 3),
        feature_names=["a", "b", "c"],
    )
    to_json(prev_summary, prev_dir / "training_summary.json")

    new_dir = parent / "v-new_phase4a"
    new_dir.mkdir()

    # Within tolerance — passes, summary written.
    result = build_check_and_save_versioned(
        log, new_dir,
        composite="precipitation/ea_test", phase="4a", version="v-new",
        rows_train=10_500, rows_val=1_950, rows_test=2_050,
        train_features=_make_features(100, 3),
        feature_names=["a", "b", "c"],
    )
    assert result.passed
    assert (new_dir / "training_summary.json").exists()


def test_build_check_and_save_versioned_does_not_write_on_fail(tmp_path):
    log = logging.getLogger("test")
    parent = tmp_path / "precipitation" / "ea_test"
    parent.mkdir(parents=True)

    prev_dir = parent / "v-old_phase4a"
    prev_dir.mkdir()
    (prev_dir / "training_metadata.json").write_text(json.dumps({"Phase": "4a"}))
    prev_summary = build_summary(
        composite="precipitation/ea_test", phase="4a", version="v-old",
        rows_train=10_000, rows_val=2_000, rows_test=2_000,
        train_features=_make_features(100, 3),
        feature_names=["a", "b", "c"],
    )
    to_json(prev_summary, prev_dir / "training_summary.json")

    new_dir = parent / "v-new_phase4a"
    new_dir.mkdir()

    # Row drop way past tolerance — guard fails.
    result = build_check_and_save_versioned(
        log, new_dir,
        composite="precipitation/ea_test", phase="4a", version="v-new",
        rows_train=2_000, rows_val=400, rows_test=400,
        train_features=_make_features(100, 3),
        feature_names=["a", "b", "c"],
    )
    assert not result.passed
    assert not (new_dir / "training_summary.json").exists(), (
        "guard FAIL must not write the new summary — preserves baseline"
    )


def test_build_check_and_save_singleton_overwrites_on_pass(tmp_path):
    log = logging.getLogger("test")
    summary_path = tmp_path / "training_summary.json"

    # Seed the existing file.
    prev = build_summary(
        composite="precipitation_5a", phase="5a", version="v-old",
        rows_train=10_000, rows_val=2_000, rows_test=2_000,
        train_features=_make_features(100, 3),
        feature_names=["a", "b", "c"],
    )
    to_json(prev, summary_path)

    # Within tolerance — overwrites.
    result = build_check_and_save_singleton(
        log, summary_path,
        composite="precipitation_5a", phase="5a", version="v-new",
        rows_train=10_500, rows_val=1_950, rows_test=2_050,
        train_features=_make_features(100, 3),
        feature_names=["a", "b", "c"],
    )
    assert result.passed
    saved = from_json(summary_path)
    assert saved.Version == "v-new", "singleton path must overwrite on pass"


def test_build_check_and_save_singleton_does_not_overwrite_on_fail(tmp_path):
    log = logging.getLogger("test")
    summary_path = tmp_path / "training_summary.json"

    prev = build_summary(
        composite="precipitation_5a", phase="5a", version="v-old",
        rows_train=10_000, rows_val=2_000, rows_test=2_000,
        train_features=_make_features(100, 3),
        feature_names=["a", "b", "c"],
    )
    to_json(prev, summary_path)

    # Row drop > 30% — guard fails, file should still say v-old.
    result = build_check_and_save_singleton(
        log, summary_path,
        composite="precipitation_5a", phase="5a", version="v-new",
        rows_train=2_000, rows_val=400, rows_test=400,
        train_features=_make_features(100, 3),
        feature_names=["a", "b", "c"],
    )
    assert not result.passed
    saved = from_json(summary_path)
    assert saved.Version == "v-old", "guard FAIL must preserve previous singleton baseline"


def test_build_check_and_save_skips_when_features_empty(tmp_path):
    log = logging.getLogger("test")
    version_dir = tmp_path / "ea_test" / "v-current"
    version_dir.mkdir(parents=True)

    result = build_check_and_save_versioned(
        log, version_dir,
        composite="precipitation/ea_test", phase="4a", version="v-current",
        rows_train=0, rows_val=0, rows_test=0,
        train_features=np.zeros((0, 0)),
        feature_names=[],
    )
    assert result.passed
    assert "Empty train slice" in result.note
    assert not (version_dir / "training_summary.json").exists()


# ----- on-disk previous-summary lookup ---------------------------------


def test_try_load_previous_summary_versioned_filters_by_phase(tmp_path):
    parent = tmp_path
    # Two predecessor dirs — one 4a, one 3a. Looking up 4a must NOT
    # return the 3a summary even though it's lex-newer.
    for ver, phase in [("v-2026-05-09_phase4a", "4a"), ("v-2026-05-10_phase3a", "3a")]:
        d = parent / ver
        d.mkdir()
        (d / "training_metadata.json").write_text(json.dumps({"Phase": phase}))
        s = build_summary(
            composite=f"precipitation/test", phase=phase, version=ver,
            rows_train=1000, rows_val=200, rows_test=200,
            train_features=_make_features(100, 3),
            feature_names=["a", "b", "c"],
        )
        to_json(s, d / "training_summary.json")

    found = try_load_previous_summary_versioned(parent, phase="4a", exclude_version="v-doesnt-matter")
    assert found is not None
    assert found.Phase == "4a"
    assert found.Version == "v-2026-05-09_phase4a"


def test_try_load_previous_summary_versioned_returns_none_on_empty_dir(tmp_path):
    found = try_load_previous_summary_versioned(tmp_path, phase="4a", exclude_version="anything")
    assert found is None


def test_try_load_previous_summary_singleton_returns_none_on_first_train(tmp_path):
    found = try_load_previous_summary_singleton(tmp_path / "training_summary.json")
    assert found is None
