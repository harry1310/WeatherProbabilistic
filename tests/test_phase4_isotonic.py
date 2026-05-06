"""Phase 4.5 isotonic post-processing — unit + invariant tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.isotonic import (  # noqa: E402
    IsotonicCalibratorBundle, apply_per_cell, calibration_error,
    fit_per_cell, load_bundle, reliability_bins, save_bundle,
    split_calibration_eval,
)


# ------------------------------------------------------------------
# Synthetic fixtures — small, fast, no I/O
# ------------------------------------------------------------------

def _make_synthetic(n_per_cell: int = 200, seed: int = 0) -> pd.DataFrame:
    """Two stations × three leads, biased Bayesian-style miscalibration:
    raw probabilities are systematically too low (a Bayesian under-
    prediction artefact). Calibration should pull them upward."""
    rng = np.random.default_rng(seed)
    rows = []
    base_time = pd.Timestamp("2025-06-01 00:00:00")
    for station in ["Bellever", "Hexworthy"]:
        for lead in [24, 48, 72]:
            # True wet rate ~28%
            y = (rng.uniform(0, 1, n_per_cell) < 0.28).astype("int8")
            # Underpredicted probability: shrink true rate by 0.6
            base_p = 0.28 * 0.6
            p = np.clip(base_p + rng.normal(0, 0.05, n_per_cell)
                        + 0.4 * y * rng.uniform(0.5, 1.0, n_per_cell), 0.001, 0.999)
            for i in range(n_per_cell):
                rows.append(dict(
                    valid_time=base_time + pd.Timedelta(hours=i),
                    station=station, lead=lead,
                    p_wet=float(p[i]), observed_wet=int(y[i]),
                ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# split_calibration_eval
# ------------------------------------------------------------------

def test_split_is_disjoint_and_chronological():
    df = _make_synthetic()
    calib, eval_ = split_calibration_eval(df)
    # 50/50 per cell
    for cell in df.groupby(["station", "lead"]).size().index:
        c = calib[(calib.station == cell[0]) & (calib.lead == cell[1])]
        e = eval_[(eval_.station == cell[0]) & (eval_.lead == cell[1])]
        assert len(c) == len(e), f"{cell}: calib={len(c)} eval={len(e)}"
        # Chronological — every calib row precedes every eval row in this cell
        assert c["valid_time"].max() < e["valid_time"].min(), \
            f"{cell}: calib max time >= eval min time → leakage!"
    # Total partition is whole input, no overlap
    assert len(calib) + len(eval_) == len(df)


def test_split_handles_unequal_cell_sizes():
    df = _make_synthetic(n_per_cell=99)  # odd → cut at 49/50
    calib, eval_ = split_calibration_eval(df)
    for (station, lead), grp in df.groupby(["station", "lead"]):
        c = calib[(calib.station == station) & (calib.lead == lead)]
        e = eval_[(eval_.station == station) & (eval_.lead == lead)]
        assert len(c) + len(e) == len(grp)
        assert abs(len(c) - len(e)) <= 1


# ------------------------------------------------------------------
# fit_per_cell + apply_per_cell
# ------------------------------------------------------------------

def test_fit_returns_one_calibrator_per_cell():
    df = _make_synthetic()
    calib, _ = split_calibration_eval(df)
    bundle = fit_per_cell(calib)
    expected = {(lead, station) for station in ["Bellever", "Hexworthy"] for lead in [24, 48, 72]}
    assert set(bundle.calibrators.keys()) == expected


def test_calibrators_are_monotonic():
    """IsotonicRegression is monotonic by construction; pin the invariant."""
    df = _make_synthetic()
    calib, _ = split_calibration_eval(df)
    bundle = fit_per_cell(calib)
    grid = np.linspace(0.0, 1.0, 1001)
    for key, cal in bundle.calibrators.items():
        out = cal.predict(grid)
        diffs = np.diff(out)
        assert (diffs >= -1e-12).all(), \
            f"{key}: non-monotonic — min diff = {diffs.min()}"


def test_calibrator_outputs_in_unit_interval():
    df = _make_synthetic()
    calib, eval_ = split_calibration_eval(df)
    bundle = fit_per_cell(calib)
    out = apply_per_cell(bundle, eval_)
    assert (out["p_wet_cal"] >= 0).all() and (out["p_wet_cal"] <= 1).all()


def test_calibration_corrects_systematic_bias():
    """Underpredicted synthetic → calibration error should be much smaller after fit."""
    df = _make_synthetic(n_per_cell=500, seed=42)
    calib, eval_ = split_calibration_eval(df)
    bundle = fit_per_cell(calib)
    out = apply_per_cell(bundle, eval_)
    raw_err = calibration_error(out["p_wet"].to_numpy(), out["observed_wet"].to_numpy())
    cal_err = calibration_error(out["p_wet_cal"].to_numpy(), out["observed_wet"].to_numpy())
    # Synthetic was deliberately badly miscalibrated; after fitting the
    # calibration error should drop by at least 50%.
    assert cal_err < raw_err * 0.5, f"raw_err={raw_err:.4f} cal_err={cal_err:.4f}"


# ------------------------------------------------------------------
# save / load round-trip
# ------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    df = _make_synthetic()
    calib, eval_ = split_calibration_eval(df)
    bundle = fit_per_cell(calib)
    save_bundle(bundle, tmp_path / "bundle")
    reloaded = load_bundle(tmp_path / "bundle")
    assert set(reloaded.calibrators.keys()) == set(bundle.calibrators.keys())
    # Same predictions on a probe grid
    probe = np.linspace(0.01, 0.99, 50)
    for key in bundle.calibrators:
        np.testing.assert_allclose(
            bundle.calibrators[key].predict(probe),
            reloaded.calibrators[key].predict(probe),
            atol=1e-12,
        )


# ------------------------------------------------------------------
# Reliability + ECE sanity
# ------------------------------------------------------------------

def test_calibration_error_zero_for_perfect_predictions():
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0, 1, n) < p).astype("int8")
    # Perfect probabilities → ECE close to 0 (but non-zero from binning noise)
    err = calibration_error(p, y, n_bins=10)
    assert err < 0.05, f"ECE for ground-truth probabilities = {err:.4f}"


def test_reliability_bins_drop_empty_bins():
    p = np.array([0.05, 0.07, 0.95])  # only bin 0 + bin 9 populated
    y = np.array([0, 0, 1])
    bins = reliability_bins(p, y, n_bins=10)
    assert len(bins) == 2
    assert sorted(bins["bin"].tolist()) == [0, 9]


# ------------------------------------------------------------------
# Phase 3 Model A artefacts unchanged invariant
# ------------------------------------------------------------------

def test_phase3_artefacts_untouched_by_calibration_workflow(tmp_path):
    """Invariant: nothing in the calibration module writes into the
    bayesian_predictions/ tree. We only ever read from there."""
    import inspect
    import src.calibration.isotonic as iso
    src_text = inspect.getsource(iso)
    # No write paths anywhere in the module reference the bayesian_predictions tree
    forbidden = ["bayesian_predictions/lead_", "bayesian_predictions\\\\lead_"]
    for forbidden_str in forbidden:
        assert forbidden_str not in src_text, \
            f"isotonic module references {forbidden_str} — risk of overwriting Phase 3 artefacts"
