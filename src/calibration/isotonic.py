"""Isotonic post-processing for Bayesian P(wet) predictions (Phase 4.5).

Pure functions: split / fit / apply / evaluate / save / load. The script
that drives them lives at scripts/run_phase4_isotonic.py.

Calibration *only* adjusts point estimates. The underlying posterior over
parameters and the per-row credible-interval structure are unaffected —
isotonic post-processing is a monotone rescaling on the scalar
posterior-mean output. Phase 5's Monte Carlo work can use the calibrated
point estimates with the original CI structure intact.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


CalibKey = tuple[int, str]  # (lead_hours, station_code)


@dataclass
class IsotonicCalibratorBundle:
    """One IsotonicRegression per (lead, station) cell + provenance metadata."""
    calibrators: dict[CalibKey, IsotonicRegression]
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, lead: int, station: str, raw_probabilities: np.ndarray) -> np.ndarray:
        cal = self.calibrators[(lead, station)]
        return cal.predict(np.asarray(raw_probabilities, dtype="float64"))


def split_calibration_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(lead, station) chronological 50/50 split of test rows.

    The "calibration set" here is NOT a held-out validation slice in the
    original-training sense — it's test rows that the Bayesian model didn't
    see during fitting, used to fit the post-hoc map. The "eval set" is the
    second chronological half, used for Brier/calibration-error reporting
    and for apples-to-apples comparison against LightGBM evaluated on the
    same rows.

    Splitting per-cell ensures every cell has both a calibration half and
    an eval half even if cell sizes vary. The split is on row position
    after sorting by valid_time, so each half is a contiguous time window.
    """
    calib_chunks, eval_chunks = [], []
    for (lead, station), grp in df.groupby(["lead", "station"], sort=True):
        g = grp.sort_values("valid_time").reset_index(drop=True)
        cut = len(g) // 2
        calib_chunks.append(g.iloc[:cut])
        eval_chunks.append(g.iloc[cut:])
    return (
        pd.concat(calib_chunks, ignore_index=True),
        pd.concat(eval_chunks, ignore_index=True),
    )


def fit_per_cell(calib_df: pd.DataFrame) -> IsotonicCalibratorBundle:
    """Fit one IsotonicRegression(p_wet -> observed_wet) per (lead, station)."""
    calibrators: dict[CalibKey, IsotonicRegression] = {}
    summary_rows = []
    for (lead, station), grp in calib_df.groupby(["lead", "station"], sort=True):
        x = grp["p_wet"].to_numpy(dtype="float64")
        y = grp["observed_wet"].to_numpy(dtype="float64")
        # out_of_bounds='clip' = at predict time, x outside the fit-time
        # range is clipped to the nearest endpoint rather than NaN'd.
        cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        cal.fit(x, y)
        calibrators[(int(lead), str(station))] = cal
        summary_rows.append(dict(
            lead=int(lead), station=str(station),
            n_calib=int(len(grp)),
            calib_wet_rate=float(y.mean()),
            calib_p_mean=float(x.mean()),
            n_knots=len(cal.X_thresholds_),
        ))
    return IsotonicCalibratorBundle(
        calibrators=calibrators,
        metadata=dict(
            fit_date_utc=datetime.now(timezone.utc).isoformat(),
            n_calibrators=len(calibrators),
            cells=summary_rows,
        ),
    )


def apply_per_cell(bundle: IsotonicCalibratorBundle, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Apply each cell's calibrator to the cell's raw p_wet, return a copy
    with the new column `p_wet_cal` alongside the original `p_wet`."""
    out = eval_df.copy()
    out["p_wet_cal"] = np.nan
    for (lead, station), grp in eval_df.groupby(["lead", "station"], sort=True):
        mask = (out["lead"] == lead) & (out["station"] == station)
        cal = bundle.calibrators.get((int(lead), str(station)))
        if cal is None:
            continue
        out.loc[mask, "p_wet_cal"] = cal.predict(grp["p_wet"].to_numpy(dtype="float64"))
    return out


def reliability_bins(probs: np.ndarray, observed: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width binning [0,1]; returns mean predicted prob, observed
    wet rate, and bin count per bin. Empty bins are dropped."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(dict(
            bin=b,
            edge_lo=float(bins[b]),
            edge_hi=float(bins[b + 1]),
            n=int(m.sum()),
            p_mean=float(probs[m].mean()),
            obs_rate=float(observed[m].mean()),
        ))
    return pd.DataFrame(rows)


def calibration_error(probs: np.ndarray, observed: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted L1 distance between bin-mean
    predicted probability and bin-mean observed wet rate. Same metric the
    Phase 4 compare script reports."""
    bins = reliability_bins(probs, observed, n_bins)
    if bins.empty:
        return float("nan")
    weights = bins["n"] / bins["n"].sum()
    return float((weights * (bins["p_mean"] - bins["obs_rate"]).abs()).sum())


def save_bundle(bundle: IsotonicCalibratorBundle, root: Path) -> None:
    """Save one calibrator.pkl per cell + a single bundle_metadata.json."""
    import json
    root.mkdir(parents=True, exist_ok=True)
    for (lead, station), cal in bundle.calibrators.items():
        cell_dir = root / f"lead_{lead}h_{station}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        with open(cell_dir / "calibrator.pkl", "wb") as f:
            pickle.dump(cal, f)
    (root / "bundle_metadata.json").write_text(json.dumps(bundle.metadata, indent=2))


def load_bundle(root: Path) -> IsotonicCalibratorBundle:
    import json
    metadata = json.loads((root / "bundle_metadata.json").read_text())
    calibrators: dict[CalibKey, IsotonicRegression] = {}
    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith("lead_"):
            continue
        # lead_24h_Bellever -> 24, "Bellever"
        body = cell_dir.name[len("lead_"):]
        lead_str, station = body.split("h_", 1)
        with open(cell_dir / "calibrator.pkl", "rb") as f:
            calibrators[(int(lead_str), station)] = pickle.load(f)
    return IsotonicCalibratorBundle(calibrators=calibrators, metadata=metadata)
