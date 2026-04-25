"""Phase 2 - No pooling (Model A).

Fit a separate Bayesian logistic regression for each station, using only
that station's data. The three fits know nothing about each other; their
parameters are entirely independent.

Why look at this
----------------
No pooling is the minimum-information / maximum-flexibility extreme.
Each station gets exactly the model best for itself - but with no help
from the other stations, so estimates at any data-thin station are
noisier than they need to be. If two stations are actually similar, no
pooling wastes that similarity.

Implementation here is a thin loop: we re-use the full-pooling model
once per station with that station's slice of the training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.phase2_full_pooling import FullPoolingFit, fit_full_pooling, predict_full_pooling


@dataclass
class NoPoolingFit:
    """Container of one full-pooling-shaped fit per station."""

    fits: list[FullPoolingFit]  # parallel to station_codes
    station_codes: list[str]
    feature_names: list[str]


def fit_no_pooling(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    station_idx_train: np.ndarray,
    station_codes: list[str],
    feature_names: list[str],
    *,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    random_seed: int = 42,
) -> NoPoolingFit:
    fits: list[FullPoolingFit] = []
    for s_idx, code in enumerate(station_codes):
        mask = station_idx_train == s_idx
        print(f"  fitting {code}: n={int(mask.sum()):,}")
        # Different seed per station so chains are independent.
        fit = fit_full_pooling(
            X_train_s[mask],
            y_train[mask],
            feature_names,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed + s_idx,
        )
        fits.append(fit)
    return NoPoolingFit(fits=fits, station_codes=station_codes, feature_names=feature_names)


def predict_no_pooling(
    fit: NoPoolingFit,
    X_test_s: np.ndarray,
    station_idx_test: np.ndarray,
) -> np.ndarray:
    """Return per-row mean posterior P(wet) using each row's station fit."""
    out = np.empty(len(X_test_s), dtype="float64")
    for s_idx, station_fit in enumerate(fit.fits):
        mask = station_idx_test == s_idx
        if mask.sum() == 0:
            continue
        out[mask] = predict_full_pooling(station_fit, X_test_s[mask])
    return out
