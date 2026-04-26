"""Phase 3 - Model A: per-lead independent partial pooling.

For each lead in {24, 48, 72} h we fit a Phase 2-shape partial-pooling
model across the three stations on that lead's slice. The three fits
are entirely independent — no information flows between leads.

This is the *baseline* model for Phase 3. Its job is to answer the
question "what predictive performance do we get if leads are treated as
unrelated?" so that the joint 2D hierarchy (Model B) can be judged
fairly against it.

Each individual fit is identical to Phase 2's partial-pool model
(`src/models/phase2_partial_pooling.py`) — same priors, same
non-centring, same sampler config. We literally call `fit_partial_pooling`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.models.phase2_partial_pooling import (
    PartialPoolingFit,
    fit_partial_pooling,
    predict_partial_pooling,
)


@dataclass
class PerLeadFit:
    """Container holding one Phase 2-shape partial-pool fit per lead."""

    fits_by_lead: dict[int, PartialPoolingFit]  # lead_hours -> fit
    feature_names: list[str]
    station_codes: list[str]
    lead_hours: list[int]


def fit_per_lead(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    station_idx_train: np.ndarray,
    lead_idx_train: np.ndarray,
    station_codes: list[str],
    lead_hours: list[int],
    feature_names: list[str],
    *,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
    progressbar: bool = False,
) -> PerLeadFit:
    """Run `fit_partial_pooling` once per lead on its slice of the data."""
    fits: dict[int, PartialPoolingFit] = {}
    for l_idx, lead in enumerate(lead_hours):
        mask = lead_idx_train == l_idx
        n = int(mask.sum())
        print(
            f"  [{time.strftime('%H:%M:%S')}] Model A lead {lead}h: starting fit_partial_pooling (n_train={n})",
            flush=True,
        )
        t0 = time.time()
        # Different seed per lead so chains across leads are independent
        # (matters in case nutpie's RNG is influenced by repeat calls).
        fit = fit_partial_pooling(
            X_train_s[mask],
            y_train[mask],
            station_idx_train[mask],
            station_codes,
            feature_names,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed + l_idx,
            progressbar=progressbar,
        )
        print(
            f"  [{time.strftime('%H:%M:%S')}] Model A lead {lead}h: fit complete in {time.time() - t0:.1f}s",
            flush=True,
        )
        fits[lead] = fit
    return PerLeadFit(
        fits_by_lead=fits,
        feature_names=feature_names,
        station_codes=station_codes,
        lead_hours=list(lead_hours),
    )


def predict_per_lead(
    fit: PerLeadFit,
    X_test_s: np.ndarray,
    station_idx_test: np.ndarray,
    lead_idx_test: np.ndarray,
) -> np.ndarray:
    """Per-row mean posterior P(wet) using each row's (station, lead) cell."""
    out = np.empty(len(X_test_s), dtype="float64")
    for l_idx, lead in enumerate(fit.lead_hours):
        mask = lead_idx_test == l_idx
        if not mask.any():
            continue
        out[mask] = predict_partial_pooling(
            fit.fits_by_lead[lead],
            X_test_s[mask],
            station_idx_test[mask],
        )
    return out
