"""Phase 2 - Full pooling (Model B).

Treat the three stations as identical. We pool all rows from all stations
into one logistic regression, ignoring which station each row came from.
The model has *one* intercept and *one* coefficient per feature; those
single estimates apply to every station.

Why look at this
----------------
Full pooling is the maximum-information / minimum-flexibility extreme.
It uses every row to estimate one shared set of parameters, so estimates
are sharp - but if stations actually differ (a wetter site, a different
diurnal pattern), the model has nowhere to express that.

The model structure is the same as Phase 1's single-station model -
because that's exactly what "ignore station identity" means.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pymc as pm

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")


@dataclass
class FullPoolingFit:
    idata: object
    feature_names: list[str]


def fit_full_pooling(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    random_seed: int = 42,
) -> FullPoolingFit:
    coords = {"feature": feature_names, "obs": np.arange(len(y_train))}
    with pm.Model(coords=coords):
        X_data = pm.Data("X", X_train_s, dims=("obs", "feature"))
        y_data = pm.Data("y", y_train.astype("int64"), dims="obs")

        # Same priors as Phase 1.
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")

        logit_p = intercept + pm.math.dot(X_data, beta)
        pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_data, dims="obs")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler="nutpie",
            random_seed=random_seed,
            progressbar=False,
        )

    return FullPoolingFit(idata=idata, feature_names=feature_names)


def predict_full_pooling(fit: FullPoolingFit, X_test_s: np.ndarray) -> np.ndarray:
    """Return per-row mean posterior P(wet) of shape (n_test,)."""
    intercept = fit.idata.posterior["intercept"].values  # (chain, draw)
    beta = fit.idata.posterior["beta"].values            # (chain, draw, feature)
    # (chain, draw, n_test) = intercept[..., None] + (X @ beta_T)
    logit = intercept[..., None] + np.einsum("nf,cdf->cdn", X_test_s, beta)
    p = 1.0 / (1.0 + np.exp(-logit))
    return p.mean(axis=(0, 1))
