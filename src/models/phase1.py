"""Phase 1 Bayesian logistic regression.

This file is the heart of Phase 1. It defines a PyMC model for
P(precip >= 0.1 mm) at Bellever Dartmoor at lead 24h, samples its
posterior with NUTS, and produces an arviz `InferenceData` object that
the diagnostics + posterior-predictive scripts then operate on.

A note on what "Bayesian" actually changes here
----------------------------------------------
A frequentist logistic regression (sklearn) returns *one* coefficient
per feature, chosen to maximise the likelihood of the training data.
That single value is treated as "the" answer.

A Bayesian model returns a *distribution* over each coefficient,
representing our remaining uncertainty about its value after seeing the
data. The distribution is computed via Bayes' theorem:

    posterior  ~  prior  *  likelihood

The prior encodes what we believed before seeing the data; the
likelihood says how plausible the data is for any candidate parameter
value. The posterior combines them. Because the integral underlying
this has no closed form for non-trivial models, we approximate the
posterior by drawing samples from it via Markov chain Monte Carlo
(NUTS, the No-U-Turn Sampler).

Why we standardise features
---------------------------
NUTS samples poorly when posterior parameters live on very different
scales: the step size that works for a tiny coefficient is far too
small for a large one. Standardising features (subtract train mean,
divide by train std) puts every coefficient on roughly the same scale,
so a single global step size works everywhere. We do the same
standardisation for the frequentist baseline to keep them comparable.

Sampler implementation note
---------------------------
The brief says to use PyMC's default NUTS sampler. That sampler
normally compiles its log-density to C via PyTensor; on this Windows
machine no C++ toolchain is installed, so we use `nutpie`, a Rust
re-implementation of the same NUTS algorithm. The algorithm is
identical; only the language the gradient is computed in differs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler


# Silence the harmless "no C compiler" warning - we're using nutpie.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")


@dataclass
class BayesFitResult:
    idata: object  # arviz.InferenceData
    scaler: StandardScaler
    feature_names: list[str]


def fit_bayesian_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    *,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 2,
    random_seed: int = 42,
) -> BayesFitResult:
    """Fit a Bayesian logistic regression via NUTS and return InferenceData."""

    # Standardise on the training set only. The fitted scaler is returned so
    # the caller can apply the same transform to the test set later.
    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values).astype("float64")
    n_features = X_train_s.shape[1]

    coords = {"feature": feature_names, "obs": np.arange(len(y_train))}

    with pm.Model(coords=coords) as model:
        # Inputs as MutableData so we can swap test data in for posterior
        # predictive sampling without rebuilding the model.
        X_data = pm.Data("X", X_train_s, dims=("obs", "feature"))
        y_data = pm.Data("y", y_train.values.astype("int64"), dims="obs")

        # Priors --------------------------------------------------------
        # Intercept N(0, 2): centred on logit(0.5) = 0, i.e. no preference
        # between wet and dry. SD of 2 spans most of the realistic range
        # (sigmoid(-4..4) covers ~0.02..0.98). Weakly informative: lets the
        # data move it.
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)

        # Coefficients N(0, 1) per feature. With standardised features a
        # coefficient of 1 means "a one-standard-deviation increase in this
        # feature shifts logit(p) by 1", which is a substantial effect.
        # SD=1 says we have no idea of sign or magnitude but doubt the
        # effect is enormous - precisely the meaning of "weakly informative".
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")

        # Linear predictor and likelihood -------------------------------
        logit_p = intercept + pm.math.dot(X_data, beta)
        # `pm.Bernoulli` with `logit_p=` applies the sigmoid internally
        # in a numerically stable way. `observed=` makes it a likelihood
        # term rather than a free random variable.
        pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_data, dims="obs")

        # Sampling ------------------------------------------------------
        # NUTS adapts step size and trajectory length during `tune`
        # iterations (warmup), then collects `draws` posterior samples.
        # Two chains let us diagnose convergence: if both chains explore
        # the same region of parameter space, we have evidence the sampler
        # has actually found the posterior rather than getting stuck in
        # one corner.
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler="nutpie",
            random_seed=random_seed,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    return BayesFitResult(idata=idata, scaler=scaler, feature_names=feature_names)


def predict_posterior(
    fit: BayesFitResult,
    X: pd.DataFrame,
    *,
    random_seed: int = 43,
):
    """Draw posterior predictive samples for new inputs.

    Returns an arviz InferenceData with `posterior_predictive` group whose
    `y_obs` variable has shape (chain, draw, n_obs) - one Bernoulli draw
    per posterior sample per observation. Averaging over (chain, draw)
    gives the posterior mean P(wet) for each row.
    """
    import arviz as az  # local import to keep top-level deps tidy

    X_s = fit.scaler.transform(X.values).astype("float64")
    n_obs = X_s.shape[0]

    # Pull the original model from the InferenceData and rebind the data
    # containers to the test inputs. We need a y placeholder of the right
    # length even though it's not used for prediction.
    coords = {"feature": fit.feature_names, "obs": np.arange(n_obs)}
    with pm.Model(coords=coords) as new_model:
        X_data = pm.Data("X", X_s, dims=("obs", "feature"))
        y_data = pm.Data("y", np.zeros(n_obs, dtype="int64"), dims="obs")
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")
        logit_p = intercept + pm.math.dot(X_data, beta)
        pm.Deterministic("p", pm.math.sigmoid(logit_p), dims="obs")
        pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_data, dims="obs")

        ppc = pm.sample_posterior_predictive(
            fit.idata,
            var_names=["p", "y_obs"],
            random_seed=random_seed,
            progressbar=False,
        )

    return ppc
