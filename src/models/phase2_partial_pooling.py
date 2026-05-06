"""Phase 2 - Partial pooling (Model C).

The hierarchical Bayesian model. Each station has its own intercept and
its own coefficient for each feature, but those station-level parameters
are drawn from a *shared population distribution* whose own parameters
(the "hyperparameters") are estimated from the data.

The big idea
------------
Instead of treating stations as identical (full pooling) or as totally
unrelated (no pooling), we say "stations are similar in some way that I
don't yet quantify; let the data tell me how similar". Mathematically:

    intercept_s   ~  Normal(mu_intercept, sigma_intercept)         per station s
    beta_s,k      ~  Normal(mu_beta_k,  sigma_beta_k)              per station s, feature k

`mu_*` is the *population average*: roughly "what does this parameter
look like across stations on average?".
`sigma_*` is *between-station variation*: roughly "how much do stations
disagree about this parameter?". A small `sigma_intercept` means
stations are very similar in their base wet rate; a large `sigma_intercept`
means they're not.

Crucially, `sigma_*` is itself estimated from the data. If the stations
look identical, the model will learn small `sigma_*` and behave like
full pooling. If they look very different, it will learn large `sigma_*`
and behave like no pooling. Partial pooling slides smoothly between the
two extremes by *learning where on the spectrum to sit*.

Non-centred parameterisation: why we don't write the model as above
-------------------------------------------------------------------
The "centred" formulation (as written above) creates correlated geometry
in the posterior: when `sigma_*` is small, `intercept_s` is forced close
to `mu_intercept`, but the joint distribution `(mu, sigma, intercept_s)`
has a thin "funnel" shape that NUTS struggles to traverse - hence
divergent transitions.

The standard fix is to reparameterise: instead of sampling the
station-level parameter directly, sample a *standard-normal offset* and
construct the parameter algebraically:

    z_s            ~  Normal(0, 1)
    intercept_s    =  mu_intercept + sigma_intercept * z_s

Mathematically equivalent (`intercept_s` still has distribution
`Normal(mu, sigma)`) but the *sampling geometry* is much nicer: NUTS
moves around `(mu, sigma, z)` instead of `(mu, sigma, intercept)`, and
the funnel disappears. This is the single most common reparameterisation
in hierarchical Bayesian modelling, almost always worth doing by default.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pymc as pm

from src.heartbeat import Heartbeat

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")


@dataclass
class PartialPoolingFit:
    idata: object
    feature_names: list[str]
    station_codes: list[str]


def fit_partial_pooling(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    station_idx_train: np.ndarray,
    station_codes: list[str],
    feature_names: list[str],
    *,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
    progressbar: bool = False,
) -> PartialPoolingFit:
    coords = {
        "feature": feature_names,
        "station": station_codes,
        "obs": np.arange(len(y_train)),
    }
    n_features = X_train_s.shape[1]

    with pm.Model(coords=coords):
        X_data = pm.Data("X", X_train_s, dims=("obs", "feature"))
        y_data = pm.Data("y", y_train.astype("int64"), dims="obs")
        s_idx = pm.Data("station_idx", station_idx_train.astype("int64"), dims="obs")

        # ---- Hyperpriors --------------------------------------------------
        # Population-level mean (mu_*) gets the same weakly informative prior
        # as a single-level Phase 1 coefficient: tells us "we expect the
        # cross-station average to look similar to a sensible single fit".
        mu_intercept = pm.Normal("mu_intercept", mu=0.0, sigma=2.0)
        mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0, dims="feature")

        # Between-station SD (sigma_*) must be positive, hence HalfNormal.
        # The active stations (Bellever, Hexworthy, Bovey) sit within
        # ~10 km on Dartmoor at similar elevation/microclimate, so we
        # genuinely expect their coefficients on standardised features to
        # be close. HalfNormal(sigma=0.5) puts the 95% interval on the
        # between-station SD at roughly (0, 1) logit units - permissive
        # enough to learn real differences, tight enough to discourage
        # the funnel geometry that stalls NUTS. (HalfNormal(1.0) was the
        # original choice; partial-pool sampling stalled with it.)
        sigma_intercept = pm.HalfNormal("sigma_intercept", sigma=0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5, dims="feature")

        # ---- Non-centred station-level parameters -------------------------
        # z's are standard-normal offsets. We multiply by sigma_* and add
        # mu_* below to get the actual station parameter values - this is
        # the non-centred reparameterisation.
        z_intercept = pm.Normal("z_intercept", mu=0.0, sigma=1.0, dims="station")
        z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, dims=("station", "feature"))

        # Deterministic transforms - tracked so we can read posteriors of the
        # actual station-level intercept / coefficients in arviz output.
        intercept_s = pm.Deterministic(
            "intercept_s",
            mu_intercept + sigma_intercept * z_intercept,
            dims="station",
        )
        beta_s = pm.Deterministic(
            "beta_s",
            mu_beta[None, :] + sigma_beta[None, :] * z_beta,
            dims=("station", "feature"),
        )

        # ---- Likelihood --------------------------------------------------
        # For each row, pick the right station's intercept and coefficients.
        logit_p = intercept_s[s_idx] + (X_data * beta_s[s_idx]).sum(axis=-1)
        pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_data, dims="obs")

        print(
            f"    [{time.strftime('%H:%M:%S')}] partial_pool: entering pm.sample "
            f"(chains={chains}, draws={draws}, tune={tune}, n_obs={len(y_train)}, n_features={n_features})",
            flush=True,
        )
        t0 = time.time()
        with Heartbeat(f"partial_pool n_obs={len(y_train)}", interval=30):
            # nutpie progress: emit every 50 draws via stderr (its native sink). Run
            # script redirects stderr to a separate file so progress output doesn't
            # tangle with the main log. progress_template uses {chain}/{draws} so
            # the file reads cleanly without TTY carriage-returns.
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                nuts_sampler="nutpie",
                nuts_sampler_kwargs={
                    "progress_rate": 50,
                    "progress_template": "[chain {chain}] draws {draws} | divs {divergences} | step {step_size:.4f}\n",
                },
                random_seed=random_seed,
                target_accept=target_accept,
                progressbar=progressbar,
            )
        print(
            f"    [{time.strftime('%H:%M:%S')}] partial_pool: pm.sample returned in {time.time() - t0:.1f}s",
            flush=True,
        )

    return PartialPoolingFit(
        idata=idata, feature_names=feature_names, station_codes=station_codes
    )


def predict_partial_pooling(
    fit: PartialPoolingFit,
    X_test_s: np.ndarray,
    station_idx_test: np.ndarray,
) -> np.ndarray:
    """Per-row mean posterior P(wet) using each row's station-level coefficients."""
    intercept_s = fit.idata.posterior["intercept_s"].values  # (chain, draw, station)
    beta_s = fit.idata.posterior["beta_s"].values            # (chain, draw, station, feature)

    # Gather station-specific params for each test row, then evaluate.
    # We loop over stations to keep memory predictable; with 3 stations
    # it's fine.
    out = np.empty(len(X_test_s), dtype="float64")
    for s_idx in range(intercept_s.shape[-1]):
        mask = station_idx_test == s_idx
        if not mask.any():
            continue
        ints = intercept_s[..., s_idx]                        # (chain, draw)
        bets = beta_s[..., s_idx, :]                          # (chain, draw, feature)
        logit = ints[..., None] + np.einsum("nf,cdf->cdn", X_test_s[mask], bets)
        out[mask] = (1.0 / (1.0 + np.exp(-logit))).mean(axis=(0, 1))
    return out


def predict_partial_pooling_summary(
    fit: PartialPoolingFit,
    X_test_s: np.ndarray,
    station_idx_test: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.50, 0.90, 0.95),
) -> dict[str, np.ndarray]:
    """Per-row posterior summary of P(wet) — mean, std, and arbitrary quantiles.

    Mirrors `predict_partial_pooling` but keeps the full posterior over the
    chain × draw axis instead of collapsing to the mean. Returns a dict with:
      - "mean": shape (n,)   — same as predict_partial_pooling
      - "std":  shape (n,)   — posterior standard deviation
      - "q{p}": shape (n,)   — posterior quantile at level p (for each p in `quantiles`)

    The credible interval width that matters for Option 2's confidence signal
    is `q0.95 - q0.05` (or `q0.90 - q0.10` for an 80% CI). Wider = the model
    is less certain about this row's P(wet). Phase 4 found that narrow-CI
    rows have ~4-5x lower Brier than wide-CI rows on the same test slice,
    so this width transfers as a usable forecast-skill flag downstream.

    Memory shape: holds (chain, draw, n_per_station) probabilities in
    a single intermediate array. With ~8000 draws (4 chains × 2000) and
    a few hundred test rows per station, this is well under 100MB.
    """
    intercept_s = fit.idata.posterior["intercept_s"].values  # (chain, draw, station)
    beta_s = fit.idata.posterior["beta_s"].values            # (chain, draw, station, feature)

    n = len(X_test_s)
    mean = np.empty(n, dtype="float64")
    std = np.empty(n, dtype="float64")
    quants = {q: np.empty(n, dtype="float64") for q in quantiles}

    for s_idx in range(intercept_s.shape[-1]):
        mask = station_idx_test == s_idx
        if not mask.any():
            continue
        ints = intercept_s[..., s_idx]                        # (chain, draw)
        bets = beta_s[..., s_idx, :]                          # (chain, draw, feature)
        logit = ints[..., None] + np.einsum("nf,cdf->cdn", X_test_s[mask], bets)
        p = 1.0 / (1.0 + np.exp(-logit))                      # (chain, draw, n_mask)
        # Flatten chain + draw into a single posterior axis for quantile
        # computation. Order doesn't matter because the chains are exchangeable
        # under Bayesian inference (no within-chain dependency we care about
        # for marginal quantiles of the posterior predictive).
        flat = p.reshape(-1, p.shape[-1])                     # (chain*draw, n_mask)
        mean[mask] = flat.mean(axis=0)
        std[mask] = flat.std(axis=0)
        qs = np.quantile(flat, list(quantiles), axis=0)       # (n_quantiles, n_mask)
        for qi, q in enumerate(quantiles):
            quants[q][mask] = qs[qi]

    out = {"mean": mean, "std": std}
    for q, arr in quants.items():
        out[f"q{q:g}"] = arr
    return out
