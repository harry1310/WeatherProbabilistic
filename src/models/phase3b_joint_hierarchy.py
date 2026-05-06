"""Phase 3 - Model B: joint two-dimensional hierarchy (station × lead).

Phase 2 had one hierarchical dimension: stations. The intercept and each
β were per-station, drawn from a shared `Normal(μ, σ)`. The σ
hyperparameter told us how much stations differed.

Phase 3 adds a second dimension — lead time — and asks the same question
twice. Each parameter (intercept, β per feature) is now a value per
(station, lead) *cell*, and we decompose its variation into four pieces:

    parameter[s, l] = μ                         # population mean
                    + σ_station   · z_s         # which station you're in
                    + σ_lead      · z_l         # which lead you're at
                    + σ_inter     · z_{s,l}     # joint-cell residual

Each `z_*` is drawn `Normal(0, 1)` (non-centred parameterisation; see
below). The σ hyperparameters are estimated from the data and tell us:

* `σ_station`     - how much do stations disagree about this parameter,
                    averaged over leads?
* `σ_lead`        - how much does this parameter change with lead time,
                    averaged over stations?
* `σ_inter`       - how much *additional* variation is there in
                    (station, lead) cells beyond what the two main
                    effects predict?

If `σ_inter` is small for a feature, its (station, lead) effects are
essentially additive: "Bellever's 72h coefficient = Bellever effect
+ 72h effect", with no special interaction. If `σ_inter` is large, the
combination matters non-additively.

This is the scientific payoff of Phase 3. Predictively it may match
per-lead partial-pooling (Model A) for the same reason Phase 2's partial
pooling matched no-pooling: each (station, lead) cell already has 8-10k
rows so there's not much borrow-strength benefit. The σ posteriors are
where the interesting findings live.

Why non-centred *especially* matters in 2D
------------------------------------------
A 1D hierarchy has one funnel: the joint (μ, σ, parameter) for each
parameter family. A 2D hierarchy nests three funnels per family — one
for the station dimension, one for the lead dimension, and one for the
interaction. Each one needs reparameterising. The model below samples
all three `z_*` arrays as standard normals and constructs the per-cell
parameter values algebraically, which is what makes NUTS tractable on
this scale.

Priors
------
σ_station_*  ~ HalfNormal(0.5)   # same as Phase 2's partial-pool prior
σ_lead_*     ~ HalfNormal(0.5)   # same prior — we don't know a-priori
                                  # whether lead-variation is bigger or
                                  # smaller than station-variation
σ_inter_*    ~ HalfNormal(0.3)   # tighter — interactions should be
                                  # smaller than main effects unless
                                  # there's strong physical reason
                                  # otherwise. The brief justifies this.

The HalfNormal(0.5) prior was empirically tractable in Phase 2;
HalfNormal(1.0) tipped the sampler over a runtime cliff. We start
Phase 3 with the proven-tractable 0.5 to keep the model from having
to fight three funnels at once.
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
class JointHierarchyFit:
    idata: object
    feature_names: list[str]
    station_codes: list[str]
    lead_hours: list[int]


def fit_joint_hierarchy(
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
    sigma_station_prior: float = 0.5,
    sigma_lead_prior: float = 0.5,
    sigma_interaction_prior: float = 0.3,
) -> JointHierarchyFit:
    coords = {
        "feature": feature_names,
        "station": station_codes,
        "lead": [f"{h}h" for h in lead_hours],
        "obs": np.arange(len(y_train)),
    }
    n_features = X_train_s.shape[1]
    n_stations = len(station_codes)
    n_leads = len(lead_hours)

    with pm.Model(coords=coords):
        X_data = pm.Data("X", X_train_s, dims=("obs", "feature"))
        y_data = pm.Data("y", y_train.astype("int64"), dims="obs")
        s_idx = pm.Data("station_idx", station_idx_train.astype("int64"), dims="obs")
        l_idx = pm.Data("lead_idx", lead_idx_train.astype("int64"), dims="obs")

        # ---- Population means -------------------------------------------------
        # These are the "single fit you'd write down if you ignored the
        # hierarchy entirely" parameters. Same priors as Phase 1/2.
        mu_intercept = pm.Normal("mu_intercept", mu=0.0, sigma=2.0)
        mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0, dims="feature")

        # ---- σ hyperparameters ------------------------------------------------
        # Three SD families per parameter (intercept and each β):
        #   _station_   - how much the parameter varies across stations
        #   _lead_      - how much it varies across leads
        #   _interaction_ - residual (station, lead) deviation beyond the
        #                   sum of the two main effects.
        sigma_station_intercept = pm.HalfNormal(
            "sigma_station_intercept", sigma=sigma_station_prior
        )
        sigma_lead_intercept = pm.HalfNormal(
            "sigma_lead_intercept", sigma=sigma_lead_prior
        )
        sigma_interaction_intercept = pm.HalfNormal(
            "sigma_interaction_intercept", sigma=sigma_interaction_prior
        )

        sigma_station_beta = pm.HalfNormal(
            "sigma_station_beta", sigma=sigma_station_prior, dims="feature"
        )
        sigma_lead_beta = pm.HalfNormal(
            "sigma_lead_beta", sigma=sigma_lead_prior, dims="feature"
        )
        sigma_interaction_beta = pm.HalfNormal(
            "sigma_interaction_beta", sigma=sigma_interaction_prior, dims="feature"
        )

        # ---- Non-centred station / lead / interaction offsets -----------------
        # `z_*` are unit Normals; the actual per-cell parameter values are
        # built algebraically below. This reparameterisation is what makes
        # NUTS able to traverse the (μ, σ, cell) joint posterior without
        # divergences on the funnel geometry.
        z_station_intercept = pm.Normal(
            "z_station_intercept", mu=0.0, sigma=1.0, dims="station"
        )
        z_lead_intercept = pm.Normal(
            "z_lead_intercept", mu=0.0, sigma=1.0, dims="lead"
        )
        z_interaction_intercept = pm.Normal(
            "z_interaction_intercept", mu=0.0, sigma=1.0, dims=("station", "lead")
        )
        z_station_beta = pm.Normal(
            "z_station_beta", mu=0.0, sigma=1.0, dims=("station", "feature")
        )
        z_lead_beta = pm.Normal(
            "z_lead_beta", mu=0.0, sigma=1.0, dims=("lead", "feature")
        )
        z_interaction_beta = pm.Normal(
            "z_interaction_beta", mu=0.0, sigma=1.0,
            dims=("station", "lead", "feature"),
        )

        # ---- Per-cell parameters (deterministic transforms) -------------------
        # intercept[s, l] = μ + σ_st·z_s + σ_ld·z_l + σ_int·z_{s,l}
        intercept_sl = pm.Deterministic(
            "intercept_sl",
            mu_intercept
            + sigma_station_intercept * z_station_intercept[:, None]
            + sigma_lead_intercept * z_lead_intercept[None, :]
            + sigma_interaction_intercept * z_interaction_intercept,
            dims=("station", "lead"),
        )
        # β[s, l, k] = μ_β[k] + σ_st_β[k]·z_st[s,k] + σ_ld_β[k]·z_ld[l,k]
        #             + σ_int_β[k]·z_int[s,l,k]
        beta_sl = pm.Deterministic(
            "beta_sl",
            mu_beta[None, None, :]
            + sigma_station_beta[None, None, :] * z_station_beta[:, None, :]
            + sigma_lead_beta[None, None, :] * z_lead_beta[None, :, :]
            + sigma_interaction_beta[None, None, :] * z_interaction_beta,
            dims=("station", "lead", "feature"),
        )

        # ---- Likelihood -------------------------------------------------------
        # For each observation, gather its cell's intercept and β vector.
        logit_p = (
            intercept_sl[s_idx, l_idx]
            + (X_data * beta_sl[s_idx, l_idx]).sum(axis=-1)
        )
        pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_data, dims="obs")

        print(
            f"  [{time.strftime('%H:%M:%S')}] Model B: entering pm.sample "
            f"(chains={chains}, draws={draws}, tune={tune}, n_obs={len(y_train)}, "
            f"n_stations={n_stations}, n_leads={n_leads}, n_features={n_features})",
            flush=True,
        )
        t0 = time.time()
        with Heartbeat(f"Model B n_obs={len(y_train)}", interval=30):
            # nutpie progress: every 50 draws via stderr (its native sink). The
            # script redirects stderr separately so progress doesn't tangle with stdout.
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
            f"  [{time.strftime('%H:%M:%S')}] Model B: pm.sample returned in {time.time() - t0:.1f}s",
            flush=True,
        )

    return JointHierarchyFit(
        idata=idata,
        feature_names=feature_names,
        station_codes=station_codes,
        lead_hours=list(lead_hours),
    )


def predict_joint_hierarchy(
    fit: JointHierarchyFit,
    X_test_s: np.ndarray,
    station_idx_test: np.ndarray,
    lead_idx_test: np.ndarray,
) -> np.ndarray:
    """Per-row mean posterior P(wet) using each row's (station, lead) cell."""
    intercept = fit.idata.posterior["intercept_sl"].values  # (chain, draw, station, lead)
    beta = fit.idata.posterior["beta_sl"].values            # (chain, draw, station, lead, feature)

    out = np.empty(len(X_test_s), dtype="float64")
    n_stations, n_leads = intercept.shape[-2], intercept.shape[-1]
    for s in range(n_stations):
        for l in range(n_leads):
            mask = (station_idx_test == s) & (lead_idx_test == l)
            if not mask.any():
                continue
            ints = intercept[..., s, l]                # (chain, draw)
            bets = beta[..., s, l, :]                  # (chain, draw, feature)
            logit = ints[..., None] + np.einsum("nf,cdf->cdn", X_test_s[mask], bets)
            out[mask] = (1.0 / (1.0 + np.exp(-logit))).mean(axis=(0, 1))
    return out
