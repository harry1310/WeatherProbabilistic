"""Phase 1 environment check.

Verifies the Bayesian stack is working before we build anything real.

Why this exists
---------------
PyMC has three moving parts that can all break on a fresh Windows box:

1. PyMC itself (the modelling DSL)
2. PyTensor (the tensor backend - normally compiles C code on first use)
3. A NUTS sampler implementation

On this machine g++ is not installed, so PyTensor would fall back to pure
Python and sampling anything real would take hours. To sidestep that we use
`nutpie`, a Rust-based NUTS sampler that ships as a pre-built wheel and
does not require a C++ toolchain. It plugs into PyMC via
`pm.sample(nuts_sampler="nutpie")`.

This script runs a trivial coin-flip model end to end: define prior +
likelihood, sample the posterior, render a trace plot. If this succeeds
the stack is healthy.
"""

from __future__ import annotations

import os
from pathlib import Path

# Silence the harmless "cxx not set" warning PyTensor emits when there is
# no C++ compiler. We are not using the C backend anyway.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import arviz as az
import matplotlib

matplotlib.use("Agg")  # headless: save figures to disk rather than opening a window
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"pymc    {pm.__version__}")
    print(f"arviz   {az.__version__}")
    print(f"numpy   {np.__version__}")

    # --- Coin-flip model --------------------------------------------------
    # Ten flips, six heads. A Beta(1,1) prior is uniform on [0,1] - we have
    # no prior preference about fairness. The Bernoulli likelihood updates
    # that uniform prior into a posterior that concentrates around 0.6.
    flips = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    print(f"flips   n={len(flips)} heads={flips.sum()}")

    with pm.Model() as coin_model:
        # Prior: our belief about `p` BEFORE seeing data. Uniform here.
        p = pm.Beta("p", alpha=1.0, beta=1.0)
        # Likelihood: how the data depends on `p`.
        pm.Bernoulli("y", p=p, observed=flips)

        # Posterior sampling. NUTS (No-U-Turn Sampler) is the modern default
        # MCMC method; we use nutpie's Rust implementation to avoid needing
        # a C++ compiler. Two chains let us diagnose convergence by checking
        # they agree.
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            nuts_sampler="nutpie",
            random_seed=42,
            progressbar=False,
        )

    # --- Summary + diagnostics -------------------------------------------
    summary = az.summary(idata, var_names=["p"])
    print("\nPosterior summary:")
    print(summary.to_string())

    # R-hat near 1.0 means the chains agree (good). ESS is the effective
    # sample size after accounting for autocorrelation.
    rhat = float(summary.loc["p", "r_hat"])
    ess = float(summary.loc["p", "ess_bulk"])
    assert rhat < 1.05, f"R-hat too high: {rhat}"
    assert ess > 400, f"ESS too low: {ess}"

    # --- Trace plot -------------------------------------------------------
    az.plot_trace(idata, var_names=["p"])
    out = REPORTS_DIR / "env_check_trace.pdf"
    plt.tight_layout()
    plt.savefig(out)
    plt.close("all")
    print(f"\nTrace plot saved: {out}")

    print("\nEnvironment OK.")


if __name__ == "__main__":
    main()
