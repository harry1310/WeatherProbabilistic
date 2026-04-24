"""MCMC diagnostics for the Phase 1 Bayesian model.

Each diagnostic answers a different question about whether we can trust
the posterior samples NUTS produced. Failing any of these is a signal
that the samples should not be used as-is - the model usually needs
reparameterisation, tighter priors, or feature standardisation
(already done here).

R-hat (potential scale reduction factor)
    Compares the variance of samples *within* each chain to the variance
    *between* chains. If chains are sampling the same posterior, these
    should be similar and R-hat ~= 1.0. R-hat > 1.05 means chains
    disagree about the posterior - they're stuck in different regions
    or haven't mixed.

Effective sample size (ESS)
    MCMC samples are autocorrelated: consecutive draws are not
    independent. ESS estimates how many *independent* samples your draws
    are equivalent to. Low ESS means high autocorrelation - estimates
    are noisier than the raw count suggests. Rule of thumb: ESS > 1000
    per parameter is comfortable; < 400 is concerning.

Divergent transitions
    NUTS uses a Hamiltonian dynamics simulation under the hood. When
    the posterior has sharp curvature or thin "funnels", the simulator
    can drift wildly - these are flagged as divergences. Any divergence
    is a warning that the local geometry is being explored badly. Many
    divergences invalidate the run.

Trace plot
    A visual check: each chain should look like a "hairy caterpillar"
    bouncing tightly around the posterior mean, with chains overlapping.
    Trends, drift, or chains in different bands all indicate trouble.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class DiagnosticsReport:
    summary: pd.DataFrame
    max_rhat: float
    min_ess_bulk: float
    n_divergences: int
    trace_plot_path: Path
    issues: list[str]


def run_diagnostics(idata, var_names: list[str], trace_plot_path: Path) -> DiagnosticsReport:
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.94)

    max_rhat = float(summary["r_hat"].max())
    min_ess_bulk = float(summary["ess_bulk"].min())

    # Divergences live in the sample_stats group.
    n_divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_divergences = int(idata.sample_stats["diverging"].sum())

    issues: list[str] = []
    if max_rhat > 1.01:
        issues.append(f"R-hat exceeds 1.01 (max={max_rhat:.3f})")
    if min_ess_bulk < 1000:
        issues.append(f"ESS bulk below 1000 (min={min_ess_bulk:.0f})")
    if n_divergences > 0:
        issues.append(f"{n_divergences} divergent transition(s)")

    az.plot_trace(idata, var_names=var_names, compact=True)
    plt.tight_layout()
    trace_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(trace_plot_path)
    plt.close("all")

    return DiagnosticsReport(
        summary=summary,
        max_rhat=max_rhat,
        min_ess_bulk=min_ess_bulk,
        n_divergences=n_divergences,
        trace_plot_path=trace_plot_path,
        issues=issues,
    )


def print_report(report: DiagnosticsReport) -> None:
    print("\nMCMC diagnostics:")
    print(f"  max R-hat       {report.max_rhat:.4f}   (good if < 1.01)")
    print(f"  min ESS bulk    {report.min_ess_bulk:.0f}     (good if > 1000)")
    print(f"  divergences     {report.n_divergences}        (good if 0)")
    if report.issues:
        print("  ISSUES:")
        for issue in report.issues:
            print(f"    - {issue}")
    else:
        print("  All checks passed.")
    print(f"\nTrace plot saved: {report.trace_plot_path}")
    print("\nPosterior summary:")
    print(report.summary.to_string())
