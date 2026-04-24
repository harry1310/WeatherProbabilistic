"""Phase 1 end-to-end runner.

Loads data, fits the frequentist baseline, fits the Bayesian model,
runs diagnostics, runs posterior predictive on the test set, writes a
comparison table and a markdown report.

Run with:
    .venv/Scripts/python.exe scripts/run_phase1.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Silence the harmless "no C compiler" warning before PyTensor imports.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

# Make `src.*` importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

from src.baseline import fit_frequentist_baseline  # noqa: E402
from src.data import prepare_phase1_dataset  # noqa: E402
from src.diagnostics import print_report, run_diagnostics  # noqa: E402
from src.models.phase1 import fit_bayesian_logistic, predict_posterior  # noqa: E402


REPORTS_DIR = ROOT / "reports"
ARTEFACTS_DIR = REPORTS_DIR / "phase1_artefacts"
ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 70)
    print("Phase 1: Bayesian Logistic Regression for P(wet) at Bellever, lead 24h")
    print("=" * 70)

    # ---- Data --------------------------------------------------------
    ds = prepare_phase1_dataset()

    # ---- Frequentist baseline ---------------------------------------
    print("\n--- Frequentist baseline ---")
    freq = fit_frequentist_baseline(
        ds.X_train, ds.y_train, ds.X_test, ds.y_test, ds.feature_names
    )
    print(f"  intercept    {freq.intercept:+.4f}")
    print(f"  test Brier   {freq.test_brier:.4f}")
    print(f"  test logloss {freq.test_log_loss:.4f}")
    print("  coefficients (standardised features):")
    for name, val in freq.coefficients.items():
        print(f"    {name:<32s}{val:+.4f}")

    # ---- Bayesian fit ------------------------------------------------
    print("\n--- Bayesian logistic regression (NUTS via nutpie) ---")
    fit = fit_bayesian_logistic(
        ds.X_train, ds.y_train, ds.feature_names,
        draws=2000, tune=2000, chains=2, random_seed=42,
    )

    # ---- Diagnostics -------------------------------------------------
    diag = run_diagnostics(
        fit.idata,
        var_names=["intercept", "beta"],
        trace_plot_path=REPORTS_DIR / "phase1_diagnostics.pdf",
    )
    print_report(diag)

    # Save the InferenceData (NetCDF) for reproducibility.
    idata_path = ARTEFACTS_DIR / "posterior.nc"
    fit.idata.to_netcdf(idata_path)
    print(f"\nPosterior saved: {idata_path}")

    # ---- Posterior predictive ---------------------------------------
    print("\n--- Posterior predictive on test set ---")
    ppc = predict_posterior(fit, ds.X_test, random_seed=43)

    # `p` is the per-row sigmoid(logit_p) at each posterior sample.
    # Posterior mean of p = point-estimate probability for each test row.
    p_samples = ppc.posterior_predictive["p"].values  # (chain, draw, obs)
    p_mean = p_samples.mean(axis=(0, 1))
    p_lo, p_hi = np.quantile(p_samples, [0.03, 0.97], axis=(0, 1))

    bayes_brier = float(brier_score_loss(ds.y_test.values, p_mean))
    bayes_logloss = float(log_loss(ds.y_test.values, p_mean))
    print(f"  test Brier   {bayes_brier:.4f}")
    print(f"  test logloss {bayes_logloss:.4f}")

    # PPC sanity check: fraction of wet hours in posterior predictive draws
    # vs observed fraction. Big mismatch would suggest miscalibration.
    y_pp = ppc.posterior_predictive["y_obs"].values  # (chain, draw, obs)
    pp_wet_fraction = y_pp.mean()  # over all chains/draws/obs
    obs_wet_fraction = ds.y_test.mean()
    print(f"  observed wet fraction (test):  {obs_wet_fraction:.3f}")
    print(f"  posterior pred wet fraction:    {pp_wet_fraction:.3f}")

    # ---- Comparison table -------------------------------------------
    delta_brier_pct = (bayes_brier - freq.test_brier) / freq.test_brier * 100.0

    print("\n--- Comparison ---")
    print(f"                       Frequentist     Bayesian (point)")
    print(f"  Brier                {freq.test_brier:.4f}          {bayes_brier:.4f}")
    print(f"  Log loss             {freq.test_log_loss:.4f}          {bayes_logloss:.4f}")
    print(f"  Brier delta vs freq: {delta_brier_pct:+.2f}%")

    if abs(delta_brier_pct) > 10.0:
        print("  WARNING: > 10% Brier difference - investigate.")

    # Posterior summary for coefficients (mean, std, 94% CI), comparing to
    # frequentist coefficients side-by-side.
    posterior_beta = fit.idata.posterior["beta"]
    posterior_intercept = fit.idata.posterior["intercept"]

    rows = []
    rows.append({
        "param": "intercept",
        "freq": freq.intercept,
        "bayes_mean": float(posterior_intercept.mean()),
        "bayes_sd": float(posterior_intercept.std()),
        "bayes_lo94": float(posterior_intercept.quantile(0.03)),
        "bayes_hi94": float(posterior_intercept.quantile(0.97)),
    })
    for i, name in enumerate(ds.feature_names):
        post = posterior_beta.isel(feature=i)
        rows.append({
            "param": name,
            "freq": float(freq.coefficients[name]),
            "bayes_mean": float(post.mean()),
            "bayes_sd": float(post.std()),
            "bayes_lo94": float(post.quantile(0.03)),
            "bayes_hi94": float(post.quantile(0.97)),
        })
    coef_table = pd.DataFrame(rows)
    coef_table.to_csv(ARTEFACTS_DIR / "coefficient_comparison.csv", index=False)
    print("\nCoefficient comparison (sklearn vs Bayesian posterior):")
    print(coef_table.to_string(index=False))

    # ---- Persist key metrics for the report -------------------------
    metrics = {
        "n_train": int(len(ds.y_train)),
        "n_test": int(len(ds.y_test)),
        "wet_fraction_train": float(ds.y_train.mean()),
        "wet_fraction_test": float(ds.y_test.mean()),
        "feature_names": ds.feature_names,
        "freq_brier": freq.test_brier,
        "freq_log_loss": freq.test_log_loss,
        "freq_intercept": freq.intercept,
        "freq_coefs": freq.coefficients.to_dict(),
        "bayes_brier": bayes_brier,
        "bayes_log_loss": bayes_logloss,
        "delta_brier_pct": delta_brier_pct,
        "max_rhat": diag.max_rhat,
        "min_ess_bulk": diag.min_ess_bulk,
        "n_divergences": diag.n_divergences,
        "diagnostic_issues": diag.issues,
        "obs_wet_fraction_test": float(obs_wet_fraction),
        "ppc_wet_fraction": float(pp_wet_fraction),
        "train_start": str(ds.valid_time_train.min()),
        "train_end": str(ds.valid_time_train.max()),
        "test_start": str(ds.valid_time_test.min()),
        "test_end": str(ds.valid_time_test.max()),
        "p_mean_test_first10": p_mean[:10].tolist(),
        "p_lo94_test_first10": p_lo[:10].tolist(),
        "p_hi94_test_first10": p_hi[:10].tolist(),
    }
    with (ARTEFACTS_DIR / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nArtefacts written to {ARTEFACTS_DIR}")
    print("Run complete.")


if __name__ == "__main__":
    main()
