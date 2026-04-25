"""Phase 2 end-to-end runner.

Fits three models (no-pooling, full-pooling, partial-pooling) on the
three-station dataset, runs diagnostics, computes per-station posterior
predictive metrics + reliability binning, and tabulates the headline
comparison.

Run with:
    .venv/Scripts/python.exe scripts/run_phase2.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import arviz as az  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

from src.data import prepare_phase2_dataset  # noqa: E402
from src.diagnostics import run_diagnostics  # noqa: E402
from src.models.phase2_full_pooling import (  # noqa: E402
    FullPoolingFit,
    fit_full_pooling,
    predict_full_pooling,
)
from src.models.phase2_no_pooling import (  # noqa: E402
    NoPoolingFit,
    fit_no_pooling,
    predict_no_pooling,
)
from src.models.phase2_partial_pooling import (  # noqa: E402
    fit_partial_pooling,
    predict_partial_pooling,
)


REPORTS_DIR = ROOT / "reports"
ARTEFACTS_DIR = REPORTS_DIR / "phase2_artefacts"
ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Sampler defaults applied to all three models.
SAMPLER_DRAWS = 2000
SAMPLER_TUNE = 2000
SAMPLER_CHAINS = 4
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Per-station metric helpers
# ---------------------------------------------------------------------------

def per_station_metrics(
    y_test: np.ndarray, p_test: np.ndarray, station_idx_test: np.ndarray, station_codes: list[str]
) -> pd.DataFrame:
    rows = []
    for s_idx, code in enumerate(station_codes):
        mask = station_idx_test == s_idx
        if not mask.any():
            continue
        rows.append(
            {
                "station": code,
                "n": int(mask.sum()),
                "obs_wet_frac": float(y_test[mask].mean()),
                "pred_wet_frac": float(p_test[mask].mean()),
                "brier": float(brier_score_loss(y_test[mask], p_test[mask])),
                "log_loss": float(log_loss(y_test[mask], p_test[mask])),
            }
        )
    return pd.DataFrame(rows)


def reliability_binning(
    y: np.ndarray, p: np.ndarray, station_idx: np.ndarray, station_codes: list[str], n_bins: int = 10
) -> pd.DataFrame:
    """For each station, bin predicted probabilities and compute observed wet rate."""
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for s_idx, code in enumerate(station_codes):
        mask = station_idx == s_idx
        ps = p[mask]
        ys = y[mask]
        bins = np.clip(np.digitize(ps, edges, right=False) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            in_bin = bins == b
            n_in = int(in_bin.sum())
            if n_in == 0:
                continue
            rows.append(
                {
                    "station": code,
                    "bin_lo": float(edges[b]),
                    "bin_hi": float(edges[b + 1]),
                    "n": n_in,
                    "mean_pred": float(ps[in_bin].mean()),
                    "obs_wet_frac": float(ys[in_bin].mean()),
                }
            )
    return pd.DataFrame(rows)


def diagnostics_summary(idata, var_names: list[str]) -> tuple[float, float, int]:
    summary = az.summary(idata, var_names=var_names)
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    n_div = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_div = int(idata.sample_stats["diverging"].sum())
    return max_rhat, min_ess, n_div


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("Phase 2: Hierarchical Bayesian logistic regression across 3 stations")
    print("=" * 78)

    ds = prepare_phase2_dataset()

    diagnostics_records = []
    metrics_records = {}

    # ---- Full pooling -----------------------------------------------------
    print("\n--- Model B: Full pooling ---")
    fp_path = ARTEFACTS_DIR / "full_pooling_posterior.nc"
    if fp_path.exists():
        print(f"  loading saved posterior from {fp_path.name} (skip refit)")
        fp = FullPoolingFit(idata=az.from_netcdf(fp_path), feature_names=ds.feature_names)
    else:
        fp = fit_full_pooling(
            ds.X_train_s, ds.y_train.to_numpy(), ds.feature_names,
            draws=SAMPLER_DRAWS, tune=SAMPLER_TUNE, chains=SAMPLER_CHAINS, random_seed=RANDOM_SEED,
        )
        fp.idata.to_netcdf(fp_path)
    fp_diag = run_diagnostics(
        fp.idata, ["intercept", "beta"],
        REPORTS_DIR / "phase2_diagnostics_full_pooling.pdf",
    )
    diagnostics_records.append({
        "model": "full_pooling",
        "max_rhat": fp_diag.max_rhat,
        "min_ess_bulk": fp_diag.min_ess_bulk,
        "n_divergences": fp_diag.n_divergences,
        "issues": fp_diag.issues,
    })
    print(f"  rhat={fp_diag.max_rhat:.3f}  ess={fp_diag.min_ess_bulk:.0f}  div={fp_diag.n_divergences}")
    p_fp = predict_full_pooling(fp, ds.X_test_s)

    # ---- No pooling -------------------------------------------------------
    print("\n--- Model A: No pooling (3 separate fits) ---")
    np_paths = {
        code: ARTEFACTS_DIR / f"no_pooling_posterior_{code.lower()}.nc"
        for code in ds.station_codes
    }
    if all(p.exists() for p in np_paths.values()):
        print("  loading saved per-station posteriors (skip refit)")
        np_fits = [
            FullPoolingFit(idata=az.from_netcdf(np_paths[code]), feature_names=ds.feature_names)
            for code in ds.station_codes
        ]
        np_fit = NoPoolingFit(
            fits=np_fits, station_codes=ds.station_codes, feature_names=ds.feature_names
        )
    else:
        np_fit = fit_no_pooling(
            ds.X_train_s, ds.y_train.to_numpy(), ds.station_idx_train,
            ds.station_codes, ds.feature_names,
            draws=SAMPLER_DRAWS, tune=SAMPLER_TUNE, chains=SAMPLER_CHAINS, random_seed=RANDOM_SEED,
        )
        for code, sf in zip(np_fit.station_codes, np_fit.fits):
            sf.idata.to_netcdf(np_paths[code])
    # Combine per-station fits' diagnostics into worst-case numbers.
    np_max_rhat = 0.0
    np_min_ess = 1e9
    np_n_div = 0
    for code, sf in zip(np_fit.station_codes, np_fit.fits):
        r, e, d = diagnostics_summary(sf.idata, ["intercept", "beta"])
        print(f"  {code}: rhat={r:.3f}  ess={e:.0f}  div={d}")
        np_max_rhat = max(np_max_rhat, r)
        np_min_ess = min(np_min_ess, e)
        np_n_div += d
    diagnostics_records.append({
        "model": "no_pooling",
        "max_rhat": np_max_rhat,
        "min_ess_bulk": np_min_ess,
        "n_divergences": np_n_div,
        "issues": [],
    })
    p_np = predict_no_pooling(np_fit, ds.X_test_s, ds.station_idx_test)

    # ---- Partial pooling --------------------------------------------------
    print("\n--- Model C: Partial pooling (hierarchical, non-centred) ---")
    pp = fit_partial_pooling(
        ds.X_train_s, ds.y_train.to_numpy(), ds.station_idx_train,
        ds.station_codes, ds.feature_names,
        draws=SAMPLER_DRAWS, tune=SAMPLER_TUNE, chains=SAMPLER_CHAINS,
        target_accept=0.9, random_seed=RANDOM_SEED,
        progressbar=True,
    )
    pp_diag = run_diagnostics(
        pp.idata,
        ["mu_intercept", "sigma_intercept", "mu_beta", "sigma_beta", "intercept_s"],
        REPORTS_DIR / "phase2_diagnostics_partial_pooling.pdf",
    )
    diagnostics_records.append({
        "model": "partial_pooling",
        "max_rhat": pp_diag.max_rhat,
        "min_ess_bulk": pp_diag.min_ess_bulk,
        "n_divergences": pp_diag.n_divergences,
        "issues": pp_diag.issues,
    })
    print(f"  rhat={pp_diag.max_rhat:.3f}  ess={pp_diag.min_ess_bulk:.0f}  div={pp_diag.n_divergences}")
    p_pp = predict_partial_pooling(pp, ds.X_test_s, ds.station_idx_test)
    pp.idata.to_netcdf(ARTEFACTS_DIR / "partial_pooling_posterior.nc")

    # ---- Per-station metrics ---------------------------------------------
    y_test = ds.y_test.to_numpy()
    metrics_fp = per_station_metrics(y_test, p_fp, ds.station_idx_test, ds.station_codes)
    metrics_np = per_station_metrics(y_test, p_np, ds.station_idx_test, ds.station_codes)
    metrics_pp = per_station_metrics(y_test, p_pp, ds.station_idx_test, ds.station_codes)

    print("\n--- Per-station test Brier (lower is better) ---")
    headline = pd.DataFrame({
        "station": ds.station_codes,
        "no_pooling": metrics_np.set_index("station").reindex(ds.station_codes)["brier"].values,
        "full_pooling": metrics_fp.set_index("station").reindex(ds.station_codes)["brier"].values,
        "partial_pooling": metrics_pp.set_index("station").reindex(ds.station_codes)["brier"].values,
    })
    print(headline.to_string(index=False))

    headline_logloss = pd.DataFrame({
        "station": ds.station_codes,
        "no_pooling": metrics_np.set_index("station").reindex(ds.station_codes)["log_loss"].values,
        "full_pooling": metrics_fp.set_index("station").reindex(ds.station_codes)["log_loss"].values,
        "partial_pooling": metrics_pp.set_index("station").reindex(ds.station_codes)["log_loss"].values,
    })
    print("\n--- Per-station test log loss ---")
    print(headline_logloss.to_string(index=False))

    print("\n--- Calibration (predicted vs observed wet fraction) ---")
    calib = pd.DataFrame({
        "station": ds.station_codes,
        "obs": metrics_np.set_index("station").reindex(ds.station_codes)["obs_wet_frac"].values,
        "no_pool_pred": metrics_np.set_index("station").reindex(ds.station_codes)["pred_wet_frac"].values,
        "full_pool_pred": metrics_fp.set_index("station").reindex(ds.station_codes)["pred_wet_frac"].values,
        "partial_pool_pred": metrics_pp.set_index("station").reindex(ds.station_codes)["pred_wet_frac"].values,
    })
    print(calib.to_string(index=False))

    headline.to_csv(ARTEFACTS_DIR / "headline_brier.csv", index=False)
    headline_logloss.to_csv(ARTEFACTS_DIR / "headline_logloss.csv", index=False)
    calib.to_csv(ARTEFACTS_DIR / "calibration.csv", index=False)

    # ---- Reliability binning ---------------------------------------------
    rel_fp = reliability_binning(y_test, p_fp, ds.station_idx_test, ds.station_codes)
    rel_np = reliability_binning(y_test, p_np, ds.station_idx_test, ds.station_codes)
    rel_pp = reliability_binning(y_test, p_pp, ds.station_idx_test, ds.station_codes)
    rel_fp["model"] = "full_pooling"
    rel_np["model"] = "no_pooling"
    rel_pp["model"] = "partial_pooling"
    pd.concat([rel_fp, rel_np, rel_pp], ignore_index=True).to_csv(
        ARTEFACTS_DIR / "reliability_bins.csv", index=False
    )

    # ---- Hyperparameter posteriors (the conceptual punchline) ------------
    print("\n--- Hyperparameter posteriors (partial pooling) ---")
    sigma_intercept = pp.idata.posterior["sigma_intercept"].values.flatten()
    print(
        f"  sigma_intercept   mean={sigma_intercept.mean():.3f}  "
        f"sd={sigma_intercept.std():.3f}  "
        f"94% CI [{np.quantile(sigma_intercept, 0.03):.3f}, {np.quantile(sigma_intercept, 0.97):.3f}]"
    )
    sigma_beta = pp.idata.posterior["sigma_beta"]  # (chain, draw, feature)
    sigma_rows = []
    for i, name in enumerate(ds.feature_names):
        sb = sigma_beta.isel(feature=i).values.flatten()
        sigma_rows.append({
            "feature": name,
            "sigma_mean": float(sb.mean()),
            "sigma_sd": float(sb.std()),
            "sigma_lo94": float(np.quantile(sb, 0.03)),
            "sigma_hi94": float(np.quantile(sb, 0.97)),
        })
    sigma_df = pd.DataFrame(sigma_rows).sort_values("sigma_mean", ascending=False)
    print("\nBetween-station SD per feature (sigma_beta):")
    print(sigma_df.to_string(index=False))
    sigma_df.to_csv(ARTEFACTS_DIR / "sigma_beta_summary.csv", index=False)

    # Population-level means and per-station betas - for the report.
    mu_intercept = pp.idata.posterior["mu_intercept"].values.flatten()
    print(
        f"\n  mu_intercept    mean={mu_intercept.mean():.3f}  "
        f"sd={mu_intercept.std():.3f}"
    )
    mu_beta = pp.idata.posterior["mu_beta"]
    mu_rows = []
    for i, name in enumerate(ds.feature_names):
        mb = mu_beta.isel(feature=i).values.flatten()
        mu_rows.append({
            "feature": name,
            "mu_mean": float(mb.mean()),
            "mu_sd": float(mb.std()),
            "mu_lo94": float(np.quantile(mb, 0.03)),
            "mu_hi94": float(np.quantile(mb, 0.97)),
        })
    pd.DataFrame(mu_rows).to_csv(ARTEFACTS_DIR / "mu_beta_summary.csv", index=False)

    # Per-station beta posterior table for inspection.
    beta_s = pp.idata.posterior["beta_s"]  # (chain, draw, station, feature)
    intercept_s = pp.idata.posterior["intercept_s"]
    rows = []
    for s_idx, code in enumerate(ds.station_codes):
        rows.append({
            "param": "intercept",
            "station": code,
            "mean": float(intercept_s.isel(station=s_idx).mean()),
            "sd": float(intercept_s.isel(station=s_idx).std()),
        })
        for f_idx, name in enumerate(ds.feature_names):
            v = beta_s.isel(station=s_idx, feature=f_idx).values.flatten()
            rows.append({
                "param": name,
                "station": code,
                "mean": float(v.mean()),
                "sd": float(v.std()),
            })
    per_station_betas = pd.DataFrame(rows)
    per_station_betas.to_csv(ARTEFACTS_DIR / "per_station_coefficients.csv", index=False)

    # ---- Sanity check vs phase 1 -----------------------------------------
    phase1_metrics_path = REPORTS_DIR / "phase1_artefacts" / "metrics.json"
    phase1_brier_bellever = None
    if phase1_metrics_path.exists():
        with phase1_metrics_path.open() as f:
            phase1_metrics = json.load(f)
        phase1_brier_bellever = phase1_metrics.get("bayes_brier")

    bellever_idx = ds.station_codes.index("Bellever")
    bellever_mask = ds.station_idx_test == bellever_idx
    bellever_brier_np = float(brier_score_loss(y_test[bellever_mask], p_np[bellever_mask]))
    bellever_brier_pp = float(brier_score_loss(y_test[bellever_mask], p_pp[bellever_mask]))
    print("\n--- Phase 1 sanity check (Bellever only) ---")
    if phase1_brier_bellever is not None:
        print(f"  phase 1 Brier        {phase1_brier_bellever:.4f}")
    print(f"  phase 2 no-pool Brier {bellever_brier_np:.4f}")
    print(f"  phase 2 partial Brier {bellever_brier_pp:.4f}")

    # ---- Persist consolidated metrics for the report ---------------------
    summary = {
        "n_train_total": int(len(ds.y_train)),
        "n_test_total": int(len(ds.y_test)),
        "feature_names": ds.feature_names,
        "station_codes": ds.station_codes,
        "n_train_per_station": {
            code: int((ds.station_idx_train == i).sum()) for i, code in enumerate(ds.station_codes)
        },
        "n_test_per_station": {
            code: int((ds.station_idx_test == i).sum()) for i, code in enumerate(ds.station_codes)
        },
        "wet_fraction_train": {
            code: float(ds.y_train[ds.station_idx_train == i].mean())
            for i, code in enumerate(ds.station_codes)
        },
        "wet_fraction_test": {
            code: float(ds.y_test[ds.station_idx_test == i].mean())
            for i, code in enumerate(ds.station_codes)
        },
        "diagnostics": diagnostics_records,
        "headline_brier": headline.to_dict(orient="records"),
        "headline_logloss": headline_logloss.to_dict(orient="records"),
        "calibration": calib.to_dict(orient="records"),
        "sigma_beta": sigma_df.to_dict(orient="records"),
        "mu_beta": mu_rows,
        "sigma_intercept_mean": float(sigma_intercept.mean()),
        "sigma_intercept_sd": float(sigma_intercept.std()),
        "mu_intercept_mean": float(mu_intercept.mean()),
        "mu_intercept_sd": float(mu_intercept.std()),
        "phase1_bellever_brier": phase1_brier_bellever,
        "phase2_bellever_no_pool_brier": bellever_brier_np,
        "phase2_bellever_partial_pool_brier": bellever_brier_pp,
    }
    with (ARTEFACTS_DIR / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- Trace / forest plot for sigma_beta ------------------------------
    az.plot_forest(
        pp.idata, var_names=["sigma_beta", "sigma_intercept"], combined=True
    )
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "phase2_sigma_forest.pdf")
    plt.close("all")

    print(f"\nArtefacts in: {ARTEFACTS_DIR}")
    print("Run complete.")


if __name__ == "__main__":
    main()
