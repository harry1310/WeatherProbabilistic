"""Phase 3 end-to-end runner.

Fits Model A (per-lead independent partial pooling) and Model B (joint
2D station × lead hierarchy) on the multi-lead dataset, runs diagnostics,
computes per-(station, lead) test metrics, and emits the comparison
tables and hyperparameter summaries the report depends on.

Run with:

    .venv/Scripts/python.exe scripts/run_phase3.py            # full data
    .venv/Scripts/python.exe scripts/run_phase3.py --subset 1000   # dev
    .venv/Scripts/python.exe scripts/run_phase3.py --skip-existing # reuse .nc

Posteriors and CSVs land in `reports/phase3_artefacts/`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

# Greek letters (σ, μ, β) and × appear in our prints — Windows defaults stdout
# to cp1252 which can't encode the Greek glyphs, so a print would crash mid-run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import arviz as az  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

from src.data import prepare_phase3_dataset  # noqa: E402
from src.diagnostics import run_diagnostics  # noqa: E402
from src.models.phase2_partial_pooling import PartialPoolingFit  # noqa: E402
from src.models.phase3a_per_lead import (  # noqa: E402
    PerLeadFit,
    fit_per_lead,
    predict_per_lead,
)
from src.models.phase3b_joint_hierarchy import (  # noqa: E402
    fit_joint_hierarchy,
    predict_joint_hierarchy,
    JointHierarchyFit,
)


REPORTS_DIR = ROOT / "reports"
ARTEFACTS_DIR = REPORTS_DIR / "phase3_artefacts"
ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
PER_LEAD_DIR = ARTEFACTS_DIR / "phase3a_per_lead"
PER_LEAD_DIR.mkdir(parents=True, exist_ok=True)


SAMPLER_DRAWS = 2000
SAMPLER_TUNE = 2000
SAMPLER_CHAINS = 4
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Per-cell metric helpers (station × lead)
# ---------------------------------------------------------------------------

def per_cell_metrics(
    y_test: np.ndarray,
    p_test: np.ndarray,
    station_idx_test: np.ndarray,
    lead_idx_test: np.ndarray,
    station_codes: list[str],
    lead_hours: list[int],
) -> pd.DataFrame:
    rows = []
    for s_idx, code in enumerate(station_codes):
        for l_idx, lh in enumerate(lead_hours):
            mask = (station_idx_test == s_idx) & (lead_idx_test == l_idx)
            if not mask.any():
                continue
            rows.append(
                {
                    "station": code,
                    "lead_h": lh,
                    "n": int(mask.sum()),
                    "obs_wet_frac": float(y_test[mask].mean()),
                    "pred_wet_frac": float(p_test[mask].mean()),
                    "brier": float(brier_score_loss(y_test[mask], p_test[mask])),
                    "log_loss": float(log_loss(y_test[mask], p_test[mask])),
                }
            )
    return pd.DataFrame(rows)


def reliability_binning(
    y: np.ndarray,
    p: np.ndarray,
    station_idx: np.ndarray,
    lead_idx: np.ndarray,
    station_codes: list[str],
    lead_hours: list[int],
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for s_idx, code in enumerate(station_codes):
        for l_idx, lh in enumerate(lead_hours):
            mask = (station_idx == s_idx) & (lead_idx == l_idx)
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
                        "lead_h": lh,
                        "bin_lo": float(edges[b]),
                        "bin_hi": float(edges[b + 1]),
                        "n": n_in,
                        "mean_pred": float(ps[in_bin].mean()),
                        "obs_wet_frac": float(ys[in_bin].mean()),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Subset helper for development
# ---------------------------------------------------------------------------

def subset_per_cell(ds, n_per_cell: int, seed: int = 0):
    """Return train/test masks limiting each (station, lead) cell to the
    earliest N rows. Used only when running with --subset for dev."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(ds.X_train_s), dtype=bool)
    for s in range(len(ds.station_codes)):
        for l in range(len(ds.lead_hours)):
            idxs = np.where((ds.station_idx_train == s) & (ds.lead_idx_train == l))[0]
            if len(idxs) > n_per_cell:
                idxs = idxs[:n_per_cell]  # earliest, not random — keeps chronology
            keep[idxs] = True
    return keep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 runner")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit each (station, lead) cell to this many train rows (dev only).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse saved .nc posteriors instead of refitting.")
    parser.add_argument("--target-accept", type=float, default=0.9,
                        help="NUTS target acceptance (default 0.9).")
    parser.add_argument("--no-progressbar", action="store_true",
                        help="Disable nutpie progressbar (useful when teeing to a file).")
    args = parser.parse_args()

    print("=" * 78)
    print("Phase 3: Hierarchical Bayesian logistic regression - 3 stations × 3 leads")
    print("=" * 78)

    ds = prepare_phase3_dataset()

    if args.subset:
        keep = subset_per_cell(ds, args.subset)
        print(f"\n[subset] keeping {keep.sum():,} of {len(keep):,} train rows ({args.subset} per cell)")
        X_train_s = ds.X_train_s[keep]
        y_train = ds.y_train.to_numpy()[keep]
        st_idx_tr = ds.station_idx_train[keep]
        ld_idx_tr = ds.lead_idx_train[keep]
    else:
        X_train_s = ds.X_train_s
        y_train = ds.y_train.to_numpy()
        st_idx_tr = ds.station_idx_train
        ld_idx_tr = ds.lead_idx_train

    y_test = ds.y_test.to_numpy()

    diagnostics_records = []
    progressbar = not args.no_progressbar

    # ---- Model A: per-lead partial pooling --------------------------------
    print("\n--- Model A: Per-lead independent partial pooling ---")
    per_lead_paths = {
        lh: PER_LEAD_DIR / f"lead_{lh}h.nc" for lh in ds.lead_hours
    }
    if args.skip_existing and all(p.exists() for p in per_lead_paths.values()):
        print("  loading saved per-lead posteriors (skip refit)")
        fits_by_lead: dict[int, PartialPoolingFit] = {
            lh: PartialPoolingFit(
                idata=az.from_netcdf(per_lead_paths[lh]),
                feature_names=ds.feature_names,
                station_codes=ds.station_codes,
            )
            for lh in ds.lead_hours
        }
        a_fit = PerLeadFit(
            fits_by_lead=fits_by_lead,
            feature_names=ds.feature_names,
            station_codes=ds.station_codes,
            lead_hours=list(ds.lead_hours),
        )
    else:
        a_fit = fit_per_lead(
            X_train_s, y_train, st_idx_tr, ld_idx_tr,
            ds.station_codes, ds.lead_hours, ds.feature_names,
            draws=SAMPLER_DRAWS, tune=SAMPLER_TUNE, chains=SAMPLER_CHAINS,
            target_accept=args.target_accept, random_seed=RANDOM_SEED,
            progressbar=progressbar,
        )
        for lh, fit in a_fit.fits_by_lead.items():
            fit.idata.to_netcdf(per_lead_paths[lh])

    # Per-lead diagnostics: worst-case across the three fits, plus a per-lead
    # trace PDF via run_diagnostics so funnels / divergent geometry can be
    # eyeballed (Model B already gets this; matching it for Model A so both
    # halves of the comparison have the same diagnostic surface).
    a_max_rhat, a_min_ess, a_n_div = 0.0, 1e9, 0
    a_var_names = ["mu_intercept", "sigma_intercept", "mu_beta", "sigma_beta"]
    a_per_lead_diag: dict[int, object] = {}
    for lh, fit in a_fit.fits_by_lead.items():
        diag = run_diagnostics(
            fit.idata,
            a_var_names,
            REPORTS_DIR / f"phase3a_diagnostics_lead_{lh}h.pdf",
        )
        a_per_lead_diag[lh] = diag
        print(
            f"  lead {lh}h: rhat={diag.max_rhat:.3f}  ess={diag.min_ess_bulk:.0f}  "
            f"div={diag.n_divergences}  issues={'; '.join(diag.issues) if diag.issues else 'none'}",
            flush=True,
        )
        a_max_rhat = max(a_max_rhat, diag.max_rhat)
        a_min_ess = min(a_min_ess, diag.min_ess_bulk)
        a_n_div += diag.n_divergences
    diagnostics_records.append({
        "model": "phase3a_per_lead",
        "max_rhat": a_max_rhat,
        "min_ess_bulk": a_min_ess,
        "n_divergences": a_n_div,
        "issues": sorted({i for d in a_per_lead_diag.values() for i in d.issues}),
    })

    p_a = predict_per_lead(a_fit, ds.X_test_s, ds.station_idx_test, ds.lead_idx_test)

    # ---- Model B: joint 2D hierarchy --------------------------------------
    print("\n--- Model B: Joint station × lead hierarchy (non-centred) ---")
    b_path = ARTEFACTS_DIR / "phase3b_joint_hierarchy_posterior.nc"
    if args.skip_existing and b_path.exists():
        print(f"  loading saved posterior from {b_path.name} (skip refit)")
        b_fit = JointHierarchyFit(
            idata=az.from_netcdf(b_path),
            feature_names=ds.feature_names,
            station_codes=ds.station_codes,
            lead_hours=list(ds.lead_hours),
        )
    else:
        b_fit = fit_joint_hierarchy(
            X_train_s, y_train, st_idx_tr, ld_idx_tr,
            ds.station_codes, ds.lead_hours, ds.feature_names,
            draws=SAMPLER_DRAWS, tune=SAMPLER_TUNE, chains=SAMPLER_CHAINS,
            target_accept=args.target_accept, random_seed=RANDOM_SEED,
            progressbar=progressbar,
        )
        b_fit.idata.to_netcdf(b_path)

    b_diag = run_diagnostics(
        b_fit.idata,
        ["mu_intercept", "sigma_station_intercept", "sigma_lead_intercept",
         "sigma_interaction_intercept", "mu_beta", "sigma_station_beta",
         "sigma_lead_beta", "sigma_interaction_beta"],
        REPORTS_DIR / "phase3_diagnostics.pdf",
    )
    diagnostics_records.append({
        "model": "phase3b_joint_hierarchy",
        "max_rhat": b_diag.max_rhat,
        "min_ess_bulk": b_diag.min_ess_bulk,
        "n_divergences": b_diag.n_divergences,
        "issues": b_diag.issues,
    })
    print(f"  rhat={b_diag.max_rhat:.3f}  ess={b_diag.min_ess_bulk:.0f}  div={b_diag.n_divergences}")

    p_b = predict_joint_hierarchy(b_fit, ds.X_test_s, ds.station_idx_test, ds.lead_idx_test)

    # ---- Per-cell test metrics --------------------------------------------
    metrics_a = per_cell_metrics(
        y_test, p_a, ds.station_idx_test, ds.lead_idx_test, ds.station_codes, ds.lead_hours
    )
    metrics_b = per_cell_metrics(
        y_test, p_b, ds.station_idx_test, ds.lead_idx_test, ds.station_codes, ds.lead_hours
    )

    # Combined headline table: A vs B side by side
    headline = pd.DataFrame({
        "station": metrics_a["station"],
        "lead_h": metrics_a["lead_h"],
        "n_test": metrics_a["n"],
        "obs_wet": metrics_a["obs_wet_frac"],
        "model_a_brier": metrics_a["brier"],
        "model_b_brier": metrics_b["brier"],
        "model_a_logloss": metrics_a["log_loss"],
        "model_b_logloss": metrics_b["log_loss"],
        "a_pred_wet": metrics_a["pred_wet_frac"],
        "b_pred_wet": metrics_b["pred_wet_frac"],
    })
    print("\n--- Per (station, lead) test metrics ---")
    print(headline.to_string(index=False))
    headline.to_csv(ARTEFACTS_DIR / "headline_metrics.csv", index=False)

    # Reliability binning per cell, both models combined
    rel_a = reliability_binning(
        y_test, p_a, ds.station_idx_test, ds.lead_idx_test, ds.station_codes, ds.lead_hours
    )
    rel_b = reliability_binning(
        y_test, p_b, ds.station_idx_test, ds.lead_idx_test, ds.station_codes, ds.lead_hours
    )
    rel_a["model"] = "phase3a_per_lead"
    rel_b["model"] = "phase3b_joint_hierarchy"
    pd.concat([rel_a, rel_b], ignore_index=True).to_csv(
        ARTEFACTS_DIR / "reliability_bins.csv", index=False
    )

    # ---- Model B hyperparameter analysis ----------------------------------
    print("\n--- Model B hyperparameter posteriors ---")

    # sigma_intercept by family
    sigma_int = {
        "station": b_fit.idata.posterior["sigma_station_intercept"].values.flatten(),
        "lead":    b_fit.idata.posterior["sigma_lead_intercept"].values.flatten(),
        "inter":   b_fit.idata.posterior["sigma_interaction_intercept"].values.flatten(),
    }
    int_summary = {
        f"sigma_intercept_{k}_mean": float(v.mean()) for k, v in sigma_int.items()
    }
    int_summary.update({
        f"sigma_intercept_{k}_lo94": float(np.quantile(v, 0.03)) for k, v in sigma_int.items()
    })
    int_summary.update({
        f"sigma_intercept_{k}_hi94": float(np.quantile(v, 0.97)) for k, v in sigma_int.items()
    })

    # sigma_beta decomposition per feature
    sigma_st = b_fit.idata.posterior["sigma_station_beta"]   # (chain, draw, feature)
    sigma_ld = b_fit.idata.posterior["sigma_lead_beta"]
    sigma_in = b_fit.idata.posterior["sigma_interaction_beta"]
    sigma_rows = []
    for i, name in enumerate(ds.feature_names):
        st = sigma_st.isel(feature=i).values.flatten()
        ld = sigma_ld.isel(feature=i).values.flatten()
        ia = sigma_in.isel(feature=i).values.flatten()
        sigma_rows.append({
            "feature": name,
            "sigma_station_mean": float(st.mean()),
            "sigma_station_lo94": float(np.quantile(st, 0.03)),
            "sigma_station_hi94": float(np.quantile(st, 0.97)),
            "sigma_lead_mean": float(ld.mean()),
            "sigma_lead_lo94": float(np.quantile(ld, 0.03)),
            "sigma_lead_hi94": float(np.quantile(ld, 0.97)),
            "sigma_interaction_mean": float(ia.mean()),
            "sigma_interaction_lo94": float(np.quantile(ia, 0.03)),
            "sigma_interaction_hi94": float(np.quantile(ia, 0.97)),
        })
    sigma_df = pd.DataFrame(sigma_rows).sort_values("sigma_station_mean", ascending=False)
    print("\nsigma decomposition per feature (mean of posterior):")
    print(
        sigma_df[["feature", "sigma_station_mean", "sigma_lead_mean", "sigma_interaction_mean"]]
        .to_string(index=False)
    )
    sigma_df.to_csv(ARTEFACTS_DIR / "sigma_decomposition.csv", index=False)

    # Population means (μ_β) for completeness
    mu_beta_post = b_fit.idata.posterior["mu_beta"]
    mu_rows = []
    for i, name in enumerate(ds.feature_names):
        v = mu_beta_post.isel(feature=i).values.flatten()
        mu_rows.append({
            "feature": name,
            "mu_mean": float(v.mean()),
            "mu_sd": float(v.std()),
            "mu_lo94": float(np.quantile(v, 0.03)),
            "mu_hi94": float(np.quantile(v, 0.97)),
        })
    pd.DataFrame(mu_rows).to_csv(ARTEFACTS_DIR / "mu_beta_summary.csv", index=False)

    # Per-cell β posterior for the report's lead-effect plots
    beta_sl = b_fit.idata.posterior["beta_sl"]   # (chain, draw, station, lead, feature)
    intercept_sl = b_fit.idata.posterior["intercept_sl"]
    rows = []
    for s_idx, code in enumerate(ds.station_codes):
        for l_idx, lh in enumerate(ds.lead_hours):
            rows.append({
                "param": "intercept",
                "station": code,
                "lead_h": lh,
                "mean": float(intercept_sl.isel(station=s_idx, lead=l_idx).mean()),
                "sd": float(intercept_sl.isel(station=s_idx, lead=l_idx).std()),
            })
            for f_idx, name in enumerate(ds.feature_names):
                v = beta_sl.isel(station=s_idx, lead=l_idx, feature=f_idx).values.flatten()
                rows.append({
                    "param": name,
                    "station": code,
                    "lead_h": lh,
                    "mean": float(v.mean()),
                    "sd": float(v.std()),
                })
    pd.DataFrame(rows).to_csv(ARTEFACTS_DIR / "per_cell_coefficients.csv", index=False)

    # Forest plot: sigma_station, sigma_lead, sigma_interaction (beta)
    az.plot_forest(
        b_fit.idata,
        var_names=["sigma_station_beta", "sigma_lead_beta", "sigma_interaction_beta"],
        combined=True,
    )
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "phase3_sigma_forest.pdf")
    plt.close("all")

    # ---- Sanity check vs Phase 2 (Bellever, 24h) --------------------------
    print("\n--- Sanity check: (Bellever, 24h) Brier across phases ---")
    bel_idx = ds.station_codes.index("Bellever")
    cell_mask = (ds.station_idx_test == bel_idx) & (ds.lead_idx_test == 0)  # lead 24h is index 0

    brier_a = float(brier_score_loss(y_test[cell_mask], p_a[cell_mask]))
    brier_b = float(brier_score_loss(y_test[cell_mask], p_b[cell_mask]))

    phase2_metrics_path = REPORTS_DIR / "phase2_artefacts" / "metrics.json"
    phase2_partial = phase2_metrics_path.exists() and json.loads(phase2_metrics_path.read_text()).get("phase2_bellever_partial_pool_brier")
    phase2_no_pool = phase2_metrics_path.exists() and json.loads(phase2_metrics_path.read_text()).get("phase2_bellever_no_pool_brier")
    phase1 = phase2_metrics_path.exists() and json.loads(phase2_metrics_path.read_text()).get("phase1_bellever_brier")

    print(f"  phase 1 single-station         {phase1}")
    print(f"  phase 2 no-pool                {phase2_no_pool}")
    print(f"  phase 2 partial-pool           {phase2_partial}")
    print(f"  phase 3 model A (24h)          {brier_a:.4f}")
    print(f"  phase 3 model B (24h)          {brier_b:.4f}")

    # ---- Persist consolidated metrics for the report ----------------------
    summary = {
        "subset": args.subset,
        "target_accept": args.target_accept,
        "n_train_total": int(len(y_train)),
        "n_test_total": int(len(y_test)),
        "feature_names": ds.feature_names,
        "station_codes": ds.station_codes,
        "lead_hours": list(ds.lead_hours),
        "diagnostics": diagnostics_records,
        "headline_metrics": headline.to_dict(orient="records"),
        "sigma_decomposition": sigma_df.to_dict(orient="records"),
        "mu_beta": mu_rows,
        "sigma_intercept": int_summary,
        "phase1_bellever_brier": phase1,
        "phase2_bellever_no_pool_brier": phase2_no_pool,
        "phase2_bellever_partial_pool_brier": phase2_partial,
        "phase3a_bellever_24h_brier": brier_a,
        "phase3b_bellever_24h_brier": brier_b,
    }
    with (ARTEFACTS_DIR / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nArtefacts in: {ARTEFACTS_DIR}")
    print("Run complete.")


if __name__ == "__main__":
    main()
