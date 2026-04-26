"""Phase 4 Bayesian 5-model trainer.

Refits Phase 3 Model A (per-lead independent partial-pooling across 3 stations)
on the 5-model Phase3Dataset (UKMO removed). Same priors, same non-centred
parameterisation, same nutpie sampler config as Phase 3. Saves per-row test
predictions to
``reports/phase4_artefacts/bayesian_predictions/lead_{N}h.parquet``
with columns ``valid_time, station, lead, p_wet, observed_wet``.

Also saves the full posterior NetCDF per lead so the comparison step can
compute credible intervals + uncertainty quality metrics from the saved
posteriors instead of re-fitting.

Sampling: ~50 min per lead × 3 leads = ~2.5h on this box. Heartbeat-only
visibility (per Phase 3 Model A behaviour) — Model A's per-lead durations
are bounded so the lack of per-draw progress is acceptable.

Run with (background recommended):

    PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -u scripts/run_phase4_bayesian.py \\
        > reports/phase4_bayesian_run.log 2>&1
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

# Greek letters in pm.sample diagnostics; force UTF-8 on Windows console.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import arviz as az  # noqa: E402

from src.data import MODELS_NO_UKMO, prepare_phase3_dataset  # noqa: E402
from src.models.phase3a_per_lead import fit_per_lead, predict_per_lead  # noqa: E402

OUT_DIR = ROOT / "reports" / "phase4_artefacts" / "bayesian_predictions"
POSTERIOR_DIR = OUT_DIR / "posteriors"

SAMPLER_DRAWS = 2000
SAMPLER_TUNE = 2000
SAMPLER_CHAINS = 4
RANDOM_SEED = 42


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Phase 4 Bayesian 5-model: loading dataset")
    ds = prepare_phase3_dataset(models=MODELS_NO_UKMO, verbose=False)
    print(f"  train rows: {len(ds.X_train):,}  test rows: {len(ds.X_test):,}")
    print(f"  features ({len(ds.feature_names)}): {ds.feature_names}")
    print(f"  stations: {ds.station_codes}")
    print(f"  leads: {ds.lead_hours}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting Model A (per-lead partial pool, 5-model variant)")
    t0 = time.time()
    fit = fit_per_lead(
        X_train_s=ds.X_train_s,
        y_train=ds.y_train.values,
        station_idx_train=ds.station_idx_train,
        lead_idx_train=ds.lead_idx_train,
        station_codes=ds.station_codes,
        lead_hours=list(ds.lead_hours),
        feature_names=ds.feature_names,
        draws=SAMPLER_DRAWS, tune=SAMPLER_TUNE, chains=SAMPLER_CHAINS,
        target_accept=0.9, random_seed=RANDOM_SEED, progressbar=True,
    )
    print(f"[{time.strftime('%H:%M:%S')}] All leads fit in {(time.time() - t0) / 60:.1f} min")

    # Per-lead diagnostics (rhat / ess / divergences)
    for lh, f in fit.fits_by_lead.items():
        s = az.summary(f.idata, var_names=["mu_intercept", "sigma_intercept", "mu_beta", "sigma_beta"])
        rhat = float(s["r_hat"].max())
        ess = float(s["ess_bulk"].min())
        ndiv = int(f.idata.sample_stats["diverging"].sum()) if "diverging" in f.idata.sample_stats else 0
        print(f"  lead {lh}h: rhat={rhat:.3f}  ess={ess:.0f}  div={ndiv}")
        f.idata.to_netcdf(POSTERIOR_DIR / f"lead_{lh}h.nc")

    # Per-row test predictions (mean of posterior P(wet)).
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing per-row test predictions")
    p_test = predict_per_lead(fit, ds.X_test_s, ds.station_idx_test, ds.lead_idx_test)

    for li, lead in enumerate(ds.lead_hours):
        mask = ds.lead_idx_test == li
        out = pd.DataFrame({
            "valid_time": pd.to_datetime(ds.valid_time_test.values[mask]),
            "station": [ds.station_codes[i] for i in ds.station_idx_test[mask]],
            "lead": int(lead),
            "p_wet": p_test[mask],
            "observed_wet": ds.y_test.values[mask].astype("int8"),
        })
        out_path = OUT_DIR / f"lead_{lead}h.parquet"
        out.to_parquet(out_path, index=False)
        print(f"  lead {lead}h: wrote {len(out):,} rows to {out_path}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Phase 4 Bayesian done.")


if __name__ == "__main__":
    main()
