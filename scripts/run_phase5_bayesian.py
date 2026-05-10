"""Phase 5 Bayesian — lead-as-feature retrain.

Same hierarchical partial-pooling shape as Phase 4 Model A, but instead
of fitting three independent per-lead posteriors and gluing them at
predict time, this fits ONE posterior on rows pooled across all three
leads with the lead value (raw hours, z-scored alongside the other
features) appended to the feature vector.

Why
---
Phase 4 Model A's per-discrete-lead structure forces the live predict
path to filter forecast rows to ``LeadHours ∈ {24, 48, 72}`` and inner-
join across 5 NWPs at the matching valid-time grid. The intersection
caps lead-24 surfaceable predictions at ~2/day on the live forecast
tree (00Z + 12Z, with GEM Seamless's 2-cycles-per-day as the structural
bottleneck) — and on bad collector days drops to 1/day. Hence the
"1 Bayesian point per day at lead 24h" thread.

Lead-as-feature swaps that for: a single posterior takes (precip_x5,
hour_sin, hour_cos, lead) → P(wet). At predict time any forecast row
at any (cycle, lead) can be scored — the lead column just becomes
whatever value matches that row. The 5-model inner-join still happens
on (valid_time, lead) for training-data assembly (we still need every
NWP to have an opinion on each row, otherwise the row drops), but the
LIVE predict no longer has to land on a specific lead-bucket grid.
For an hourly forecast curve with 5 NWPs the chart fills out hour-by-
hour from the same single posterior.

Saves
-----
Single posterior NetCDF at
  reports/phase5a_artefacts/posteriors/lead_feature.nc
plus a per-row test predictions parquet keyed by (valid_time, station,
lead) so evaluation can stay per-cell for direct comparison against
Phase 4's per-lead Brier numbers.

Run with:
    PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -u scripts/run_phase5_bayesian.py \\
        > reports/phase5_bayesian_run.log 2>&1
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import arviz as az  # noqa: E402

from src.data import MODELS_NO_UKMO, prepare_phase3_dataset  # noqa: E402
from src.retrain_guard import build_check_and_save_singleton  # noqa: E402
from src.models.phase2_partial_pooling import (  # noqa: E402
    fit_partial_pooling, predict_partial_pooling,
)

OUT_DIR = ROOT / "reports" / "phase5a_artefacts"
POSTERIOR_DIR = OUT_DIR / "posteriors"
PREDICTIONS_DIR = OUT_DIR / "predictions"

SAMPLER_DRAWS = 2000
SAMPLER_TUNE = 2000
SAMPLER_CHAINS = 4
RANDOM_SEED = 42


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--add-spread-features",
        action="store_true",
        help=("Include precip_max + precip_agreement_wet_01 in the feature "
              "set (10-feature variant). Pilot-validated 2026-05-10 to drop "
              "the dry-tail floor materially vs the 8-feature default. Off "
              "by default until promoted to production."),
    )
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    variant = "10-feat (spread)" if args.add_spread_features else "8-feat"
    print(f"[{time.strftime('%H:%M:%S')}] Phase 5 Bayesian — lead-as-feature, "
          f"5-model variant, {variant}")
    ds = prepare_phase3_dataset(
        models=MODELS_NO_UKMO, lead_as_feature=True,
        add_spread_features=args.add_spread_features,
        verbose=False,
    )
    print(f"  train rows: {len(ds.X_train):,}  test rows: {len(ds.X_test):,}")
    print(f"  features ({len(ds.feature_names)}): {ds.feature_names}")
    print(f"  stations: {ds.station_codes}")
    print(f"  leads: {ds.lead_hours}  (pooled into one fit; lead is feature index "
          f"{ds.feature_names.index('lead')})")

    # RetrainGuard wired in 2026-05-10 (Phase 1c, Python side). Fires
    # BEFORE the 3-hour MCMC sample so a stale upstream-data shift
    # doesn't burn the runner. 5a is overwrite-style (one bundle, gets
    # rewritten each run) so we use the singleton variant — the existing
    # training_summary.json at OUT_DIR is the previous baseline; pass
    # overwrites it, fail leaves it alone.
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    _log = _logging.getLogger("phase5a")

    # Per-station label rates from y_train + station_idx_train. Site
    # convention is the EA-prefixed slug; use the station_codes labels
    # verbatim (5a's station_codes are already the slugs).
    label_rates = {}
    y_train_arr = ds.y_train.values if hasattr(ds.y_train, "values") else np.asarray(ds.y_train)
    station_idx_arr = np.asarray(ds.station_idx_train)
    for i, code in enumerate(ds.station_codes):
        mask = station_idx_arr == i
        if mask.any():
            label_rates[code] = float(y_train_arr[mask].mean())

    version = datetime.now(timezone.utc).strftime("v%Y-%m-%d_%H%M%S_phase5a")
    variant_tag = "10feat-spread" if args.add_spread_features else "8feat"

    guard_result = build_check_and_save_singleton(
        _log,
        summary_path=OUT_DIR / "training_summary.json",
        composite="precipitation_5a",
        phase="5a",
        version=f"{version}_{variant_tag}",
        rows_train=len(ds.X_train),
        rows_val=0,                      # 5a has no separate val slice; train+test only
        rows_test=len(ds.X_test),
        train_features=ds.X_train_s,
        feature_names=list(ds.feature_names),
        label_rates=label_rates,
    )
    if not guard_result.passed:
        print("Phase 5a guard FAIL — skipping sample. Existing posterior + bundle stay live.")
        sys.exit(4)

    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting single partial-pool posterior across all leads")
    t0 = time.time()
    fit = fit_partial_pooling(
        X_train_s=ds.X_train_s,
        y_train=ds.y_train.values,
        station_idx_train=ds.station_idx_train,
        station_codes=ds.station_codes,
        feature_names=ds.feature_names,
        draws=SAMPLER_DRAWS, tune=SAMPLER_TUNE, chains=SAMPLER_CHAINS,
        target_accept=0.9, random_seed=RANDOM_SEED, progressbar=True,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Posterior fit in {(time.time()-t0)/60:.1f} min")

    s = az.summary(fit.idata, var_names=["mu_intercept", "sigma_intercept", "mu_beta", "sigma_beta"])
    rhat = float(s["r_hat"].max())
    ess = float(s["ess_bulk"].min())
    ndiv = int(fit.idata.sample_stats["diverging"].sum()) if "diverging" in fit.idata.sample_stats else 0
    print(f"  rhat={rhat:.3f}  ess={ess:.0f}  div={ndiv}")
    fit.idata.to_netcdf(POSTERIOR_DIR / "lead_feature.nc")

    # Per-row test predictions, keyed by (valid_time, station, lead) so
    # the comparison parquet has the same shape as Phase 4's per-lead
    # files and the evaluator code stays simple.
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing per-row test predictions")
    p_test = predict_partial_pooling(fit, ds.X_test_s, ds.station_idx_test)

    out = pd.DataFrame({
        "valid_time": pd.to_datetime(ds.valid_time_test.values),
        "station": [ds.station_codes[i] for i in ds.station_idx_test],
        "lead": [ds.lead_hours[i] for i in ds.lead_idx_test],
        "p_wet": p_test,
        "observed_wet": ds.y_test.values.astype("int8"),
    })
    out_path = PREDICTIONS_DIR / "test_predictions.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  wrote {len(out):,} rows to {out_path}")

    # Per-lead Brier breakdown so the comparison vs Phase 4 reads at a
    # glance.
    print(f"\n[{time.strftime('%H:%M:%S')}] Per-lead test Brier (vs climatology baseline):")
    print(f"{'lead':>5} {'station':>10} {'n':>5} {'Brier':>8} {'wet rate':>9}")
    for lead in ds.lead_hours:
        for code in ds.station_codes:
            mask = (out["lead"] == lead) & (out["station"] == code)
            sub = out[mask]
            if len(sub) == 0:
                continue
            brier = float(((sub["p_wet"] - sub["observed_wet"]) ** 2).mean())
            wet_rate = float(sub["observed_wet"].mean())
            print(f"{lead:>5} {code:>10} {len(sub):>5} {brier:>8.4f} {wet_rate:>9.3f}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Phase 5 Bayesian done.")


if __name__ == "__main__":
    main()
