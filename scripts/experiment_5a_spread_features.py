"""Phase 5a feature-add experiment — add precip_max + precip_agreement_wet_01.

Hypothesis: 5a's logistic regression hits a dry-tail floor at ~0.13–0.18 P(wet)
because raw precip features (z=-0.34σ when all NWPs are at zero) only pull
the logit down by ~-0.7 from the Bellever intercept of -1.05. Adding two
non-linear spread features (precip_max, precip_agreement_wet_01) — pre-
computed scalars that go to ~0 only when EVERY NWP is at zero — gives LR
the AND-detection it currently lacks, so the floor should drop materially.

What this script does
---------------------
1. Loads the same training data as production 5a but with add_spread_features=True
   (10 features instead of 8).
2. Re-fits the lead-as-feature partial-pool posterior (~20-40 min on this box
   with nutpie).
3. Builds live forecast features (including precip_max + agreement) for today's
   anchor across the same 5 NWPs.
4. Scores today's forecast and reports per-station mean/median P(wet).
5. Compares with production 5a (~0.16 Bellever median) and 3a (~0.02).

Artefacts go to reports/phase5a_artefacts_spread/ — disjoint from the
production live_bundle/ + posteriors/ trees so this can run alongside
production without disturbing it. Promote to production by:
  - mv reports/phase5a_artefacts_spread/posteriors/lead_feature.nc \\
       reports/phase5a_artefacts/posteriors/lead_feature.nc
  - mv reports/phase5a_artefacts_spread/live_bundle/* \\
       reports/phase5a_artefacts/live_bundle/
  - thread add_spread_features=True through extend_5a.py + predict_5a.py
    (dataset prep is already optional-flag-gated in src/data.py).

Run:
    .venv/Scripts/python.exe -u scripts/experiment_5a_spread_features.py
"""
from __future__ import annotations

import json
import os
import pickle
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
sys.path.insert(0, str(ROOT / "scripts"))

import arviz as az  # noqa: E402

from src.data import (  # noqa: E402
    LOCATION,
    MODELS_NO_UKMO,
    WEATHERBLEND_DATA_ROOT,
    WET_THRESHOLD_MM,
    prepare_phase3_dataset,
)
from src.models.phase2_partial_pooling import (  # noqa: E402
    fit_partial_pooling,
    predict_partial_pooling_summary,
)

from _shared import resolve_station  # noqa: E402

OUT_DIR = ROOT / "reports" / "phase5a_artefacts_spread"
POSTERIOR_DIR = OUT_DIR / "posteriors"
LIVE_BUNDLE_DIR = OUT_DIR / "live_bundle"

# Pilot config: half the production chain count, quarter the draws — this is
# directional only. Posterior CI widths and per-coefficient ESS will be lower
# than a full run, but per-row mean P(wet) is robust enough at 100k+ training
# rows to answer "do the spread features drop the dry-tail floor?". Pair with
# chain_method="vectorized" (default in fit_partial_pooling now) so the two
# chains run as one batched JAX compute, not sequentially.
SAMPLER_DRAWS = 500
SAMPLER_TUNE = 500
SAMPLER_CHAINS = 4   # pmap'd across 4 virtual CPU devices (XLA_FLAGS set in
                      # src/models/phase2_partial_pooling.py at import time)
RANDOM_SEED = 42
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def _load_one_model_live_runs(model: str, window_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Mirror predict_5a's live loader — all hourly cycles for the model
    in a window, deduped to latest cycle per (ValidTime, Lead)."""
    model_dir = WEATHERBLEND_DATA_ROOT / "forecasts" / f"location={LOCATION}" / f"model={model}"
    frames = []
    for d in window_dates:
        date_str = d.strftime("%Y-%m-%d")
        date_dir = model_dir / f"date={date_str}"
        if not date_dir.exists():
            continue
        for path in sorted(date_dir.glob("run=*.parquet")):
            df = pd.read_parquet(
                path, columns=["RunTimeUtc", "ValidTimeUtc", "LeadHours", "Precipitation"],
            )
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ValidTimeUtc", "LeadHours", f"precip_{model}"])
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ValidTimeUtc", "LeadHours", "RunTimeUtc"])
        .drop_duplicates(subset=["ValidTimeUtc", "LeadHours"], keep="last")
        .drop(columns=["RunTimeUtc"])
        .rename(columns={"Precipitation": f"precip_{model}"})
        .reset_index(drop=True)
    )


def build_live_feature_frame(anchor: pd.Timestamp, feature_names: list[str]) -> pd.DataFrame:
    """Mirror predict_5a.build_feature_frame, but compute precip_max +
    precip_agreement_wet_01 on the inner-joined live precip columns."""
    window_dates = [anchor + pd.Timedelta(days=d) for d in range(-4, 2)]
    frames: list[pd.DataFrame] = []
    for model in MODELS_NO_UKMO:
        df = _load_one_model_live_runs(model, window_dates)
        if df.empty:
            raise RuntimeError(f"no live forecasts found for {model}")
        frames.append(df)
    forecasts = frames[0]
    for fc in frames[1:]:
        forecasts = forecasts.merge(fc, on=["ValidTimeUtc", "LeadHours"], how="inner")
    if forecasts.empty:
        return forecasts

    precip_cols = [f"precip_{m}" for m in MODELS_NO_UKMO]
    pm_arr = forecasts[precip_cols].to_numpy(dtype="float64")
    forecasts["precip_max"] = np.nanmax(pm_arr, axis=1)
    present = (~np.isnan(pm_arr)).sum(axis=1)
    wet_count = (pm_arr >= WET_THRESHOLD_MM).sum(axis=1)
    forecasts["precip_agreement_wet_01"] = np.where(
        present > 0, wet_count / np.maximum(present, 1), np.nan
    )

    hours = forecasts["ValidTimeUtc"].dt.hour
    forecasts["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    forecasts["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    forecasts["lead"] = forecasts["LeadHours"].astype("float64")

    # Drop rows where any feature is NaN — e.g. if a model has NaN precip
    # the spread cols may be NaN. Same-shape filter as production
    # `dropna(subset=precip_cols)` in _select_features.
    return forecasts.dropna(subset=feature_names).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Build dataset with spread features
    print(f"[{time.strftime('%H:%M:%S')}] Loading Phase 3 dataset (5-model + lead + spread)")
    ds = prepare_phase3_dataset(
        models=MODELS_NO_UKMO, lead_as_feature=True, add_spread_features=True,
        verbose=False,
    )
    print(f"  train rows: {len(ds.X_train):,}  test rows: {len(ds.X_test):,}")
    print(f"  features ({len(ds.feature_names)}): {ds.feature_names}")

    # 2) Fit posterior
    print(f"\n[{time.strftime('%H:%M:%S')}] Fitting partial-pool posterior")
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
    print(f"[{time.strftime('%H:%M:%S')}] Sampling done in {(time.time()-t0)/60:.1f} min")
    fit.idata.to_netcdf(POSTERIOR_DIR / "lead_feature.nc")
    s = az.summary(fit.idata,
                   var_names=["mu_intercept", "sigma_intercept", "mu_beta", "sigma_beta"])
    rhat = float(s["r_hat"].max())
    ess = float(s["ess_bulk"].min())
    ndiv = int(fit.idata.sample_stats["diverging"].sum()) if "diverging" in fit.idata.sample_stats else 0
    print(f"  rhat={rhat:.3f}  ess={ess:.0f}  divergences={ndiv}")

    # Persist scaler + metadata so a follow-up predict can reuse without
    # re-running prepare_phase3_dataset (~30s itself).
    with open(LIVE_BUNDLE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(ds.scaler, f)
    metadata = {
        "feature_names": list(ds.feature_names),
        "station_codes": list(ds.station_codes),
        "station_full_names": list(ds.station_full_names),
        "lead_hours": [int(L) for L in ds.lead_hours],
        "lead_feature_index": ds.feature_names.index("lead"),
        "scaler_mean": [float(x) for x in ds.scaler.mean_.tolist()],
        "scaler_scale": [float(x) for x in ds.scaler.scale_.tolist()],
    }
    (LIVE_BUNDLE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  scaler + metadata → {LIVE_BUNDLE_DIR}")

    # 3) Build live features for today
    anchor = pd.Timestamp.utcnow().normalize().tz_localize(None)
    print(f"\n[{time.strftime('%H:%M:%S')}] Building live features (anchor={anchor.date()})")
    live = build_live_feature_frame(anchor, ds.feature_names)
    print(f"  {len(live):,} live rows after inner-join + dropna(features)")
    if len(live) == 0:
        print("  no live rows — skipping inference")
        return

    # 4) Score per station — same pattern as predict_5a
    X_live = live[ds.feature_names].to_numpy(dtype="float64")
    X_live_s = ds.scaler.transform(X_live).astype("float64")

    print(f"\n[{time.strftime('%H:%M:%S')}] Scoring per station")
    print(f"{'station':>30s} {'rows':>6s} {'mean':>7s} {'median':>7s} {'min':>6s} {'max':>6s}")
    rows_by_station = {}
    for s_idx, (full_name, code) in enumerate(zip(ds.station_full_names, ds.station_codes)):
        station_idx = np.full(len(X_live_s), s_idx, dtype="int64")
        summary = predict_partial_pooling_summary(
            fit, X_live_s, station_idx, quantiles=QUANTILES,
        )
        out = pd.DataFrame({
            "ValidTimeUtc": live["ValidTimeUtc"].values,
            "LeadHours":    live["LeadHours"].astype("int64").values,
            "ProbWet":      summary["mean"],
            "ProbWetQ50":   summary["q0.5"],
            "ProbWetQ10":   summary["q0.1"],
            "ProbWetQ90":   summary["q0.9"],
            "Station":      full_name,
        })
        rows_by_station[full_name] = out

        today = pd.Timestamp(anchor.date())
        today_rows = out[out["ValidTimeUtc"].dt.date == today.date()]
        if len(today_rows) == 0:
            print(f"  {full_name:>30s} <no rows valid today>")
            continue
        print(f"  {full_name:>30s} {len(today_rows):>6d} {today_rows['ProbWet'].mean():>7.3f} "
              f"{today_rows['ProbWet'].median():>7.3f} "
              f"{today_rows['ProbWet'].min():>6.3f} {today_rows['ProbWet'].max():>6.3f}")

    # 5) Per-station today, per-lead band breakdown
    print(f"\n[{time.strftime('%H:%M:%S')}] +24h band breakdown (today, lead in [12, 36))")
    print(f"{'station':>30s} {'rows':>6s} {'mean':>7s} {'median':>7s} {'max':>6s}")
    for full_name, out in rows_by_station.items():
        today_band = out[(out["ValidTimeUtc"].dt.date == anchor.date())
                         & (out["LeadHours"] >= 12) & (out["LeadHours"] < 36)]
        if len(today_band) == 0:
            continue
        # Match renderer tiebreaker (highest LeadHours per valid_time)
        chosen = today_band.sort_values(["ValidTimeUtc", "LeadHours"]).groupby("ValidTimeUtc").tail(1)
        print(f"  {full_name:>30s} {len(chosen):>6d} {chosen['ProbWet'].mean():>7.3f} "
              f"{chosen['ProbWet'].median():>7.3f} {chosen['ProbWet'].max():>6.3f}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()
