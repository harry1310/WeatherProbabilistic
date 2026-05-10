"""Phase 4a — TRAINING ONLY. Fits the dbarts BART precip blender once
per station, persists the saved state + preprocess + metadata under
data/models/precipitation/{station}/{version}/, and exits. Live
prediction is the job of predict_4a.py, which loads the saved state on
its own cadence (4×/day).

This split was unblocked once we verified the dbarts state round-trips
bit-exactly via storeState + saveRDS → readRDS + warm-scaffold +
setState (see scripts/smoke_dbarts_roundtrip.py + memory
reference_dbarts_serialize_caveat.md). Pre-split, train+predict ran in a
single R session because earlier (incorrect) experiments suggested
predict-from-fresh-session always fell back to Y.mean(); the missing
piece was the explicit storeState() call.

Bundle layout written per station per version:
  data/models/precipitation/{station}/{version}/
    state.rds              — saveRDS(list(state = fit$fit$state))
    arrays.npz             — X_train_s (already scaled+imputed) + y_train
                             — the warm scaffold needs IDENTICAL training
                             inputs to reproduce the original binary
                             detection + cutpoint inference, so we save
                             the scaled arrays (not raw) to skip preprocess
                             reapplication on the warm fit
    preprocess.json        — kept feature names + indices, median fill
                             values, StandardScaler mean/scale — the
                             predict-side reapplies these to live raw
                             features before predict()
    training_metadata.json — Models card + Spec page payload (existing schema)
    feature_schema.json    — per-lead schema for the Spec page (existing)

CLI:
    train_4a.py [--stations slug1 ...]  [--anchor YYYY-MM-DD]

The anchor is informational only — training does not slice by it; it's
written into the version string + metadata for traceability.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
_r_home = os.environ.get("R_HOME", r"C:\Program Files\R\R-4.6.0")
os.environ.setdefault("R_HOME", _r_home)
_r_bin = os.path.join(_r_home, "bin", "x64")
if hasattr(os, "add_dll_directory") and os.path.isdir(_r_bin):
    os.add_dll_directory(_r_bin)
os.environ["PATH"] = _r_bin + os.pathsep + os.environ.get("PATH", "")
_user_lib = os.path.join(os.environ.get("USERPROFILE", os.environ.get("HOME", "")),
                         "R", "win-library", "4.6")
os.environ.setdefault("R_LIBS_USER", _user_lib)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scipy.stats import norm  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import default_converter, numpy2ri, pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import LOCATION, WEATHERBLEND_DATA_ROOT  # noqa: E402
from src.retrain_guard import build_check_and_save_versioned  # noqa: E402

from _shared import (  # noqa: E402
    FEATURE_NAMES,
    MODELS_LEAN,
    add_synoptic_features,
    build_features_via_duckdb,
    resolve_station,
    time_split,
)

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")

# Champion config locked from the Phase 6 9-cell bake-off.
NTREE = 500
K = 3.0
NSKIP = 200
NDPOST = 1000
SEED = 42
PHASE = "4a"

STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
# Lead 12 dropped 2026-05-10 — Open-Meteo's offset_day (previous_runs)
# archive doesn't return lead 12 forecasts that go far enough back to
# survive the train/val/test split, so per-lead test stats came back with
# TestRows=0 for lead 12 in the v2026-05-09 bundle. The pooled fit still
# included lead 12 nominally but had effectively no learning signal there,
# making predict-time output pure extrapolation along the `lead` axis.
# Keeping {24, 48, 72, 96, 120} where every bucket has ~14k train rows
# and a real BSS vs climatology.
LEADS = [24, 48, 72, 96, 120]


def build_pooled_training_features(station_friendly: str) -> pd.DataFrame:
    """Pool every lead in LEADS into one DataFrame with a `lead` column.

    SORT ORDER: time_split is positional 70/15/15. Sorting by
    (ValidTimeUtc, lead) ensures every lead contributes proportionally
    to each split slice — without this, leads would land in disjoint
    blocks and per-lead test Brier would collapse to NaN for most leads.
    """
    frames = []
    for lead in LEADS:
        df = build_features_via_duckdb(station_friendly, lead)
        df, _syn_feats = add_synoptic_features(station_friendly, lead, df)
        df["lead"] = float(lead)
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values(["ValidTimeUtc", "lead"], kind="mergesort").reset_index(drop=True)
    return pooled


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def train_one_station(station_friendly: str) -> dict:
    """Fit dbarts on pooled training features, evaluate on the held-out
    test slice (passed as x.test in the same fit), return everything the
    bundle writer needs.
    """
    syn_feats = ["wind_dir_sin_mean", "wind_dir_cos_mean", "surface_pressure_mean"]
    feats = list(FEATURE_NAMES) + syn_feats + ["lead"]

    print(f"  building pooled training features ({len(LEADS)} leads)...", flush=True)
    df_pooled = build_pooled_training_features(station_friendly)
    train_df, val_df, test_df = time_split(df_pooled)

    X_train_full = train_df[feats].to_numpy(dtype="float64")
    y_train      = train_df["wet"].to_numpy(dtype="float64")
    X_test_full  = test_df[feats].to_numpy(dtype="float64")
    y_test       = test_df["wet"].to_numpy(dtype="int8")

    # Drop columns that are all-NaN in training — fixes the kept-columns
    # mask we'll re-apply on the predict side. Live features MUST drop
    # the same indices or the saved state's tree splits won't line up.
    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept = np.where(~col_all_nan)[0]
    feature_names_eff = [feats[i] for i in kept]
    X_train = X_train_full[:, kept]
    X_test  = X_test_full[:, kept]
    print(f"  train: {len(y_train):,} | val: {len(val_df):,} | test: {len(y_test):,} | "
          f"features eff: {len(feature_names_eff)}", flush=True)

    median = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), median, X_train)
    X_test  = np.where(np.isnan(X_test),  median, X_test)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float64)
    X_test_s  = scaler.transform(X_test).astype(np.float64)

    print(f"  fitting dbarts (ntree={NTREE}, k={K}, nskip={NSKIP}, ndpost={NDPOST})...",
          flush=True)
    t0 = time.time()
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train_s)
        y_train_r = ro.conversion.py2rpy(y_train)
        x_test_r  = ro.conversion.py2rpy(X_test_s)
    # verbose=True so dbarts emits per-block sampler progress (tens of
    # lines over the 200-burn + 1000-post run). Without this the fit
    # is silent for ~8 min on Ubuntu CI / ~30 min on Windows; no way to
    # tell a slow fit from a hung one. flush=True up-stream + R's
    # own stderr flushing means lines land in the CI log in real time.
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_test_r,
        ntree=NTREE, k=K, nskip=NSKIP, ndpost=NDPOST,
        keeptrees=True, verbose=True, seed=SEED,
    )
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat_test = np.array(ro.conversion.rpy2py(yhat_test_r))
    p_test = norm.cdf(yhat_test).mean(axis=0)
    print(f"  fit done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Per-lead test Brier for training_metadata.PerLead.
    per_lead = []
    test_lead = test_df["lead"].astype(int).to_numpy()
    train_clim = float(train_df["wet"].mean())
    for L in LEADS:
        mask = test_lead == L
        n = int(mask.sum())
        if n == 0:
            per_lead.append({"LeadHours": L, "TestRows": 0,
                             "BlendTestMae": float("nan"), "BSS": float("nan")})
            continue
        p_l = p_test[mask]
        y_l = y_test[mask]
        b = _brier(p_l, y_l)
        clim_b = _brier(np.full_like(y_l, train_clim, dtype="float64"), y_l)
        bss = (clim_b - b) / clim_b if clim_b > 0 else float("nan")
        per_lead.append({
            "LeadHours":   L,
            "TestRows":    n,
            "ValRows":     int((val_df["lead"] == L).sum()),
            "TrainRows":   int((train_df["lead"] == L).sum()),
            "BlendTestMae": b,
            "BSS":          bss,
            "ClimatologyBrier": clim_b,
            "BestSingle":   "(BART blender — pooled across NWPs)",
            "BestSingleTestMae": float("nan"),
        })

    # Stash fit in globalenv for storeState — refclass methods aren't
    # subscriptable from rpy2's S4 view, must invoke through ro.r().
    ro.globalenv["fit"] = fit
    ro.r('fit$fit$storeState()')

    return {
        "fit_in_globalenv": True,  # fit is now ro.globalenv["fit"]
        "per_lead":   per_lead,
        "train_df":   train_df,
        "val_df":     val_df,
        "test_df":    test_df,
        "feature_names_eff": feature_names_eff,
        "kept_indices":      kept.tolist(),
        "feature_list_full": feats,
        "median":     median,
        "scaler_mean":  scaler.mean_,
        "scaler_scale": scaler.scale_,
        "X_train_s":  X_train_s,
        "y_train":    y_train,
        # Per-row test predictions for downstream bake-offs (e.g. 3a+4a
        # linear pool). Same shape and column conventions as 5a's
        # test_predictions.parquet so a single bake-off script can
        # consume both. y_test cast to int8 to match.
        "p_test":     p_test,
        "y_test":     y_test,
    }


def write_bundle(out_dir: Path, station_slug: str, station_friendly: str,
                 version: str, result: dict, anchor: datetime) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) state.rds — saveRDS through R itself (the state is an R list of
    # integer matrices; pickling it would lose the structure).
    state_path = (out_dir / "state.rds").as_posix()
    ro.globalenv["state_only"] = ro.r('list(state = fit$fit$state)')
    ro.r(f'saveRDS(state_only, "{state_path}")')

    # 2) arrays.npz — the warm scaffold needs the IDENTICAL training
    # inputs to reproduce binary detection + cutpoints. Save the SCALED
    # versions so the warm fit replays them verbatim with no preprocess
    # round-trip risk.
    np.savez_compressed(
        out_dir / "arrays.npz",
        X_train_s=result["X_train_s"].astype(np.float64),
        y_train=result["y_train"].astype(np.float64),
    )

    # 3) preprocess.json — applied to LIVE raw features before predict()
    # to map them into the same feature space the saved state was fit in.
    preprocess = {
        "feature_list_full":  result["feature_list_full"],
        "kept_indices":       result["kept_indices"],
        "feature_names_eff":  result["feature_names_eff"],
        "median":             result["median"].tolist(),
        "scaler_mean":        result["scaler_mean"].tolist(),
        "scaler_scale":       result["scaler_scale"].tolist(),
        "ntree":              NTREE,
        "k":                  K,
        "ndpost":             NDPOST,
        "seed":               SEED,
        "leads":              LEADS,
    }
    (out_dir / "preprocess.json").write_text(json.dumps(preprocess, indent=2))

    # 4) training_metadata.json + feature_schema.json — Models card + Spec
    # page + verify pipeline payload. Schema unchanged from the previous
    # train+predict combo so WeatherBlend's loader picks 4a up unchanged.
    syn_feats = ["wind_dir_sin_mean", "wind_dir_cos_mean", "surface_pressure_mean"]
    base_feats = list(FEATURE_NAMES) + syn_feats + ["lead"]
    train_df = result["train_df"]
    val_df   = result["val_df"]
    test_df  = result["test_df"]
    per_lead = result["per_lead"]
    feature_names_eff = result["feature_names_eff"]

    metadata = {
        "Version":     version,
        "Target":      "precipitation",
        "Phase":       PHASE,
        "DataSource":  "open_meteo_previous_runs+ea_rainfall+dbarts_bart_lead_as_feature",
        "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
        "Hyperparameters": {
            "library":   "dbarts (R) via rpy2",
            "ntree":     NTREE,
            "k":         K,
            "nskip":     NSKIP,
            "ndpost":    NDPOST,
            "seed":      SEED,
            "objective": "binary (probit-link BART)",
            "leadAsFeature": True,
            "trainPredictSplit": True,
        },
        "DeviationsFromBrief": [
            "BART (Bayesian Additive Regression Trees) via R dbarts package, "
            "called from Python via rpy2.",
            "Train and predict are now separate processes — train_4a.py emits "
            "a saved state.rds (storeState + saveRDS) and predict_4a.py loads "
            "it via a warm scaffold + setState on each 6-hourly cycle.",
            "Lead pooled across all six leads via a `lead` feature column "
            "(Phase 5 pattern); one BART per station instead of per (station, "
            "lead).",
            "22-feature 3a base + 3 synoptic flow features (wind_dir_sin_mean, "
            "wind_dir_cos_mean, surface_pressure_mean).",
        ],
        "PerLead": {str(d["LeadHours"]): {
            **d,
            "DataRangeTrain": (
                f"{train_df['ValidTimeUtc'].min()} → {train_df['ValidTimeUtc'].max()}"
                if len(train_df) else ""),
            "DataRangeVal":   (
                f"{val_df['ValidTimeUtc'].min()} → {val_df['ValidTimeUtc'].max()}"
                if len(val_df) else ""),
            "DataRangeTest":  (
                f"{test_df['ValidTimeUtc'].min()} → {test_df['ValidTimeUtc'].max()}"
                if len(test_df) else ""),
            "TestCalendarMonths": 4,
        } for d in per_lead if d["LeadHours"] > 0},
    }

    nwp_models = [m for m, _ in MODELS_LEAN]
    schema_per_lead = {
        str(L): {
            "Target": "precipitation",
            "FeatureSet": f"phase4a-bart-l{L:02}",
            "LeadHours":  L,
            "RequiredModels": [],
            "OptionalModels": nwp_models,
            "Models":         nwp_models,
            "FeatureNames":   feature_names_eff,
            "DataSource":     "open_meteo_previous_runs",
            "Tier":           "4a-bart",
            "UkvStrategy":    None,
        } for L in LEADS
    }
    schema = {"Leads": schema_per_lead}

    metadata = _json_sanitize_nans(metadata)
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str, allow_nan=False))
    (out_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, allow_nan=False))

    # 5) test_predictions.parquet — per-row held-out probabilities for
    # downstream bake-offs (e.g. 3a+4a linear pool). Same column
    # conventions as 5a's test_predictions.parquet so a single bake-off
    # script can inner-join across phases. lead is the per-row lead
    # value (lead-as-feature 4a pools all leads in one fit, so each test
    # row already carries its own lead value).
    p_test = result["p_test"]
    y_test = result["y_test"]
    test_pred_df = pd.DataFrame({
        "valid_time": pd.to_datetime(test_df["ValidTimeUtc"].values),
        "station":    station_slug,
        "lead":       test_df["lead"].astype(int).values,
        "p_wet":      p_test,
        "observed_wet": y_test.astype("int8"),
    })
    test_pred_df.to_parquet(out_dir / "test_predictions.parquet", index=False)

    sizes = {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file()}
    print(f"  bundle → {out_dir}")
    for name, size in sorted(sizes.items()):
        print(f"    {name:25s} {size:>10,} bytes")


def _json_sanitize_nans(obj):
    """Replace non-finite floats (NaN, ±inf) with 0.0 for strict-JSON
    consumers — the WeatherBlend C# renderer's System.Text.Json rejects
    literal NaN/Infinity tokens and silently drops the model summary.
    See ModelArtifact.cs convention.
    """
    import math
    if isinstance(obj, dict):
        return {k: _json_sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize_nans(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return 0.0
    if hasattr(obj, "__float__"):
        try:
            f = float(obj)
        except (TypeError, ValueError):
            return obj
        if not math.isfinite(f):
            return 0.0
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", default=None,
                   help="Anchor date YYYY-MM-DD UTC for the version string only.")
    p.add_argument("--stations", nargs="*", default=None,
                   help="Station subset (default: all 3 active).")
    p.add_argument("--models-root", default=str(WEATHERBLEND_DATA_ROOT / "models"),
                   help="Models tree root for bundle output.")
    args = p.parse_args()

    if args.anchor:
        anchor = datetime.fromisoformat(args.anchor).replace(tzinfo=timezone.utc)
    else:
        anchor = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    anchor = anchor.replace(tzinfo=None)

    stations = args.stations or STATIONS
    models_root = Path(args.models_root)
    version = datetime.now(timezone.utc).strftime("v%Y-%m-%d_%H%M%S_phase4a")

    print(f"[{time.strftime('%H:%M:%S')}] Phase 4a TRAIN (split from predict, state-persisting)")
    print(f"  anchor:  {anchor.isoformat()}")
    print(f"  version: {version}")
    print(f"  stations: {stations}")
    print(f"  leads (pooled feature): {LEADS}")

    # RetrainGuard wired in 2026-05-10 (Phase 1c of AUTO_RETRAIN_PLAN.md
    # — Python side). Each station guards independently against the
    # latest 4a version under that station's models dir; if the guard
    # fires, that station's bundle is skipped (no write_bundle, no
    # state.rds, no metadata) and the loop continues with the others.
    # If any station fails, exit 4 at the end so the [ci-fail]
    # retrain-python issue fires.
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    log = _logging.getLogger("train_4a")

    bundles_written = 0
    guard_failures = 0
    for station_input in stations:
        station_slug, station_friendly = resolve_station(station_input)
        print(f"\n[{time.strftime('%H:%M:%S')}] {station_friendly}")
        result = train_one_station(station_friendly)
        bundle_dir = models_root / "precipitation" / station_slug / version
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Guard BEFORE the heavy bundle write. Inputs are the SCALED
        # train arrays (the warm-scaffold path the predict side uses, so
        # NaN / mean / std stats reflect what the model actually fit on).
        # On pass the guard writes training_summary.json into bundle_dir;
        # write_bundle() runs after and adds the rest of the artefacts.
        y_train = result["y_train"]
        guard_result = build_check_and_save_versioned(
            log,
            version_dir=bundle_dir,
            composite=f"precipitation/{station_slug}",
            phase=PHASE,
            version=version,
            rows_train=len(result["train_df"]),
            rows_val=len(result["val_df"]),
            rows_test=len(result["test_df"]),
            train_features=result["X_train_s"],
            feature_names=result["feature_names_eff"],
            label_rates={station_slug: float(np.mean(y_train))} if len(y_train) else {},
        )
        if not guard_result.passed:
            print(
                f"  guard FAIL — skipping bundle for {station_slug}; previous version stays current."
            )
            guard_failures += 1
            ro.r('rm(fit, state_only); gc()')
            continue

        write_bundle(bundle_dir, station_slug, station_friendly, version, result, anchor)
        bundles_written += 1
        # Free the R-side fit before the next station so peak RAM stays bounded.
        ro.r('rm(fit, state_only); gc()')

    print()
    print(f"Phase 4a train complete. Bundles written: {bundles_written} (guard failures: {guard_failures})")
    if bundles_written == 0:
        sys.exit(1)
    if guard_failures > 0:
        # Some stations succeeded; non-zero exit so the workflow webhook
        # opens a [ci-fail] issue listing the failed cells. Successful
        # stations remain promoted (they wrote bundles).
        sys.exit(4)


if __name__ == "__main__":
    main()
