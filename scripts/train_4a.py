"""Phase 4a — TRAINING ONLY. Fits one dbarts BART per (station, lead)
and persists per-cell state + preprocess + metadata under
data/models/precipitation/{station}/{version}_phase4a/.

Per-cell architecture: each (station, lead) gets its own BART fit on
its own ~14k-row slice, with no `lead` feature column. Matches 3a's
per-cell architecture and recovers the -4.0% aggregate Brier advantage
that the previous lead-pooled 4a gave up (see WB bake-off 2026-05-10).

The lead-pooled 4a flavour was retired 2026-05-11 in favour of
per-cell — the aligned-row bake-off showed per-cell beats lead-pooled
by ~6% Brier and the diversity-ceiling analysis showed the linear
pool of the two flavours added literally zero, so keeping both was
redundant. Old lead-pooled bundles (`v..._phase4a` with a flat
state.rds at the root) become stale immediately after the first
per-cell retrain — find_latest_bundle in predict_4a.py filters them
out by requiring per-lead state_lead_NNh.rds files.

Deployment posture: 4a stays a CHALLENGER alongside 3a (3a Current).
Bovey-specific underperformance (per-cell BART loses by 0.8-4.6% on
Bovey cells) is accepted as a known caveat at this stage.

Bundle layout per station per version::

  data/models/precipitation/{station}/{version}_phase4a/
    state_lead_24h.rds         saveRDS(list(state = fit$fit$state)) per lead
    state_lead_48h.rds
    state_lead_72h.rds
    state_lead_96h.rds
    state_lead_120h.rds
    arrays_lead_24h.npz        X_train_s + y_train (scaled, imputed) per lead
    arrays_lead_48h.npz        (needed for warm-scaffold predict-side replay)
    ... (one per lead)
    preprocess.json            {"24": {feature_names_eff, kept_indices, median,
                                       scaler_mean, scaler_scale}, "48": {...}, ...}
                                — per-lead preprocess since each lead has its
                                own NaN pattern and scaler params.
    training_metadata.json     PerLead populated for 5 leads (Models card)
    training_summary.json      RetrainGuard sidecar (per-station-aggregated)
    feature_schema.json        per-lead spec (Spec page)
    test_predictions.parquet   per-row (valid_time, station, lead, p_wet,
                                observed_wet) across all 5 leads
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
from run_phase6_bart_bakeoff import brier  # noqa: E402

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")

NTREE = 500
K = 3.0
NSKIP = 200
NDPOST = 1000
SEED = 42
PHASE = "4a"

STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
# All 5 leads (24/48/72/96/120). The original 9-cell research scope was
# {24, 48, 72} only; we extend to 96/120 here because the per-cell
# architecture should scale cleanly and the long-lead skill is what
# the Models card needs to show. As a challenger deployment, the
# downside of extending is minimal — worst case the long-lead per-cell
# numbers underperform and the Models card just shows that.
LEADS = [24, 48, 72, 96, 120]


def _prepare_cell(df: pd.DataFrame, feature_list: list[str]):
    """Per-cell prep mirrors run_phase6_bart_9cell's prepare_full but
    returns the scaler/median/kept_indices for bundle persistence. The
    predict side needs these to map live features into the same scaled
    space the saved state was fit in."""
    train_df, val_df, test_df = time_split(df)
    X_train_full = train_df[feature_list].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="float64")
    X_test_full = test_df[feature_list].to_numpy(dtype="float64")
    y_test = test_df["wet"].to_numpy(dtype="int8")
    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept = np.where(~col_all_nan)[0]
    X_train = X_train_full[:, kept]
    X_test = X_test_full[:, kept]
    median = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), median, X_train)
    X_test = np.where(np.isnan(X_test), median, X_test)
    scaler = StandardScaler().fit(X_train)
    feature_names_eff = [feature_list[i] for i in kept]
    return {
        "X_train_s":         scaler.transform(X_train),
        "y_train":           y_train,
        "X_test_s":          scaler.transform(X_test),
        "y_test":            y_test,
        "train_df":          train_df,
        "val_df":            val_df,
        "test_df":           test_df,
        "feature_list_full": feature_list,
        "kept_indices":      kept.tolist(),
        "feature_names_eff": feature_names_eff,
        "median":            median.tolist(),
        "scaler_mean":       scaler.mean_.tolist(),
        "scaler_scale":      scaler.scale_.tolist(),
    }


def _fit_and_store(cell: dict, lead: int) -> dict:
    """Fit dbarts on the prepped cell, score test, store state in R's
    globalenv under a lead-specific name so we can saveRDS it after."""
    X_train_s = cell["X_train_s"].astype(np.float64)
    y_train   = cell["y_train"].astype(np.float64)
    X_test_s  = cell["X_test_s"].astype(np.float64)
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train_s)
        y_train_r = ro.conversion.py2rpy(y_train)
        x_test_r  = ro.conversion.py2rpy(X_test_s)
    t0 = time.time()
    # verbose=True so dbarts emits per-block progress to stderr — without
    # it the per-cell fit is silent for several minutes and CI/local
    # logs can't tell a slow fit from a hung one. Same convention as
    # train_4a.py and run_phase6_bart_9cell.
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_test_r,
        ntree=NTREE, k=K, nskip=NSKIP, ndpost=NDPOST,
        keeptrees=True, verbose=True, seed=SEED,
    )
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat_test = np.array(ro.conversion.rpy2py(yhat_test_r))
    p_test = norm.cdf(yhat_test).mean(axis=0)
    wall = time.time() - t0

    # Store state in R-side globalenv under a lead-keyed name. After
    # ALL cells for this station are fit, we saveRDS each in turn.
    # storeState mutates fit$fit$state to a serialisable form.
    fit_var = f"fit_lead_{lead}h"
    ro.globalenv[fit_var] = fit
    ro.r(f'{fit_var}$fit$storeState()')
    return {
        "p_test":   p_test,
        "y_test":   cell["y_test"],
        "wall_s":   wall,
        "fit_var":  fit_var,
    }


def train_one_station(station_friendly: str, station_slug: str) -> dict:
    """Fit per-cell BART for every lead in LEADS for one station.
    Returns aggregate result dict with per-cell prep + fit info, ready
    for write_per_cell_bundle to persist."""
    syn_feats = ["wind_dir_sin_mean", "wind_dir_cos_mean", "surface_pressure_mean"]
    base_feats = list(FEATURE_NAMES) + syn_feats   # NOTE: no "lead" — per-cell

    per_cell: dict[int, dict] = {}
    per_lead_stats: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for lead in LEADS:
        print(f"\n  [{time.strftime('%H:%M:%S')}] lead {lead}h — building features", flush=True)
        df = build_features_via_duckdb(station_friendly, lead)
        df, syn_feats_added = add_synoptic_features(station_friendly, lead, df)
        feats = list(FEATURE_NAMES) + syn_feats_added
        cell = _prepare_cell(df, feats)
        print(f"    rows: train {len(cell['y_train']):,} · val {len(cell['val_df']):,} · "
              f"test {len(cell['y_test']):,} · features kept {len(cell['feature_names_eff'])}", flush=True)

        fit_out = _fit_and_store(cell, lead)
        b = brier(fit_out["p_test"], fit_out["y_test"].astype(np.float64))
        clim_b = brier(np.full_like(fit_out["y_test"], cell["train_df"]["wet"].mean(), dtype="float64"),
                       fit_out["y_test"].astype(np.float64))
        bss = (clim_b - b) / clim_b if clim_b > 0 else float("nan")
        print(f"    fit done in {fit_out['wall_s']:.1f}s — BART Brier={b:.4f} (BSS {bss:+.4f}, clim {clim_b:.4f})", flush=True)

        per_cell[lead] = {**cell, **fit_out}
        per_lead_stats.append({
            "LeadHours":         lead,
            "TestRows":          int(len(fit_out["y_test"])),
            "ValRows":           int(len(cell["val_df"])),
            "TrainRows":         int(len(cell["y_train"])),
            "BlendTestMae":      b,                                  # repurposed: Brier
            "BSS":               bss,
            "ClimatologyBrier":  clim_b,
            "BestSingle":        "(per-cell BART)",
            "BestSingleTestMae": float("nan"),
            "DataRangeTrain":    (f"{cell['train_df']['ValidTimeUtc'].min()} → "
                                  f"{cell['train_df']['ValidTimeUtc'].max()}"
                                  if len(cell['train_df']) else ""),
            "DataRangeVal":      (f"{cell['val_df']['ValidTimeUtc'].min()} → "
                                  f"{cell['val_df']['ValidTimeUtc'].max()}"
                                  if len(cell['val_df']) else ""),
            "DataRangeTest":     (f"{cell['test_df']['ValidTimeUtc'].min()} → "
                                  f"{cell['test_df']['ValidTimeUtc'].max()}"
                                  if len(cell['test_df']) else ""),
            "TestCalendarMonths": 4,
        })
        pred_frames.append(pd.DataFrame({
            "valid_time":   pd.to_datetime(cell["test_df"]["ValidTimeUtc"].values),
            "station":      station_slug,
            "lead":         lead,
            "p_wet":        fit_out["p_test"],
            "observed_wet": fit_out["y_test"].astype("int8"),
        }))

    test_predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    # Aggregate train slice for the RetrainGuard summary. Use lead 24h's
    # slice as the canonical per-station train (largest, same shape as
    # 3a per-station guard). Per-lead feature stats would be richer but
    # the guard helper expects one (rows, features) matrix.
    canonical_lead = LEADS[0]
    canonical = per_cell[canonical_lead]
    train_features = canonical["X_train_s"]
    feature_names = canonical["feature_names_eff"]

    return {
        "per_cell":          per_cell,
        "per_lead_stats":    per_lead_stats,
        "test_predictions":  test_predictions,
        "train_features":    train_features,        # for RetrainGuard
        "feature_names":     feature_names,
        "y_train":           canonical["y_train"],
    }


def write_per_cell_bundle(out_dir: Path, station_slug: str, station_friendly: str,
                          version: str, result: dict, anchor: datetime) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) state.rds per lead — saveRDS through R itself.
    for lead, cell in result["per_cell"].items():
        fit_var = cell["fit_var"]
        rds_path = (out_dir / f"state_lead_{lead}h.rds").as_posix()
        ro.globalenv["state_only"] = ro.r(f'list(state = {fit_var}$fit$state)')
        ro.r(f'saveRDS(state_only, "{rds_path}")')

    # 2) arrays_lead_NNh.npz per lead — warm-scaffold inputs.
    for lead, cell in result["per_cell"].items():
        np.savez_compressed(
            out_dir / f"arrays_lead_{lead}h.npz",
            X_train_s=cell["X_train_s"].astype(np.float64),
            y_train=cell["y_train"].astype(np.float64),
        )

    # 3) preprocess.json — per-lead block, all in one file.
    preprocess = {
        "ntree": NTREE, "k": K, "ndpost": NDPOST, "seed": SEED,
        "leads": LEADS,
        "per_lead": {
            str(lead): {
                "feature_list_full":  cell["feature_list_full"],
                "kept_indices":       cell["kept_indices"],
                "feature_names_eff":  cell["feature_names_eff"],
                "median":             cell["median"],
                "scaler_mean":        cell["scaler_mean"],
                "scaler_scale":       cell["scaler_scale"],
            }
            for lead, cell in result["per_cell"].items()
        },
    }
    (out_dir / "preprocess.json").write_text(json.dumps(preprocess, indent=2))

    # 4) training_metadata.json + feature_schema.json — Models card + Spec.
    nwp_models = [m for m, _ in MODELS_LEAN]
    metadata = {
        "Version":      version,
        "Target":       "precipitation",
        "Phase":        PHASE,
        "DataSource":   "open_meteo_previous_runs+ea_rainfall+dbarts_bart_per_cell",
        "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
        "Hyperparameters": {
            "library":           "dbarts (R) via rpy2",
            "ntree":             NTREE,
            "k":                 K,
            "nskip":             NSKIP,
            "ndpost":            NDPOST,
            "seed":              SEED,
            "objective":         "binary (probit-link BART)",
            "leadAsFeature":     False,
            "trainPredictSplit": True,
            "perCell":           True,
        },
        "DeviationsFromBrief": [
            "Per-cell BART: one fit per (station, lead). Matches 3a's per-cell "
            "architecture; deployed as a challenger alongside 3a (3a stays Current).",
            "5 leads: {24, 48, 72, 96, 120}. Original 9-cell research used "
            "{24, 48, 72} only; extending here as per the deployment plan.",
            "PerLeadStats fields repurposed: BlendTestMae=BART Brier, "
            "BlendTestRmse=climatology Brier (NOT MAE/RMSE — site reuses fields).",
        ],
        "PerLead": {
            str(d["LeadHours"]): d for d in result["per_lead_stats"]
        },
    }
    schema_per_lead = {
        str(L): {
            "Target":         "precipitation",
            "FeatureSet":     f"phase4a-percell-l{L:02}",
            "LeadHours":      L,
            "RequiredModels": [],
            "OptionalModels": nwp_models,
            "Models":         nwp_models,
            "FeatureNames":   result["per_cell"][L]["feature_names_eff"],
            "DataSource":     "open_meteo_previous_runs",
            "Tier":           "4a-percell",
            "UkvStrategy":    None,
        }
        for L in LEADS if L in result["per_cell"]
    }
    metadata = _json_sanitize_nans(metadata)
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str, allow_nan=False))
    (out_dir / "feature_schema.json").write_text(
        json.dumps({"Leads": schema_per_lead}, indent=2, allow_nan=False))

    # 5) test_predictions.parquet — aggregated across 5 leads.
    if not result["test_predictions"].empty:
        result["test_predictions"].to_parquet(out_dir / "test_predictions.parquet", index=False)

    sizes = {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file()}
    print(f"  bundle → {out_dir}")
    for name, size in sorted(sizes.items()):
        print(f"    {name:30s} {size:>10,} bytes")


def _json_sanitize_nans(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_nans(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--anchor", default=None,
                   help="Anchor date YYYY-MM-DD UTC for the version string. Default: today.")
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

    print(f"[{time.strftime('%H:%M:%S')}] Phase 4a per-cell TRAIN")
    print(f"  anchor:   {anchor.isoformat()}")
    print(f"  version:  {version}")
    print(f"  stations: {stations}")
    print(f"  leads:    {LEADS}")
    print(f"  fits:     {len(stations)} stations × {len(LEADS)} leads = {len(stations)*len(LEADS)} BART fits total")

    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    log = _logging.getLogger("train_4a_per_cell")

    bundles_written = 0
    guard_failures = 0
    for station_input in stations:
        station_slug, station_friendly = resolve_station(station_input)
        print(f"\n[{time.strftime('%H:%M:%S')}] {station_friendly} — {len(LEADS)} per-cell fits")
        result = train_one_station(station_friendly, station_slug)

        bundle_dir = models_root / "precipitation" / station_slug / version
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # RetrainGuard against the previous 4a (or first-ever if no
        # prior). Phase tag `4a` — old lead-pooled bundles also used
        # this tag; try_load_previous_summary will pick up whichever
        # is latest. That's fine: even if the previous run was lead-
        # pooled with different feature stats, the guard's NaN-pct +
        # label-rate + row-count bands are robust to architectural
        # shifts, and the first per-cell retrain inherits the lead-
        # pooled summary as its baseline (acceptable seed).
        y_train = result["y_train"]
        guard_result = build_check_and_save_versioned(
            log,
            version_dir=bundle_dir,
            composite=f"precipitation/{station_slug}",
            phase=PHASE,
            version=version,
            rows_train=len(result["train_features"]),
            rows_val=0,
            rows_test=int(result["test_predictions"].shape[0] // len(LEADS)) if len(result["test_predictions"]) else 0,
            train_features=result["train_features"],
            feature_names=result["feature_names"],
            label_rates={station_slug: float(np.mean(y_train))} if len(y_train) else {},
        )
        if not guard_result.passed:
            print(f"  guard FAIL — skipping bundle for {station_slug}.")
            guard_failures += 1
            # Free R-side state before next station so peak RAM is bounded.
            for lead in LEADS:
                ro.r(f'rm(fit_lead_{lead}h)')
            ro.r('gc()')
            continue

        write_per_cell_bundle(bundle_dir, station_slug, station_friendly, version, result, anchor)
        bundles_written += 1

        # Free R-side fits before the next station — peak RAM scales
        # linearly otherwise (5 BARTs × 14k rows × keeptrees=TRUE).
        for lead in LEADS:
            ro.r(f'rm(fit_lead_{lead}h)')
        ro.r('rm(state_only); gc()')

    print()
    print(f"Phase 4a per-cell train complete. Bundles written: {bundles_written} "
          f"(guard failures: {guard_failures})")
    if bundles_written == 0:
        sys.exit(1)
    if guard_failures > 0:
        sys.exit(4)


if __name__ == "__main__":
    main()
