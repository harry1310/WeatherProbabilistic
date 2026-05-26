"""Phase 3f — TRAINING ONLY. Fits one NGBoost-LogNormal regressor per
(station, lead) on the wet-only rows of EA hourly rainfall truth and
persists per-cell pickles + preprocess + metadata under
``data/models/rainfall_amount/{station_slug}/{version}_phase3f/``.

3f is a **two-stage distributional forecast** for rainfall_amount
(hourly mm/h):

  F(x) = (1 − π) · δ_0(x) + π · LogNormal(μ_log, σ_log)(x)

Stage 1 (π = P(hour is wet)) is supplied by the bound Phase 3a champion
at predict time — NOT fit here. Stage 2 (μ_log, σ_log) is what THIS
trainer fits via NGBoost-LogNormal, learning feature-dependent
heteroscedasticity from the wet-only rows of EA hourly rainfall truth.

Identity is FIXED: 3f IS "NGBoost-LogNormal over Phase 3a's hourly
P(wet) marginal". The 3f→3a source binding lives in
``WeatherBlend/src/WeatherBlend/Train/PrecipIntensity/IntensityModelSources.cs``
— same hardcoded-source-of-truth discipline the deleted-2026-05-25
``DryWindowMcSources`` established. If we ever want NGBoost-over-3c
or NGBoost-over-Membury-4a, that's a NEW phase id, not a
reconfigured 3f.

Deployment posture: Membury-only, role=champion in phases.yaml's
``rainfall_amount`` target block. The bake-off (2026-05-22/23) settled
on NGBoost-LogNormal as the stage-2 winner across Membury's 3 EA
gauges; this script productionises that exact recipe.

Bundle layout per station per version::

  data/models/rainfall_amount/{station_slug}/{version}_phase3f/
    stage2_lognormal_lead24h.pkl   pickled NGBRegressor per lead
    stage2_lognormal_lead48h.pkl
    stage2_lognormal_lead72h.pkl
    stage2_lognormal_lead96h.pkl
    stage2_lognormal_lead120h.pkl
    preprocess.json                {"24": {feature_names, scaler_mean, scaler_scale,
                                           imputation: "row_mean_precip+col_median_else",
                                           best_iter}, "48": {...}, ...}
    feature_schema.json            per-lead spec (Spec page consumer)
    training_metadata.json         Phase=3f, precip_3a_version stamp, PerLead stats
    training_summary.json          RetrainGuard sidecar (per-station-aggregated)
    test_predictions.parquet       per-row (valid_time, station, lead, mu_log,
                                   sigma_log, observed_mm, observed_wet) across
                                   all 5 leads — verify_3f.py reads this for CRPS
                                   + PIT + coverage + reliability when paired with
                                   the bound 3a's pi.

Bake-off reference CRPS we expect this trainer to reproduce within ±2%
(aggregate across 3 Membury stations, mean over wet+dry test rows):
  +24h: 0.1185   +48h: 0.1296   +72h: 0.1412
Source: ``scripts/run_membury_two_stage_ngboost.py``,
``WP/reports/membury_intensity_lgbm/two_stage_ngboost_*.csv``.

Plan: ``WeatherBlend/docs/RAINFALL_AMOUNT_3F_PLAN.md``.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure UTF-8 stdout/stderr so non-ASCII (μ, σ) doesn't crash on CP1252 hosts.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from ngboost import NGBRegressor  # noqa: E402
from ngboost.distns import LogNormal as NGBLogNormal  # noqa: E402
from ngboost.scores import LogScore  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import WEATHERBLEND_DATA_ROOT, WET_THRESHOLD_MM  # noqa: E402
from src.retrain_guard import build_check_and_save_versioned  # noqa: E402
from src.manifest_promote import promote_station_version  # noqa: E402

# WB-side scripts share helpers via _shared.py — reuse the same
# build_features_via_duckdb + time_split + resolve_station that
# train_4a.py uses so 3f's training data is BIT-IDENTICAL to 3a's at
# the row level (modulo the wet-only filter applied below). Any drift
# between 3f's stage-2 features and 3a's stage-1 features would break
# the mixed-distribution math at predict time.
sys.path.insert(0, str(ROOT / "scripts"))
from _shared import (  # noqa: E402
    MODELS_LEAN,
    build_features_via_duckdb,
    resolve_station,
    time_split,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

PHASE = "3f"
TARGET = "rainfall_amount"
ACTIVE_LOCATION = "membury_devon"

# Membury rainfall stations — match phases.yaml's `locations:` filter
# + the bake-off scope.
STATIONS = ("Chards Snowdon Hill", "Goren", "Raymonds Hill")

# Leads scope — same as 3a/4a. Verify scores per-lead so we cover the
# full +120h horizon the site's intensity ribbon can render.
LEADS = (24, 48, 72, 96, 120)

# 15-feature input vector (7 NWP precip + 4 spread + 4 calendar). Order
# matches ``_shared.build_features_via_duckdb``'s output AND the
# bake-off's ``run_membury_two_stage_ngboost.py`` feature list — predict
# MUST compose in this order or scaler params are off.
PRECIP_COLS = [f"precip_{short}" for _, short in MODELS_LEAN]
SPREAD_COLS = ["precip_mean", "precip_std", "precip_max", "precip_agreement_wet_01"]
CALENDAR_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FEATURE_NAMES = PRECIP_COLS + SPREAD_COLS + CALENDAR_COLS

# NGBoost hyperparameters — mirror the bake-off's NGB_BASE. Pinned so a
# future ngboost minor bump doesn't silently shift Brier numbers.
NGB_BASE = dict(
    n_estimators=500,
    learning_rate=0.01,
    verbose=False,
    random_state=42,
)
EARLY_STOPPING_ROUNDS = 30

# Minimum row thresholds — below these we skip the cell with a warning
# rather than fit a model on too-thin data. Wet-row counts because the
# stage-2 fit is wet-only.
MIN_WET_TRAIN_ROWS = 100
MIN_WET_VAL_ROWS = 20

log = logging.getLogger("train_3f")


# ----------------------------------------------------------------------------
# Imputation + per-lead fit
# ----------------------------------------------------------------------------

def emos_impute(X: np.ndarray, n_precip: int = len(PRECIP_COLS)) -> np.ndarray:
    """Row-mean impute the NWP precip columns; column-median fill the rest.

    Bit-identical to the bake-off's ``emos_impute`` so the predict-time
    NaN handling stays consistent with the trained distribution. Without
    imputation NGBoost would refuse the fit (no native NaN support like
    LightGBM has).
    """
    X = X.copy()
    P = X[:, :n_precip]
    row_mean = np.nanmean(P, axis=1)
    row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
    idx = np.where(np.isnan(P))
    P[idx] = row_mean[idx[0]]
    X[:, :n_precip] = P
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = med[idx[1]]
    return X


def fit_one_lead(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    lead: int,
    station_slug: str,
) -> dict | None:
    """Fit NGBoost-LogNormal on the wet rows of train/val. Returns the
    fitted regressor + scaler params + train summary. Returns None when
    the cell has too few wet rows to fit responsibly (caller logs +
    skips)."""
    df_tr_w = df_tr[df_tr["wet"] == 1].copy()
    df_va_w = df_va[df_va["wet"] == 1].copy()
    if len(df_tr_w) < MIN_WET_TRAIN_ROWS or len(df_va_w) < MIN_WET_VAL_ROWS:
        log.warning(
            "  %s lead %dh: only %d wet train / %d wet val rows (need ≥%d/%d) — skipping cell.",
            station_slug, lead, len(df_tr_w), len(df_va_w),
            MIN_WET_TRAIN_ROWS, MIN_WET_VAL_ROWS,
        )
        return None

    X_tr = df_tr_w[FEATURE_NAMES].to_numpy(dtype="float64")
    X_va = df_va_w[FEATURE_NAMES].to_numpy(dtype="float64")
    y_tr = df_tr_w["precip_mm_hour"].to_numpy(dtype="float64")
    y_va = df_va_w["precip_mm_hour"].to_numpy(dtype="float64")

    # NGBoost LogNormal requires y > 0. After the wet filter all y are
    # ≥ WET_THRESHOLD_MM but floating-point + 4-of-4 rounding can leave
    # values exactly at the threshold — clip just-above-zero for safety.
    y_tr = np.maximum(y_tr, 1e-6)
    y_va = np.maximum(y_va, 1e-6)

    X_tr = emos_impute(X_tr)
    X_va = emos_impute(X_va)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)

    t0 = time.time()
    ngb = NGBRegressor(Dist=NGBLogNormal, Score=LogScore, **NGB_BASE)
    ngb.fit(
        X_tr_s, y_tr,
        X_val=X_va_s, Y_val=y_va,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    fit_secs = time.time() - t0
    best_iter = getattr(ngb, "best_val_loss_itr", None)

    log.info(
        "  %s lead %dh: NGBoost fit done in %.1fs, best_val_loss_itr=%s, "
        "n_train_wet=%d n_val_wet=%d",
        station_slug, lead, fit_secs, best_iter, len(X_tr), len(X_va),
    )

    return dict(
        regressor=ngb,
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        n_train_wet=len(X_tr),
        n_val_wet=len(X_va),
        best_iter=best_iter,
        fit_seconds=fit_secs,
    )


def score_test_slice(
    cell: dict,
    df_te: pd.DataFrame,
    lead: int,
    station_slug: str,
) -> pd.DataFrame:
    """Predict (μ_log, σ_log) per test row + return a DataFrame ready
    for the bundle's ``test_predictions.parquet``. Includes BOTH wet
    and dry rows so verify_3f.py can score the mixed distribution
    (π · LogNormal + (1−π) · δ_0) against EA truth at the cell."""
    if df_te.empty:
        return pd.DataFrame()

    X_te = df_te[FEATURE_NAMES].to_numpy(dtype="float64")
    X_te = emos_impute(X_te)
    X_te_s = (X_te - np.array(cell["scaler_mean"])) / np.array(cell["scaler_scale"])

    dist = cell["regressor"].pred_dist(
        X_te_s, max_iter=cell["best_iter"] if cell["best_iter"] else None
    )
    # NGBoost LogNormal: params['s'] = σ (log-std), params['scale'] = exp(μ).
    # Recover μ_log = log(scale), σ_log = s. Both are float64 arrays of
    # length len(X_te).
    s = np.asarray(dist.params["s"], dtype="float64")
    scale = np.asarray(dist.params["scale"], dtype="float64")
    mu_log = np.log(scale)
    sigma_log = s

    return pd.DataFrame({
        "valid_time": pd.to_datetime(df_te["ValidTimeUtc"]).dt.tz_localize(None),
        "station": station_slug,
        "lead": lead,
        "mu_log": mu_log,
        "sigma_log": sigma_log,
        "observed_mm": df_te["precip_mm_hour"].astype("float64").to_numpy(),
        "observed_wet": df_te["wet"].astype("int8").to_numpy(),
    })


# ----------------------------------------------------------------------------
# Per-station orchestration
# ----------------------------------------------------------------------------

def train_one_station(station_friendly: str, station_slug: str) -> dict:
    """Train every (station, lead) cell for one station. Returns a dict
    with per-cell fits + aggregated test predictions + train slice
    features the RetrainGuard needs."""
    per_cell: dict[int, dict] = {}
    test_pred_frames: list[pd.DataFrame] = []
    train_features_first_lead: np.ndarray | None = None
    train_label_rate_first_lead: float = 0.0
    rows_train_total = 0
    rows_val_total = 0
    rows_test_total = 0

    for lead in LEADS:
        log.info("[%s] %s lead %dh — loading rows via DuckDB",
                 time.strftime("%H:%M:%S"), station_slug, lead)
        df = build_features_via_duckdb(station_friendly, lead)
        log.info("  loaded %d rows (wet %.1f%%)",
                 len(df), 100.0 * df["wet"].mean() if len(df) else 0.0)
        if df.empty:
            log.warning("  %s lead %dh: ZERO rows from build_features_via_duckdb — skipping cell.",
                        station_slug, lead)
            continue

        df_tr, df_va, df_te = time_split(df)
        rows_train_total += len(df_tr)
        rows_val_total += len(df_va)
        rows_test_total += len(df_te)

        # RetrainGuard features: lift the FIRST trained lead's TRAIN slice
        # (wet+dry, not the wet-only filter) so the guard's row count +
        # NaN ratio + label-rate match what the upstream feature builder
        # produced. Must be a 2D numpy ndarray — training_summary's
        # compute_feature_stats calls X.shape and X.mean(axis=0).
        # train_4a passes the same shape via canonical["X_train_s"].
        if train_features_first_lead is None and len(df_tr):
            train_features_first_lead = df_tr[FEATURE_NAMES].to_numpy(dtype="float32")
            train_label_rate_first_lead = float(df_tr["wet"].mean())

        cell = fit_one_lead(df_tr, df_va, lead, station_slug)
        if cell is None:
            continue

        per_cell[lead] = cell
        tp = score_test_slice(cell, df_te, lead, station_slug)
        if not tp.empty:
            test_pred_frames.append(tp)

    test_predictions = (
        pd.concat(test_pred_frames, ignore_index=True)
        if test_pred_frames else pd.DataFrame()
    )

    return dict(
        per_cell=per_cell,
        test_predictions=test_predictions,
        # `or [...]` truth-test on an ndarray raises; check for None.
        train_features=(
            train_features_first_lead
            if train_features_first_lead is not None
            else np.zeros((0, len(FEATURE_NAMES)), dtype="float32")
        ),
        train_label_rate=train_label_rate_first_lead,
        rows_train_total=rows_train_total,
        rows_val_total=rows_val_total,
        rows_test_total=rows_test_total,
    )


# ----------------------------------------------------------------------------
# Bundle writer
# ----------------------------------------------------------------------------

def resolve_bound_3a_version(
    models_root: Path, station_slug: str,
) -> str | None:
    """Read ``data/models/precipitation/MANIFEST.json`` and return the
    newest Active entry on the station whose name contains no phase
    suffix — that's the 3a champion (3a bundles ship unsuffixed per the
    .NET BuildStationVersionDir convention; 3c/3d/3o/4a/4b/5a all
    carry _phaseXX suffixes).

    Returns None when the manifest has no 3a entry — caller logs +
    proceeds without the stamp (the bundle's metadata then carries an
    empty string, and predict_3f will warn at runtime). Better than
    crashing at train time over a manifest hiccup."""
    manifest_path = models_root / "precipitation" / "MANIFEST.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    station_entry = manifest.get("Stations", {}).get(station_slug, {})
    active = station_entry.get("Active", [])
    # 3a versions DON'T carry a `_phaseXX` suffix — filter by that.
    plain = [v for v in active if "_phase" not in v]
    if not plain:
        return None
    # Lexicographic max — version names start with v{yyyy-MM-dd_HHmmss}
    # so lexicographic order == chronological order.
    return max(plain)


def write_bundle(
    out_dir: Path,
    station_slug: str,
    station_friendly: str,
    version: str,
    result: dict,
    bound_3a_version: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pickle each lead's NGBoost regressor.
    for lead, cell in result["per_cell"].items():
        pkl_path = out_dir / f"stage2_lognormal_lead{lead}h.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(cell["regressor"], f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2) preprocess.json — per-lead block.
    preprocess = {
        "feature_names": FEATURE_NAMES,
        "imputation": "row_mean_precip+col_median_else",
        "n_precip_cols": len(PRECIP_COLS),
        "per_lead": {
            str(lead): {
                "scaler_mean":  cell["scaler_mean"],
                "scaler_scale": cell["scaler_scale"],
                "best_iter":    cell["best_iter"],
                "n_train_wet":  cell["n_train_wet"],
                "n_val_wet":    cell["n_val_wet"],
            }
            for lead, cell in result["per_cell"].items()
        },
    }
    (out_dir / "preprocess.json").write_text(
        json.dumps(preprocess, indent=2, default=str, allow_nan=False)
    )

    # 3) training_metadata.json + feature_schema.json.
    nwp_models = [m for m, _ in MODELS_LEAN]
    per_lead_stats = {
        str(lead): {
            "LeadHours":   lead,
            "TrainRows":   cell["n_train_wet"],
            "ValRows":     cell["n_val_wet"],
            "TestRows":    0,  # filled below from the test_predictions
            "BestSingle":  "ngboost_lognormal",
            "BlendTestMae": float("nan"),  # verify_3f.py computes the actual CRPS
            "BlendTestRmse": float("nan"),
            "BlendTestBias": float("nan"),
            "BestSingleValMae":  float("nan"),
            "BestSingleTestMae": float("nan"),
            "DataRangeTrain": "",
            "DataRangeVal":   "",
            "DataRangeTest":  "",
            "TestCalendarMonths": 0,
        }
        for lead, cell in result["per_cell"].items()
    }
    # Fill TestRows from the actual emitted test parquet.
    if not result["test_predictions"].empty:
        for lead in result["per_cell"]:
            n_te = int((result["test_predictions"]["lead"] == lead).sum())
            per_lead_stats[str(lead)]["TestRows"] = n_te

    metadata = {
        "Version":      version,
        "Target":       TARGET,
        "Phase":        PHASE,
        "LocationName": ACTIVE_LOCATION,
        "DataSource":   "open_meteo_previous_runs+ea_rainfall+ngboost_lognormal_per_cell",
        "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
        "Hyperparameters": {
            "library":               "ngboost==0.5.10",
            "distribution":          "LogNormal (params: s=σ_log, scale=exp(μ_log))",
            "score":                 "LogScore",
            "n_estimators":          NGB_BASE["n_estimators"],
            "learning_rate":         NGB_BASE["learning_rate"],
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "random_state":          NGB_BASE["random_state"],
            "imputation":            "row_mean_precip+col_median_else",
            "wet_only_train":        True,
            "wet_threshold_mm":      WET_THRESHOLD_MM,
            "stage1_phase":          "3a",
            "precip_3a_version":     bound_3a_version or "",
        },
        "DeviationsFromBrief": [
            "Membury-only champion in phases.yaml's rainfall_amount target. "
            "Bonehill rollout deferred — original 2026-05-22/23 bake-off "
            "was Membury-only.",
            "Two-stage distributional: NGBoost-LogNormal stage-2 over the "
            "bound 3a champion's hourly P(wet). Mixed predictive distribution "
            "(1-π)·δ_0 + π·LogNormal mixed at predict time.",
            "5 leads: {24, 48, 72, 96, 120}. Bake-off scope was {24, 48, 72} "
            "only; 96 + 120 extended here per the production deployment plan.",
            "PerLeadStats Brier/Mae fields left as NaN — 3f's primary skill "
            "metric is CRPS (computed by verify_3f.py against the mixed "
            "distribution + truth). Site renders verify CRPS rolling, NOT "
            "trained-time numbers.",
        ],
        "PerLead": per_lead_stats,
    }
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str, allow_nan=False)
    )

    schema_per_lead = {
        str(L): {
            "Target":         TARGET,
            "FeatureSet":     "phase3f-ngboost-lognormal",
            "LeadHours":      L,
            "RequiredModels": [],
            "OptionalModels": nwp_models,
            "Models":         nwp_models,
            "FeatureNames":   FEATURE_NAMES,
            "DataSource":     "open_meteo_previous_runs+ea_rainfall",
            "Tier":           "3f-lean",
            "UkvStrategy":    None,
        }
        for L in LEADS if L in result["per_cell"]
    }
    (out_dir / "feature_schema.json").write_text(
        json.dumps({"Leads": schema_per_lead}, indent=2, allow_nan=False)
    )

    # 4) test_predictions.parquet — verify_3f.py reads this.
    if not result["test_predictions"].empty:
        result["test_predictions"].to_parquet(
            out_dir / "test_predictions.parquet", index=False
        )

    sizes = {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file()}
    log.info("  bundle → %s", out_dir)
    for name, size in sorted(sizes.items()):
        log.info("    %-32s %s bytes", name, f"{size:,}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--anchor", default=None,
                        help="Anchor date YYYY-MM-DD UTC for the version string. Default: today.")
    parser.add_argument("--stations", nargs="*", default=None,
                        help="Station subset (default: all 3 Membury active).")
    parser.add_argument("--models-root", default=str(WEATHERBLEND_DATA_ROOT / "models"),
                        help="Models tree root for bundle output.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.anchor:
        anchor = datetime.fromisoformat(args.anchor).replace(tzinfo=timezone.utc)
    else:
        anchor = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    anchor = anchor.replace(tzinfo=None)

    stations = args.stations or STATIONS
    models_root = Path(args.models_root)
    version = datetime.now(timezone.utc).strftime("v%Y-%m-%d_%H%M%S_phase3f")

    log.info("[%s] Phase 3f per-cell TRAIN", time.strftime("%H:%M:%S"))
    log.info("  anchor:   %s", anchor.isoformat())
    log.info("  version:  %s", version)
    log.info("  stations: %s", stations)
    log.info("  leads:    %s", LEADS)
    log.info("  fits:     %d stations × %d leads = %d NGBoost fits total",
             len(stations), len(LEADS), len(stations) * len(LEADS))

    bundles_written = 0
    guard_failures = 0
    for station_input in stations:
        station_slug, station_friendly = resolve_station(station_input)
        log.info("[%s] %s — %d per-cell fits", time.strftime("%H:%M:%S"), station_friendly, len(LEADS))

        result = train_one_station(station_friendly, station_slug)
        if not result["per_cell"]:
            log.warning("  no cells trained for %s — skipping bundle write.", station_slug)
            continue

        bound_3a_version = resolve_bound_3a_version(models_root, station_slug)
        if bound_3a_version is None:
            log.warning(
                "  %s: no 3a champion found in manifest — predict_3f.py will warn at runtime. "
                "Stamping empty precip_3a_version in bundle metadata.",
                station_slug,
            )

        bundle_dir = models_root / TARGET / station_slug / version
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # RetrainGuard — same shape as 4a, against the previous 3f bundle.
        # Rows ±30% / NaN% absolute 0.20 / label-rate 0.10 defaults match
        # the existing dbarts/INLA guard config.
        guard_result = build_check_and_save_versioned(
            log,
            version_dir=bundle_dir,
            composite=f"{TARGET}/{station_slug}",
            phase=PHASE,
            version=version,
            rows_train=result["rows_train_total"],
            rows_val=result["rows_val_total"],
            rows_test=result["rows_test_total"],
            train_features=result["train_features"],
            feature_names=FEATURE_NAMES,
            label_rates={station_slug: result["train_label_rate"]},
            location_name=ACTIVE_LOCATION,
        )
        if not guard_result.passed:
            log.error("  guard FAIL — skipping bundle for %s.", station_slug)
            guard_failures += 1
            continue

        write_bundle(bundle_dir, station_slug, station_friendly, version, result, bound_3a_version)
        bundles_written += 1

        # Promote into MANIFEST.json's Active so predict_3f reads this
        # version on the next cycle. Phase 3f ships as CHAMPION per
        # phases.yaml — first-class, sole entry in the rainfall_amount
        # target — so Current also gets stamped.
        new_active = promote_station_version(
            models_root=models_root,
            target=TARGET,
            station_slug=station_slug,
            version=version,
            phase_tag="phase3f",
            role="champion",
        )
        log.info("  promoted into manifest → station Active = %s", new_active["Active"])

    log.info("")
    log.info("Phase 3f train complete. Bundles written: %d (guard failures: %d)",
             bundles_written, guard_failures)
    if bundles_written == 0:
        sys.exit(1)
    if guard_failures > 0:
        sys.exit(4)


if __name__ == "__main__":
    main()
