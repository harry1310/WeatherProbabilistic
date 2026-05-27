"""Phase 3f — PREDICT ONLY. Scores the NGBoost-LogNormal stage-2
distribution per (station, lead), joins π from the bound Phase 3a
champion's live predictions parquet, and emits one mixed-distribution
row per (ValidTime, Lead) under
``data/predictions/rainfall_amount/{station}/model_version={v}/date=YYYY-MM-DD/``.

Two-stage mixed predictive distribution:

  F(x) = (1 − π) · δ_0(x) + π · LogNormal(μ_log, σ_log)(x)

  π          = ProbWet from Phase 3a champion @ (ValidTime, Lead)
  (μ, σ)     = NGBoost-LogNormal stage-2 prediction (this script)

Pi binding is mandatory — Phase 3a is HARDCODED as the stage-1 source
for 3f (see ``train_3f.py`` docstring + Phase 3 plan §2.5). If 3a's
parquet for the anchor is missing the script skips that station
rather than emit a one-sided wet-only forecast.

Bundle layout (produced by ``train_3f.py``):
  data/models/rainfall_amount/{station}/{version}_phase3f/
    stage2_lognormal_lead{24,48,72,96,120}h.pkl   one pickled NGBRegressor per lead
    preprocess.json                                per-lead scaler params + best_iter
    training_metadata.json                         carries Hyperparameters.precip_3a_version

CLI::

    predict_3f.py [--anchor YYYY-MM-DD] [--stations slug1 ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import duckdb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import (  # noqa: E402
    WEATHERBLEND_DATA_ROOT,
    WET_THRESHOLD_MM,
    stations_for_location,
)

from _shared import (  # noqa: E402
    MODELS_LEAN,
    lead_day_bucket,
    resolve_station,
)

PHASE = "3f"
TARGET = "rainfall_amount"

# Membury-only by default. WB_LOCATION can swap if 3f ever ships
# for Bonehill — phases.yaml's locations filter is the gate.
ACTIVE_LOCATION = os.environ.get("WB_LOCATION", "membury_devon")

STATIONS = list(stations_for_location(ACTIVE_LOCATION))
LEADS = [24, 48, 72, 96, 120]
HORIZON_DAYS = 7

# Same 15-feature vector train_3f.py composes. Order MUST match —
# the scaler params in preprocess.json are positional.
PRECIP_COLS = [f"precip_{short}" for _, short in MODELS_LEAN]
SPREAD_COLS = ["precip_mean", "precip_std", "precip_max", "precip_agreement_wet_01"]
CALENDAR_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FEATURE_NAMES = PRECIP_COLS + SPREAD_COLS + CALENDAR_COLS

# Exceedance thresholds + quantile fan — match RainfallAmountPredictionRow.cs.
EXCEED_THRESHOLDS_MM = [0.1, 1.0, 5.0, 10.0]
QUANTILES = [0.025, 0.10, 0.50, 0.90, 0.975]

log = logging.getLogger("predict_3f")


# ----------------------------------------------------------------------------
# Bundle discovery
# ----------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^v\d{4}-\d{2}-\d{2}_\d{6}_phase3f$")


def find_latest_bundle(models_root: Path, station_slug: str) -> Path:
    """Lexicographic newest v..._phase3f under
    ``models_root/rainfall_amount/{station_slug}/`` that has the
    per-lead pickle layout + preprocess.json. Raises if none found."""
    parent = models_root / TARGET / station_slug
    if not parent.is_dir():
        raise FileNotFoundError(
            f"no model dir for station {station_slug!r} under {parent}. "
            f"Run train_3f.py first."
        )
    name_matches = sorted(
        (d for d in parent.iterdir() if d.is_dir() and _VERSION_RE.match(d.name)),
        key=lambda d: d.name,
        reverse=True,
    )
    required_artefacts = ["preprocess.json", "training_metadata.json"] + [
        f"stage2_lognormal_lead{L}h.pkl" for L in LEADS
    ]
    for candidate in name_matches:
        # Allow a SUBSET of leads — a fit might skip a lead with
        # too-few wet rows; require at least preprocess + metadata +
        # ≥1 lead pickle to call the bundle usable.
        if not (candidate / "preprocess.json").is_file():
            continue
        if not (candidate / "training_metadata.json").is_file():
            continue
        if not any((candidate / f"stage2_lognormal_lead{L}h.pkl").is_file() for L in LEADS):
            continue
        return candidate
    raise FileNotFoundError(
        f"no usable *_phase3f bundle under {parent}. "
        f"Required: preprocess.json + training_metadata.json + ≥1 stage2_lognormal_lead*.pkl. "
        f"Run train_3f.py to mint one."
    )


# ----------------------------------------------------------------------------
# Live features — same DuckDB SQL shape as predict_4a.py
# ----------------------------------------------------------------------------

def build_live_features(anchor: datetime) -> pd.DataFrame:
    """One row per ValidTimeUtc in the horizon, freshest run per
    (valid_time, model), 7 NWP precip pivot + 4 spread + 4 calendar
    + day-bucketed lead. Mirrors ``predict_4a.build_live_features``."""
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
    precip_pivot = ",\n        ".join(
        f"MAX(CASE WHEN Model = '{full}' THEN Precipitation END) AS precip_{short}"
        for full, short in MODELS_LEAN
    )
    horizon_end = anchor + pd.Timedelta(days=HORIZON_DAYS)
    sql = f"""
    WITH latest AS (
        SELECT ValidTimeUtc, Model, Precipitation,
               ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model
                                  ORDER BY RunTimeUtc DESC) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{ACTIVE_LOCATION}'
          AND RunTimeSource = 'reported'
          AND Model IN {model_in}
          AND ValidTimeUtc > timestamp '{anchor.isoformat()}'
          AND ValidTimeUtc <= timestamp '{horizon_end.isoformat()}'
    )
    SELECT ValidTimeUtc,
        {precip_pivot}
    FROM latest WHERE rn = 1
    GROUP BY ValidTimeUtc
    ORDER BY ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()
    if len(df) == 0:
        return df

    precip_cols = [f"precip_{short}" for _, short in MODELS_LEAN]
    pm_arr = df[precip_cols].to_numpy(dtype="float64")
    present = (~np.isnan(pm_arr)).sum(axis=1)
    sumv = np.nansum(pm_arr, axis=1)
    sumsq = np.nansum(pm_arr ** 2, axis=1)
    mean_safe = np.where(present > 0, sumv / np.maximum(present, 1), np.nan)
    var = np.maximum(0.0, sumsq / np.maximum(present, 1) - mean_safe ** 2)
    df["precip_mean"] = np.nanmean(pm_arr, axis=1)
    df["precip_std"] = np.where(present > 1, np.sqrt(var), 0.0)
    df["precip_max"] = np.nanmax(pm_arr, axis=1)
    wet_count = (pm_arr >= WET_THRESHOLD_MM).sum(axis=1)
    df["precip_agreement_wet_01"] = np.where(present > 0, wet_count / np.maximum(present, 1), np.nan)

    hour_angle = 2.0 * np.pi * df["ValidTimeUtc"].dt.hour / 24.0
    doy_angle = 2.0 * np.pi * (df["ValidTimeUtc"].dt.dayofyear - 1) / 365.0
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["doy_sin"] = np.sin(doy_angle)
    df["doy_cos"] = np.cos(doy_angle)

    df["lead"] = df["ValidTimeUtc"].apply(lambda v: lead_day_bucket(v, anchor))
    df = df[df["lead"].isin(LEADS)].reset_index(drop=True)
    df["lead"] = df["lead"].astype(int)
    return df


# ----------------------------------------------------------------------------
# Bound 3a join
# ----------------------------------------------------------------------------

def _resolve_3a_parquet_with_fallback(
    predictions_root: Path,
    station_slug: str,
    precip_3a_version: str,
    anchor: datetime,
) -> tuple[Path, str]:
    """Return (parquet path, resolved version) for today's bound 3a π.

    Preferred path: the version stamped into the 3f bundle's metadata.
    Fallback: if the stamped version's parquet is missing at today's
    anchor, scan the station's predictions tree for any unsuffixed
    (= 3a — 3c/3d/3o/4a/4b all carry _phaseXX) sibling that DOES have
    today's parquet and pick the freshest by lexicographic max.

    Motivated by the 2026-05-27 06:35Z predict-3f failure: 3f trained
    at 2026-05-26 10:24 stamped 3a v2026-05-24_141845, but Sunday's
    auto-retrain re-trained 3a at 10:27 → newer 3a writes predictions
    to v2026-05-26_102758/date=today, leaving the stamped version's
    folder without today's parquet. Stale-stamp shouldn't black-hole
    every 3f cycle until the next 3f train rebinds.

    Raises FileNotFoundError when neither the stamp nor any unsuffixed
    sibling has today's parquet — predict_3f without a π source can't
    proceed.
    """
    date_str = anchor.date().isoformat()
    station_dir = predictions_root / "precipitation" / station_slug
    preferred = (
        station_dir / f"model_version={precip_3a_version}"
                    / f"date={date_str}" / "predictions.parquet"
    )
    if preferred.is_file():
        return preferred, precip_3a_version

    # Fallback: any unsuffixed model_version=* sibling whose today/
    # parquet exists. Order by version-name lex max so the freshest
    # 3a champion wins. _phaseXX-suffixed dirs are 3c/3d/3o/4a/4b —
    # not 3a, never used here.
    candidates: list[tuple[str, Path]] = []
    if station_dir.is_dir():
        for d in station_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("model_version="):
                continue
            version = d.name[len("model_version="):]
            if "_phase" in version:
                continue
            today_path = d / f"date={date_str}" / "predictions.parquet"
            if today_path.is_file():
                candidates.append((version, today_path))
    if not candidates:
        raise FileNotFoundError(
            f"bound Phase 3a predictions parquet missing: {preferred}. "
            f"3f cannot mix without π — neither the stamped version "
            f"({precip_3a_version}) nor any unsuffixed sibling has today's "
            f"parquet. Either today's predict-all step hasn't run yet for "
            f"3a, or the predictions tree is empty for this station."
        )
    candidates.sort(key=lambda c: c[0])
    chosen_version, chosen_path = candidates[-1]
    log.warning(
        "  bound 3a stamp %s has no parquet at %s; using on-disk fallback %s",
        precip_3a_version, date_str, chosen_version,
    )
    return chosen_path, chosen_version


def load_bound_3a_pi(
    predictions_root: Path,
    station_slug: str,
    precip_3a_version: str,
    anchor: datetime,
) -> tuple[pd.DataFrame, str]:
    """Read TODAY's 3a champion predictions parquet for this station
    and return ((ValidTimeUtc, LeadHours, ProbWet) DataFrame, resolved
    3a version) — the π values we mix into the 3f predictive
    distribution. The resolved version may differ from the stamped one
    when the bundle's stamp went stale between 3f train and 3f predict
    (see :func:`_resolve_3a_parquet_with_fallback`). Raises if neither
    the stamped version nor any unsuffixed sibling has today's parquet;
    predict_3f without a stage-1 source is meaningless.

    DEDUPES on (ValidTimeUtc, LeadHours) keeping the freshest
    PredictionMadeAtUtc — today's 3a parquet typically holds 4-5
    intra-day cycles (predict.yml fires every 6h) and an un-deduped
    merge against df_live (24 rows per lead) would broadcast to
    (24 * cycles) rows per lead, blowing up downstream numpy ops
    with a (120,) vs (24,) shape mismatch (regression 2026-05-26).
    Same ROW_NUMBER pattern Phase4bPredictCommand.cs uses on its
    3o parquet read.
    """
    parquet, resolved_version = _resolve_3a_parquet_with_fallback(
        predictions_root, station_slug, precip_3a_version, anchor)
    con = duckdb.connect(":memory:")
    df = con.execute(f"""
        WITH ranked AS (
          SELECT ValidTimeUtc, LeadHours, ProbWet, PredictionMadeAtUtc,
                 ROW_NUMBER() OVER (
                   PARTITION BY ValidTimeUtc, LeadHours
                   ORDER BY PredictionMadeAtUtc DESC
                 ) AS rn
          FROM read_parquet('{parquet.as_posix()}')
        )
        SELECT ValidTimeUtc, LeadHours, ProbWet
        FROM ranked WHERE rn = 1
    """).fetch_df()
    con.close()
    df["LeadHours"] = df["LeadHours"].astype(int)
    return df, resolved_version


# ----------------------------------------------------------------------------
# Mixed-distribution math
# ----------------------------------------------------------------------------

def mixed_mean(pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """E[X] under F(x) = (1-π) · δ_0 + π · LogNormal(μ,σ).
    = π · exp(μ + σ²/2)."""
    return pi * np.exp(mu + 0.5 * sigma ** 2)


def mixed_quantile(
    alphas: list[float],
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> dict[float, np.ndarray]:
    """Per-α inverse mixed CDF.

    F(x) = (1-π) + π · LogNormalCDF(x) for x > 0; (1-π) at x = 0.
    For α ≤ (1-π): quantile = 0.
    For α >  (1-π): solve π · LogNormalCDF(x) = α - (1-π)
                     ⇒ LogNormalCDF(x) = (α - (1-π)) / π
                     ⇒ x = exp(μ + σ · Φ⁻¹((α - (1-π)) / π))
    """
    out: dict[float, np.ndarray] = {}
    one_minus_pi = 1.0 - pi
    for alpha in alphas:
        q = np.zeros_like(pi)
        # Avoid div-by-zero when π is exactly 0 — those rows stay 0.
        mask = (alpha > one_minus_pi) & (pi > 0.0)
        if mask.any():
            inner = (alpha - one_minus_pi[mask]) / pi[mask]
            # Numerical safety: clip into (0, 1) for ppf.
            inner = np.clip(inner, 1e-12, 1.0 - 1e-12)
            q[mask] = np.exp(mu[mask] + sigma[mask] * norm.ppf(inner))
        out[alpha] = q
    return out


def mixed_exceedance(
    thresholds: list[float],
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> dict[float, np.ndarray]:
    """P(X ≥ t) = π · (1 - LogNormalCDF(t)). LogNormalCDF(t) =
    Φ((log t − μ) / σ). Threshold t=0 not supported (we use ≥ 0.1 etc)."""
    out: dict[float, np.ndarray] = {}
    for t in thresholds:
        if t <= 0:
            raise ValueError(f"Exceedance threshold must be > 0; got {t}")
        # σ > 0 by construction (LogNormal scale); safe to divide.
        z = (math.log(t) - mu) / sigma
        ln_cdf = norm.cdf(z)
        out[t] = pi * (1.0 - ln_cdf)
    return out


# ----------------------------------------------------------------------------
# Per-lead score
# ----------------------------------------------------------------------------

def score_one_lead(
    bundle_dir: Path,
    lead: int,
    df_lead: pd.DataFrame,
    preprocess_lead: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply emos_impute + scaler to df_lead's 15-col feature matrix,
    return (μ_log, σ_log) for each row. Skips the cell (returns
    None) if the lead's pickle isn't in this bundle."""
    pkl_path = bundle_dir / f"stage2_lognormal_lead{lead}h.pkl"
    if not pkl_path.is_file():
        log.warning("    lead %dh: no pickle %s — skipping", lead, pkl_path.name)
        return None, None

    with open(pkl_path, "rb") as f:
        ngb = pickle.load(f)

    X = df_lead[FEATURE_NAMES].to_numpy(dtype="float64")

    # emos_impute: row-mean for precip cols, column-median for rest.
    # Bit-identical to train_3f.emos_impute so the predict-time NaN
    # handling matches what the scaler was fit on.
    n_precip = len(PRECIP_COLS)
    P = X[:, :n_precip]
    row_mean = np.nanmean(P, axis=1)
    row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
    idx = np.where(np.isnan(P))
    P[idx] = row_mean[idx[0]]
    X[:, :n_precip] = P
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = med[idx[1]]

    mean = np.array(preprocess_lead["scaler_mean"], dtype="float64")
    scale = np.array(preprocess_lead["scaler_scale"], dtype="float64")
    X_s = (X - mean) / scale

    best_iter = preprocess_lead.get("best_iter")
    dist = ngb.pred_dist(X_s, max_iter=best_iter if best_iter else None)
    s = np.asarray(dist.params["s"], dtype="float64")
    scale_param = np.asarray(dist.params["scale"], dtype="float64")
    mu_log = np.log(scale_param)
    sigma_log = s
    return mu_log, sigma_log


# ----------------------------------------------------------------------------
# Per-station orchestration
# ----------------------------------------------------------------------------

def predict_one_station(
    bundle_dir: Path,
    station_slug: str,
    predictions_root: Path,
    anchor: datetime,
) -> pd.DataFrame:
    meta_path = bundle_dir / "training_metadata.json"
    metadata = json.loads(meta_path.read_text())

    bundle_loc = (metadata.get("LocationName") or "").strip()
    if not bundle_loc:
        raise ValueError(
            f"bundle {bundle_dir.name} has no LocationName — refusing to score."
        )
    if bundle_loc.lower() != ACTIVE_LOCATION.lower():
        raise ValueError(
            f"bundle {bundle_dir.name} trained on '{bundle_loc}' but "
            f"predict using NWP from '{ACTIVE_LOCATION}' — refusing to score."
        )

    hyper = metadata.get("Hyperparameters", {}) or {}
    precip_3a_version = (hyper.get("precip_3a_version") or "").strip()
    if not precip_3a_version:
        raise ValueError(
            f"bundle {bundle_dir.name} has no precip_3a_version stamp — "
            f"cannot resolve bound 3a parquet. Retrain to repair (train_3f.py "
            f"stamps the resolved 3a champion at promote time)."
        )

    preprocess = json.loads((bundle_dir / "preprocess.json").read_text())
    per_lead_preprocess = preprocess.get("per_lead") or {}

    log.info("  bundle: %s", bundle_dir.name)
    log.info("  bound 3a (stamp): %s", precip_3a_version)

    log.info("  loading π from bound Phase 3a parquet...")
    df_3a, resolved_3a_version = load_bound_3a_pi(
        predictions_root, station_slug, precip_3a_version, anchor)
    if resolved_3a_version != precip_3a_version:
        log.info("  bound 3a (resolved): %s (stamp was stale)", resolved_3a_version)
    log.info("  π rows: %d (leads %s)",
             len(df_3a), sorted(df_3a["LeadHours"].unique().tolist()))

    log.info("  building live features (horizon %dd)...", HORIZON_DAYS)
    df_live = build_live_features(anchor)
    if len(df_live) == 0:
        log.warning("  no live feature rows past anchor — emitting nothing")
        return pd.DataFrame()
    log.info("  live: %d rows across leads %s",
             len(df_live), sorted(df_live["lead"].unique().tolist()))

    out_rows: list[pd.DataFrame] = []
    for lead in LEADS:
        if str(lead) not in per_lead_preprocess:
            log.info("    lead %dh: not in bundle's preprocess — skipping", lead)
            continue
        df_lead = df_live[df_live["lead"] == lead].reset_index(drop=True)
        if df_lead.empty:
            log.info("    lead %dh: no live rows for this lead — skipping", lead)
            continue

        mu_log, sigma_log = score_one_lead(bundle_dir, lead, df_lead, per_lead_preprocess[str(lead)])
        if mu_log is None:
            continue

        # Join π by (ValidTime, Lead). 3a's parquet uses ValidTimeUtc + LeadHours;
        # df_lead's lead is the day-bucketed integer matching 3a's writer.
        df_lead_join = df_lead[["ValidTimeUtc", "lead"]].rename(columns={"lead": "LeadHours"})
        df_lead_join = df_lead_join.merge(df_3a, on=["ValidTimeUtc", "LeadHours"], how="left")
        # Belt-and-braces defence against the 2026-05-26 broadcast bug:
        # if load_bound_3a_pi ever ships un-deduped (V, L) keys again
        # the left-merge silently multiplies df_lead's row count and
        # the next numpy op blows up with a confusing shape mismatch.
        # Fail loud + early with a useful message instead.
        if len(df_lead_join) != len(df_lead):
            raise ValueError(
                f"lead {lead}h: 3a merge multiplied df_lead's {len(df_lead)} rows "
                f"to {len(df_lead_join)} — duplicate (ValidTimeUtc, LeadHours) keys "
                f"in 3a parquet. load_bound_3a_pi must dedupe."
            )
        pi = df_lead_join["ProbWet"].to_numpy(dtype="float64")
        # Missing π rows: warn + drop (can't compute mixed dist without it).
        missing_pi = np.isnan(pi)
        if missing_pi.any():
            log.warning("    lead %dh: %d rows missing π from 3a join — dropped",
                        lead, int(missing_pi.sum()))
            keep = ~missing_pi
            if not keep.any():
                continue
            mu_log = mu_log[keep]
            sigma_log = sigma_log[keep]
            pi = pi[keep]
            df_lead = df_lead.iloc[keep].reset_index(drop=True)

        mean = mixed_mean(pi, mu_log, sigma_log)
        median = pi * np.exp(mu_log)  # plan-spec "robust headline median"
        qs = mixed_quantile(QUANTILES, pi, mu_log, sigma_log)
        exs = mixed_exceedance(EXCEED_THRESHOLDS_MM, pi, mu_log, sigma_log)

        out_rows.append(pd.DataFrame({
            "ValidTimeUtc":  df_lead["ValidTimeUtc"].values,
            "LeadHours":     lead,
            "Pi":            pi,
            "MuLog":         mu_log,
            "SigmaLog":      sigma_log,
            "MeanMmPerHr":   mean,
            "MedianMmPerHr": median,
            "P2_5MmPerHr":   qs[0.025],
            "P10MmPerHr":    qs[0.10],
            "P50MmPerHr":    qs[0.50],
            "P90MmPerHr":    qs[0.90],
            "P97_5MmPerHr":  qs[0.975],
            "PExceed0_1":    exs[0.1],
            "PExceed1":      exs[1.0],
            "PExceed5":      exs[5.0],
            "PExceed10":     exs[10.0],
        }))
        log.info("    lead %dh: scored %d rows", lead, len(out_rows[-1]))

    if not out_rows:
        return pd.DataFrame()
    out = pd.concat(out_rows, ignore_index=True).sort_values(
        ["ValidTimeUtc", "LeadHours"], kind="mergesort"
    ).reset_index(drop=True)
    # Stamp the version we ACTUALLY read π from, not the bundle's
    # (possibly stale) stamp — downstream readers expect this to point
    # at the on-disk parquet they can re-load.
    out["Precip3aVersion"] = resolved_3a_version
    return out


# ----------------------------------------------------------------------------
# Parquet writer
# ----------------------------------------------------------------------------

def write_predictions_parquet(
    predictions_root: Path,
    station_slug: str,
    bundle_name: str,
    anchor: datetime,
    predictions: pd.DataFrame,
) -> Path:
    """Write under
    ``data/predictions/rainfall_amount/{station}/model_version={v}/date={d}/predictions.parquet``."""
    date_str = anchor.date().isoformat()
    out_dir = (
        predictions_root
        / TARGET
        / station_slug
        / f"model_version={bundle_name}"
        / f"date={date_str}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.parquet"
    predictions = predictions.copy()
    predictions["LocationName"] = ACTIVE_LOCATION
    predictions["TruthStation"] = station_slug
    predictions["ModelVersion"] = bundle_name
    predictions["PredictionMadeAtUtc"] = datetime.now(timezone.utc)

    column_order = [
        "LocationName", "TruthStation", "ModelVersion",
        "PredictionMadeAtUtc", "ValidTimeUtc", "LeadHours",
        "Pi", "MuLog", "SigmaLog",
        "MeanMmPerHr", "MedianMmPerHr",
        "P2_5MmPerHr", "P10MmPerHr", "P50MmPerHr", "P90MmPerHr", "P97_5MmPerHr",
        "PExceed0_1", "PExceed1", "PExceed5", "PExceed10",
        "Precip3aVersion",
    ]
    predictions = predictions[column_order]
    predictions.to_parquet(out_path, index=False)
    return out_path


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", default=None,
                   help="Anchor date YYYY-MM-DD UTC (default: today).")
    p.add_argument("--stations", nargs="*", default=None,
                   help="Station subset (default: all Membury active).")
    p.add_argument("--predictions-root", default=str(WEATHERBLEND_DATA_ROOT / "predictions"),
                   help="Predictions tree root.")
    p.add_argument("--models-root", default=str(WEATHERBLEND_DATA_ROOT / "models"),
                   help="Models tree root holding the saved bundles.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.anchor:
        anchor = datetime.fromisoformat(args.anchor).replace(tzinfo=timezone.utc)
    else:
        anchor = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    anchor = anchor.replace(tzinfo=None)

    stations = args.stations or STATIONS
    predictions_root = Path(args.predictions_root)
    models_root = Path(args.models_root)

    log.info("[%s] Phase 3f PREDICT", time.strftime("%H:%M:%S"))
    log.info("  anchor:   %s", anchor.isoformat())
    log.info("  location: %s", ACTIVE_LOCATION)
    log.info("  stations: %s", stations)
    log.info("  leads:    %s", LEADS)
    log.info("  horizon:  %d days", HORIZON_DAYS)

    rows_written = 0
    stations_ok = 0
    for station_input in stations:
        station_slug, station_friendly = resolve_station(station_input)
        log.info("[%s] %s", time.strftime("%H:%M:%S"), station_friendly)
        try:
            bundle_dir = find_latest_bundle(models_root, station_slug)
        except FileNotFoundError as e:
            log.warning("  %s", e)
            continue

        try:
            preds = predict_one_station(bundle_dir, station_slug, predictions_root, anchor)
        except (FileNotFoundError, ValueError) as e:
            log.error("  %s — skipping station", e)
            continue

        if preds.empty:
            log.warning("  no predictions emitted for %s", station_slug)
            continue
        out_path = write_predictions_parquet(
            predictions_root, station_slug, bundle_dir.name, anchor, preds
        )
        log.info("  wrote %d rows → %s", len(preds), out_path)
        rows_written += len(preds)
        stations_ok += 1

    log.info("")
    log.info("Phase 3f predict complete. Stations OK: %d/%d. Rows written: %d",
             stations_ok, len(stations), rows_written)
    if rows_written == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
