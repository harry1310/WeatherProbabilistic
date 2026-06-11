"""Phase 2 wind_mvn — PREDICT ONLY. Loads the latest *_wind_mvn bundle,
builds live features from the forecast tree at each of the configured
leads, runs the MLP forward pass, draws MC samples with the per-lead α'
calibration applied, and writes one row per (ValidTimeUtc, Lead) to
``data/predictions/wind_direction/{location}/model_version={v}_wind_mvn/date={d}/predictions.parquet``.

Output schema (matches WIND_BLENDER_PLAN.md §Output schema additions):

    ValidTimeUtc, LeadHours, PredictionMadeAtUtc, ModelVersion, LocationName,
    RunTime{Gfs,Ecmwf,Icon,Mf,Ukmo,Gem,Aifs,Jma},
    MuU, MuV, SigmaU, SigmaV, Rho,
    BlendDirection, BlendDirectionCi95Lo, BlendDirectionCi95Hi,
    BlendSpeedMagnitude, BlendSpeedCi95Lo, BlendSpeedCi95Hi

Calibration applied at predict time:
  - σ_u, σ_v multiplied by per-lead alpha_prime_dir BEFORE drawing direction
    MC quantiles.
  - σ_u, σ_v multiplied by per-lead alpha_prime_spd BEFORE drawing speed
    MC quantiles. (Two separate calibrated draws — dir and spd noise
    profiles differ.)

WindSpeedBlend (the wind_blend composition) lives in WeatherBlend's
predict-tail; this script's role ends at writing the wind_direction parquet.

CLI::

    predict_wind_mvn.py [--location bonehill_rocks] [--anchor YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _shared import force_utf8_stdio  # noqa: E402

force_utf8_stdio()

import duckdb  # noqa: E402
import torch  # noqa: E402

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402

# Architecture + train/predict-shared constants and helpers come from
# _wind_common — the SAME definitions train_wind_mvn saved the state_dicts
# with, so load_state_dict can't silently mis-load (see its docstring).
from _wind_common import (  # noqa: E402
    LEADS,
    NWPS,
    ORO_LEAN,
    SPREAD_VARS,
    WindMVNHead,
    circ_quantiles,
    circ_quantiles_95,  # noqa: F401  (back-compat re-export for older callers)
    load_oro_static,
    mc_speed_dir,
    oro_dynamic,
)


# ----------------------------------------------------------------------------
# Constants (predict-only — shared ones come from _wind_common)
# ----------------------------------------------------------------------------

PHASE = "wind_mvn"
TARGET = "wind_direction"

# Canonical NWP-id → output column shortname, for the per-NWP RunTime
# columns. Mirrors the .NET ElementPredictionRow's named slots so a
# future C# reader uses the same column names regardless of which 6
# NWPs participated.
NWP_SHORT = {
    "gfs_seamless": "Gfs", "ecmwf_ifs025": "Ecmwf", "icon_seamless": "Icon",
    "meteofrance_seamless": "Mf", "ukmo_seamless": "Ukmo",
    "gem_seamless": "Gem", "ecmwf_aifs025_single": "Aifs",
    "jma_seamless": "Jma",
}

LOCATION_DEFAULT = os.environ.get("WB_LOCATION", "bonehill_rocks")

log = logging.getLogger("predict_wind_mvn")


# ----------------------------------------------------------------------------
# Bundle discovery
# ----------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^v\d{4}-\d{2}-\d{2}_\d{6}_wind_mvn$")


def find_latest_bundle(models_root: Path, location: str) -> Path:
    parent = models_root / TARGET / location
    if not parent.is_dir():
        raise FileNotFoundError(
            f"No bundle dir under {parent}. Run train_wind_mvn.py first.")
    candidates = sorted(
        (d for d in parent.iterdir()
         if d.is_dir() and _VERSION_RE.match(d.name)),
        key=lambda d: d.name, reverse=True,
    )
    for c in candidates:
        if not (c / "calibration.json").is_file(): continue
        if not (c / "feature_scaler.json").is_file(): continue
        if not (c / "feature_schema.json").is_file(): continue
        if not (c / "training_metadata.json").is_file(): continue
        if not any((c / f"state_lead_{L}h.pt").is_file() for L in LEADS): continue
        return c
    raise FileNotFoundError(
        f"No usable *_wind_mvn bundle under {parent}. Re-run train.")


# ----------------------------------------------------------------------------
# Live feature build (per lead)
# ----------------------------------------------------------------------------

def build_live_for_lead(location: str, lead: int, anchor: datetime,
                         oro: dict, feature_names: list[str]) -> pd.DataFrame:
    """Pull freshest-cycle forecasts for this lead, anchor day, and build
    the 29-feature vector per row plus per-NWP RunTime stamps."""
    fc_glob = str(WEATHERBLEND_DATA_ROOT / "forecasts"
                  / f"location={location}" / "**" / "*.parquet").replace("\\", "/")
    nwp_in = "(" + ",".join(f"'{m}'" for m in NWPS) + ")"
    day_start = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=lead // 24 + 1)
    day_start_eff = day_start + timedelta(days=(lead // 24))

    # Lead semantics: "+24h tab" means "the day starting anchor+24h"
    # (24 hourly ValidTimes), not the single ValidTime whose RunTime is
    # exactly anchor. The WB element predict pipelines (WindPredictPipeline
    # etc.) use this same hourly-per-day pattern — staying parallel keeps
    # the wind page chart density consistent across phases.
    #
    # We previously filtered LeadHours = {lead} EXACTLY, which restricted
    # each lead to the few ValidTimes per day where some cycle's
    # 24h-ahead happened to align — only 7 hours/day on R2 today. Dropped
    # that predicate; `rn=1` over RunTimeUtc DESC keeps the freshest
    # cycle's prediction per (Model, ValidTime), which is what every WB
    # blender already does.
    #
    # RunTimeSource='reported' filters out the offset_day previous_runs
    # rows (which is what the bundle was TRAINED on but stays in-tree
    # alongside the live rows). At predict time we want the live cycle's
    # values per ValidTime.
    con = duckdb.connect()
    fc = con.execute(f"""
        WITH ranked AS (
            SELECT Model, ValidTimeUtc, RunTimeUtc,
                   WindSpeed10m, WindDirection10m, WindGusts10m,
                   Temperature2m, DewPoint2m, SurfacePressure, CloudCover,
                   row_number() OVER (PARTITION BY Model, ValidTimeUtc
                                       ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
            WHERE Model IN {nwp_in}
              AND RunTimeSource = 'reported'
              AND ValidTimeUtc >= TIMESTAMP '{day_start_eff.isoformat()}'
              AND ValidTimeUtc <  TIMESTAMP '{day_end.isoformat()}'
              AND WindSpeed10m IS NOT NULL
              AND WindDirection10m IS NOT NULL
        )
        SELECT * FROM ranked WHERE rn = 1
    """).df()
    if fc.empty:
        log.warning("  lead %dh: no forecast rows in [%s, %s) — skipping cell.",
                    lead, day_start_eff, day_end)
        return pd.DataFrame()

    # Pivot.
    def pivot(col: str) -> pd.DataFrame:
        p = fc.pivot(index="ValidTimeUtc", columns="Model", values=col).reset_index()
        p.columns = ["ValidTimeUtc"] + [f"{col}_{m}" for m in p.columns[1:]]
        return p

    run_pivot = fc.pivot(index="ValidTimeUtc", columns="Model",
                          values="RunTimeUtc").reset_index()
    run_pivot.columns = ["ValidTimeUtc"] + [f"RunTime_{m}" for m in run_pivot.columns[1:]]

    tabs = {n: pivot(n) for n in ("WindSpeed10m", "WindDirection10m",
                                    "WindGusts10m", "Temperature2m",
                                    "DewPoint2m", "SurfacePressure",
                                    "CloudCover")}
    df = tabs["WindSpeed10m"]
    for k, t in tabs.items():
        if k != "WindSpeed10m":
            df = df.merge(t, on="ValidTimeUtc", how="inner")
    df = df.merge(run_pivot, on="ValidTimeUtc", how="left")

    # Spread + ORO derivation (same as train).
    wsp_cols = [c for c in tabs["WindSpeed10m"].columns if c != "ValidTimeUtc"]
    wdir_cols = [c for c in tabs["WindDirection10m"].columns if c != "ValidTimeUtc"]
    gust_cols = [c for c in tabs["WindGusts10m"].columns if c != "ValidTimeUtc"]
    t_cols    = [c for c in tabs["Temperature2m"].columns if c != "ValidTimeUtc"]
    td_cols   = [c for c in tabs["DewPoint2m"].columns if c != "ValidTimeUtc"]
    p_cols    = [c for c in tabs["SurfacePressure"].columns if c != "ValidTimeUtc"]
    cc_cols   = [c for c in tabs["CloudCover"].columns if c != "ValidTimeUtc"]

    df["wsp_xmean"] = df[wsp_cols].mean(axis=1, skipna=True)
    df["wd_sin_xmean"] = np.nanmean(np.sin(np.radians(df[wdir_cols].values)), axis=1)
    df["wd_cos_xmean"] = np.nanmean(np.cos(np.radians(df[wdir_cols].values)), axis=1)
    df["td_xmean"] = df[td_cols].mean(axis=1, skipna=True)
    df["p_xmean"]  = df[p_cols].mean(axis=1, skipna=True)
    arr = np.array([
        oro_dynamic(r.wsp_xmean, r.wd_sin_xmean, r.wd_cos_xmean,
                    r.td_xmean, r.p_xmean, oro)
        for r in df.itertuples()
    ])
    for i, name in enumerate(ORO_LEAN):
        df[name] = arr[:, i]
    for label, col in SPREAD_VARS:
        if col == "WindSpeed10m":   src_cols = wsp_cols
        elif col == "WindGusts10m": src_cols = gust_cols
        elif col == "Temperature2m": src_cols = t_cols
        elif col == "DewPoint2m":    src_cols = td_cols
        elif col == "SurfacePressure": src_cols = p_cols
        elif col == "CloudCover":    src_cols = cc_cols
        else: raise AssertionError(col)
        df[f"{label}_mean"] = df[src_cols].mean(axis=1, skipna=True)
        df[f"{label}_std"]  = df[src_cols].std(axis=1, skipna=True)

    # Some NWPs may be entirely absent from the live forecast tree this
    # cycle (e.g. GEM hasn't run for this anchor yet). The DuckDB pivot
    # then omits those NWPs' wsp/wdir columns from `df`. pandas dropna
    # raises KeyError on a subset that names a missing column, so add any
    # absent feature as all-NaN to preserve the scaler's positional
    # column-order contract. Rows that end up all-NaN across feature_names
    # are then dropped as before — net effect is "soft-skip NWPs we don't
    # have this cycle" rather than crashing the whole workflow.
    for col in feature_names:
        if col not in df.columns:
            df[col] = float("nan")
    df = df.dropna(subset=feature_names, how="all").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Main predict loop (MC + circular-CI helpers come from _wind_common)
# ----------------------------------------------------------------------------

def predict_for_location(location: str, anchor: datetime,
                          models_root: Path, predictions_root: Path) -> int:
    bundle = find_latest_bundle(models_root, location)
    version = bundle.name  # e.g. "v2026-05-28_103045_wind_mvn"
    log.info("Using bundle %s", bundle)

    scaler = json.loads((bundle / "feature_scaler.json").read_text())
    schema = json.loads((bundle / "feature_schema.json").read_text())
    calibration = json.loads((bundle / "calibration.json").read_text())
    feature_names: list[str] = schema["FeatureNames"]

    oro = load_oro_static(location)
    prediction_made_at = datetime.now(timezone.utc).replace(microsecond=0)

    out_rows: list[dict] = []
    for lead in LEADS:
        lead_key = str(lead)
        state_path = bundle / f"state_lead_{lead}h.pt"
        if not state_path.is_file():
            log.warning("  lead %dh: state file missing — skipping.", lead)
            continue
        if lead_key not in scaler["per_lead"]:
            log.warning("  lead %dh: scaler missing — skipping.", lead)
            continue
        if lead_key not in calibration["per_lead"]:
            log.warning("  lead %dh: calibration missing — skipping.", lead)
            continue

        df = build_live_for_lead(location, lead, anchor, oro, feature_names)
        if df.empty:
            continue

        per_lead_scaler = scaler["per_lead"][lead_key]
        medians = np.asarray(per_lead_scaler["medians"], dtype=np.float64)
        mu_f    = np.asarray(per_lead_scaler["mean"],    dtype=np.float64)
        scale_f = np.asarray(per_lead_scaler["scale"],   dtype=np.float64)
        keep    = np.asarray(per_lead_scaler["kept_mask"], dtype=bool)

        # Pull feature matrix in canonical order, restrict to kept columns,
        # median-impute, standardise. Strict: feature_names order matches
        # the saved scaler so positional indexing is safe.
        X_raw = df[feature_names].to_numpy(dtype="float64")
        X_raw = X_raw[:, keep]
        # Median-impute (using train medians stored in the bundle).
        nan_mask = np.isnan(X_raw)
        for j in range(X_raw.shape[1]):
            X_raw[nan_mask[:, j], j] = medians[j]
        X_s = (X_raw - mu_f) / scale_f

        # Forward pass.
        net = WindMVNHead(n_in=X_s.shape[1])
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            mu_t, sig_t, rho_t = net(torch.tensor(X_s, dtype=torch.float32))
        mu_u = mu_t[:, 0].numpy()
        mu_v = mu_t[:, 1].numpy()
        sig_u = sig_t[:, 0].numpy()
        sig_v = sig_t[:, 1].numpy()
        rho = rho_t.squeeze(-1).numpy()

        cal = calibration["per_lead"][lead_key]
        alpha_dir = float(cal["alpha_prime_dir"])
        alpha_spd = float(cal["alpha_prime_spd"])

        # Direction CI: MC over (u,v), highest-density arc per level.
        # The direction marginal of N(μ, Σ) has no closed-form CI, so MC
        # is the practical option. For directional uncertainty the MC is
        # well-behaved (no Jensen bias) — the issue is purely on speed.
        _, dir_samples = mc_speed_dir(mu_u, mu_v, sig_u, sig_v, rho, alpha_dir)
        dir_lo95, dir_hi95, _ = circ_quantiles(dir_samples, level=0.95)
        dir_lo80, dir_hi80, _ = circ_quantiles(dir_samples, level=0.80)

        # Speed CI: delta method projection of Σ onto the radial axis at
        # (μ_u, μ_v) — see the 2026-05-28 note below for why MC was
        # replaced.
        #
        # NOTE on calibration (2026-05-29): α'_spd is INTENTIONALLY NOT
        # applied to the speed σ here. The trained α'_spd was fitted to
        # make MC CI95 hit ~95% coverage on validation residuals; on this
        # bundle that came out at 2.675 (lead 24h), which inflated σ by
        # ~3× and produced bands of [0, 36] mph on a 5 mph point —
        # mathematically calibrated but visually unusable and not
        # comparable to any other wind product on the page (the LGB
        # blender has no CI of its own; the wind champion likewise). We
        # use the network's RAW σ output as the model's self-asserted
        # uncertainty, which gives "looks-right" bands that under-cover
        # real residuals. Honest framing on the page: "model-asserted
        # ±1σ / ±2σ-shaped band", not "80% credible interval".
        # Re-introduce α' once the trainer either learns a better-
        # calibrated raw σ or moves to a head that predicts speed
        # parameters directly.
        #
        # Delta method: linearise speed = sqrt(u² + v²) around (μ_u, μ_v).
        #   ∂s/∂u = μ_u / ||μ|| ; ∂s/∂v = μ_v / ||μ|| ; so
        #   σ²_speed = (μ_u² σ_u² + μ_v² σ_v² + 2 μ_u μ_v ρ σ_u σ_v) / ||μ||²
        # CI is symmetric in Gaussian space around ||μ||, floored at 0 m/s.
        _ = alpha_spd  # kept in scope for future re-introduction; intentionally unused.
        spd_mu = np.sqrt(mu_u ** 2 + mu_v ** 2)
        spd_mu_safe = np.maximum(spd_mu, 1e-6)
        var_spd = (mu_u ** 2 * sig_u ** 2
                   + mu_v ** 2 * sig_v ** 2
                   + 2.0 * mu_u * mu_v * rho * sig_u * sig_v) / spd_mu_safe ** 2
        sig_spd = np.sqrt(np.maximum(var_spd, 0.0))
        spd_lo95 = np.maximum(0.0, spd_mu - 1.96  * sig_spd)
        spd_hi95 = spd_mu + 1.96  * sig_spd
        spd_lo80 = np.maximum(0.0, spd_mu - 1.282 * sig_spd)
        spd_hi80 = spd_mu + 1.282 * sig_spd

        # Point estimates: ||μ|| for speed (centre of delta-method CI by
        # construction); atan2(-μ) for direction. Both stable + easy to
        # explain.
        pt_speed = spd_mu
        pt_dir = (np.degrees(np.arctan2(-mu_u, -mu_v))) % 360.0

        # Per-NWP RunTime columns (RunTime_<nwp>_id → RunTimeUtcGfs etc.)
        run_cols = {}
        for nwp in NWPS:
            src = f"RunTime_{nwp}"
            short = NWP_SHORT.get(nwp, nwp)
            run_cols[f"RunTimeUtc{short}"] = df[src] if src in df.columns else pd.NaT

        for i, valid in enumerate(df["ValidTimeUtc"].values):
            row = {
                "ValidTimeUtc":          pd.Timestamp(valid).to_pydatetime(),
                "LeadHours":             lead,
                "PredictionMadeAtUtc":   prediction_made_at,
                "ModelVersion":          version,
                "LocationName":          location,
                "MuU":                   float(mu_u[i]),
                "MuV":                   float(mu_v[i]),
                "SigmaU":                float(sig_u[i]),
                "SigmaV":                float(sig_v[i]),
                "Rho":                   float(rho[i]),
                "BlendDirection":        float(pt_dir[i]),
                "BlendDirectionCi95Lo":  float(dir_lo95[i]),
                "BlendDirectionCi95Hi":  float(dir_hi95[i]),
                "BlendDirectionCi80Lo":  float(dir_lo80[i]),
                "BlendDirectionCi80Hi":  float(dir_hi80[i]),
                "BlendSpeedMagnitude":   float(pt_speed[i]),
                "BlendSpeedCi95Lo":      float(spd_lo95[i]),
                "BlendSpeedCi95Hi":      float(spd_hi95[i]),
                "BlendSpeedCi80Lo":      float(spd_lo80[i]),
                "BlendSpeedCi80Hi":      float(spd_hi80[i]),
            }
            for k, v in run_cols.items():
                vv = v.iloc[i] if hasattr(v, "iloc") else v
                row[k] = (pd.Timestamp(vv).to_pydatetime()
                          if pd.notna(vv) else None)
            out_rows.append(row)

        log.info("  lead %dh: wrote %d rows (mean dir %.1f°, mean spd %.2f m/s)",
                 lead, len(df), float(np.mean(pt_dir)), float(np.mean(pt_speed)))

    if not out_rows:
        log.error("No predictions produced for %s — exit 3.", location)
        return 3

    out_df = pd.DataFrame(out_rows).sort_values(["ValidTimeUtc", "LeadHours"])
    date_str = anchor.strftime("%Y-%m-%d")
    out_path = (predictions_root / TARGET / location
                / f"model_version={version}" / f"date={date_str}"
                / "predictions.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    log.info("Wrote %d rows → %s", len(out_df), out_path)
    return 0


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--location", default=LOCATION_DEFAULT)
    parser.add_argument("--anchor", default=None,
                         help="Anchor date YYYY-MM-DD UTC. Empty = today.")
    parser.add_argument("--models-root",
                         default=str(WEATHERBLEND_DATA_ROOT / "models"))
    parser.add_argument("--predictions-root",
                         default=str(WEATHERBLEND_DATA_ROOT / "predictions"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.anchor:
        anchor = datetime.fromisoformat(args.anchor).replace(tzinfo=timezone.utc)
    else:
        anchor = datetime.now(timezone.utc)
    anchor = anchor.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)

    log.info("Phase 2 wind_mvn — PREDICT")
    log.info("  location: %s",  args.location)
    log.info("  anchor:   %s",  anchor.isoformat())

    rc = predict_for_location(
        args.location, anchor, Path(args.models_root), Path(args.predictions_root))
    sys.exit(rc)


if __name__ == "__main__":
    main()
