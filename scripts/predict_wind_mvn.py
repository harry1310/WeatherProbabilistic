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
import math
import os
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import duckdb  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402


# ----------------------------------------------------------------------------
# Constants (must match train_wind_mvn.py)
# ----------------------------------------------------------------------------

PHASE = "wind_mvn"
TARGET = "wind_direction"

LEADS = (24, 48, 72)
NWPS = (
    "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless",
    "ukmo_seamless", "jma_seamless",
)
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

ORO_LEAN = ["oro_wind_sin", "oro_wind_cos", "oro_upwind_gain",
            "oro_uplift", "oro_uplift_x_q"]
SPREAD_VARS = [("wsp_spd",  "WindSpeed10m"),
               ("gust_spd", "WindGusts10m"),
               ("t_spd",    "Temperature2m"),
               ("td_spd",   "DewPoint2m"),
               ("p_spd",    "SurfacePressure"),
               ("cc_spd",   "CloudCover")]

N_HIDDEN = 64
DROPOUT = 0.10
N_MC = 500
SEED = 42

LOCATION_DEFAULT = os.environ.get("WB_LOCATION", "bonehill_rocks")

log = logging.getLogger("predict_wind_mvn")


# ----------------------------------------------------------------------------
# MLP — same class as train_wind_mvn.py (must match for state_dict load)
# ----------------------------------------------------------------------------

class WindMVNHead(nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, N_HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN, N_HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
        )
        self.mu_head = nn.Linear(N_HIDDEN, 2)
        self.scale_head = nn.Linear(N_HIDDEN, 2)
        self.rho_head = nn.Linear(N_HIDDEN, 1)

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        mu = self.mu_head(z)
        log_sigma = self.scale_head(z).clamp(min=-3.0, max=3.0)
        sigma = torch.exp(log_sigma)
        rho = 0.99 * torch.tanh(self.rho_head(z))
        return mu, sigma, rho


# ----------------------------------------------------------------------------
# Orographic — copy of train_wind_mvn.py's helpers (kept local for predict
# slimness — no import from train script which pulls torch.optim etc.)
# ----------------------------------------------------------------------------

_SECTOR_RAD = {
    "N": 0.0, "NE": math.pi / 4, "E": math.pi / 2, "SE": 3 * math.pi / 4,
    "S": math.pi, "SW": 5 * math.pi / 4, "W": 3 * math.pi / 2,
    "NW": 7 * math.pi / 4,
}


def load_oro_static(location: str) -> dict:
    return json.loads((WEATHERBLEND_DATA_ROOT / "static" / "orographic"
                       / f"{location}.json").read_text())


def upwind_gain_at(rad: float, upwind_gain_5km: dict) -> float:
    if math.isnan(rad):
        return 0.0
    d = rad % (2 * math.pi)
    best, best_diff = "N", math.pi * 3
    for s, c in _SECTOR_RAD.items():
        diff = min(abs(d - c), 2 * math.pi - abs(d - c))
        if diff < best_diff:
            best_diff, best = diff, s
    return float(upwind_gain_5km[best])


def oro_dynamic(wsp: float, wd_sin: float, wd_cos: float,
                td: float, p: float, oro: dict) -> list[float]:
    grad_dx = float(oro["terrain_gradient_dx"])
    grad_dy = float(oro["terrain_gradient_dy"])
    rad = (math.atan2(wd_sin, wd_cos)
           if not (math.isnan(wd_sin) or math.isnan(wd_cos)) else float("nan"))
    gain = upwind_gain_at(rad, oro["upwind_gain_5km"])
    if math.isnan(wsp) or math.isnan(wd_sin) or math.isnan(wd_cos):
        uplift = 0.0
    else:
        uplift = max(0.0, (-wsp * wd_sin) * grad_dx + (-wsp * wd_cos) * grad_dy)
    if math.isnan(td) or math.isnan(p) or p <= 0:
        q = 0.0
    else:
        e = 6.112 * math.exp(17.62 * td / (td + 243.12))
        q = max(0.0, 0.622 * e / (p - 0.378 * e) * 1000.0)
    return [
        0.0 if math.isnan(wd_sin) else wd_sin,
        0.0 if math.isnan(wd_cos) else wd_cos,
        gain, uplift, uplift * q,
    ]


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
            FROM read_parquet('{fc_glob}', union_by_name=true)
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
# CI helpers
# ----------------------------------------------------------------------------

def circ_quantiles(angles_deg: np.ndarray, level: float = 0.95):
    """Highest-density circular credible interval at the requested level.

    Finds the shortest arc covering `level` fraction of the samples per
    column. Returns (lo_deg, hi_deg, arc_width_deg).
    """
    M, N = angles_deg.shape
    lo = np.empty(N); hi = np.empty(N); width = np.empty(N)
    target = int(np.ceil(level * M))
    for j in range(N):
        s = np.sort(angles_deg[:, j])
        s_ext = np.concatenate([s, s + 360.0])
        widths = s_ext[target - 1:target - 1 + M] - s_ext[:M]
        k = int(np.argmin(widths))
        lo[j] = s_ext[k] % 360.0
        hi[j] = s_ext[k + target - 1] % 360.0
        width[j] = widths[k]
    return lo, hi, width


# Back-compat alias — older callers may import the 95-only name.
def circ_quantiles_95(angles_deg: np.ndarray):
    return circ_quantiles(angles_deg, level=0.95)


def _mc_speed_dir(mu_u: np.ndarray, mu_v: np.ndarray,
                  sig_u: np.ndarray, sig_v: np.ndarray, rho: np.ndarray,
                  alpha: float, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    z = rng.standard_normal((N_MC, len(mu_u), 2))
    s_u = sig_u * alpha
    s_v = sig_v * alpha
    sqrt_1mr2 = np.sqrt(1.0 - rho ** 2)
    u = mu_u[None, :] + s_u[None, :] * z[:, :, 0]
    v = mu_v[None, :] + s_v[None, :] * (rho[None, :] * z[:, :, 0]
                                          + sqrt_1mr2[None, :] * z[:, :, 1])
    speed = np.sqrt(u ** 2 + v ** 2)
    direction = (np.degrees(np.arctan2(-u, -v))) % 360.0
    return speed, direction


# ----------------------------------------------------------------------------
# Main predict loop
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
        _, dir_samples = _mc_speed_dir(mu_u, mu_v, sig_u, sig_v, rho, alpha_dir)
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
