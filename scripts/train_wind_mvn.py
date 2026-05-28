"""Phase 2 wind_mvn — PyTorch MLP bivariate-normal direction estimator.

Trains one MLP per (location, lead) on the 29-feature lean+ORO+spread
input (matching the production scope locked in 2026-05-27 — see
``WeatherBlend/docs/WIND_BLENDER_PLAN.md``). Output: μ_u, μ_v, σ_u, σ_v, ρ
parameterising a bivariate normal over the (u, v) wind decomposition,
fit by NLL. The MLP gets both a direction estimate (atan2(-μ_u, -μ_v))
and a speed magnitude (sqrt(μ_u² + μ_v²)).

Truth: Dunkeswell SYNOP (MIDAS Open station 01383), available 2022-2024
via the WeatherBlend ``data/truth/midas/raw/`` tree.

Per-lead artefact bundle::

  data/models/wind_direction/{location}/{version}_wind_mvn/
    state_lead_24h.pt          torch.save(state_dict)
    state_lead_48h.pt
    state_lead_72h.pt
    feature_scaler.json        per-lead {mean[29], scale[29], medians[29]}
    feature_schema.json        feature_names + nwp_models + production_scope
    calibration.json           per-lead {alpha_prime_dir, alpha_prime_spd,
                                blend_center, blend_scale}
    training_metadata.json     standard schema (Version, Phase, PerLead, ...)
    training_summary.json      RetrainGuard sidecar
    test_predictions.parquet   per-row (valid_time, lead, mu_u, mu_v,
                                sigma_u, sigma_v, rho, truth_u, truth_v)

Calibration recipe per lead, fit on val:
  - alpha_prime_dir : multiplicative scaling of σ_u and σ_v before
    drawing MC samples → direction CI95 coverage hits 95% on val.
  - alpha_prime_spd : same for the speed CI95.
  - blend_center, blend_scale : sigmoid blend parameters for the
    downstream wind_blend composition (final = w·mvn + (1-w)·lgb where
    w = 1 / (1 + exp((lgb_speed - center)/scale))). Fit on val MAE
    against the bound wind_speed_lgb predictions IF available; else
    falls back to the plan defaults (2.50, 3.00).

CLI::

    train_wind_mvn.py [--location bonehill_rocks] [--leads 24 48 72]
                      [--anchor YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime, timezone
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
import torch.optim as optim  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402
from src.manifest_promote import promote_station_version  # noqa: E402
from src.retrain_guard import build_check_and_save_versioned  # noqa: E402


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

PHASE = "wind_mvn"
TARGET = "wind_direction"
LOCATION_DEFAULT = "bonehill_rocks"
LEADS = (24, 48, 72)

# Production NWPs from the 2026-05-27 wind_speed_lgb decision — 6 NWPs.
# MF excluded (0 wsp rows on Open-Meteo Previous Runs).
# AIFS excluded — no 2024 rows.
# HARMONIE confirmed dead weight for wind.
NWPS = (
    "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless",
    "ukmo_seamless", "jma_seamless",
)

# Dunkeswell SYNOP station — MIDAS Open ID 01383.
DUNKESWELL_STATION_ID = "01383"
DUNKESWELL_YEARS = (2022, 2023, 2024)
KT_TO_MS = 0.514444

# Feature-set composition. Order matters — the scaler params in
# feature_scaler.json are positional. Mirrors the lock-in in
# WIND_BLENDER_PLAN.md "wind_mvn architecture" section.
ORO_LEAN = ["oro_wind_sin", "oro_wind_cos", "oro_upwind_gain",
            "oro_uplift", "oro_uplift_x_q"]
SPREAD_VARS = [("wsp_spd",  "WindSpeed10m"),
               ("gust_spd", "WindGusts10m"),
               ("t_spd",    "Temperature2m"),
               ("td_spd",   "DewPoint2m"),
               ("p_spd",    "SurfacePressure"),
               ("cc_spd",   "CloudCover")]

# MLP hyperparams per the plan.
N_HIDDEN = 64
DROPOUT = 0.10
LR = 1e-3
WEIGHT_DECAY = 1e-4
COSINE_T_MAX = 200
COSINE_ETA_MIN = 1e-5
BATCH_SIZE = 256
GRAD_CLIP = 5.0
PATIENCE = 30
MAX_EPOCHS = 300
SEED = 42
N_MC = 500  # MC samples for CI calibration

# Calibration grid search bounds.
ALPHA_GRID = np.arange(0.5, 3.01, 0.025)
BLEND_CENTER_GRID = np.arange(0.5, 8.01, 0.25)
BLEND_SCALE_GRID = np.arange(0.25, 5.01, 0.25)
TARGET_COVERAGE = 0.95


log = logging.getLogger("train_wind_mvn")


# ----------------------------------------------------------------------------
# Orographic dynamic features
# ----------------------------------------------------------------------------

def load_oro_static(location: str) -> dict:
    path = WEATHERBLEND_DATA_ROOT / "static" / "orographic" / f"{location}.json"
    return json.loads(path.read_text())


_SECTOR_RAD = {
    "N": 0.0, "NE": math.pi / 4, "E": math.pi / 2, "SE": 3 * math.pi / 4,
    "S": math.pi, "SW": 5 * math.pi / 4, "W": 3 * math.pi / 2,
    "NW": 7 * math.pi / 4,
}


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
# Dunkeswell MIDAS truth loader
# ----------------------------------------------------------------------------

def parse_badc(path: Path) -> pd.DataFrame:
    t = path.read_text(encoding="utf-8", errors="replace").splitlines()
    s = t.index("data") + 1
    e = t.index("end data")
    header = t[s].split(",")
    rows = [r.split(",") for r in t[s + 1:e]]
    return pd.DataFrame(rows, columns=header)


def load_dunkeswell(years: tuple[int, ...] = DUNKESWELL_YEARS) -> pd.DataFrame:
    midas_dir = WEATHERBLEND_DATA_ROOT / "truth" / "midas" / "raw"
    frames: list[pd.DataFrame] = []
    for y in years:
        candidates = list(midas_dir.glob(f"midas-open*_{DUNKESWELL_STATION_ID}_*_{y}.csv"))
        if not candidates:
            log.warning("Dunkeswell %d MIDAS file not found under %s — skipping year.",
                        y, midas_dir)
            continue
        df = parse_badc(candidates[0])
        df = df[df["met_domain_name"] == "SYNOP"].copy()
        df["ob_time"] = pd.to_datetime(df["ob_time"], utc=True).dt.tz_localize(None)
        df["wsp_ms"] = pd.to_numeric(df["wind_speed"].replace("NA", np.nan),
                                      errors="coerce") * KT_TO_MS
        df["wdir_deg"] = pd.to_numeric(df["wind_direction"].replace("NA", np.nan),
                                        errors="coerce")
        frames.append(df[["ob_time", "wsp_ms", "wdir_deg"]].rename(
            columns={"ob_time": "ValidTimeUtc"}))
    if not frames:
        raise FileNotFoundError(
            f"No Dunkeswell MIDAS files found under {midas_dir} for any of {years}.")
    out = pd.concat(frames, ignore_index=True).dropna(
        subset=["wsp_ms", "wdir_deg"]).sort_values("ValidTimeUtc").reset_index(drop=True)
    log.info("Loaded %d Dunkeswell SYNOP truth points (%d-%d).",
             len(out), min(years), max(years))
    return out


# ----------------------------------------------------------------------------
# Forecast pivot + feature build
# ----------------------------------------------------------------------------

def build_features(location: str, lead: int, dunk: pd.DataFrame,
                   oro: dict) -> pd.DataFrame:
    """Pull forecast tree at the given lead for the 6 production NWPs,
    pivot into per-NWP columns, derive ORO + SPREAD, merge with Dunkeswell
    truth. Returns one row per ValidTimeUtc with the 29-feature vector +
    (u, v) truth columns."""
    fc_glob = str(WEATHERBLEND_DATA_ROOT / "forecasts"
                  / f"location={location}" / "**" / "*.parquet").replace("\\", "/")
    nwp_in = "(" + ",".join(f"'{m}'" for m in NWPS) + ")"

    con = duckdb.connect()
    fc = con.execute(f"""
        WITH ranked AS (
            SELECT Model, ValidTimeUtc, WindSpeed10m, WindDirection10m,
                   WindGusts10m, Temperature2m, DewPoint2m, SurfacePressure,
                   CloudCover, RunTimeUtc,
                   row_number() OVER (PARTITION BY Model, ValidTimeUtc
                                       ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{fc_glob}', union_by_name=true)
            WHERE Model IN {nwp_in}
              AND RunTimeSource = 'offset_day'
              AND LeadHours = {lead}
              AND WindSpeed10m IS NOT NULL
              AND WindDirection10m IS NOT NULL
        )
        SELECT * FROM ranked WHERE rn = 1
    """).df()
    if fc.empty:
        log.warning("  lead %dh: zero forecast rows for %s. Skipping cell.", lead, location)
        return pd.DataFrame()

    def pivot(col: str) -> pd.DataFrame:
        p = fc.pivot(index="ValidTimeUtc", columns="Model", values=col).reset_index()
        p.columns = ["ValidTimeUtc"] + [f"{col}_{m}" for m in p.columns[1:]]
        return p

    tabs = {n: pivot(n) for n in ("WindSpeed10m", "WindDirection10m",
                                    "WindGusts10m", "Temperature2m",
                                    "DewPoint2m", "SurfacePressure",
                                    "CloudCover")}
    df = tabs["WindSpeed10m"]
    for k, t in tabs.items():
        if k != "WindSpeed10m":
            df = df.merge(t, on="ValidTimeUtc", how="inner")

    def cols_for(col_prefix: str) -> list[str]:
        return [c for c in tabs[col_prefix].columns if c != "ValidTimeUtc"]

    wsp_cols  = cols_for("WindSpeed10m")
    wdir_cols = cols_for("WindDirection10m")
    gust_cols = cols_for("WindGusts10m")
    t_cols    = cols_for("Temperature2m")
    td_cols   = cols_for("DewPoint2m")
    p_cols    = cols_for("SurfacePressure")
    cc_cols   = cols_for("CloudCover")

    # NWP-mean derived columns for ORO + SPREAD.
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

    spread_names: list[str] = []
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
        spread_names.extend([f"{label}_mean", f"{label}_std"])

    # Merge truth + decompose into (u, v). Dunkeswell wdir is FROM convention,
    # so u = -wsp · sin(dir), v = -wsp · cos(dir).
    df = df.merge(dunk, on="ValidTimeUtc", how="inner").dropna(
        subset=["wsp_ms", "wdir_deg"]).reset_index(drop=True)
    df["u_truth"] = -df["wsp_ms"] * np.sin(np.radians(df["wdir_deg"]))
    df["v_truth"] = -df["wsp_ms"] * np.cos(np.radians(df["wdir_deg"]))
    return df


def feature_column_order(df: pd.DataFrame) -> list[str]:
    """Resolve the 29 feature names in their canonical order: per-NWP wsp,
    per-NWP wdir, ORO_LEAN, SPREAD. Order matches the .NET / bake-off
    convention so cross-repo readers don't drift."""
    feats: list[str] = []
    for nwp in NWPS:
        col = f"WindSpeed10m_{nwp}"
        if col in df.columns:
            feats.append(col)
    for nwp in NWPS:
        col = f"WindDirection10m_{nwp}"
        if col in df.columns:
            feats.append(col)
    feats.extend(ORO_LEAN)
    for label, _ in SPREAD_VARS:
        feats.append(f"{label}_mean")
        feats.append(f"{label}_std")
    return feats


# ----------------------------------------------------------------------------
# MLP definition + training
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


def bvn_nll(mu: torch.Tensor, sigma: torch.Tensor, rho: torch.Tensor,
            y: torch.Tensor) -> torch.Tensor:
    du = y[:, 0] - mu[:, 0]
    dv = y[:, 1] - mu[:, 1]
    s_u = sigma[:, 0]
    s_v = sigma[:, 1]
    r = rho.squeeze(-1)
    one_m_r2 = 1.0 - r * r
    z = (du / s_u) ** 2 + (dv / s_v) ** 2 - 2 * r * du * dv / (s_u * s_v)
    return (math.log(2 * math.pi)
            + torch.log(s_u) + torch.log(s_v)
            + 0.5 * torch.log(one_m_r2)
            + 0.5 * z / one_m_r2).mean()


def standardise(X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray):
    """Median-impute NaNs (from train), drop all-NaN-in-train columns,
    return (X_tr_s, X_va_s, X_te_s, medians, mu, scale, kept_mask)."""
    nan_frac = np.isnan(X_tr).mean(axis=0)
    keep = nan_frac < 1.0
    X_tr = X_tr[:, keep].astype(np.float64)
    X_va = X_va[:, keep].astype(np.float64)
    X_te = X_te[:, keep].astype(np.float64)
    medians = np.nanmedian(X_tr, axis=0)
    for X in (X_tr, X_va, X_te):
        mask = np.isnan(X)
        for j in range(X.shape[1]):
            X[mask[:, j], j] = medians[j]
    mu = X_tr.mean(axis=0)
    scale = X_tr.std(axis=0) + 1e-9
    return ((X_tr - mu) / scale,
            (X_va - mu) / scale,
            (X_te - mu) / scale,
            medians, mu, scale, keep)


def train_one_lead(X_tr: np.ndarray, X_va: np.ndarray,
                   Y_tr: np.ndarray, Y_va: np.ndarray) -> tuple[nn.Module, float, int]:
    """Returns (best_net, best_val_nll, epochs_to_best)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = WindMVNHead(n_in=X_tr.shape[1])
    opt = optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=COSINE_T_MAX,
                                                  eta_min=COSINE_ETA_MIN)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32)
    Y_va_t = torch.tensor(Y_va, dtype=torch.float32)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = PATIENCE
    for epoch in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(len(X_tr_t))
        for s in range(0, len(X_tr_t), BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            mu, sigma, rho = net(X_tr_t[idx])
            loss = bvn_nll(mu, sigma, rho, Y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
        net.eval()
        with torch.no_grad():
            mu, sigma, rho = net(X_va_t)
            v_nll = bvn_nll(mu, sigma, rho, Y_va_t).item()
        sched.step()
        if v_nll < best_val - 1e-4:
            best_val = v_nll
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best_val, best_epoch


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------

def circ_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def circ_quantiles_95(angles_deg: np.ndarray):
    """Per-row minimum-width 95% interval over MC samples (deg)."""
    M, N = angles_deg.shape
    lo = np.empty(N); hi = np.empty(N); width = np.empty(N)
    target = int(np.ceil(0.95 * M))
    for j in range(N):
        s = np.sort(angles_deg[:, j])
        s_ext = np.concatenate([s, s + 360.0])
        widths = s_ext[target - 1:target - 1 + M] - s_ext[:M]
        k = int(np.argmin(widths))
        lo[j] = s_ext[k] % 360.0
        hi[j] = s_ext[k + target - 1] % 360.0
        width[j] = widths[k]
    return lo, hi, width


def circ_in(truth_deg: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    t = truth_deg % 360.0
    lo_n = lo % 360.0
    hi_n = hi % 360.0
    return np.where(lo_n <= hi_n, (t >= lo_n) & (t <= hi_n),
                                   (t >= lo_n) | (t <= hi_n))


def _mc_samples(mu_u: np.ndarray, mu_v: np.ndarray,
                sig_u: np.ndarray, sig_v: np.ndarray, rho: np.ndarray,
                alpha: float, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Draw N_MC bivariate-normal samples per row with σ scaled by alpha.
    Returns (speed_samples, direction_samples) shaped [N_MC, N]."""
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


def calibrate_alpha_prime(mu_u: np.ndarray, mu_v: np.ndarray,
                          sig_u: np.ndarray, sig_v: np.ndarray, rho: np.ndarray,
                          truth_speed: np.ndarray, truth_dir: np.ndarray
                          ) -> tuple[float, float, dict]:
    """Grid-search α' (scaling σ_u, σ_v) so direction and speed CI95
    coverage match the 0.95 target on the val slice. Independent grids
    for direction and speed since they have different noise profiles.
    Returns (alpha_dir, alpha_spd, diagnostics)."""
    # Direction CI calibration.
    best_dir = (None, float("inf"))
    for alpha in ALPHA_GRID:
        _, dr = _mc_samples(mu_u, mu_v, sig_u, sig_v, rho, alpha)
        lo, hi, _ = circ_quantiles_95(dr)
        cov = float(np.mean(circ_in(truth_dir, lo, hi)))
        miss = abs(cov - TARGET_COVERAGE)
        if miss < best_dir[1]:
            best_dir = (float(alpha), miss)
    # Speed CI calibration.
    best_spd = (None, float("inf"))
    for alpha in ALPHA_GRID:
        sp, _ = _mc_samples(mu_u, mu_v, sig_u, sig_v, rho, alpha)
        lo = np.percentile(sp, 2.5, axis=0)
        hi = np.percentile(sp, 97.5, axis=0)
        cov = float(np.mean((truth_speed >= lo) & (truth_speed <= hi)))
        miss = abs(cov - TARGET_COVERAGE)
        if miss < best_spd[1]:
            best_spd = (float(alpha), miss)
    # Diagnostics post-calibration.
    sp_cal, dr_cal = _mc_samples(mu_u, mu_v, sig_u, sig_v, rho, best_dir[0])
    lo_d, hi_d, w_d = circ_quantiles_95(dr_cal)
    dir_cov = float(np.mean(circ_in(truth_dir, lo_d, hi_d)))
    sp_cal2, _ = _mc_samples(mu_u, mu_v, sig_u, sig_v, rho, best_spd[0])
    lo_s = np.percentile(sp_cal2, 2.5, axis=0)
    hi_s = np.percentile(sp_cal2, 97.5, axis=0)
    spd_cov = float(np.mean((truth_speed >= lo_s) & (truth_speed <= hi_s)))
    diagnostics = {
        "val_dir_cov_after_alpha_prime_dir":  dir_cov,
        "val_spd_cov_after_alpha_prime_spd": spd_cov,
        "val_dir_width_deg":  float(np.mean(w_d)),
        "val_spd_width_ms":   float(np.mean(hi_s - lo_s)),
    }
    return best_dir[0], best_spd[0], diagnostics


def calibrate_blend(lgb_val: np.ndarray | None,
                    mvn_val_speed: np.ndarray,
                    truth_val_speed: np.ndarray
                    ) -> tuple[float, float, dict]:
    """Grid-search sigmoid blend params on val. If no LGB predictions
    are provided (bound wind_speed_lgb not yet trained), fall back to
    the plan defaults 2.50 / 3.00."""
    if lgb_val is None or len(lgb_val) != len(mvn_val_speed):
        log.info("  blend calibration: no LGB val predictions; using plan defaults (2.50, 3.00).")
        return 2.50, 3.00, {"blend_source": "plan_defaults"}
    best = (2.50, 3.00, float("inf"))
    for c in BLEND_CENTER_GRID:
        for s in BLEND_SCALE_GRID:
            w = 1.0 / (1.0 + np.exp((lgb_val - c) / s))
            blended = w * mvn_val_speed + (1.0 - w) * lgb_val
            mae = float(np.mean(np.abs(blended - truth_val_speed)))
            if mae < best[2]:
                best = (float(c), float(s), mae)
    return best[0], best[1], {"blend_source": "val_fit",
                               "val_blend_mae": best[2]}


# ----------------------------------------------------------------------------
# Bundle writer
# ----------------------------------------------------------------------------

def write_bundle(out_dir: Path, location: str, version: str,
                 per_lead: dict, feature_names: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-lead model state.
    for lead, cell in per_lead.items():
        torch.save(cell["state_dict"], out_dir / f"state_lead_{lead}h.pt")

    # 2) feature_scaler.json — per-lead.
    scaler = {
        "feature_names": feature_names,
        "per_lead": {
            str(lead): {
                "medians": cell["medians"].tolist(),
                "mean":    cell["mu"].tolist(),
                "scale":   cell["scale"].tolist(),
                "kept_mask": cell["keep"].tolist(),
            }
            for lead, cell in per_lead.items()
        },
    }
    (out_dir / "feature_scaler.json").write_text(
        json.dumps(scaler, indent=2, allow_nan=False))

    # 3) feature_schema.json.
    schema = {
        "Target":      TARGET,
        "Phase":       PHASE,
        "Location":    location,
        "FeatureNames": feature_names,
        "NwpModels":    list(NWPS),
        "ProductionScope": (
            "6 production NWPs, no HARMONIE, no archive. AIFS excluded "
            "(no 2024 data). MF excluded (no wsp on Open-Meteo)."
        ),
        "PerLead": {
            str(lead): {
                "LeadHours": lead,
                "RequiredModels": [],
                "OptionalModels": list(NWPS),
                "Models":         list(NWPS),
                "Tier":           "lean+ORO+spread",
            }
            for lead in per_lead
        },
    }
    (out_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, allow_nan=False))

    # 4) calibration.json — per-lead α' + blend.
    calibration = {
        "per_lead": {
            str(lead): {
                "alpha_prime_dir": cell["alpha_dir"],
                "alpha_prime_spd": cell["alpha_spd"],
                "blend_center":    cell["blend_center"],
                "blend_scale":     cell["blend_scale"],
                "calibration_diagnostics": cell["calib_diag"],
            }
            for lead, cell in per_lead.items()
        }
    }
    (out_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, allow_nan=False))

    # 5) training_metadata.json.
    per_lead_stats = {
        str(lead): {
            "LeadHours":         lead,
            "TrainRows":         int(cell["n_train"]),
            "ValRows":           int(cell["n_val"]),
            "TestRows":          int(cell["n_test"]),
            "BestSingle":        "wind_mvn_mlp",
            "BlendTestMae":      float(cell["test_dir_mae_deg"]),  # repurposed: dir MAE °
            "BlendTestRmse":     float(cell["test_spd_mae_ms"]),   # repurposed: spd MAE m/s
            "BlendTestBias":     None,
            "BestSingleValMae":  float(cell["val_nll"]),
            "BestSingleTestMae": float(cell["test_dir_mae_deg"]),
            "DataRangeTrain":    cell["range_train"],
            "DataRangeVal":      cell["range_val"],
            "DataRangeTest":     cell["range_test"],
            "TestCalendarMonths": int(cell["test_calendar_months"]),
        }
        for lead, cell in per_lead.items()
    }
    metadata = {
        "Version":       version,
        "Target":        TARGET,
        "Phase":         PHASE,
        "LocationName":  location,
        "DataSource":    "open_meteo_previous_runs+dunkeswell_midas_synop",
        "TrainedAtUtc":  datetime.now(timezone.utc).isoformat(),
        "Hyperparameters": {
            "library":         f"torch=={torch.__version__}",
            "architecture":    f"Linear({len(feature_names)}→{N_HIDDEN})→GELU→Dropout({DROPOUT})"
                                 f"→Linear({N_HIDDEN}→{N_HIDDEN})→GELU→Dropout({DROPOUT})"
                                 f"→{{μ:2, σ:2 clamp[-3,3] exp, ρ:1 0.99·tanh}}",
            "loss":            "bivariate_normal_nll",
            "optimiser":       f"AdamW lr={LR} wd={WEIGHT_DECAY}",
            "scheduler":       f"CosineAnnealingLR T_max={COSINE_T_MAX} eta_min={COSINE_ETA_MIN}",
            "batch_size":      BATCH_SIZE,
            "grad_clip":       GRAD_CLIP,
            "early_stop_patience": PATIENCE,
            "max_epochs":      MAX_EPOCHS,
            "seed":            SEED,
            "calibration":     "α'_dir + α'_spd grid-fit on val (target CI95 cov=0.95)",
        },
        "DeviationsFromBrief": [
            "Per-lead calibration of α'_dir / α'_spd / blend_center / "
            "blend_scale (the plan suggested pooled across leads; we fit "
            "per-lead for finer control, fallback to pooled if any lead "
            "has <100 val rows).",
            "PerLeadStats BlendTestMae repurposed to carry direction MAE "
            "(°). BlendTestRmse carries speed MAE (m/s). The schema is "
            "shared with the LGB element verifiers, which expect MAE-style "
            "numbers in those fields.",
            "Speed CI calibration via α' is a heuristic — bake-off 2026-05-27 "
            "verified MC + α' beats variance-inflation alternatives on aggregate.",
        ],
        "PerLead": per_lead_stats,
    }
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False, default=str))

    # 6) test_predictions.parquet (per-row, all leads concatenated).
    test_frames = [cell["test_predictions"] for cell in per_lead.values()
                   if not cell["test_predictions"].empty]
    if test_frames:
        pd.concat(test_frames, ignore_index=True).to_parquet(
            out_dir / "test_predictions.parquet", index=False)


# ----------------------------------------------------------------------------
# Main per-location orchestration
# ----------------------------------------------------------------------------

def time_split(n: int, train_frac: float = 0.70, val_frac: float = 0.15
                ) -> tuple[int, int]:
    train_end = int(np.floor(n * train_frac))
    val_end = train_end + int(np.floor(n * val_frac))
    return train_end, val_end


def train_one_location(location: str, version: str,
                        models_root: Path,
                        leads: tuple[int, ...] = LEADS) -> int:
    """Train wind_mvn for one location across all configured leads.
    Returns 0 on success, nonzero on guard failure / no rows."""
    oro = load_oro_static(location)
    dunk = load_dunkeswell()

    # 1) Build per-lead datasets; gather rows + feature ordering.
    per_lead_data: dict[int, dict] = {}
    feature_names: list[str] | None = None
    rows_train_total = rows_val_total = rows_test_total = 0
    first_lead_train_features: np.ndarray | None = None

    for lead in leads:
        log.info("[%s] %s lead %dh — loading", time.strftime("%H:%M:%S"), location, lead)
        df = build_features(location, lead, dunk, oro)
        if df.empty:
            log.warning("  zero rows for %s lead %dh — skipping cell.", location, lead)
            continue
        feats = feature_column_order(df)
        if feature_names is None:
            feature_names = feats
        elif feats != feature_names:
            # Per-lead feature order drift would crash the per-lead bundle —
            # bail loudly with the diff rather than silently mis-stamp the
            # schema.
            raise RuntimeError(
                f"feature column order drift between leads. "
                f"baseline={feature_names!r} lead {lead}={feats!r}")

        n = len(df)
        i_tr, i_va = time_split(n)
        X = df[feature_names].to_numpy(dtype="float64")
        Y = df[["u_truth", "v_truth"]].to_numpy(dtype="float64")
        train_idx = slice(0, i_tr)
        val_idx   = slice(i_tr, i_va)
        test_idx  = slice(i_va, n)

        per_lead_data[lead] = {
            "df":   df,
            "X_tr": X[train_idx], "X_va": X[val_idx], "X_te": X[test_idx],
            "Y_tr": Y[train_idx], "Y_va": Y[val_idx], "Y_te": Y[test_idx],
            "n_train": i_tr, "n_val": i_va - i_tr, "n_test": n - i_va,
            "valid_train_range": (df.iloc[0]["ValidTimeUtc"],
                                   df.iloc[i_tr - 1]["ValidTimeUtc"]),
            "valid_val_range":   (df.iloc[i_tr]["ValidTimeUtc"],
                                   df.iloc[i_va - 1]["ValidTimeUtc"]),
            "valid_test_range":  (df.iloc[i_va]["ValidTimeUtc"],
                                   df.iloc[n - 1]["ValidTimeUtc"]),
        }
        rows_train_total += i_tr
        rows_val_total   += (i_va - i_tr)
        rows_test_total  += (n - i_va)
        if first_lead_train_features is None and i_tr > 0:
            first_lead_train_features = X[train_idx].astype("float32")
        log.info("  lead %dh: aligned %d rows -> train %d / val %d / test %d",
                 lead, n, i_tr, i_va - i_tr, n - i_va)

    if feature_names is None or not per_lead_data:
        log.error("No training rows for any lead — abort.")
        return 3

    log.info("Feature count: %d (locked production scope)", len(feature_names))

    # 2) Per-lead train + calibrate.
    per_lead_out: dict[int, dict] = {}
    for lead, d in per_lead_data.items():
        log.info("[%s] %s lead %dh — train", time.strftime("%H:%M:%S"), location, lead)
        X_tr, X_va, X_te, medians, mu, scale, keep = standardise(
            d["X_tr"], d["X_va"], d["X_te"])
        Y_tr = d["Y_tr"]; Y_va = d["Y_va"]; Y_te = d["Y_te"]
        net, val_nll, best_epoch = train_one_lead(X_tr, X_va, Y_tr, Y_va)
        log.info("  best val_nll=%.3f at epoch %d", val_nll, best_epoch)

        # Per-lead val MVN params for α' fit, and test predictions.
        net.eval()
        with torch.no_grad():
            mu_v, sig_v, rho_v = net(torch.tensor(X_va, dtype=torch.float32))
            mu_t, sig_t, rho_t = net(torch.tensor(X_te, dtype=torch.float32))
        mu_u_va,  mu_v_va  = mu_v[:, 0].numpy(),  mu_v[:, 1].numpy()
        sig_u_va, sig_v_va = sig_v[:, 0].numpy(), sig_v[:, 1].numpy()
        rho_va             = rho_v.squeeze(-1).numpy()
        mu_u_te,  mu_v_te  = mu_t[:, 0].numpy(),  mu_t[:, 1].numpy()
        sig_u_te, sig_v_te = sig_t[:, 0].numpy(), sig_t[:, 1].numpy()
        rho_te             = rho_t.squeeze(-1).numpy()

        df_va = d["df"].iloc[d["n_train"]:d["n_train"] + d["n_val"]]
        df_te = d["df"].iloc[d["n_train"] + d["n_val"]:]
        truth_spd_va = df_va["wsp_ms"].values
        truth_dir_va = df_va["wdir_deg"].values
        truth_spd_te = df_te["wsp_ms"].values
        truth_dir_te = df_te["wdir_deg"].values

        # α' calibration on val.
        alpha_dir, alpha_spd, calib_diag = calibrate_alpha_prime(
            mu_u_va, mu_v_va, sig_u_va, sig_v_va, rho_va,
            truth_spd_va, truth_dir_va,
        )
        log.info("  calibration: α'_dir=%.3f α'_spd=%.3f  val dir cov=%.3f spd cov=%.3f",
                 alpha_dir, alpha_spd,
                 calib_diag["val_dir_cov_after_alpha_prime_dir"],
                 calib_diag["val_spd_cov_after_alpha_prime_spd"])

        # Blend calibration (no LGB val preds yet — falls back to plan defaults).
        mvn_va_speed = np.sqrt(mu_u_va ** 2 + mu_v_va ** 2)
        blend_c, blend_s, blend_diag = calibrate_blend(
            lgb_val=None,  # wind_speed_lgb not yet integrated; TODO when shipped
            mvn_val_speed=mvn_va_speed,
            truth_val_speed=truth_spd_va,
        )

        # Test point metrics.
        pt_speed_te = np.sqrt(mu_u_te ** 2 + mu_v_te ** 2)
        pt_dir_te   = (np.degrees(np.arctan2(-mu_u_te, -mu_v_te))) % 360.0
        spd_mae_te = float(np.mean(np.abs(pt_speed_te - truth_spd_te)))
        dir_mae_te = float(np.mean(circ_diff(pt_dir_te, truth_dir_te)))
        log.info("  test point: direction MAE = %.3f°,  speed MAE = %.4f m/s",
                 dir_mae_te, spd_mae_te)

        # Persist per-row test predictions for verify.
        test_predictions = pd.DataFrame({
            "ValidTimeUtc": df_te["ValidTimeUtc"].values,
            "Lead":         lead,
            "mu_u":         mu_u_te,
            "mu_v":         mu_v_te,
            "sigma_u":      sig_u_te,
            "sigma_v":      sig_v_te,
            "rho":          rho_te,
            "truth_u":      df_te["u_truth"].values,
            "truth_v":      df_te["v_truth"].values,
            "truth_speed":  truth_spd_te,
            "truth_dir":    truth_dir_te,
        })

        per_lead_out[lead] = {
            "state_dict":   net.state_dict(),
            "medians":      medians, "mu": mu, "scale": scale, "keep": keep,
            "alpha_dir":    alpha_dir, "alpha_spd": alpha_spd,
            "blend_center": blend_c, "blend_scale": blend_s,
            "calib_diag":   {**calib_diag, **blend_diag},
            "val_nll":      val_nll,
            "n_train":      d["n_train"], "n_val": d["n_val"], "n_test": d["n_test"],
            "test_dir_mae_deg": dir_mae_te,
            "test_spd_mae_ms":  spd_mae_te,
            "range_train":  f"{d['valid_train_range'][0]} -> {d['valid_train_range'][1]}",
            "range_val":    f"{d['valid_val_range'][0]} -> {d['valid_val_range'][1]}",
            "range_test":   f"{d['valid_test_range'][0]} -> {d['valid_test_range'][1]}",
            "test_calendar_months": int(
                df_te["ValidTimeUtc"].dt.to_period("M").nunique() if d["n_test"] else 0
            ),
            "test_predictions": test_predictions,
        }

    # 3) RetrainGuard before any manifest write.
    bundle_dir = models_root / TARGET / location / f"{version}_wind_mvn"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    guard = build_check_and_save_versioned(
        log,
        version_dir=bundle_dir,
        composite=f"{TARGET}/{location}",
        phase=PHASE,
        version=version,
        rows_train=rows_train_total,
        rows_val=rows_val_total,
        rows_test=rows_test_total,
        train_features=(first_lead_train_features
                        if first_lead_train_features is not None
                        else np.zeros((0, len(feature_names)), dtype="float32")),
        feature_names=feature_names,
        label_rates={location: 1.0},  # MLP is regression, no label-rate
        location_name=location,
    )
    if not guard.passed:
        log.error("Retrain guard FAIL — bundle not promoted.")
        return 4

    # 4) Write bundle + promote.
    write_bundle(bundle_dir, location, version, per_lead_out, feature_names)
    log.info("Bundle → %s", bundle_dir)
    promote_station_version(
        models_root=models_root,
        target=TARGET,
        station_slug=location,
        version=f"{version}_wind_mvn",
        phase_tag="wind_mvn",
        role="champion",
    )
    log.info("Promoted into %s/MANIFEST.json under station=%s",
             TARGET, location)
    return 0


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--location", default=LOCATION_DEFAULT,
                         help="Location slug. Default: bonehill_rocks.")
    parser.add_argument("--leads", type=int, nargs="*", default=list(LEADS),
                         help=f"Lead-hour set. Default: {LEADS}.")
    parser.add_argument("--models-root",
                         default=str(WEATHERBLEND_DATA_ROOT / "models"),
                         help="Models tree root for bundle output.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    location = args.location
    leads = tuple(args.leads)
    models_root = Path(args.models_root)
    version = datetime.now(timezone.utc).strftime("v%Y-%m-%d_%H%M%S")
    log.info("Phase 2 wind_mvn — TRAIN")
    log.info("  location:    %s", location)
    log.info("  leads:       %s", leads)
    log.info("  version:     %s_wind_mvn", version)
    log.info("  models_root: %s", models_root)

    rc = train_one_location(location, version, models_root, leads=leads)
    sys.exit(rc)


if __name__ == "__main__":
    main()
