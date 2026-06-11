"""Shared wind_mvn model definition + feature helpers (train AND predict).

WHY THIS MODULE EXISTS: predict_wind_mvn loads the trainer's per-lead
state_dicts with ``net.load_state_dict(...)``, which only round-trips when
the architecture matches BIT-FOR-BIT. Until 2026-06-10 the WindMVNHead
class — plus the sector/orographic helpers, the MC samplers and the
NWP/feature-order constants it depends on — was defined TWICE, once in
``train_wind_mvn.py`` and once in ``predict_wind_mvn.py``. That is exactly
how weights silently mis-load: a hyperparameter tweak on the train side
(hidden width, dropout, head shape) leaves the predict-side copy stale and
torch either errors or, worse, partially loads. One definition imported by
both sides removes the failure mode.

The constants here are the ones both sides MUST agree on: the scaler
params in ``feature_scaler.json`` are positional over the feature column
order derived from NWPS + ORO_LEAN + SPREAD_VARS, so train/predict drift
corrupts predictions without raising.

Import-weight note: this module pulls torch + torch.nn only (no
torch.optim, no duckdb) so the predict scripts stay as slim as their old
local copies were — the original reason the helpers were copied rather
than imported from the train script.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Module-object import (NOT `from src.data import WEATHERBLEND_DATA_ROOT`):
# the smoke tests repoint the data root via env + importlib.reload of
# src.data WITHOUT reloading this module, so the root must be read off the
# (reloaded-in-place) module object at call time, not copied at import.
from src import data as _wb_data  # noqa: E402


# ----------------------------------------------------------------------------
# Constants both sides must agree on
# ----------------------------------------------------------------------------

# Trained lead set — bundles carry one state_lead_{L}h.pt per lead.
LEADS = (24, 48, 72)

# Production NWPs from the 2026-05-27 wind_speed_lgb decision — 6 NWPs.
# MF excluded (0 wsp rows on Open-Meteo Previous Runs).
# AIFS excluded — no 2024 rows.
# HARMONIE confirmed dead weight for wind.
NWPS = (
    "gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless",
    "ukmo_seamless", "jma_seamless",
)

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

# MLP architecture shape — these two define what the state_dict must match.
# The rest of the training hyperparams (LR, patience, ...) live in
# train_wind_mvn.py; predict never needs them.
N_HIDDEN = 64
DROPOUT = 0.10

N_MC = 500  # MC samples per row for the CI draws
SEED = 42


# ----------------------------------------------------------------------------
# MLP definition
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
# Orographic static + dynamic features
# ----------------------------------------------------------------------------

def load_oro_static(location: str) -> dict:
    path = (_wb_data.WEATHERBLEND_DATA_ROOT / "static" / "orographic"
            / f"{location}.json")
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
# MC sampling + circular CI helpers
# ----------------------------------------------------------------------------

def mc_speed_dir(mu_u: np.ndarray, mu_v: np.ndarray,
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
