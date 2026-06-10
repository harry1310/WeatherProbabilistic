"""4a-oro bake-off — does the 3o dynamic terrain block lift per-cell BART?

Plan: memory/project_4a_oro_bakeoff_plan.md. Drafted 2026-05-29.

2×2 matrix per (station, lead):

  {baseline 4a, 4a-dyn} × {min_valid_time=2024-01-01, no cutoff}

  baseline 4a   = current production 4a feature set (22 lean + 3 synoptic)
  4a-dyn        = baseline + 5 dynamic terrain features from
                  PrecipRichOroFeatureBuilder.ComposeTerrainBlock:
                    oro_wind_sin, oro_wind_cos,
                    oro_upwind_gain_per_wind_5km_m,
                    oro_uplift_m_per_s, oro_uplift_x_q_g_per_kg.
                  Static terrain features and oro_station_id are constants
                  per cell in a per-station BART so dropped from Step 1
                  (only the dynamic block carries within-cell signal).

3 Bonehill EA gauges × 5 leads × 2 scopes × 2 variants = 60 BART fits.

Scoring strategy:

  * within scope: oro_dyn vs baseline on the same train/test split → the
    oro delta itself.
  * cross-scope baseline: all-data baseline re-scored on the 2024+-only
    subset of its test set, vs the 2024+ baseline on its (already 2024+)
    test set → "does adding the pre-2024 JMA-only rows help?". The
    convention from Harry 2026-05-29 is to prefer 2024+ and only fall
    back to no-cutoff if the cross-scope delta is clearly negative.

Output: WP/reports/oro_4a_bakeoff_{date}/
  bart_oro_cells.csv  one row per (station, lead, scope, variant).
                       Includes brier_aligned_2024plus for cross-scope work.
  summary.txt         within-scope (oro vs baseline) + cross-scope
                       (alldata vs 2024+) tables and aggregate deltas.

Step 2 — pooled-with-terrain 4a-oro — runs only if Step 1 disappoints. See plan.

Usage:
    # smoke (1 cell × 1 scope × 2 variants = 2 fits, ~5-10 min)
    python scripts/run_phase6_dbarts_oro.py --smoke

    # full sweep (60 fits — run via Agent / overnight)
    python scripts/run_phase6_dbarts_oro.py
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

# R env setup — mirrors train_4a.py exactly.
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

import duckdb  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import default_converter, numpy2ri, pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402

from _shared import (  # noqa: E402
    ACTIVE_LOCATION,
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

# Match train_4a.py hyperparameters exactly so the baseline-here is the
# same fit as production 4a (modulo seed-level RNG noise from a separate
# process invocation).
NTREE = int(os.environ.get("WB_BART_NTREE", "500"))
K = float(os.environ.get("WB_BART_K", "3.0"))
NSKIP = int(os.environ.get("WB_BART_NSKIP", "200"))
NDPOST = int(os.environ.get("WB_BART_NDPOST", "1000"))
SEED = int(os.environ.get("WB_BART_SEED", "42"))

STATIONS = ["ea_bellever_dartmoor", "ea_dartmoor_nr_hexworthy", "ea_bovey_tracey"]
LEADS = [24, 48, 72, 96, 120]

ORO_STATIC_ROOT = WEATHERBLEND_DATA_ROOT / "static" / "orographic"

ORO_DYNAMIC_FEATURE_NAMES = [
    "oro_wind_sin",
    "oro_wind_cos",
    "oro_upwind_gain_per_wind_5km_m",
    "oro_uplift_m_per_s",
    "oro_uplift_x_q_g_per_kg",
]

SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

CUTOFF_2024 = datetime(2024, 1, 1)


# ---------------------------------------------------------------------------
# Oro static feature loader (the v1 dynamic-aux subset of OroStaticFeatures)
# ---------------------------------------------------------------------------

def load_oro_static(slug: str) -> dict:
    """Read WB/data/static/orographic/{slug}.json and return the fields the
    dynamic terrain block consumes: upwind_gain_5km dict + terrain_gradient_dx/dy.
    Mirrors the relevant subset of WB's OroStaticFeatures.LoadOne."""
    path = ORO_STATIC_ROOT / f"{slug}.json"
    rec = json.loads(path.read_text())
    return {
        "slug":                slug,
        "upwind_gain_5km":     {s: float(rec["upwind_gain_5km"][s]) for s in SECTORS},
        "terrain_gradient_dx": float(rec["terrain_gradient_dx"]),
        "terrain_gradient_dy": float(rec["terrain_gradient_dy"]),
    }


def upwind_gain_at(oro: dict, wind_dir_rad_from: float) -> float:
    """8-sector nearest-bin lookup. NaN input → 0 (neutral prior).
    Mirrors OroStaticFeatures.UpwindGainAt exactly."""
    if np.isnan(wind_dir_rad_from):
        return 0.0
    deg = wind_dir_rad_from * (180.0 / np.pi)
    deg = ((deg % 360.0) + 360.0) % 360.0
    idx = int(np.floor((deg + 22.5) / 45.0)) % 8
    return oro["upwind_gain_5km"][SECTORS[idx]]


# ---------------------------------------------------------------------------
# NWP-mean aux pull — Python port of PrecipRichOroFeatureBuilder.LoadAuxNwpMeans
# ---------------------------------------------------------------------------

def load_aux_nwp_means(lead_hours: int, min_valid_time=None) -> pd.DataFrame:
    """Per (lead, valid_time) NWP-ensemble-mean wind sin/cos/speed, dewpoint,
    pressure from the offset_day (Open-Meteo Previous Runs) forecast tree.

    Returns: ValidTimeUtc, wind_sin, wind_cos, wind_speed, dew_c, pres_hpa.
    Temperature is intentionally omitted (not used by the dynamic-block formulas).

    Mirrors WB's PrecipRichOroFeatureBuilder.LoadAuxNwpMeans line-for-line
    with min_valid_time forwarded through (same convention as
    _shared.add_synoptic_features)."""
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    model_in_clause = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
    valid_filter = (
        f"          AND ValidTimeUtc >= TIMESTAMP '{min_valid_time.strftime('%Y-%m-%d %H:%M:%S')}'\n"
        if min_valid_time is not None else ""
    )
    sql = f"""
    WITH latest AS (
        SELECT
            ValidTimeUtc, Model,
            WindDirection10m, WindSpeed10m, DewPoint2m, SurfacePressure,
            ROW_NUMBER() OVER (
                PARTITION BY ValidTimeUtc, Model
                ORDER BY RunTimeUtc DESC
            ) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{ACTIVE_LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours = {lead_hours}
          AND Model IN {model_in_clause}
{valid_filter}    )
    SELECT
        ValidTimeUtc,
        AVG(SIN(RADIANS(WindDirection10m))) AS wind_sin,
        AVG(COS(RADIANS(WindDirection10m))) AS wind_cos,
        AVG(WindSpeed10m)    AS wind_speed,
        AVG(DewPoint2m)      AS dew_c,
        AVG(SurfacePressure) AS pres_hpa
    FROM latest
    WHERE rn = 1
    GROUP BY ValidTimeUtc
    ORDER BY ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()
    return df


# ---------------------------------------------------------------------------
# Dynamic terrain block — vectorised port of WB's ComposeTerrainBlock
# (only the 5 dynamic features; static block + oro_station_id are constants
# per cell and add zero discriminative signal to a per-station BART)
# ---------------------------------------------------------------------------

def compose_dynamic_terrain(df_aux: pd.DataFrame, oro: dict) -> pd.DataFrame:
    """Vectorised port of the 5 dynamic features in
    PrecipRichOroFeatureBuilder.ComposeTerrainBlock. df_aux is the output of
    load_aux_nwp_means. Returns ValidTimeUtc + the 5 oro_* columns."""
    wind_sin   = df_aux["wind_sin"].to_numpy(dtype="float64")
    wind_cos   = df_aux["wind_cos"].to_numpy(dtype="float64")
    wind_speed = df_aux["wind_speed"].to_numpy(dtype="float64")
    dew_c      = df_aux["dew_c"].to_numpy(dtype="float64")
    pres_hpa   = df_aux["pres_hpa"].to_numpy(dtype="float64")

    # Mean wind direction "from" via atan2 of mean sin/cos (matches the C# side).
    wind_dir_rad = np.where(
        np.isnan(wind_sin) | np.isnan(wind_cos),
        np.nan,
        np.arctan2(wind_sin, wind_cos),
    )
    upwind_gain = np.array([upwind_gain_at(oro, d) for d in wind_dir_rad], dtype="float64")

    # u_east = -ws · sin(dir_from); v_north = -ws · cos(dir_from)
    has_wind = ~(np.isnan(wind_speed) | np.isnan(wind_sin) | np.isnan(wind_cos))
    u_east  = np.where(has_wind, -wind_speed * wind_sin, 0.0)
    v_north = np.where(has_wind, -wind_speed * wind_cos, 0.0)
    w = u_east * oro["terrain_gradient_dx"] + v_north * oro["terrain_gradient_dy"]
    uplift = np.where(has_wind, np.maximum(0.0, w), 0.0)

    # Specific humidity q (g/kg) from Magnus saturation vapor pressure at
    # dewpoint, divided by surface pressure. NaN-safe: any missing input
    # collapses q to 0 (no humidity contribution) — matches C# convention.
    has_q  = ~(np.isnan(dew_c) | np.isnan(pres_hpa)) & (pres_hpa > 0)
    e_hpa  = np.where(has_q, 6.112 * np.exp(17.62 * dew_c / (dew_c + 243.12)), 0.0)
    q_kgkg = np.where(has_q, 0.622 * e_hpa / (pres_hpa - 0.378 * e_hpa), 0.0)
    q_gkg  = np.maximum(0.0, q_kgkg * 1000.0)

    return pd.DataFrame({
        "ValidTimeUtc":                   df_aux["ValidTimeUtc"].values,
        "oro_wind_sin":                   np.where(np.isnan(wind_sin), 0.0, wind_sin),
        "oro_wind_cos":                   np.where(np.isnan(wind_cos), 0.0, wind_cos),
        "oro_upwind_gain_per_wind_5km_m": upwind_gain,
        "oro_uplift_m_per_s":             uplift,
        "oro_uplift_x_q_g_per_kg":        uplift * q_gkg,
    })


# ---------------------------------------------------------------------------
# Per-cell prep + BART fit — same shape as run_phase6_bart_9cell.py
# ---------------------------------------------------------------------------

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) -
                          np.asarray(y, dtype=np.float64)) ** 2))


def prepare_cell(df: pd.DataFrame, feature_list: list[str]) -> dict:
    """70/15/15 time-split, drop all-NaN columns, NaN→median impute,
    StandardScaler — same recipe as train_4a._prepare_cell."""
    train_df, val_df, test_df = time_split(df)
    X_train_full = train_df[feature_list].to_numpy(dtype="float64")
    y_train = train_df["wet"].to_numpy(dtype="float64")
    X_test_full = test_df[feature_list].to_numpy(dtype="float64")
    y_test = test_df["wet"].to_numpy(dtype="int8")
    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept = np.where(~col_all_nan)[0]
    X_train = X_train_full[:, kept]
    X_test  = X_test_full[:, kept]
    median = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), median, X_train)
    X_test  = np.where(np.isnan(X_test), median, X_test)
    scaler = StandardScaler().fit(X_train)
    return {
        "X_train_s":   scaler.transform(X_train),
        "y_train":     y_train,
        "X_test_s":    scaler.transform(X_test),
        "y_test":      y_test,
        "test_df":     test_df,
        "n_feats_eff": int(len(kept)),
    }


def fit_bart(cell: dict, seed: int = SEED) -> tuple[np.ndarray, float]:
    """Fit a single dbarts BART, return per-test probit-mean P(wet) + wall time."""
    X_train = cell["X_train_s"].astype(np.float64)
    y_train = cell["y_train"].astype(np.float64)
    X_test  = cell["X_test_s"].astype(np.float64)
    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train)
        y_train_r = ro.conversion.py2rpy(y_train)
        x_test_r  = ro.conversion.py2rpy(X_test)
    t0 = time.time()
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_test_r,
        ntree=NTREE, k=K, nskip=NSKIP, ndpost=NDPOST,
        keeptrees=False, verbose=False, seed=seed,
    )
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat = np.array(ro.conversion.rpy2py(yhat_test_r))
    p_test = norm.cdf(yhat).mean(axis=0)
    # Free R-side fit immediately — peak RAM matters at 60 fits in a row.
    ro.r('gc()')
    return p_test, time.time() - t0


# ---------------------------------------------------------------------------
# Per (station, lead, scope): build features once, run 2 BART fits
# (baseline + 4a-dyn). Returns 2 rows per cell.
# ---------------------------------------------------------------------------

def run_cell(station_friendly: str, station_slug: str, lead: int,
             min_valid_time, scope_label: str, oro: dict) -> list[dict]:
    print(f"  [{time.strftime('%H:%M:%S')}] building features (scope={scope_label})", flush=True)
    df = build_features_via_duckdb(station_friendly, lead, min_valid_time=min_valid_time)
    if len(df) < 1000:
        print(f"  WARN: only {len(df)} rows — skipping cell", flush=True)
        return []
    df, syn_feats = add_synoptic_features(station_friendly, lead, df,
                                          min_valid_time=min_valid_time)
    base_feats = list(FEATURE_NAMES) + syn_feats

    aux = load_aux_nwp_means(lead, min_valid_time=min_valid_time)
    oro_block = compose_dynamic_terrain(aux, oro)
    df_oro = df.merge(oro_block, on="ValidTimeUtc", how="left")
    oro_feats = base_feats + ORO_DYNAMIC_FEATURE_NAMES

    out: list[dict] = []
    for variant_label, feats, source_df in [
        ("baseline", base_feats, df),
        ("oro_dyn",  oro_feats,  df_oro),
    ]:
        cell = prepare_cell(source_df, feats)
        clim = float(np.mean(cell["y_train"]))
        clim_brier = brier(np.full_like(cell["y_test"], clim, dtype="float64"),
                           cell["y_test"].astype("float64"))
        p_test, wall = fit_bart(cell)
        b = brier(p_test, cell["y_test"].astype("float64"))
        bss = (clim_brier - b) / clim_brier if clim_brier > 0 else float("nan")
        # Cross-scope alignment: also score on the 2024+-only subset of this
        # test set. For the 2024+ variant this is just the full test set; for
        # alldata it's the relevant subset for apples-to-apples comparison.
        test_times = pd.to_datetime(cell["test_df"]["ValidTimeUtc"].values)
        mask_2024 = test_times >= np.datetime64(CUTOFF_2024)
        n_test_2024 = int(mask_2024.sum())
        b_aligned = (brier(p_test[mask_2024], cell["y_test"].astype("float64")[mask_2024])
                     if n_test_2024 > 0 else float("nan"))
        print(f"    {variant_label:8s} | feats {cell['n_feats_eff']:2d} | "
              f"train {len(cell['y_train']):,} | test {len(cell['y_test']):,} | "
              f"Brier {b:.4f} (BSS {bss:+.4f}) | aligned {b_aligned:.4f} "
              f"({n_test_2024:,} rows) | {wall:.1f}s",
              flush=True)
        out.append({
            "station":                station_slug,
            "lead":                   lead,
            "scope":                  scope_label,
            "variant":                variant_label,
            "n_train":                int(len(cell["y_train"])),
            "n_test":                 int(len(cell["y_test"])),
            "n_test_2024plus":        n_test_2024,
            "n_feats_eff":            cell["n_feats_eff"],
            "brier":                  round(b, 4),
            "bss":                    round(bss, 4),
            "brier_aligned_2024plus": round(b_aligned, 4) if not np.isnan(b_aligned) else None,
            "clim_brier":             round(clim_brier, 4),
            "wall_s":                 round(wall, 1),
        })
    return out


# ---------------------------------------------------------------------------
# Summarisation: within-scope (oro vs baseline) + cross-scope baseline delta
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame) -> str:
    lines = [
        "4a-oro bake-off summary",
        "=======================",
        "",
        f"NTREE={NTREE} K={K} NSKIP={NSKIP} NDPOST={NDPOST} SEED={SEED}",
        f"Stations: {STATIONS}",
        f"Leads:    {LEADS}",
        "",
        "Negative delta = oro / extra data wins.",
        "",
    ]

    for scope in ("2024plus", "alldata"):
        lines.append(f"--- Within scope: {scope} (oro_dyn vs baseline) ---")
        sub = df[df.scope == scope]
        if sub.empty:
            lines.append("  (no rows)\n")
            continue
        pivot = sub.pivot_table(index=["station", "lead"], columns="variant",
                                values="brier", aggfunc="first")
        if "baseline" in pivot and "oro_dyn" in pivot:
            pivot["delta"] = pivot["oro_dyn"] - pivot["baseline"]
            pivot["delta_pct"] = (pivot["delta"] / pivot["baseline"]) * 100
        lines.append(pivot.to_string())
        if "delta" in pivot:
            lines.append("")
            lines.append(f"  Aggregate Δ Brier ({scope}): "
                         f"{pivot['delta'].mean():+.4f} "
                         f"({pivot['delta_pct'].mean():+.2f}%)")
        lines.append("")

    lines.append("--- Cross-scope baseline (alldata vs 2024+, aligned to 2024+ test rows) ---")
    base = df[df.variant == "baseline"]
    if not base.empty:
        rows = []
        for (st, ld), g in base.groupby(["station", "lead"]):
            row: dict = {"station": st, "lead": ld}
            r_2024 = g[g.scope == "2024plus"]
            r_all  = g[g.scope == "alldata"]
            if not r_2024.empty:
                row["baseline_2024plus_brier"] = float(r_2024.brier.iloc[0])
                row["n_test_2024plus"] = int(r_2024.n_test.iloc[0])
            if not r_all.empty:
                row["baseline_alldata_brier_aligned"] = r_all.brier_aligned_2024plus.iloc[0]
                row["n_test_aligned"] = int(r_all.n_test_2024plus.iloc[0])
            rows.append(row)
        cs = pd.DataFrame(rows)
        if {"baseline_2024plus_brier", "baseline_alldata_brier_aligned"}.issubset(cs.columns):
            cs["delta"] = (cs["baseline_alldata_brier_aligned"]
                           - cs["baseline_2024plus_brier"])
            cs["delta_pct"] = cs["delta"] / cs["baseline_2024plus_brier"] * 100
        lines.append(cs.to_string(index=False))
        if "delta" in cs.columns:
            lines.append("")
            lines.append(f"  Aggregate Δ Brier (alldata vs 2024+, aligned): "
                         f"{cs['delta'].mean():+.4f} ({cs['delta_pct'].mean():+.2f}%)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--stations", nargs="*", default=None,
                   help="Station subset (default: all 3 Bonehill gauges).")
    p.add_argument("--leads", nargs="*", type=int, default=None,
                   help=f"Lead subset (default: {LEADS}).")
    p.add_argument("--scopes", nargs="*", default=None,
                   help="Scope subset: 2024plus alldata (default: both).")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke: ea_bellever_dartmoor lead 24h scope 2024plus only (2 fits).")
    args = p.parse_args()

    if args.smoke:
        stations = ["ea_bellever_dartmoor"]
        leads = [24]
        scopes_req = ["2024plus"]
    else:
        stations = args.stations or STATIONS
        leads = args.leads or LEADS
        scopes_req = args.scopes or ["2024plus", "alldata"]

    scope_cutoffs = {"2024plus": CUTOFF_2024, "alldata": None}

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ROOT / "reports" / f"oro_4a_bakeoff_{date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_fits = len(stations) * len(leads) * len(scopes_req) * 2
    print(f"[{time.strftime('%H:%M:%S')}] 4a-oro bake-off (Step 1: dynamic-only, per-cell)")
    print(f"  stations: {stations}")
    print(f"  leads:    {leads}")
    print(f"  scopes:   {scopes_req}")
    print(f"  fits:     {n_fits}")
    print(f"  output:   {out_dir}")

    oro_by_slug = {slug: load_oro_static(slug) for slug in stations}

    rows: list[dict] = []
    for station_slug in stations:
        slug, friendly = resolve_station(station_slug)
        oro = oro_by_slug[station_slug]
        for lead in leads:
            for scope in scopes_req:
                print(f"\n[{time.strftime('%H:%M:%S')}] {friendly} · lead {lead}h · scope {scope}")
                try:
                    cell_rows = run_cell(friendly, slug, lead, scope_cutoffs[scope], scope, oro)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    continue
                rows.extend(cell_rows)
                # Persist progressively — a crash mid-run leaves partial results
                # for diagnosis.
                pd.DataFrame(rows).to_csv(out_dir / "bart_oro_cells.csv", index=False)

    if not rows:
        print("\nNo cells completed — aborting summary.")
        sys.exit(1)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "bart_oro_cells.csv", index=False)
    print()
    print(summary_df.to_string(index=False))

    text = summarise(summary_df)
    # encoding=utf-8 required — summarise() emits the Δ glyph and Windows'
    # default cp1252 codec raises UnicodeEncodeError otherwise.
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"\nArtefacts → {out_dir}")


if __name__ == "__main__":
    main()
