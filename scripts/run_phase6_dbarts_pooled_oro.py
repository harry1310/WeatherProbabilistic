"""4a-oro Step 2 — pooled-with-terrain BART bake-off.

Step 2 of memory/project_4a_oro_bakeoff_plan.md. Step 1 (dynamic-only per-cell)
came back essentially flat (aggregate +0.05% Brier across 3 gauges at lead 24h).
Step 2 mirrors 3o's architecture — one BART per lead trained on a pool of
4 Bonehill EA gauges (Bellever + Bovey + Hexworthy + Princetown) with the
full 9-feature terrain block (3 static + 5 dynamic + oro_station_id).

Two pooled variants in this run:

  lean-pooled-terrain   — 22 lean + 3 syn (implicit via the dump) + 9 oro = 31 feats
                          NB: the dump emits 22 + 9 = 31 (no syn block in the dump).
                          For apples-to-apples vs the Step 1 baseline we add the
                          3 syn features here via add_synoptic_features.
  rich-pooled-terrain   — 55 rich + 9 oro = 64 feats (full 3o mirror).
                          The rich-oro dump already includes everything; no syn add.

Input is the WB-side feature dump at WB/data/scratch/oro_dump/:
  {station_slug}_lead{lead}h_{lean-oro|rich-oro}.parquet      — rows
  {station_slug}_lead{lead}h_{lean-oro|rich-oro}.schema.json  — feature names

Pooling strategy:
  * time_split per-station (70/15/15) to mirror the per-cell baseline's row alignment.
  * Concat train slices across 4 stations → ~56k pooled train rows per lead.
  * Concat val slices similarly (used as nothing here — BART has no val callback,
    but kept for parity / future calibration step).
  * Test slices STAY per-station — score each station against the same pooled
    BART so the per-cell Brier comparison vs the per-cell baseline 4a is direct.

Output: WP/reports/pooled_oro_4a_bakeoff_{date}/
  test_predictions.parquet      per-row (valid_time, station, lead, variant, p_wet, observed_wet)
  per_cell_brier.csv            per-(station, lead, variant) Brier + BSS + n_test
  summary.txt                   side-by-side aggregate, comparison vs per-cell baseline filled in later

The comparison vs production 4a + Princetown per-cell baseline is a separate
post-processing step (build_step2_comparison.py — TBD).

Usage:
    # smoke (1 lead × 2 variants = 2 fits, ~30-50 min)
    python scripts/run_phase6_dbarts_pooled_oro.py --smoke

    # full sweep (5 leads × 2 variants = 10 fits, ~3-4h)
    python scripts/run_phase6_dbarts_pooled_oro.py
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

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402
from _shared import add_synoptic_features, time_split  # noqa: E402

_RCONVERT = default_converter + numpy2ri.converter + pandas2ri.converter
ro.r(f'.libPaths(c("{_user_lib.replace(os.sep, "/")}", .libPaths()))')
dbarts = importr("dbarts")

NTREE = int(os.environ.get("WB_BART_NTREE", "500"))
K = float(os.environ.get("WB_BART_K", "3.0"))
NSKIP = int(os.environ.get("WB_BART_NSKIP", "200"))
NDPOST = int(os.environ.get("WB_BART_NDPOST", "1000"))
SEED = int(os.environ.get("WB_BART_SEED", "42"))

_stations_env = os.environ.get("WB_STATIONS", "").strip()
STATIONS = ([s.strip() for s in _stations_env.split(",") if s.strip()]
            if _stations_env
            else ["ea_bellever_dartmoor", "ea_dartmoor_nr_hexworthy",
                  "ea_bovey_tracey", "ea_princetown"])
LEADS = [24, 48, 72, 96, 120]
VARIANTS = ["lean-pooled-terrain", "rich-pooled-terrain", "rich-pooled-terrain-v2",
            "rich-pooled-terrain-dynlee", "rich-pooled-terrain-v3"]

DUMP_ROOT = WEATHERBLEND_DATA_ROOT / "scratch" / "oro_dump"
ORO_STATIC_ROOT = WEATHERBLEND_DATA_ROOT / "static" / "orographic"

# v2 DEM aggregations — mirrors PrecipRichOroV2FeatureBuilder.ComposeV2TerrainBlock.
# 14 static-per-station features: 4 TPI radii + 8 sector lee obstructions +
# mean slope + aspect dominance. Pure constants per site, no NWP dependency,
# so we broadcast each station's values across every row of its dump.
V2_FEAT_NAMES = [
    "oro_tpi_200m", "oro_tpi_1000m", "oro_tpi_5000m", "oro_tpi_25000m",
    "oro_lee_obstr_n", "oro_lee_obstr_ne", "oro_lee_obstr_e", "oro_lee_obstr_se",
    "oro_lee_obstr_s", "oro_lee_obstr_sw", "oro_lee_obstr_w", "oro_lee_obstr_nw",
    "oro_mean_slope_5km", "oro_aspect_dominance_5km",
]


def load_v2_block(station_slug: str) -> dict[str, float]:
    """Load the 14 static v2 DEM features from data/static/orographic/{slug}.json.
    Missing fields default to 0.0 (matches the C# builder's Get() helper)."""
    rec = json.loads((ORO_STATIC_ROOT / f"{station_slug}.json").read_text())
    lee_dict = rec.get("lee_obstruction_10km", {}) or {}
    values = [
        rec.get("tpi_200m", 0.0), rec.get("tpi_1000m", 0.0),
        rec.get("tpi_5000m", 0.0), rec.get("tpi_25000m", 0.0),
        *(lee_dict.get(s, 0.0) for s in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        rec.get("mean_slope_5km", 0.0),
        rec.get("aspect_dominance_5km", 0.0),
    ]
    return dict(zip(V2_FEAT_NAMES, values))


_SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _wind_sector_idx(sin_v: float, cos_v: float) -> int | None:
    """Mirror PrecipRichOroFeatureBuilder.UpwindGainAt — nearest 45-deg sector
    from NWP-mean wind sin/cos. Returns None if either input is NaN (caller
    treats as 'no orographic context')."""
    if np.isnan(sin_v) or np.isnan(cos_v):
        return None
    deg = (np.degrees(np.arctan2(sin_v, cos_v)) + 360.0) % 360.0
    return int(((deg + 22.5) // 45.0)) % 8


def compose_dynamic_lee(station_slug: str, wind_sin: np.ndarray, wind_cos: np.ndarray) -> np.ndarray:
    """Wind-direction-picked lee obstruction at 10 km. Mirror of v1's
    upwind_gain_per_wind_5km_m pattern — pick the sector matching this row's
    NWP-mean wind direction from the 8 stored values."""
    rec = json.loads((ORO_STATIC_ROOT / f"{station_slug}.json").read_text())
    lee_dict = rec.get("lee_obstruction_10km", {}) or {}
    sector_vals = np.array([float(lee_dict.get(s, 0.0)) for s in _SECTORS], dtype=np.float32)
    out = np.zeros(len(wind_sin), dtype=np.float32)
    for i in range(len(wind_sin)):
        idx = _wind_sector_idx(wind_sin[i], wind_cos[i])
        out[i] = sector_vals[idx] if idx is not None else 0.0
    return out


_V3_FEAT_NAMES = [
    "climo_lapse_850_500", "climo_lapse_700_500", "climo_q_850",
    "climo_wind_500_speed", "climo_shear_850_500", "climo_thickness_proxy",
]
_V3_JSON_KEYS = ["lapse_850_500", "lapse_700_500", "q_850",
                 "wind_500_speed", "shear_850_500", "thickness_proxy"]


def compose_v3_climatology(
        station_slug: str,
        wind_sin: np.ndarray, wind_cos: np.ndarray,
        valid_time: np.ndarray) -> dict[str, np.ndarray]:
    """Per-(wind sector, month) climatology lookup — mirror of
    PrecipRichOroV3FeatureBuilder.BuildForLead. 6 features per row; NaN where
    the (sector, month) bin has insufficient samples in the offline table."""
    rec = json.loads((ORO_STATIC_ROOT / f"{station_slug}.json").read_text())
    climo = rec.get("climatology_by_sector_month", {}) or {}
    n = len(wind_sin)
    out = {f: np.full(n, np.nan, dtype=np.float32) for f in _V3_FEAT_NAMES}
    months = pd.to_datetime(valid_time).month.values if hasattr(valid_time, 'month') \
             else pd.DatetimeIndex(valid_time).month.values
    for i in range(n):
        idx = _wind_sector_idx(wind_sin[i], wind_cos[i])
        if idx is None:
            continue
        sector = _SECTORS[idx]
        month_str = str(int(months[i]))
        bin_data = climo.get(sector, {}).get(month_str, {})
        if not bin_data:
            continue
        for f, k in zip(_V3_FEAT_NAMES, _V3_JSON_KEYS):
            v = bin_data.get(k)
            if v is not None:
                out[f][i] = v
    return out


# ---------------------------------------------------------------------------
# Dump loader — reads {station}_lead{lead}h_{feature-set}.parquet + sidecar
# and returns a wide DataFrame with named feature columns.
# ---------------------------------------------------------------------------

def load_dump_wide(station_slug: str, lead: int, dump_feature_set: str) -> tuple[pd.DataFrame, list[str], dict]:
    """Load one station's dumped feature parquet + sidecar and return:
      - wide DataFrame with ValidTimeUtc, Label, TruthMmHour, and one column per feature
      - list of feature names (in dump order)
      - sidecar dict (metadata)

    Raises FileNotFoundError if the parquet or sidecar is missing — caller
    should retry after the C# dump finishes.
    """
    stem = f"{station_slug}_lead{lead}h_{dump_feature_set}"
    parquet_path = DUMP_ROOT / f"{stem}.parquet"
    schema_path  = DUMP_ROOT / f"{stem}.schema.json"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dump missing: {parquet_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Dump schema missing: {schema_path}")

    sidecar = json.loads(schema_path.read_text())
    feat_names = list(sidecar["FeatureNames"])
    df_narrow = pd.read_parquet(parquet_path)
    # Sanity: shape matches sidecar.
    if len(df_narrow) != sidecar["NRows"]:
        raise ValueError(f"{stem}: row count mismatch parquet={len(df_narrow)} vs sidecar={sidecar['NRows']}")
    # Expand Features list column → wide. Coerce dtype to float64 for sklearn.
    feat_mat = np.asarray(df_narrow["Features"].tolist(), dtype="float64")
    if feat_mat.shape[1] != len(feat_names):
        raise ValueError(f"{stem}: feature dim mismatch parquet={feat_mat.shape[1]} vs sidecar={len(feat_names)}")
    wide = pd.DataFrame(feat_mat, columns=feat_names, index=df_narrow.index)
    wide.insert(0, "ValidTimeUtc", df_narrow["ValidTimeUtc"].values)
    wide.insert(1, "wet", df_narrow["Label"].astype("int8").values)
    wide.insert(2, "truth_mm_hour", df_narrow["TruthMmHour"].astype("float32").values)
    # Mirror the wet-from-truth convention used in production (>= WET_THRESHOLD_MM).
    # The dumped Label is already the C# builder's boolean so this is redundant
    # but cheap to assert.
    return wide, feat_names, sidecar


def _resolve_friendly(slug: str) -> str:
    """Slug → friendly station name. Needed by add_synoptic_features which
    queries by StationName (friendly), not slug."""
    mapping = {
        "ea_bellever_dartmoor":     "Bellever Dartmoor",
        "ea_dartmoor_nr_hexworthy": "Dartmoor nr Hexworthy",
        "ea_bovey_tracey":          "Bovey Tracey",
        "ea_princetown":            "Princetown",
    }
    if slug not in mapping:
        raise ValueError(f"No friendly-name mapping for {slug}")
    return mapping[slug]


# ---------------------------------------------------------------------------
# Per-lead per-variant: build pooled training matrix + per-station test matrices
# ---------------------------------------------------------------------------

def build_pooled_cell(lead: int, variant: str) -> dict:
    """Load 4 station dumps, add syn features for lean variant, time_split
    per-station, concat train+val, keep test per-station.

    Returns a dict with:
        feat_names_full:  list[str], final feature list including any added syn cols
        X_train_pooled:   (n_train_pooled, n_feats) float64
        y_train_pooled:   (n_train_pooled,) float64
        test_by_station:  {slug: dict with X_test, y_test, valid_time array}
        kept_mask + median + scaler — derived from the POOLED train slice and
                          applied identically to every station's test slice.
    """
    dump_fs = "lean-oro" if variant == "lean-pooled-terrain" else "rich-oro"

    per_station: dict[str, dict] = {}
    # Two reference lists: the raw schema from the parquet (always identical
    # across stations for the same lead × feature-set), and the extended
    # final feature list once any per-variant additions (syn cols for lean)
    # are folded in. Conflating these caused a "drift" error in the first
    # version of this function.
    parquet_feat_names_ref: list[str] | None = None
    feat_names_full: list[str] | None = None
    for slug in STATIONS:
        wide, parquet_feat_names, _sidecar = load_dump_wide(slug, lead, dump_fs)
        if parquet_feat_names_ref is None:
            parquet_feat_names_ref = parquet_feat_names
        elif parquet_feat_names != parquet_feat_names_ref:
            raise ValueError(f"feature-schema drift between stations at lead {lead}h {dump_fs}: "
                             f"{slug} parquet differs from reference")

        extra_names: list[str] = []

        # Lean variant needs the 3 synoptic features that production 4a uses
        # (and the Step 1 baseline used). The lean-oro dump doesn't include
        # them — they live in a separate SQL pass. Pull + merge here.
        if variant == "lean-pooled-terrain":
            wide, syn_added = add_synoptic_features(
                _resolve_friendly(slug), lead, wide,
                min_valid_time=datetime(2024, 1, 1),
            )
            extra_names.extend(c for c in syn_added if c not in parquet_feat_names)

        # v2 variant appends 14 static DEM features per station (constants
        # broadcast to every row). Stacks on top of whichever dump_fs the
        # variant chose — for rich-pooled-terrain-v2 that's rich-oro (68
        # features) + 14 v2 = 82, matching the C# v2 builder spec.
        if variant.endswith("-v2"):
            v2_block = load_v2_block(slug)
            for fname, fval in v2_block.items():
                wide[fname] = float(fval)
            extra_names.extend(V2_FEAT_NAMES)

        # dynlee variant appends 1 dynamic feature — wind-direction-picked
        # lee obstruction. Adds the missing DYNAMIC partner to v1's existing
        # upwind_gain trick; signals "for THIS hour's flow, how much terrain
        # stands between me and the upwind boundary in a 10 km arc?"
        if variant.endswith("-dynlee"):
            wind_sin = wide["oro_wind_sin"].values
            wind_cos = wide["oro_wind_cos"].values
            wide["oro_lee_obstr_per_wind_10km_m"] = compose_dynamic_lee(slug, wind_sin, wind_cos)
            extra_names.append("oro_lee_obstr_per_wind_10km_m")

        # v3 variant appends 6 dynamic climatology features (per-row lookup
        # by NWP wind sector + calendar month). NOT a static block — varies
        # per row because the sector and month do. Test whether v3's signal
        # (which was marginal in pooled bake-off) earns its keep per-station.
        if variant.endswith("-v3"):
            wind_sin = wide["oro_wind_sin"].values
            wind_cos = wide["oro_wind_cos"].values
            v3_block = compose_v3_climatology(slug, wind_sin, wind_cos, wide["ValidTimeUtc"].values)
            for fname, fvals in v3_block.items():
                wide[fname] = fvals
            extra_names.extend(_V3_FEAT_NAMES)

        # Resolve the extended feature list once, from the first station's
        # additions. Subsequent stations are asserted to add the same set.
        if feat_names_full is None:
            feat_names_full = parquet_feat_names + extra_names
        else:
            extended_now = parquet_feat_names + extra_names
            if extended_now != feat_names_full:
                raise ValueError(f"feature-name drift at {slug} lead {lead}h: "
                                 f"{extended_now} vs {feat_names_full}")

        train_df, val_df, test_df = time_split(wide)
        per_station[slug] = {
            "train_df": train_df,
            "val_df":   val_df,
            "test_df":  test_df,
        }

    assert feat_names_full is not None

    # Pool train+val across stations; keep test per-station.
    train_pooled = pd.concat([per_station[s]["train_df"] for s in STATIONS], ignore_index=True)

    X_train_full = train_pooled[feat_names_full].to_numpy(dtype="float64")
    y_train = train_pooled["wet"].to_numpy(dtype="float64")
    # Drop columns all-NaN in the POOLED train slice. With 4 pooled stations
    # this should be strictly looser than any per-station drop — but assert
    # nothing went wrong by logging the kept count.
    col_all_nan = np.isnan(X_train_full).all(axis=0)
    kept = np.where(~col_all_nan)[0]
    X_train_kept = X_train_full[:, kept]
    median = np.nanmedian(X_train_kept, axis=0)
    X_train_kept = np.where(np.isnan(X_train_kept), median, X_train_kept)
    scaler = StandardScaler().fit(X_train_kept)
    X_train_s = scaler.transform(X_train_kept)
    feat_names_eff = [feat_names_full[i] for i in kept]

    test_by_station = {}
    for slug in STATIONS:
        td = per_station[slug]["test_df"]
        X_test_full = td[feat_names_full].to_numpy(dtype="float64")
        X_test_kept = X_test_full[:, kept]
        # Impute using POOLED train median, scale using POOLED train scaler.
        X_test_kept = np.where(np.isnan(X_test_kept), median, X_test_kept)
        X_test_s = scaler.transform(X_test_kept)
        test_by_station[slug] = {
            "X_test_s": X_test_s,
            "y_test":   td["wet"].to_numpy(dtype="int8"),
            "valid_time": pd.to_datetime(td["ValidTimeUtc"].values),
        }

    return {
        "feat_names_full": feat_names_full,
        "feat_names_eff":  feat_names_eff,
        "kept_indices":    kept.tolist(),
        "X_train_s":       X_train_s,
        "y_train":         y_train,
        "test_by_station": test_by_station,
        "n_pooled_train":  int(len(y_train)),
    }


# ---------------------------------------------------------------------------
# BART fit (no per-test scoring — done after fit, against multiple test sets)
# ---------------------------------------------------------------------------

def fit_pooled_bart(cell: dict, seed: int = SEED) -> tuple[dict, float]:
    """Fit one BART on the pooled train set, then score each station's test
    set in turn. Returns {slug: p_test (np.ndarray)} + wall time."""
    X_train = cell["X_train_s"].astype(np.float64)
    y_train = cell["y_train"].astype(np.float64)
    # Concat all test sets into one combined X for prediction, then split back.
    # dbarts.bart needs x_test at fit time (no separate predict-after-fit method
    # without storeState replay). Concat is cheap.
    test_X_blocks = []
    test_sizes = {}
    for slug in STATIONS:
        ts = cell["test_by_station"][slug]
        test_X_blocks.append(ts["X_test_s"])
        test_sizes[slug] = ts["X_test_s"].shape[0]
    X_test_concat = np.concatenate(test_X_blocks, axis=0).astype(np.float64)

    with localconverter(_RCONVERT):
        x_train_r = ro.conversion.py2rpy(X_train)
        y_train_r = ro.conversion.py2rpy(y_train)
        x_test_r  = ro.conversion.py2rpy(X_test_concat)

    t0 = time.time()
    fit = dbarts.bart(
        x_train=x_train_r, y_train=y_train_r, x_test=x_test_r,
        ntree=NTREE, k=K, nskip=NSKIP, ndpost=NDPOST,
        keeptrees=False, verbose=True, seed=seed,
    )
    yhat_test_r = fit.rx2("yhat.test")
    with localconverter(_RCONVERT):
        yhat = np.array(ro.conversion.rpy2py(yhat_test_r))
    p_test_concat = norm.cdf(yhat).mean(axis=0)
    wall = time.time() - t0
    ro.r('rm(list = ls()); gc()')

    # Split p_test_concat back by station in order.
    p_by_station = {}
    offset = 0
    for slug in STATIONS:
        n = test_sizes[slug]
        p_by_station[slug] = p_test_concat[offset:offset + n]
        offset += n
    return p_by_station, wall


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) -
                          np.asarray(y, dtype=np.float64)) ** 2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--leads", nargs="*", type=int, default=None,
                   help=f"Lead subset (default {LEADS}).")
    p.add_argument("--variants", nargs="*", default=None,
                   help=f"Variants (default {VARIANTS}).")
    p.add_argument("--smoke", action="store_true",
                   help="1 lead × both variants = 2 fits.")
    args = p.parse_args()

    if args.smoke:
        leads = [24]
        variants = VARIANTS
    else:
        leads = args.leads or LEADS
        variants = args.variants or VARIANTS

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ROOT / "reports" / f"pooled_oro_4a_bakeoff_{date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Step 2 — pooled-with-terrain BART bake-off")
    print(f"  stations: {STATIONS}")
    print(f"  leads:    {leads}")
    print(f"  variants: {variants}")
    print(f"  fits:     {len(leads) * len(variants)}")
    print(f"  output:   {out_dir}")

    all_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    for lead in leads:
        for variant in variants:
            print(f"\n[{time.strftime('%H:%M:%S')}] lead {lead}h · variant {variant}")
            try:
                cell = build_pooled_cell(lead, variant)
            except FileNotFoundError as e:
                print(f"  ERROR: {e} — has the C# dump finished?", flush=True)
                continue
            except Exception as e:
                print(f"  ERROR building cell: {e}", flush=True)
                continue

            print(f"  pooled train rows: {cell['n_pooled_train']:,}; "
                  f"feats kept {len(cell['feat_names_eff'])} of {len(cell['feat_names_full'])}",
                  flush=True)
            for slug in STATIONS:
                ts = cell["test_by_station"][slug]
                print(f"    test {slug}: {len(ts['y_test']):,} rows", flush=True)

            try:
                p_by_station, wall = fit_pooled_bart(cell)
            except Exception as e:
                print(f"  ERROR fitting BART: {e}", flush=True)
                continue

            print(f"  fit in {wall:.1f}s — per-station Brier:")
            for slug in STATIONS:
                p_test = p_by_station[slug]
                y_test = cell["test_by_station"][slug]["y_test"]
                vt     = cell["test_by_station"][slug]["valid_time"]
                b = brier(p_test, y_test.astype("float64"))
                # Climatology = pooled-train wet rate (single number across pool).
                clim = float(np.mean(cell["y_train"]))
                clim_b = brier(np.full_like(y_test, clim, dtype="float64"),
                               y_test.astype("float64"))
                bss = (clim_b - b) / clim_b if clim_b > 0 else float("nan")
                print(f"    {slug:32s} | n_test {len(y_test):,} | Brier {b:.4f} (BSS {bss:+.4f})",
                      flush=True)
                all_rows.append({
                    "lead":      lead,
                    "variant":   variant,
                    "station":   slug,
                    "n_test":    int(len(y_test)),
                    "brier":     round(b, 4),
                    "bss":       round(bss, 4),
                    "clim_brier": round(clim_b, 4),
                    "n_pooled_train": cell["n_pooled_train"],
                    "n_feats_eff":    len(cell["feat_names_eff"]),
                    "wall_s":    round(wall, 1),
                })
                pred_frames.append(pd.DataFrame({
                    "valid_time":   vt,
                    "station":      slug,
                    "lead":         lead,
                    "variant":      variant,
                    "p_wet":        p_test,
                    "observed_wet": y_test.astype("int8"),
                }))

            # Persist progressively.
            pd.DataFrame(all_rows).to_csv(out_dir / "per_cell_brier.csv", index=False)
            if pred_frames:
                pd.concat(pred_frames, ignore_index=True).to_parquet(
                    out_dir / "test_predictions.parquet", index=False)

    if not all_rows:
        print("\nNo cells completed — aborting.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "per_cell_brier.csv", index=False)
    print()
    print(df.to_string(index=False))

    # Aggregate per variant per lead (mean Brier across 4 stations).
    agg = df.groupby(["lead", "variant"], as_index=False)["brier"].mean().rename(
        columns={"brier": "brier_mean_4stns"})
    text = (
        "Step 2 pooled-with-terrain BART bake-off — raw per-cell Brier\n"
        "=============================================================\n\n"
        f"NTREE={NTREE} K={K} NSKIP={NSKIP} NDPOST={NDPOST} SEED={SEED}\n"
        f"Stations: {STATIONS}\n"
        f"Leads:    {leads}\n"
        f"Variants: {variants}\n\n"
        + df.to_string(index=False)
        + "\n\n--- Aggregate (mean Brier across 4 stations) ---\n"
        + agg.to_string(index=False)
        + "\n\nFor the apples-to-apples comparison vs production 4a + Princetown\n"
          "per-cell baseline, run scripts/build_step2_comparison.py (TBD).\n"
    )
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"Artefacts → {out_dir}")


if __name__ == "__main__":
    main()
