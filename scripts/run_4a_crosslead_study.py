"""4a cross-lead study — best model / equal-weight blend per 6h lead band.

Mirrors the 3c/3o per-lead policy methodology (WB docs/PRECIP_LEAD_POLICY_PLAN.md,
WB PrecipCrossLeadBakeoffCommand) for the per-cell BART 4a (Harry, 2026-06-10):
"do a full study of 4a across lead hours, like we did with 3c/o … what is the
best combination of 4a models to use on each lead bucket, 6-hourly chunks."

Stages (run with --stage retrain | score | both):

  retrain  Walk-forward STUDY bundles: per-cell BART per (station, lead in
           {24,48,72,96,120}) trained on offset_day rows with ValidTimeUtc ≤
           --cutoff (default 2026-03-15, matching the 3c/3o study), written to
           data/models_study/precipitation/{slug}/vstudy_phase4a/ in the exact
           production bundle layout (state.rds + arrays.npz + preprocess.json)
           so the scorer reuses predict_4a's load/impute/scale path verbatim.
           ~15 BART fits — hours; run overnight.

  score    For each τ in 12..117 step 3h: build LIVE-input features at lead τ
           (freshest live run with RunTimeUtc ≤ ValidTimeUtc − τ — the same
           cycle-selection rule the C# harness used, so results are directly
           comparable to the 3c/3o tables), predict EVERY candidate lead model
           on the same rows, join EA truth, accumulate Brier for the 5 singles
           + 10 equal-weight pairs. SELECT (< --split) picks per 6h band,
           SCORE (≥ --split) grades — margin/χ summary mirrors fit-lead-policy.
           Emits reports/crosslead_4a_study/{per_tau.csv, bands.md}.

Decision rules reported (same thresholds as LEAD_POLICY.json): margin 0.75%
vs the production bucket model, blends must also beat the best single by 0.5%.
NOTHING is productionised by this script — it reports; encoding any 4a policy
(and lead-12 predict) is a separate step after Harry reviews.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Windows redirected-output default is cp1252, which can't encode the arrows
# in progress prints — reconfigure before anything prints (same shim as
# predict_wind_speed_pi.py).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import WEATHERBLEND_DATA_ROOT, stations_for_location  # noqa: E402
from _shared import (  # noqa: E402
    ACTIVE_LOCATION, MODELS_RICH, RICH_ORO_FEATURE_NAMES, WET_THRESHOLD_MM,
    build_rich_features_via_duckdb, compose_v1_terrain_block, resolve_station,
    _load_hourly_rain, _compute_persistence, _load_oro_static, _upwind_gain_at,
)


def friendly_name_for_slug(slug: str) -> str:
    """resolve_station accepts a slug or a config name; normalise to friendly."""
    _, friendly = resolve_station(slug)
    return friendly

LEADS = [24, 48, 72, 96, 120]
TAUS = list(range(12, 118, 3))
BANDS = [(lo, lo + 6) for lo in range(12, 120, 6)]
MARGIN_PCT = 0.75
BLEND_OVER_SINGLE_PCT = 0.5
STUDY_ROOT = WEATHERBLEND_DATA_ROOT / "models_study" / "precipitation"
REPORT_DIR = ROOT / "reports" / "crosslead_4a_study"
SCRATCH = WEATHERBLEND_DATA_ROOT / "scratch" / "crosslead_4a"


def bucket_model(tau: float) -> int:
    """Production bucket model for actual lead τ (mirrors PrecipLeadPolicy.BucketModelFor)."""
    return 120 if tau >= 120 else 96 if tau >= 96 else 72 if tau >= 72 else 48 if tau >= 48 else 24


# --------------------------------------------------------------------------
# Stage 1 — walk-forward study retrain (cutoff-bounded per-cell BART)
# --------------------------------------------------------------------------

def _materialise_train_cache(cutoff: datetime) -> Path:
    """Scan-once for the RETRAIN stage (the same trick the C# policy-retrain
    + this script's scoring stage already use): the production builders glob
    the WHOLE forecast tree (~40k small parquets) per (station, lead) call —
    and the terrain aux pull globs it AGAIN — so 4 stations × 5 leads paid
    ~40 full-tree scans (~12 min of every ~15-min lead cycle; the BART fit
    itself is ~3 min). Consolidate the offset_day rows ≤ cutoff + the EA
    rainfall ONCE into a scratch data root shaped like WEATHERBLEND_DATA_ROOT,
    then point the builders at it (env swap + module reload below)."""
    root = SCRATCH / "train_root"
    fc_dir = root / "forecasts"; rn_dir = root / "truth" / "rainfall"
    oro_dir = root / "static" / "orographic"
    done_marker = root / f".cache_cutoff_{cutoff:%Y%m%d}"
    if done_marker.exists():
        print(f"train cache present → {root} (delete to refresh)", flush=True)
        return root
    fc_dir.mkdir(parents=True, exist_ok=True)
    rn_dir.mkdir(parents=True, exist_ok=True)
    oro_dir.mkdir(parents=True, exist_ok=True)

    fc_glob = str(WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet").replace("\\", "/")
    rn_glob = str(WEATHERBLEND_DATA_ROOT / "truth" / "rainfall" / "**" / "*.parquet").replace("\\", "/")
    con = duckdb.connect(":memory:")
    t0 = time.time()
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
              WHERE LocationName = '{ACTIVE_LOCATION}'
                AND RunTimeSource = 'offset_day'
                AND ValidTimeUtc <= TIMESTAMP '{cutoff:%Y-%m-%d %H:%M:%S}')
        TO '{(fc_dir / "all.parquet").as_posix()}' (FORMAT PARQUET)""")
    print(f"  forecasts consolidated in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{rn_glob}', hive_partitioning=false, union_by_name=true)
              WHERE LocationName = '{ACTIVE_LOCATION}')
        TO '{(rn_dir / "all.parquet").as_posix()}' (FORMAT PARQUET)""")
    print(f"  rainfall consolidated in {time.time()-t0:.0f}s", flush=True)
    con.close()
    import shutil
    for f in (WEATHERBLEND_DATA_ROOT / "static" / "orographic").glob("*.json"):
        shutil.copyfile(f, oro_dir / f.name)
    done_marker.write_text("ok")
    print(f"train cache written → {root}", flush=True)
    return root


def stage_retrain(cutoff: datetime) -> None:
    import importlib
    import os

    cache_root = _materialise_train_cache(cutoff)

    # Point every builder at the consolidated trees: swap the data root and
    # reload the module chain (the smoke tests' established pattern — the
    # builders bind WEATHERBLEND_DATA_ROOT at import time), and REBIND this
    # module's imported names to the reloaded functions (the loop below would
    # otherwise still call the old-root-bound objects). Restored in main()
    # via _restore_real_root so --stage both scores against the real tree.
    os.environ["WEATHERBLEND_DATA_ROOT"] = str(cache_root)
    import src.data
    import _shared
    importlib.reload(src.data)
    importlib.reload(_shared)
    import train_4a as T4
    importlib.reload(T4)
    global build_rich_features_via_duckdb, compose_v1_terrain_block
    build_rich_features_via_duckdb = _shared.build_rich_features_via_duckdb
    compose_v1_terrain_block = _shared.compose_v1_terrain_block
    from src.phase_registry import min_valid_time_for

    min_vt = min_valid_time_for("precipitation", "4a")
    slugs = list(stations_for_location(ACTIVE_LOCATION))
    print(f"STUDY retrain — {len(slugs)} stations × {LEADS} leads, "
          f"offset_day ValidTime ∈ [{min_vt}, {cutoff:%Y-%m-%d}] → {STUDY_ROOT}", flush=True)

    for slug in slugs:
        friendly = friendly_name_for_slug(slug)
        station_index = slugs.index(slug)
        out_dir = STUDY_ROOT / slug / "vstudy_phase4a"
        # Resume: a complete study bundle (preprocess + all per-lead states)
        # survives an interrupted run — skip it rather than re-fit.
        if (out_dir / "preprocess.json").exists() and all(
                (out_dir / f"state_lead_{l}h.rds").exists() for l in LEADS):
            print(f"\n[{time.strftime('%H:%M:%S')}] {friendly} ({slug}) — complete bundle present, skipping.", flush=True)
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] {friendly} ({slug}) → {out_dir}", flush=True)

        per_cell: dict[int, dict] = {}
        for lead in LEADS:
            print(f"  [{time.strftime('%H:%M:%S')}] lead {lead}h — features…", flush=True)
            df_rich = build_rich_features_via_duckdb(friendly, lead, min_valid_time=min_vt)
            df = compose_v1_terrain_block(slug, station_index, lead, df_rich, min_valid_time=min_vt)
            df = df[pd.to_datetime(df["ValidTimeUtc"]) <= cutoff].reset_index(drop=True)
            if len(df) < 1000:
                print(f"    only {len(df)} rows ≤ cutoff — skipping lead.", flush=True)
                continue
            cell = T4._prepare_cell(df, RICH_ORO_FEATURE_NAMES)
            print(f"    rows: train {len(cell['y_train']):,} val {len(cell['val_df']):,} "
                  f"test {len(cell['y_test']):,}", flush=True)
            fit_out = T4._fit_and_store(cell, lead)
            print(f"    fit {fit_out['wall_s']:.0f}s", flush=True)
            per_cell[lead] = {**cell, **fit_out}

        if not per_cell:
            print(f"  no cells trained for {slug} — skipping bundle write.", flush=True)
            continue
        result = {
            "per_cell": per_cell,
            "per_lead_stats": [],
            "test_predictions": pd.DataFrame(),
            "train_features": per_cell[min(per_cell)]["X_train_s"],
            "feature_names": per_cell[min(per_cell)]["feature_names_eff"],
            "y_train": per_cell[min(per_cell)]["y_train"],
        }
        T4.write_per_cell_bundle(out_dir, slug, friendly, "vstudy_phase4a", result,
                                 anchor=cutoff)
        print(f"  bundle written → {out_dir}", flush=True)
    print("\nSTUDY retrain DONE.", flush=True)


# --------------------------------------------------------------------------
# Stage 2 — live-OOS scoring at every τ
# --------------------------------------------------------------------------

def _materialise_live_cache(window_start: datetime, window_end: datetime) -> Path:
    """Scan-once: consolidate the live (non-offset_day) Bonehill forecast rows
    in the window into one parquet so the 36 per-τ pivots don't each re-glob
    the full tree (the same trick the C# harness uses)."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / "live_fc.parquet"
    if out.exists():
        print(f"live cache present → {out} (delete to refresh)", flush=True)
        return out
    fc_glob = str(WEATHERBLEND_DATA_ROOT / "forecasts" / f"location={ACTIVE_LOCATION}" / "**" / "*.parquet").replace("\\", "/")
    con = duckdb.connect(":memory:")
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true)
              WHERE (RunTimeSource IS NULL OR RunTimeSource <> 'offset_day')
                AND ValidTimeUtc BETWEEN TIMESTAMP '{window_start:%Y-%m-%d %H:%M:%S}'
                                     AND TIMESTAMP '{window_end:%Y-%m-%d %H:%M:%S}')
        TO '{out.as_posix()}' (FORMAT PARQUET)""")
    con.close()
    print(f"live cache written → {out}", flush=True)
    return out


def _build_live_at_tau(cache: Path, friendly: str, slug: str, station_index: int,
                       tau: int, hourly_rain: dict, oro_rec: dict,
                       window_start: datetime, window_end: datetime) -> pd.DataFrame:
    """Rich+oro (68) feature frame from LIVE inputs cycle-selected at lead τ:
    per (ValidTime, Model) the freshest run with RunTimeUtc ≤ Valid − τ.
    Mirrors build_rich_features_via_duckdb + compose_v1_terrain_block downstream
    maths exactly; only the row-selection rule differs (the C# harness's
    QueryLatestForecastRows(leadHoursLowerBound=τ) semantics)."""
    model_in = "(" + ",".join(f"'{full}'" for full, _ in MODELS_RICH) + ")"
    pivots = []
    for col, expr in [("precip", "Precipitation"), ("dew", "DewPoint2m"), ("rh", "RelativeHumidity2m"),
                      ("dew_depression", "Temperature2m - DewPoint2m"), ("pressure", "SurfacePressure")]:
        for full, short in MODELS_RICH:
            pivots.append(f"MAX(CASE WHEN Model = '{full}' THEN {expr} END) AS {col}_{short},")
    pivot_sql = "\n            ".join(pivots)
    con = duckdb.connect(":memory:")
    df = con.execute(f"""
        WITH latest AS (
            SELECT ValidTimeUtc, Model,
                   Precipitation, RelativeHumidity2m, Temperature2m, DewPoint2m,
                   CloudCoverLow, CloudCoverMid, CloudCoverHigh,
                   Cape, WindSpeed10m, SurfacePressure,
                   WindDirection10m,
                   ROW_NUMBER() OVER (PARTITION BY ValidTimeUtc, Model
                                      ORDER BY RunTimeUtc DESC) AS rn
            FROM read_parquet('{cache.as_posix()}')
            WHERE Model IN {model_in}
              AND RunTimeUtc <= ValidTimeUtc - INTERVAL '{tau} hours'
              AND ValidTimeUtc BETWEEN TIMESTAMP '{window_start:%Y-%m-%d %H:%M:%S}'
                                   AND TIMESTAMP '{window_end:%Y-%m-%d %H:%M:%S}'
        )
        SELECT ValidTimeUtc,
            {pivot_sql}
            AVG(RelativeHumidity2m)         AS rh_mean,
            AVG(Temperature2m - DewPoint2m) AS dew_depression_mean,
            AVG(CloudCoverLow)  AS cloud_low_mean,
            AVG(CloudCoverMid)  AS cloud_mid_mean,
            AVG(CloudCoverHigh) AS cloud_high_mean,
            AVG(Cape)           AS cape_mean,
            AVG(WindSpeed10m)   AS wind_speed_mean,
            AVG(SIN(WindDirection10m * pi() / 180.0)) AS wind_sin,
            AVG(COS(WindDirection10m * pi() / 180.0)) AS wind_cos,
            AVG(WindSpeed10m)    AS wind_speed,
            AVG(Temperature2m)   AS temp_c,
            AVG(DewPoint2m)      AS dew_c,
            AVG(SurfacePressure) AS pres_hpa
        FROM latest WHERE rn = 1
        GROUP BY ValidTimeUtc ORDER BY ValidTimeUtc""").fetch_df()
    con.close()
    if len(df) == 0:
        return df

    precip_cols = [f"precip_{s}" for _, s in MODELS_RICH]
    pm = df[precip_cols].to_numpy(dtype="float64")
    any_present = (~np.isnan(pm)).any(axis=1)
    df = df[any_present].copy(); pm = pm[any_present]

    df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"])
    truth = df["ValidTimeUtc"].map(lambda v: hourly_rain.get(pd.Timestamp(v)))
    keep = truth.notna()
    df = df[keep].copy(); pm = pm[keep.to_numpy()]
    df["precip_mm_hour"] = truth[keep].values

    present = (~np.isnan(pm)).sum(axis=1)
    sumv = np.nansum(pm, axis=1); sumsq = np.nansum(pm ** 2, axis=1)
    mean_safe = np.where(present > 0, sumv / np.maximum(present, 1), np.nan)
    var = np.maximum(0.0, sumsq / np.maximum(present, 1) - mean_safe ** 2)
    df["precip_mean"] = np.where(present > 0, mean_safe, np.nan)
    df["precip_std"]  = np.where(present > 1, np.sqrt(var), 0.0)
    df["precip_max"]  = np.where(present > 0, np.nanmax(pm, axis=1), np.nan)
    wet_count = (pm >= WET_THRESHOLD_MM).sum(axis=1)
    df["precip_agreement_wet_01"] = np.where(present > 0, wet_count / np.maximum(present, 1), np.nan)

    hour_angle = 2.0 * np.pi * df["ValidTimeUtc"].dt.hour / 24.0
    doy_angle  = 2.0 * np.pi * (df["ValidTimeUtc"].dt.dayofyear - 1) / 365.0
    df["hour_sin"] = np.sin(hour_angle); df["hour_cos"] = np.cos(hour_angle)
    df["doy_sin"]  = np.sin(doy_angle);  df["doy_cos"]  = np.cos(doy_angle)

    run_times = df["ValidTimeUtc"] - pd.Timedelta(hours=tau)
    pers = [_compute_persistence(hourly_rain, rt) for rt in run_times]
    df["ea_rain_prev_24h_mm"]   = [p[0] for p in pers]
    df["ea_rain_prev_72h_mm"]   = [p[1] for p in pers]
    df["ea_wet_hours_last_24h"] = [p[2] for p in pers]
    df["ea_dry_hours_trailing"] = [p[3] for p in pers]
    df["wet"] = (df["precip_mm_hour"] >= WET_THRESHOLD_MM).astype("int8")

    # Terrain block — same maths as compose_v1_terrain_block, aux columns inline.
    elev = float(oro_rec.get("elevation_vs_cell_m", 0.0))
    relief = float(oro_rec.get("relief_5km_m", 0.0))
    rugged = float(oro_rec.get("terrain_ruggedness_5km_m", 0.0))
    gdx = float(oro_rec.get("terrain_gradient_dx", 0.0))
    gdy = float(oro_rec.get("terrain_gradient_dy", 0.0))
    ws  = df["wind_speed"].to_numpy(dtype="float64")
    wsn = df["wind_sin"].to_numpy(dtype="float64")
    wcs = df["wind_cos"].to_numpy(dtype="float64")
    td  = df["dew_c"].to_numpy(dtype="float64")
    p   = df["pres_hpa"].to_numpy(dtype="float64")
    valid_wind = ~(np.isnan(ws) | np.isnan(wsn) | np.isnan(wcs))
    u_east  = np.where(valid_wind, -ws * wsn, 0.0)
    v_north = np.where(valid_wind, -ws * wcs, 0.0)
    uplift = np.where(valid_wind, np.maximum(0.0, u_east * gdx + v_north * gdy), 0.0)
    valid_q = ~(np.isnan(td) | np.isnan(p)) & (p > 0)
    e_hpa = 6.112 * np.exp(17.62 * td / (td + 243.12))
    q_gkg = np.where(valid_q, np.maximum(0.0, 0.622 * e_hpa / (p - 0.378 * e_hpa) * 1000.0), 0.0)
    upwind = np.array([_upwind_gain_at(oro_rec, s, c) for s, c in zip(wsn, wcs)], dtype="float64")
    df["oro_elevation_vs_cell_m"] = elev
    df["oro_relief_5km_m"] = relief
    df["oro_ruggedness_5km_m"] = rugged
    df["oro_wind_sin"] = np.where(np.isnan(wsn), 0.0, wsn)
    df["oro_wind_cos"] = np.where(np.isnan(wcs), 0.0, wcs)
    df["oro_upwind_gain_per_wind_5km_m"] = upwind
    df["oro_uplift_m_per_s"] = uplift
    df["oro_uplift_x_q_g_per_kg"] = uplift * q_gkg
    df["oro_station_id"] = float(station_index)
    return df.reset_index(drop=True)


class _CellPredictor:
    """Load one (station, lead) study cell ONCE (warm scaffold + setState) and
    predict many frames — predict_4a.predict_one_cell rebuilds the sampler per
    call, which at 36 τ × 15 cells would dominate the wall clock."""

    def __init__(self, bundle_dir: Path, lead: int):
        import json
        import predict_4a as P4
        import rpy2.robjects as ro
        from rpy2.robjects.conversion import localconverter
        self._ro = ro
        self._local = localconverter
        self._convert = P4._RCONVERT
        pp = json.loads((bundle_dir / "preprocess.json").read_text())["per_lead"][str(lead)]
        self.feature_list_full = pp["feature_list_full"]
        self.kept = np.array(pp["kept_indices"], dtype=int)
        self.median = np.array(pp["median"], dtype="float64")
        self.mean = np.array(pp["scaler_mean"], dtype="float64")
        self.scale = np.array(pp["scaler_scale"], dtype="float64")
        arrays = np.load(bundle_dir / f"arrays_lead_{lead}h.npz")
        ntree = int(json.loads((bundle_dir / "preprocess.json").read_text()).get("ntree", 500))
        seed = int(json.loads((bundle_dir / "preprocess.json").read_text()).get("seed", 42))
        self.var = f"warm_{bundle_dir.parent.name}_{lead}"
        with self._local(self._convert):
            xr = ro.conversion.py2rpy(arrays["X_train_s"].astype(np.float64))
            yr = ro.conversion.py2rpy(arrays["y_train"].astype(np.float64))
        warm = P4.dbarts.bart(x_train=xr, y_train=yr, ntree=ntree,
                              nskip=P4.WARM_NSKIP, ndpost=P4.WARM_NDPOST,
                              keeptrees=True, verbose=False, seed=seed)
        ro.globalenv[self.var] = warm
        ro.r(f'bundle <- readRDS("{(bundle_dir / f"state_lead_{lead}h.rds").as_posix()}")')
        ro.r(f'{self.var}$fit$setState(bundle$state)')

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_list_full].to_numpy(dtype="float64")[:, self.kept]
        X = np.where(np.isnan(X), self.median, X)
        Xs = ((X - self.mean) / self.scale).astype(np.float64)
        with self._local(self._convert):
            xr = self._ro.conversion.py2rpy(Xs)
        self._ro.globalenv["x_live_study"] = xr
        pred = self._ro.r(f'colMeans(predict({self.var}, newdata = x_live_study))')
        with self._local(self._convert):
            return np.asarray(self._ro.conversion.rpy2py(pred), dtype="float64")


def stage_score(window_start: datetime, split: datetime) -> None:
    window_end = datetime.utcnow()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cache = _materialise_live_cache(window_start, window_end)
    slugs = [s for s in stations_for_location(ACTIVE_LOCATION)
             if (STUDY_ROOT / s / "vstudy_phase4a" / "preprocess.json").exists()]
    if not slugs:
        sys.exit(f"no vstudy_phase4a bundles under {STUDY_ROOT} — run --stage retrain first.")
    print(f"scoring stations: {slugs}; τ grid {TAUS[0]}..{TAUS[-1]} step 3; "
          f"SELECT<{split:%Y-%m-%d}≤SCORE", flush=True)

    pairs = [(LEADS[i], LEADS[j]) for i in range(len(LEADS)) for j in range(i + 1, len(LEADS))]
    rows = []   # per (tau, slice, series): n, sq
    for slug in slugs:
        friendly = friendly_name_for_slug(slug)
        station_index = list(stations_for_location(ACTIVE_LOCATION)).index(slug)
        hourly = _load_hourly_rain(friendly)
        oro = _load_oro_static(slug)
        bundle = STUDY_ROOT / slug / "vstudy_phase4a"
        print(f"\n[{time.strftime('%H:%M:%S')}] {slug}: loading {len(LEADS)} BART states…", flush=True)
        cells = {lead: _CellPredictor(bundle, lead) for lead in LEADS}
        for tau in TAUS:
            df = _build_live_at_tau(cache, friendly, slug, station_index, tau,
                                    hourly, oro, window_start, window_end)
            if len(df) < 100:
                print(f"  τ={tau}h: only {len(df)} rows — skipped", flush=True)
                continue
            preds = {lead: cells[lead].predict(df) for lead in LEADS}
            y = df["wet"].to_numpy(dtype="float64")
            sel_mask = (df["ValidTimeUtc"] < split).to_numpy()
            for slice_name, mask in (("sel", sel_mask), ("sco", ~sel_mask)):
                if mask.sum() == 0:
                    continue
                ym = y[mask]
                for lead in LEADS:
                    e = preds[lead][mask] - ym
                    rows.append({"tau": tau, "slice": slice_name, "series": f"s{lead}",
                                 "n": int(mask.sum()), "sq": float((e * e).sum())})
                for lo, hi in pairs:
                    bl = 0.5 * (preds[lo][mask] + preds[hi][mask]) - ym
                    rows.append({"tau": tau, "slice": slice_name, "series": f"b{lo}x{hi}",
                                 "n": int(mask.sum()), "sq": float((bl * bl).sum())})
            print(f"  [{time.strftime('%H:%M:%S')}] τ={tau}h scored ({len(df)} rows)", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(REPORT_DIR / "per_tau_raw.csv", index=False)

    # Pool stations per (tau, slice, series), then aggregate τ → 6h bands.
    agg = raw.groupby(["tau", "slice", "series"], as_index=False).agg(n=("n", "sum"), sq=("sq", "sum"))
    lines = ["# 4a cross-lead study — 6h-band decisions (pooled stations, live OOS)",
             f"window {window_start:%Y-%m-%d}..{window_end:%Y-%m-%d}, SELECT<{split:%Y-%m-%d}≤SCORE",
             f"margin {MARGIN_PCT}% vs bucket baseline; blends must beat best single by {BLEND_OVER_SINGLE_PCT}%",
             "",
             "| band | Nsco | baseline | SELECT pick | SCORE | decision |",
             "|---|---|---|---|---|---|"]
    for lo, hi in BANDS:
        tin = [t for t in TAUS if lo <= t < hi]
        def brier(slice_name, series):
            sub = agg[(agg["slice"] == slice_name) & (agg["series"] == series) & (agg["tau"].isin(tin))]
            n = sub["n"].sum()
            return (sub["sq"].sum() / n, int(n)) if n > 0 else (float("nan"), 0)
        base_lead = bucket_model(lo)
        base_sco, n_sco = brier("sco", f"s{base_lead}")
        if n_sco < 300 or np.isnan(base_sco):
            lines.append(f"| {lo}-{hi}h | {n_sco} | — | — | — | baseline (insufficient) |")
            continue
        series_all = [f"s{l}" for l in LEADS] + [f"b{lo_}x{hi_}" for lo_, hi_ in pairs]
        pick = min(series_all, key=lambda s: brier("sel", s)[0])
        pick_sco, _ = brier("sco", pick)
        best_single_sco = min(brier("sco", f"s{l}")[0] for l in LEADS)
        is_blend = pick.startswith("b")
        passes = (pick_sco <= base_sco * (1 - MARGIN_PCT / 100.0)
                  and (not is_blend or pick_sco <= best_single_sco * (1 - BLEND_OVER_SINGLE_PCT / 100.0))
                  and pick != f"s{base_lead}")
        delta = 100.0 * (base_sco - pick_sco) / base_sco
        decision = pick.replace("s", "m").replace("b", "blend ").replace("x", "+") if passes else f"baseline m{base_lead}"
        lines.append(f"| {lo}-{hi}h | {n_sco} | m{base_lead} {base_sco:.4f} | {pick} | "
                     f"{pick_sco:.4f} | {decision}{f' (+{delta:.2f}%)' if passes else ''} |")
    (REPORT_DIR / "bands.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nreports → {REPORT_DIR}", flush=True)


def _restore_real_root(real_root: str) -> None:
    """Undo stage_retrain's cache-root swap so a same-process score stage
    reads the REAL forecast/truth trees."""
    import importlib
    import os
    os.environ["WEATHERBLEND_DATA_ROOT"] = real_root
    import src.data
    import _shared
    importlib.reload(src.data)
    importlib.reload(_shared)
    global build_rich_features_via_duckdb, compose_v1_terrain_block
    global _load_hourly_rain, _compute_persistence, _load_oro_static, _upwind_gain_at
    build_rich_features_via_duckdb = _shared.build_rich_features_via_duckdb
    compose_v1_terrain_block = _shared.compose_v1_terrain_block
    _load_hourly_rain = _shared._load_hourly_rain
    _compute_persistence = _shared._compute_persistence
    _load_oro_static = _shared._load_oro_static
    _upwind_gain_at = _shared._upwind_gain_at


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stage", choices=("retrain", "score", "both"), default="both")
    ap.add_argument("--cutoff", default="2026-03-15")
    ap.add_argument("--start", default="2026-03-19")
    ap.add_argument("--split", default=None,
                    help="SELECT/SCORE boundary (yyyy-mm-dd). Default: today − 21d.")
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff)
    start = datetime.fromisoformat(args.start)
    split = (datetime.fromisoformat(args.split) if args.split
             else datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=21))
    real_root = str(WEATHERBLEND_DATA_ROOT)
    if args.stage in ("retrain", "both"):
        try:
            stage_retrain(cutoff)
        finally:
            _restore_real_root(real_root)
    if args.stage in ("score", "both"):
        stage_score(start, split)


if __name__ == "__main__":
    main()
