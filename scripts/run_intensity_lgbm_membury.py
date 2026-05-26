"""Quick LightGBM mm/h regressor for Membury — "3c rich features, but for the
intensity question, not P(wet)".

Asks: if we train a LightGBM regressor with the Phase 3c rich feature set
(~54 features: per-NWP precip + spread/covariate aggregates + per-NWP
humidity/dew/pressure + 4 EA persistence) and predict raw mm/h (continuous),
how well does it do on Membury at leads 24/48/72h?

Two objectives are trained side by side:
  * `regression` (L2 / MSE) — vanilla squared-error
  * `tweedie`  (variance_power=1.5) — heavy-tailed, zero-inflated count-like target,
    the textbook fit for hourly precip

Per-station per-lead — matches the per-station convention of the existing
deployed Membury 3c bundles (BinaryTrainingRow.TruthMmHour is what we'd be
regressing on; the C# pipeline just doesn't use it).

Baselines reported alongside:
  * `nwp_mean`   — mean of per-NWP precip predictions (already a feature)
  * `nwp_max`    — max of per-NWP precip
  * `clim`       — constant = train-set mean of mm/h

Metrics:
  * MAE, RMSE, R2
  * MAE on wet hours only (truth >= 0.1 mm) — the hard part of the problem
  * Pearson correlation of prediction vs truth

Run:
    .venv/Scripts/python.exe -u scripts/run_intensity_lgbm_membury.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import WEATHERBLEND_DATA_ROOT, WET_THRESHOLD_MM  # noqa: E402

LOCATION = "membury_devon"
STATIONS = ("Chards Snowdon Hill", "Goren", "Raymonds Hill")
LEADS = (24, 48, 72)

# 3c lean model set (same 7 NWPs as Phase 3c).
MODELS_LEAN = [
    ("gfs_seamless",         "gfs"),
    ("ecmwf_ifs025",         "ecmwf"),
    ("icon_seamless",        "icon"),
    ("meteofrance_seamless", "mf"),
    ("gem_seamless",         "gem"),
    ("ecmwf_aifs025_single", "aifs"),
    ("jma_seamless",         "jma"),
]

OUT_DIR = ROOT / "reports" / "membury_intensity_lgbm"

LGB_BASE = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "feature_fraction": 0.9,
    "bagging_fraction": 1.0,
    "verbose": -1,
    "seed": 42,
    "num_threads": 0,
}
NUM_ITERS = 500
EARLY_STOP = 30


# ---------------------------------------------------------------------------
# Rich-feature build (mirrors PrecipRichFeatureBuilder C# SQL + ComposeRow)
# ---------------------------------------------------------------------------

def build_rich(station_friendly: str, lead_hours: int) -> pd.DataFrame:
    """Return one row per ValidTimeUtc with the 3c rich feature set plus the
    continuous mm/h truth label.
    """
    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    rn_glob = str((WEATHERBLEND_DATA_ROOT / "truth" / "rainfall" / "**" / "*.parquet")).replace("\\", "/")

    model_in = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
    pivot_lines = []
    for full, short in MODELS_LEAN:
        pivot_lines += [
            f"MAX(CASE WHEN Model = '{full}' THEN Precipitation END)                   AS precip_{short}",
            f"MAX(CASE WHEN Model = '{full}' THEN DewPoint2m END)                      AS dew_{short}",
            f"MAX(CASE WHEN Model = '{full}' THEN RelativeHumidity2m END)              AS rh_{short}",
            f"MAX(CASE WHEN Model = '{full}' THEN Temperature2m - DewPoint2m END)      AS dewdep_{short}",
            f"MAX(CASE WHEN Model = '{full}' THEN SurfacePressure END)                 AS pressure_{short}",
        ]

    sql = f"""
    WITH hourly_truth AS (
        SELECT
            date_trunc('hour', ObservedTimeUtc) AS valid_time,
            SUM(Value15MinMm) AS precip_mm_hour
        FROM read_parquet('{rn_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND StationName  = '{station_friendly}'
          AND Value15MinMm IS NOT NULL
        GROUP BY 1
        HAVING COUNT(*) = 4
    ),
    latest AS (
        SELECT
            ValidTimeUtc, Model,
            Precipitation,
            RelativeHumidity2m, Temperature2m, DewPoint2m,
            CloudCoverLow, CloudCoverMid, CloudCoverHigh,
            Cape, WindSpeed10m, SurfacePressure,
            ROW_NUMBER() OVER (
                PARTITION BY ValidTimeUtc, Model
                ORDER BY RunTimeUtc DESC
            ) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours = {lead_hours}
          AND Model IN {model_in}
    ),
    pivoted AS (
        SELECT
            ValidTimeUtc,
            {",\n            ".join(pivot_lines)},
            AVG(RelativeHumidity2m)         AS rh_mean,
            AVG(Temperature2m - DewPoint2m) AS dew_depression_mean,
            AVG(CloudCoverLow)  AS cloud_low_mean,
            AVG(CloudCoverMid)  AS cloud_mid_mean,
            AVG(CloudCoverHigh) AS cloud_high_mean,
            AVG(Cape)           AS cape_mean,
            AVG(WindSpeed10m)   AS wind_speed_mean
        FROM latest
        WHERE rn = 1
        GROUP BY ValidTimeUtc
    )
    SELECT p.*, t.precip_mm_hour
    FROM pivoted p
    JOIN hourly_truth t ON p.ValidTimeUtc = t.valid_time
    ORDER BY p.ValidTimeUtc
    """
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    # Per-station hourly rainfall dict (for EA persistence)
    rn = con.execute(f"""
        SELECT date_trunc('hour', ObservedTimeUtc) AS valid_time,
               SUM(Value15MinMm) AS mm
        FROM read_parquet('{rn_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND StationName  = '{station_friendly}'
          AND Value15MinMm IS NOT NULL
        GROUP BY 1 HAVING COUNT(*) = 4 ORDER BY 1
    """).fetch_df()
    con.close()

    # Spread features
    precip_cols = [f"precip_{short}" for _, short in MODELS_LEAN]
    pm = df[precip_cols].to_numpy(dtype="float64")
    present = (~np.isnan(pm)).sum(axis=1)
    sumv = np.nansum(pm, axis=1)
    sumsq = np.nansum(pm ** 2, axis=1)
    mean_safe = np.where(present > 0, sumv / np.maximum(present, 1), np.nan)
    var = np.maximum(0.0, sumsq / np.maximum(present, 1) - mean_safe ** 2)
    wet_count = (pm >= WET_THRESHOLD_MM).sum(axis=1)
    df["precip_mean"] = mean_safe
    df["precip_std"]  = np.where(present > 1, np.sqrt(var), 0.0)
    df["precip_max"]  = np.nanmax(pm, axis=1)
    df["precip_agreement_wet_01"] = np.where(present > 0, wet_count / np.maximum(present, 1), np.nan)

    # Calendar
    hour_angle = 2.0 * np.pi * df["ValidTimeUtc"].dt.hour / 24.0
    doy_angle  = 2.0 * np.pi * (df["ValidTimeUtc"].dt.dayofyear - 1) / 365.0
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["doy_sin"]  = np.sin(doy_angle)
    df["doy_cos"]  = np.cos(doy_angle)

    # EA persistence — anchored at runTime = ValidTimeUtc - lead.
    hourly = {pd.Timestamp(t).to_pydatetime(): float(mm)
              for t, mm in zip(rn["valid_time"].to_numpy(), rn["mm"].to_numpy())}
    vts = pd.to_datetime(df["ValidTimeUtc"]).dt.to_pydatetime()
    prev24 = np.empty(len(df)); prev72 = np.empty(len(df))
    wet24  = np.empty(len(df)); drytr  = np.empty(len(df))
    for i, vt in enumerate(vts):
        run = vt - pd.Timedelta(hours=lead_hours)
        run = run.to_pydatetime() if hasattr(run, "to_pydatetime") else run
        s24 = 0.0; s72 = 0.0; w24 = 0; c24 = True; c72 = True
        for h in range(72):
            t = run - pd.Timedelta(hours=h)
            if t in hourly:
                mm = hourly[t]
                s72 += mm
                if h < 24:
                    s24 += mm
                    if mm >= WET_THRESHOLD_MM:
                        w24 += 1
            else:
                if h < 24: c24 = False
                c72 = False
        d = 0
        for h in range(72):
            t = run - pd.Timedelta(hours=h)
            if t not in hourly: break
            if hourly[t] > WET_THRESHOLD_MM: break
            d += 1
        prev24[i] = s24 if c24 else np.nan
        prev72[i] = s72 if c72 else np.nan
        wet24[i]  = float(w24) if c24 else np.nan
        drytr[i]  = float(d)
    df["ea_rain_prev_24h_mm"]   = prev24
    df["ea_rain_prev_72h_mm"]   = prev72
    df["ea_wet_hours_last_24h"] = wet24
    df["ea_dry_hours_trailing"] = drytr

    return df.reset_index(drop=True)


FEATURE_NAMES: list[str] = (
    [f"precip_{s}" for _, s in MODELS_LEAN]
    + ["precip_mean", "precip_std", "precip_max", "precip_agreement_wet_01"]
    + ["rh_mean", "dew_depression_mean",
       "cloud_low_mean", "cloud_mid_mean", "cloud_high_mean",
       "cape_mean", "wind_speed_mean"]
    + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    + [f"dew_{s}" for _, s in MODELS_LEAN]
    + [f"rh_{s}" for _, s in MODELS_LEAN]
    + [f"dewdep_{s}" for _, s in MODELS_LEAN]
    + [f"pressure_{s}" for _, s in MODELS_LEAN]
    + ["ea_rain_prev_24h_mm", "ea_rain_prev_72h_mm",
       "ea_wet_hours_last_24h", "ea_dry_hours_trailing"]
)


def time_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    a = int(n * train_frac); b = a + int(n * val_frac)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def metrics(p: np.ndarray, y: np.ndarray, train_mean: float) -> dict:
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    wet_mask = y >= WET_THRESHOLD_MM
    mae_wet = float(np.mean(np.abs(err[wet_mask]))) if wet_mask.any() else float("nan")
    if y.std() > 0 and p.std() > 0:
        corr = float(np.corrcoef(p, y)[0, 1])
    else:
        corr = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2,
            "mae_wet": mae_wet, "corr": corr,
            "n_test": int(len(y)), "n_wet": int(wet_mask.sum())}


def fit_one(X_tr, y_tr, X_val, y_val, X_te, objective: str):
    params = dict(LGB_BASE)
    params["objective"] = objective
    params["metric"] = "rmse" if objective != "tweedie" else "tweedie"
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.5
    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_set)
    t0 = time.time()
    booster = lgb.train(
        params, train_set,
        num_boost_round=NUM_ITERS,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    elapsed = time.time() - t0
    p_te = booster.predict(X_te, num_iteration=booster.best_iteration)
    p_te = np.clip(p_te, 0.0, None)  # mm/h cannot be negative
    return p_te, elapsed, booster.best_iteration


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output -> {OUT_DIR}")
    print(f"Features in spec: {len(FEATURE_NAMES)}")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    summary_rows = []

    for station in STATIONS:
        for lead in LEADS:
            print(f"\n=== {station} | lead {lead}h ===")
            t0 = time.time()
            df = build_rich(station, lead)
            print(f"  rows: {len(df):,}  (build {time.time()-t0:.1f}s)")
            if len(df) < 500:
                print(f"  SKIP — too few rows ({len(df)})")
                continue

            tr, va, te = time_split(df)
            X_tr = tr[FEATURE_NAMES].to_numpy(dtype="float64")
            y_tr = tr["precip_mm_hour"].to_numpy(dtype="float64")
            X_va = va[FEATURE_NAMES].to_numpy(dtype="float64")
            y_va = va["precip_mm_hour"].to_numpy(dtype="float64")
            X_te = te[FEATURE_NAMES].to_numpy(dtype="float64")
            y_te = te["precip_mm_hour"].to_numpy(dtype="float64")
            train_mean = float(y_tr.mean())
            wet_rate = float((y_tr >= WET_THRESHOLD_MM).mean())
            print(f"  train n={len(y_tr):,}  wet rate={wet_rate*100:.1f}%  "
                  f"train mean mm/h={train_mean:.4f}  test mean={y_te.mean():.4f}  "
                  f"test wet rate={(y_te>=WET_THRESHOLD_MM).mean()*100:.1f}%")

            # Baselines
            p_nwp_mean = np.nan_to_num(te["precip_mean"].to_numpy(dtype="float64"),
                                       nan=train_mean)
            p_nwp_max  = np.nan_to_num(te["precip_max"].to_numpy(dtype="float64"),
                                       nan=train_mean)
            p_clim     = np.full_like(y_te, train_mean)
            p_zero     = np.zeros_like(y_te)

            for tag, p in [("nwp_mean", p_nwp_mean),
                           ("nwp_max",  p_nwp_max),
                           ("clim",     p_clim),
                           ("zero",     p_zero)]:
                m = metrics(p, y_te, train_mean)
                row = {"station": station, "lead": lead, "model": tag,
                       "best_iter": "-", "wall_s": 0.0, **m}
                summary_rows.append(row)
                print(f"  baseline {tag:9s} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                      f"R2={m['r2']:+.3f}  MAE_wet={m['mae_wet']:.3f}  r={m['corr']:+.3f}")

            for obj in ["regression", "tweedie"]:
                p_te, wall, best = fit_one(X_tr, y_tr, X_va, y_va, X_te, obj)
                m = metrics(p_te, y_te, train_mean)
                row = {"station": station, "lead": lead, "model": f"lgbm_{obj}",
                       "best_iter": best, "wall_s": round(wall, 1), **m}
                summary_rows.append(row)
                print(f"  LGBM {obj:11s} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                      f"R2={m['r2']:+.3f}  MAE_wet={m['mae_wet']:.3f}  r={m['corr']:+.3f}  "
                      f"(iters={best}, {wall:.1f}s)")

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)

    # Pivot: model × lead (averaged across stations), MAE & MAE_wet
    print("\n\n========== AGGREGATE (mean over 3 Membury stations) ==========")
    agg = (df.groupby(["model", "lead"])
             .agg(mae=("mae", "mean"),
                  rmse=("rmse", "mean"),
                  r2=("r2", "mean"),
                  mae_wet=("mae_wet", "mean"),
                  corr=("corr", "mean"))
             .reset_index())
    agg = agg.sort_values(["lead", "mae"])
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Per-cell winner table
    print("\n========== PER-CELL WINNER (by MAE) ==========")
    rank = (df.groupby(["station", "lead"])
              .apply(lambda g: g.sort_values("mae").head(2)[["model", "mae", "mae_wet"]])
              .reset_index())
    print(rank.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    agg.to_csv(OUT_DIR / "aggregate.csv", index=False)
    print(f"\nWrote summary -> {OUT_DIR/'summary.csv'}")
    print(f"Wrote aggregate -> {OUT_DIR/'aggregate.csv'}")


if __name__ == "__main__":
    main()
