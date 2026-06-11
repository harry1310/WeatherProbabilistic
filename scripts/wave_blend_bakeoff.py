"""Wave-height blend bake-off for Sennen: LEAN vs RICH feature sets.

Decides (per Harry, 2026-06-11) whether the wave blender ships with the
slim per-model feature set or the full one, and incidentally reports how
each raw wave model scores on its own (no separate model bake-off needed).

Data (all hourly, at the pinned Sennen sea cell):
  features — marine hindcast (data/marine/.../hist_forecast.parquet):
    best-available per valid-time, lead-unlabelled. Same caveat as the
    weather historical-forecast archive: optimistic vs true 1-3 day lead
    skill. The honest per-lead rows only started accruing 2026-06-11; this
    bake-off picks the feature set, not the final skill number.
  truth — era5_ocean significant wave height (data/truth/waves/).
  second opinion — Sevenstones lightvessel Hs on the test window
    (28 km W of the cell, so its errors are location-inflated for every
    candidate equally; comparative use only).

Feature sets:
  LEAN  — per wave model (5): Hs, total period, direction (sin/cos),
          + month sin/cos.
  RICH  — LEAN + per-model swell Hs / swell period / swell direction
          (sin/cos) + wind-wave Hs, + best_match site extras (tide height,
          SST, secondary swell Hs/period).

Recipe matches train_wind_speed_pi.py: point = L2-objective LGB with MAE
early stopping; band = q05/q95 quantile heads + split-CQR on a calibration
slice. Chronological splits throughout.

Run with the WP .venv python.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._shared import force_utf8_stdio  # noqa: E402

force_utf8_stdio()

WB_DATA = Path(r"C:\Projects\Weather\WeatherBlend\data")
MODELS = ["meteofrance_wave", "ecmwf_wam025", "gwam", "ewam", "ncep_gfswave025"]
BEST = "best_match"

TRAIN_END = "2025-06-30"     # train ≤ this
VAL_END = "2025-10-31"       # early-stop slice
CAL_END = "2026-01-31"       # CQR calibration slice
                              # test = after CAL_END (≈ 2026-02 → 2026-06)
COVERAGE = 0.90
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                  min_child_samples=20, subsample=1.0, random_state=42, verbose=-1)
EARLY_STOP = 30


def load() -> pd.DataFrame:
    con = duckdb.connect()
    hind = con.sql(f"""
        SELECT Model, ValidTimeUtc, WaveHeight, WavePeriod, WaveDirection,
               SwellWaveHeight, SwellWavePeriod, SwellWaveDirection,
               WindWaveHeight, SeaLevelHeightMsl, SeaSurfaceTemperature,
               SecondarySwellWaveHeight, SecondarySwellWavePeriod
        FROM read_parquet('{WB_DATA.as_posix()}/marine/location=sennen_cove/model=*/date=*/hist_forecast.parquet',
                          hive_partitioning=false)
    """).df()
    truth = con.sql(f"""
        SELECT ValidTimeUtc, WaveHeight AS hs_truth
        FROM read_parquet('{WB_DATA.as_posix()}/truth/waves/location=sennen_cove/source=era5_ocean/*/data.parquet',
                          hive_partitioning=false)
        WHERE WaveHeight IS NOT NULL
    """).df()
    buoy = con.sql(f"""
        SELECT ValidTimeUtc, WaveHeight AS hs_buoy
        FROM read_parquet('{WB_DATA.as_posix()}/truth/waves/location=sennen_cove/source=sevenstones_62107/*/data.parquet',
                          hive_partitioning=false)
        WHERE WaveHeight IS NOT NULL
    """).df()

    wide = None
    for m in MODELS + [BEST]:
        sub = hind[hind.Model == m].drop(columns=["Model"]).set_index("ValidTimeUtc")
        cols = {
            "WaveHeight": f"{m}__hs", "WavePeriod": f"{m}__tp", "WaveDirection": f"{m}__dir",
            "SwellWaveHeight": f"{m}__swh", "SwellWavePeriod": f"{m}__swp",
            "SwellWaveDirection": f"{m}__swd", "WindWaveHeight": f"{m}__wwh",
        }
        if m == BEST:
            cols |= {"SeaLevelHeightMsl": "site__tide", "SeaSurfaceTemperature": "site__sst",
                     "SecondarySwellWaveHeight": "site__sswh", "SecondarySwellWavePeriod": "site__sswp"}
        sub = sub.rename(columns=cols)[list(cols.values())]
        sub = sub[~sub.index.duplicated(keep="last")]
        wide = sub if wide is None else wide.join(sub, how="outer")

    df = wide.join(truth.set_index("ValidTimeUtc"), how="inner")
    df = df.join(buoy.set_index("ValidTimeUtc"), how="left")
    df = df.sort_index()
    # Require at least one model's Hs (the universal at-least-one rule).
    df = df[df[[f"{m}__hs" for m in MODELS]].notna().any(axis=1)]
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    for m in MODELS:
        for stem in ["dir", "swd"]:
            rad = np.deg2rad(df[f"{m}__{stem}"])
            df[f"{m}__{stem}_sin"] = np.sin(rad)
            df[f"{m}__{stem}_cos"] = np.cos(rad)
    doy = df.index.dayofyear.to_numpy()
    df["month_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["month_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def feature_cols(rich: bool) -> list[str]:
    lean = []
    for m in MODELS:
        lean += [f"{m}__hs", f"{m}__tp", f"{m}__dir_sin", f"{m}__dir_cos"]
    lean += ["month_sin", "month_cos"]
    if not rich:
        return lean
    extra = []
    for m in MODELS:
        extra += [f"{m}__swh", f"{m}__swp", f"{m}__swd_sin", f"{m}__swd_cos", f"{m}__wwh"]
    extra += ["site__tide", "site__sst", "site__sswh", "site__sswp"]
    return lean + extra


def fit_eval(df: pd.DataFrame, rich: bool) -> dict:
    cols = feature_cols(rich)
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["hs_truth"].to_numpy(dtype=np.float64)
    idx = df.index

    tr = idx <= TRAIN_END
    va = (idx > TRAIN_END) & (idx <= VAL_END)
    ca = (idx > VAL_END) & (idx <= CAL_END)
    te = idx > CAL_END

    def fit(objective, alpha=None):
        kw = dict(objective=objective, **LGB_PARAMS)
        if alpha is not None:
            kw["alpha"] = alpha
        if objective == "regression":
            kw["metric"] = "l1"
        m = lgb.LGBMRegressor(**kw)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        return m

    point = fit("regression")
    qlo = fit("quantile", 0.05)
    qhi = fit("quantile", 0.95)

    # Split-CQR on the calibration slice.
    e = np.maximum(qlo.predict(X[ca]) - y[ca], y[ca] - qhi.predict(X[ca]))
    n = len(e)
    q = np.quantile(e, min(1.0, np.ceil((n + 1) * COVERAGE) / n), method="higher")

    pred = point.predict(X[te])
    lo = qlo.predict(X[te]) - q
    hi = qhi.predict(X[te]) + q
    y_te = y[te]
    mae = float(np.abs(pred - y_te).mean())
    cover = float(((y_te >= lo) & (y_te <= hi)).mean())
    width = float((hi - lo).mean())

    # Buoy second opinion on test hours where Sevenstones reported.
    hb = df.loc[te, "hs_buoy"].to_numpy()
    okb = ~np.isnan(hb)
    mae_buoy = float(np.abs(pred[okb] - hb[okb]).mean()) if okb.sum() > 50 else float("nan")

    return dict(mae=mae, coverage=cover, width=width, mae_buoy=mae_buoy,
                n_test=int(te.sum()), n_buoy=int(okb.sum()), cqr_q=float(q),
                n_features=len(cols))


def main() -> int:
    df = add_derived(load())
    print(f"Joined rows: {len(df)}   {df.index.min()} .. {df.index.max()}")
    te = df.index > CAL_END
    y_te = df.loc[te, "hs_truth"].to_numpy()
    hb = df.loc[te, "hs_buoy"].to_numpy()
    okb = ~np.isnan(hb)

    print(f"\nTest window: > {CAL_END}  ({te.sum()} rows, {okb.sum()} with buoy)")
    print("\n--- Raw model baselines on test (vs era5_ocean truth | vs Sevenstones) ---")
    for m in MODELS + [BEST]:
        p = df.loc[te, f"{m}__hs"].to_numpy()
        ok = ~np.isnan(p)
        mae = np.abs(p[ok] - y_te[ok]).mean() if ok.sum() else float("nan")
        okb2 = okb & ~np.isnan(p)
        mb = np.abs(p[okb2] - hb[okb2]).mean() if okb2.sum() > 50 else float("nan")
        print(f"  {m:18s} MAE {mae:.3f} m  (n={ok.sum():5d})   | buoy MAE {mb:.3f} m")
    mean_p = df.loc[te, [f"{m}__hs" for m in MODELS]].mean(axis=1).to_numpy()
    print(f"  {'equal-mean':18s} MAE {np.abs(mean_p - y_te).mean():.3f} m"
          f"                  | buoy MAE {np.abs(mean_p[okb] - hb[okb]).mean():.3f} m")

    print("\n--- Blends (point MAE on test; 90% band coverage/width; buoy check) ---")
    for name, rich in [("LEAN", False), ("RICH", True)]:
        r = fit_eval(df, rich)
        print(f"  {name}: MAE {r['mae']:.3f} m | coverage {r['coverage']:.3f} "
              f"width {r['width']:.3f} m (CQR q={r['cqr_q']:+.3f}) | "
              f"buoy MAE {r['mae_buoy']:.3f} m | {r['n_features']} features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
