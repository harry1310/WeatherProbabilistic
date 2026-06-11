"""wave_height_lgb — TRAIN (Sennen sea-state Phase 2, 2026-06-11).

Significant-wave-height blender at a location's pinned marine cell:
LightGBM point head (L2 objective, MAE early stopping) + q05/q95 quantile
heads + split-CQR conformal correction, the train_wind_speed_pi.py recipe.
RICH feature set per the 2026-06-11 bake-off (wave_blend_bakeoff.py):
per-wave-model Hs/period/direction + swell decomposition + wind-wave Hs +
site extras (tide, SST, secondary swell) — beat LEAN 0.095 vs 0.102 m MAE
with on-target 0.91 band coverage; best single model (ECMWF WAM) was 0.153.

V1 TRAINS ON THE HINDCAST ARCHIVE (best-available per valid-time,
lead-unlabelled) against era5_ocean truth at the location's pinned TRUTH
cell. There is no lead-labelled wave history yet — the marine API's
previous_day columns only exist live, and collection started 2026-06-11 —
so v1 is ONE any-lead model. Consequences, stated plainly: skill numbers
are nowcast-flavoured (optimistic for 1-3 day leads) and the CQR band is
calibrated on hindcast errors, so live multi-day coverage will run under
90% until per-lead calibration lands. Revisit once the offset_day archive
has a few months of rows (~2026-09): per-lead conformal Q first, per-lead
boosters after.

Bundle (data/models/wave_height/{location}/v{ts}_wave_height_lgb/):
  point_model.txt / q_lo.txt / q_hi.txt    LightGBM boosters (any-lead)
  feature_schema.json, calibration.json, training_metadata.json,
  training_summary.json (RetrainGuard)
Promotes into models/wave_height/MANIFEST.json (champion — first model of
its target). R2 push happens in retrain-python.yml.

CLI::
    train_wave_pi.py [--location sennen_cove] [--anchor YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _shared import force_utf8_stdio  # noqa: E402

force_utf8_stdio()

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402
from src.manifest_promote import promote_station_version  # noqa: E402
from src.retrain_guard import build_check_and_save_versioned  # noqa: E402

TARGET = "wave_height"
PHASE = "wave_height_lgb"
log = logging.getLogger("train_wave_pi")

WAVE_MODELS = ["meteofrance_wave", "ecmwf_wam025", "gwam", "ewam", "ncep_gfswave025"]
BEST = "best_match"

COVERAGE = 0.90
ALPHA_LO, ALPHA_HI = 0.05, 0.95
EARLY_STOP = 30
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                  min_child_samples=20, subsample=1.0, random_state=42, verbose=-1)

# Chronological split fractions over however much archive exists: the last
# 10% is the CQR calibration slice, the 10% before that the early-stop val
# slice. (Dates, not fractions, in the bake-off — fractions here so the
# weekly Sunday retrain naturally rolls the slices forward.)
VAL_FRAC, CAL_FRAC = 0.10, 0.10


def load_training_frame(location: str) -> pd.DataFrame:
    base = WEATHERBLEND_DATA_ROOT.as_posix()
    con = duckdb.connect()
    hind = con.sql(f"""
        SELECT Model, ValidTimeUtc, WaveHeight, WavePeriod, WaveDirection,
               SwellWaveHeight, SwellWavePeriod, SwellWaveDirection,
               WindWaveHeight, SeaLevelHeightMsl, SeaSurfaceTemperature,
               SecondarySwellWaveHeight, SecondarySwellWavePeriod
        FROM read_parquet('{base}/marine/location={location}/model=*/date=*/hist_forecast.parquet',
                          hive_partitioning=false)
    """).df()
    truth = con.sql(f"""
        SELECT ValidTimeUtc, WaveHeight AS hs_truth
        FROM read_parquet('{base}/truth/waves/location={location}/source=era5_ocean/*/data.parquet',
                          hive_partitioning=false)
        WHERE WaveHeight IS NOT NULL
    """).df()

    wide = None
    for m in WAVE_MODELS + [BEST]:
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

    df = wide.join(truth.set_index("ValidTimeUtc"), how="inner").sort_index()
    df = df[df[[f"{m}__hs" for m in WAVE_MODELS]].notna().any(axis=1)]
    return add_derived(df)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    for m in WAVE_MODELS:
        for stem in ["dir", "swd"]:
            rad = np.deg2rad(df[f"{m}__{stem}"])
            df[f"{m}__{stem}_sin"] = np.sin(rad)
            df[f"{m}__{stem}_cos"] = np.cos(rad)
    doy = df.index.dayofyear.to_numpy()
    df["month_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["month_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def feature_columns() -> list[str]:
    """RICH set — canonical order; feature_schema.json mirrors this and
    predict_wave_pi builds the same order. Keep in lockstep with
    predict_wave_pi.feature_columns (single duplicated list, asserted equal
    in tests/test_wave_scripts.py)."""
    cols = []
    for m in WAVE_MODELS:
        cols += [f"{m}__hs", f"{m}__tp", f"{m}__dir_sin", f"{m}__dir_cos"]
    cols += ["month_sin", "month_cos"]
    for m in WAVE_MODELS:
        cols += [f"{m}__swh", f"{m}__swp", f"{m}__swd_sin", f"{m}__swd_cos", f"{m}__wwh"]
    cols += ["site__tide", "site__sst", "site__sswh", "site__sswp"]
    return cols


def _fit(X, y, Xv, yv, objective, alpha=None):
    kw = dict(objective=objective, **LGB_PARAMS)
    if alpha is not None:
        kw["alpha"] = alpha
    if objective == "regression":
        kw["metric"] = "l1"
    m = lgb.LGBMRegressor(**kw)
    ok, okv = ~np.isnan(y), ~np.isnan(yv)
    m.fit(X[ok], y[ok], eval_set=[(Xv[okv], yv[okv])],
          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return m


def train_location(location: str, models_root: Path, anchor: datetime) -> int:
    df = load_training_frame(location)
    if len(df) < 5000:
        log.error("Only %d joined rows for %s — refusing to train.", len(df), location)
        return 3

    cols = feature_columns()
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["hs_truth"].to_numpy(dtype=np.float64)
    n = len(df)
    i_val = int(n * (1 - VAL_FRAC - CAL_FRAC))
    i_cal = int(n * (1 - CAL_FRAC))
    log.info("%s: %d rows (%s..%s) — train %d / val %d / cal %d",
             location, n, df.index.min(), df.index.max(), i_val, i_cal - i_val, n - i_cal)

    Xtr, ytr = X[:i_val], y[:i_val]
    Xva, yva = X[i_val:i_cal], y[i_val:i_cal]
    Xca, yca = X[i_cal:], y[i_cal:]

    point = _fit(Xtr, ytr, Xva, yva, "regression")
    qlo = _fit(Xtr, ytr, Xva, yva, "quantile", ALPHA_LO)
    qhi = _fit(Xtr, ytr, Xva, yva, "quantile", ALPHA_HI)

    e = np.maximum(qlo.predict(Xca) - yca, yca - qhi.predict(Xca))
    nc = len(e)
    conformal_q = float(np.quantile(e, min(1.0, np.ceil((nc + 1) * COVERAGE) / nc), method="higher"))

    # Diagnostics on the cal slice (the freshest held-out data we have).
    pred = point.predict(Xca)
    mae = float(np.abs(pred - yca).mean())
    lo = qlo.predict(Xca) - conformal_q
    hi = qhi.predict(Xca) + conformal_q
    cover = float(((yca >= lo) & (yca <= hi)).mean())
    log.info("cal-slice: point MAE %.3f m | band coverage %.3f | conformal_q %+.3f",
             mae, cover, conformal_q)

    version = f"v{anchor:%Y-%m-%d}_{datetime.now(timezone.utc):%H%M%S}_{PHASE}"
    out_dir = models_root / TARGET / location / version
    out_dir.mkdir(parents=True, exist_ok=True)

    point.booster_.save_model(str(out_dir / "point_model.txt"))
    qlo.booster_.save_model(str(out_dir / "q_lo.txt"))
    qhi.booster_.save_model(str(out_dir / "q_hi.txt"))
    (out_dir / "feature_schema.json").write_text(json.dumps(
        {"FeatureNames": cols, "WaveModels": WAVE_MODELS, "FeatureSet": "rich"}, indent=2))
    (out_dir / "calibration.json").write_text(json.dumps({
        "coverage": COVERAGE, "conformal_q": conformal_q,
        "cal_rows": int(n - i_cal), "cal_mae": mae, "cal_band_coverage": cover,
        "lead_mode": "any-lead-v1",
        "note": "Trained on lead-unlabelled hindcast; per-lead calibration "
                "planned once the offset_day archive matures (~2026-09).",
    }, indent=2))
    (out_dir / "training_metadata.json").write_text(json.dumps({
        "Version": version, "Target": TARGET, "Phase": PHASE, "LocationName": location,
        "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
        "RowsTrain": i_val, "RowsVal": i_cal - i_val, "RowsCal": n - i_cal,
        "TruthSource": "era5_ocean", "FeatureCount": len(cols),
        "WindowStart": str(df.index.min()), "WindowEnd": str(df.index.max()),
    }, indent=2))

    guard = build_check_and_save_versioned(
        log, out_dir,
        composite=f"{TARGET}/{location}",
        phase=PHASE, version=version,
        rows_train=i_val, rows_val=i_cal - i_val, rows_test=n - i_cal,
        train_features=Xtr, feature_names=cols,
        label_rates={location: 1.0},   # regression — no label rate
        location_name=location,
    )
    if not guard.passed:
        log.error("RetrainGuard FAILED — bundle left unpromoted at %s", out_dir)
        return 4

    promote_station_version(models_root, TARGET, location, version, PHASE, role="champion")
    log.info("Promoted %s into %s/MANIFEST.json under station=%s", version, TARGET, location)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--location", default="sennen_cove")
    ap.add_argument("--anchor", default=None, help="Anchor date YYYY-MM-DD UTC for the version string.")
    ap.add_argument("--models-root", default=str(WEATHERBLEND_DATA_ROOT / "models"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    anchor = (datetime.fromisoformat(args.anchor).replace(tzinfo=timezone.utc)
              if args.anchor else datetime.now(timezone.utc))
    log.info("wave_height_lgb — TRAIN (quantile-LGB + split-CQR, any-lead v1)")
    sys.exit(train_location(args.location, Path(args.models_root), anchor))


if __name__ == "__main__":
    main()
