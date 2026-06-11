"""wave_height_lgb — PREDICT (Sennen sea-state Phase 2, 2026-06-11).

Runtime half of train_wave_pi.py: loads the latest ``*_wave_height_lgb``
bundle, builds the RICH feature row for every forecast hour from the
FRESHEST live marine run per wave model, runs point + q05/q95 boosters,
applies the split-CQR correction, and writes one row per ValidTimeUtc.

The output parquet is deliberately self-sufficient for the Sea state tab —
alongside the blended Hs point + band it passes through the site-extras the
wetting-elevation story needs at render time, all from the freshest
best_match run: tide height (sea_level_height_msl), total/swell period +
direction, secondary swell, SST. One windowed pull, one parquet, no extra
render joins.

LeadHours on each row = hours from prediction time to the valid hour
(the v1 model is any-lead — the label is for charts/verify bucketing, not
model selection; see train_wave_pi.py's lead_mode note).

Output:
  data/predictions/wave_height/model_version=v..._wave_height_lgb/date={d}/predictions.parquet

CLI::
    predict_wave_pi.py [--location sennen_cove] [--anchor YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _shared import force_utf8_stdio  # noqa: E402

force_utf8_stdio()

import lightgbm as lgb  # noqa: E402

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402
from train_wave_pi import WAVE_MODELS, BEST, add_derived, feature_columns  # noqa: E402

TARGET = "wave_height"
PHASE = "wave_height_lgb"
log = logging.getLogger("predict_wave_pi")

_VERSION_RE = re.compile(r"^v\d{4}-\d{2}-\d{2}_\d{6}_wave_height_lgb$")

# best_match pass-through surfaced on every output row for the Sea state
# tab: output column name -> the feature-frame column it reads (the
# best_match feature columns double as the pass-through source).
PASSTHROUGH_SRC = {
    "WavePeriodS": "best_match__tp", "WaveDirectionDeg": "best_match__dir",
    "SwellHeightM": "best_match__swh", "SwellPeriodS": "best_match__swp",
    "SwellDirectionDeg": "best_match__swd",
    "SecondarySwellHeightM": "site__sswh", "SecondarySwellPeriodS": "site__sswp",
    "TideHeightMsl": "site__tide", "SeaSurfaceTempC": "site__sst",
}


def find_latest_bundle(models_root: Path, location: str) -> Path:
    parent = models_root / TARGET / location
    if not parent.is_dir():
        raise FileNotFoundError(f"No bundle dir under {parent}. Run train_wave_pi.py first.")
    for c in sorted((d for d in parent.iterdir()
                     if d.is_dir() and _VERSION_RE.match(d.name)),
                    key=lambda d: d.name, reverse=True):
        if all((c / f).is_file() for f in
               ("point_model.txt", "q_lo.txt", "q_hi.txt",
                "feature_schema.json", "calibration.json", "training_summary.json")):
            return c
    raise FileNotFoundError(f"No usable *_wave_height_lgb bundle under {parent}.")


def load_latest_live(location: str) -> pd.DataFrame:
    """Freshest live run per model from the marine tree, pivoted wide per
    ValidTimeUtc. Live files are run=HH.parquet under date=<run date>;
    'freshest' = lexicographically max (date, run) per model — the same
    newest-run-wins rule the weather predict pulls use."""
    base = WEATHERBLEND_DATA_ROOT.as_posix()
    con = duckdb.connect()
    wide = None
    for m in WAVE_MODELS + [BEST]:
        glob = f"{base}/marine/location={location}/model={m}/date=*/run=*.parquet"
        try:
            sub = con.sql(f"""
                WITH ranked AS (
                  SELECT *, max(RunTimeUtc) OVER () AS max_run
                  FROM read_parquet('{glob}', hive_partitioning=false)
                  WHERE RunTimeSource = 'synthesised'
                )
                SELECT ValidTimeUtc, RunTimeUtc, WaveHeight, WavePeriod, WaveDirection,
                       SwellWaveHeight, SwellWavePeriod, SwellWaveDirection,
                       WindWaveHeight, SeaLevelHeightMsl, SeaSurfaceTemperature,
                       SecondarySwellWaveHeight, SecondarySwellWavePeriod
                FROM ranked WHERE RunTimeUtc = max_run
            """).df()
        except duckdb.IOException:
            log.warning("  %s: no live files on disk — skipping model.", m)
            continue
        if sub.empty:
            continue
        run_time = pd.Timestamp(sub["RunTimeUtc"].max())
        age_h = (pd.Timestamp.utcnow().tz_localize(None) - run_time).total_seconds() / 3600
        if age_h > 36:
            log.warning("  %s: freshest live run is %.0fh old — stale, skipping model.", m, age_h)
            continue
        sub = sub.drop(columns=["RunTimeUtc"]).set_index("ValidTimeUtc")
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
    if wide is None:
        return pd.DataFrame()
    wide = wide.sort_index()
    hs_cols = [f"{m}__hs" for m in WAVE_MODELS if f"{m}__hs" in wide.columns]
    if not hs_cols:
        return pd.DataFrame()
    return wide[wide[hs_cols].notna().any(axis=1)]


def predict_for_location(location: str, anchor: datetime,
                         models_root: Path, predictions_root: Path) -> int:
    bundle = find_latest_bundle(models_root, location)
    version = bundle.name
    log.info("Using bundle %s", bundle)

    schema = json.loads((bundle / "feature_schema.json").read_text())
    calibration = json.loads((bundle / "calibration.json").read_text())
    feature_names: list[str] = schema["FeatureNames"]
    assert feature_names == feature_columns(), \
        "feature_schema.json order != predict-side feature_columns() — train/predict drift"
    conformal_q = float(calibration["conformal_q"])

    df = load_latest_live(location)
    if df.empty:
        log.error("No live marine rows for %s — exit 3.", location)
        return 3
    # add_derived needs every raw model column to exist; outer-join gaps for
    # missing models arrive as absent columns — create them as NaN first.
    for m in WAVE_MODELS:
        for stem in ("hs", "tp", "dir", "swh", "swp", "swd", "wwh"):
            if f"{m}__{stem}" not in df.columns:
                df[f"{m}__{stem}"] = np.nan
    for c in ("site__tide", "site__sst", "site__sswh", "site__sswp"):
        if c not in df.columns:
            df[c] = np.nan
    df = add_derived(df)

    X = df[feature_names].to_numpy(dtype=np.float64)
    point = lgb.Booster(model_file=str(bundle / "point_model.txt"))
    qlo = lgb.Booster(model_file=str(bundle / "q_lo.txt"))
    qhi = lgb.Booster(model_file=str(bundle / "q_hi.txt"))

    q50 = point.predict(X)
    lo = np.maximum(0.0, qlo.predict(X) - conformal_q)
    hi = qhi.predict(X) + conformal_q

    made_at = datetime.now(timezone.utc).replace(microsecond=0)
    made_naive = made_at.replace(tzinfo=None)
    out = pd.DataFrame({
        "LocationName": location,
        "Element": TARGET,
        "ModelVersion": version,
        "PredictionMadeAtUtc": made_at,
        "ValidTimeUtc": df.index,
        "LeadHours": [int(round((pd.Timestamp(v) - made_naive).total_seconds() / 3600))
                      for v in df.index],
        "BlendValue": q50,
        "BandLoM": lo,
        "BandHiM": hi,
        "ConformalQ": conformal_q,
    })
    for outname, col in PASSTHROUGH_SRC.items():
        out[outname] = df[col].to_numpy() if col in df.columns else np.nan

    out = out[out["LeadHours"] >= 0].reset_index(drop=True)
    if out.empty:
        log.error("All live rows are in the past — exit 3.")
        return 3

    date_str = anchor.strftime("%Y-%m-%d")
    out_path = (predictions_root / TARGET / f"model_version={version}"
                / f"date={date_str}" / "predictions.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        try:
            prev = pd.read_parquet(out_path)
            prev = prev[prev["LocationName"] != location]
            out = pd.concat([prev, out], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("  could not merge existing parquet (%s) — overwriting.", exc)
    out = out.sort_values(["ValidTimeUtc"]).reset_index(drop=True)
    out.to_parquet(out_path, index=False)
    log.info("Wrote %d rows (Hs q50 %.2f..%.2f m, tide %s) → %s",
             len(out), float(np.nanmin(q50)), float(np.nanmax(q50)),
             "present" if out["TideHeightMsl"].notna().any() else "MISSING",
             out_path)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--location", default="sennen_cove")
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--models-root", default=str(WEATHERBLEND_DATA_ROOT / "models"))
    ap.add_argument("--predictions-root", default=str(WEATHERBLEND_DATA_ROOT / "predictions"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    anchor = (datetime.fromisoformat(args.anchor).replace(tzinfo=timezone.utc)
              if args.anchor else datetime.now(timezone.utc))
    anchor = anchor.replace(hour=0, minute=0, second=0, microsecond=0)

    log.info("wave_height_lgb — PREDICT")
    sys.exit(predict_for_location(args.location, anchor,
                                  Path(args.models_root), Path(args.predictions_root)))


if __name__ == "__main__":
    main()
