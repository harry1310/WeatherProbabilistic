"""wind_speed_lgb — PREDICT ONLY (Python, CQR quantile-LGB + cross-conformal).

Option B cutover (2026-06-09): the wind-speed model moved from the .NET
WindSpeedLgbPredictPipeline to Python so one model owns the point (q50) AND a
calibrated 90% prediction band. This predictor is the runtime half: it loads the
latest ``*_wind_speed_lgb`` bundle, builds live features from the forecast tree at
each lead, runs the three LightGBM quantile boosters, applies the per-lead
cross-conformal Q, and writes one ``ElementPredictionRow``-shaped row per
(ValidTimeUtc, Lead) so the EXISTING consumers keep working unchanged:

  * wind_blend (WindBlendPredictCommand) inner-joins on (ValidTime, Lead) and
    reads BlendValue + the per-NWP slots via union_by_name — so we emit the exact
    same columns the .NET WindSpeedLgbPredictPipeline did (Element="wind",
    BlendValue, ModelGfs..ModelAifs, RunTime*, Mean/Std/Range, FeatureVectorHash).
  * The CQR band (q_lo-Q, q_hi+Q) is the NEW sidecar: extra columns BandLoMs /
    BandHiMs / ConformalQ on the same parquet. union_by_name means the .NET
    readers ignore them; the wind-speed chart ribbon (WB step 6) reads them.

Output path mirrors the .NET ElementPredictCommand element tree (NO location
subdir — LocationName is a column; model_version disambiguates the phase):
  data/predictions/wind/model_version=v..._wind_speed_lgb/date={d}/predictions.parquet

The feature recipe is identical to wind_mvn's (both built via
train_wind_mvn.build_features), so we reuse predict_wind_mvn.build_live_for_lead
verbatim rather than re-deriving the 29-feature live build. LightGBM consumes raw
features (no scaler, unlike the MVN MLP) and handles missing NWPs natively.

CLI::
    predict_wind_speed_pi.py [--location bonehill_rocks] [--anchor YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _shared import force_utf8_stdio  # noqa: E402

force_utf8_stdio()

import lightgbm as lgb  # noqa: E402

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402
import predict_wind_mvn as PM  # noqa: E402  (build_live_for_lead / load_oro_static / NWPS / NWP_SHORT)

PHASE = "wind_speed_lgb"
TARGET = "wind"                 # element tree dir (NOT wind_direction)
LEADS = (24, 48, 72)

# Per-NWP wsp slots surfaced in ElementPredictionRow (7 named: Gfs..Aifs). JMA
# falls off the named-slot end exactly as the .NET pipeline did (consumed by the
# LGB, not surfaced). Mf/Aifs are null — the wind feature set has no MétéoFrance
# or AIFS member.
SLOT_NWP = {
    "Gfs": "gfs_seamless", "Ecmwf": "ecmwf_ifs025", "Icon": "icon_seamless",
    "Mf": "meteofrance_seamless", "Ukmo": "ukmo_seamless",
    "Gem": "gem_seamless", "Aifs": "ecmwf_aifs025_single",
}
SLOT_ORDER = ["Gfs", "Ecmwf", "Icon", "Mf", "Ukmo", "Gem", "Aifs"]
# The 6 NWPs whose wsp feeds Mean/Std/Range (the wind feature recipe's set).
SPREAD_NWPS = PM.NWPS  # gfs/ecmwf/icon/gem/ukmo/jma

LOCATION_DEFAULT = os.environ.get("WB_LOCATION", "bonehill_rocks")
log = logging.getLogger("predict_wind_speed_pi")

_VERSION_RE = re.compile(r"^v\d{4}-\d{2}-\d{2}_\d{6}_wind_speed_lgb$")


def find_latest_bundle(models_root: Path, location: str) -> Path:
    parent = models_root / TARGET / location
    if not parent.is_dir():
        raise FileNotFoundError(
            f"No bundle dir under {parent}. Run train_wind_speed_pi.py first.")
    candidates = sorted(
        (d for d in parent.iterdir()
         if d.is_dir() and _VERSION_RE.match(d.name)),
        key=lambda d: d.name, reverse=True,
    )
    for c in candidates:
        if not (c / "calibration.json").is_file(): continue
        if not (c / "feature_schema.json").is_file(): continue
        if not (c / "training_metadata.json").is_file(): continue
        # At least one lead must have its FULL q_lo/q_med/q_hi triple — the
        # per-lead loop below skips incomplete leads, so a bundle with no
        # complete triple would silently emit nothing.
        if not any(all((c / f"q_{tag}_lead_{L}h.txt").is_file()
                       for tag in ("lo", "med", "hi"))
                   for L in LEADS):
            continue
        return c
    raise FileNotFoundError(
        f"No usable *_wind_speed_lgb bundle under {parent}. Re-run train.")


def _feature_hash(vec: np.ndarray) -> str:
    """Provenance hash of the feature row (Python-side; intentionally NOT
    bit-compatible with the .NET FeatureHashing.HashFloats — nothing downstream
    compares hashes across runtimes, it's audit metadata only)."""
    b = np.asarray(vec, dtype=np.float64).tobytes()
    return "py:" + hashlib.sha1(b).hexdigest()[:16]


def predict_for_location(location: str, anchor: datetime,
                         models_root: Path, predictions_root: Path) -> int:
    bundle = find_latest_bundle(models_root, location)
    version = bundle.name
    log.info("Using bundle %s", bundle)

    schema = json.loads((bundle / "feature_schema.json").read_text())
    calibration = json.loads((bundle / "calibration.json").read_text())
    feature_names: list[str] = schema["FeatureNames"]
    oro = PM.load_oro_static(location)
    prediction_made_at = datetime.now(timezone.utc).replace(microsecond=0)

    out_rows: list[dict] = []
    for lead in LEADS:
        lead_key = str(lead)
        booster_paths = {tag: bundle / f"q_{tag}_lead_{lead}h.txt"
                         for tag in ("lo", "med", "hi")}
        if not all(p.is_file() for p in booster_paths.values()):
            log.warning("  lead %dh: booster(s) missing — skipping.", lead)
            continue
        if lead_key not in calibration["per_lead"]:
            log.warning("  lead %dh: calibration missing — skipping.", lead)
            continue

        df = PM.build_live_for_lead(location, lead, anchor, oro, feature_names)
        if df.empty:
            continue

        # Raw feature matrix in canonical order — LightGBM handles NaN natively,
        # so (unlike the MVN MLP) there is no scaler/median-impute step and we
        # predict on every cell build_live returns (max availability; a missing
        # NWP just leaves NaNs the boosters split on). build_live already drops
        # rows that are all-NaN across feature_names.
        X = df[feature_names].to_numpy(dtype=np.float64)

        boosters = {tag: lgb.Booster(model_file=str(p))
                    for tag, p in booster_paths.items()}
        Q = float(calibration["per_lead"][lead_key]["conformal_q"])
        q50 = boosters["med"].predict(X)
        lo = boosters["lo"].predict(X) - Q
        hi = boosters["hi"].predict(X) + Q
        lo = np.maximum(0.0, lo)  # wind speed can't be negative

        # Per-NWP wsp + runtime slots (Gfs..Aifs); Mf/Aifs absent in wind set.
        def col(prefix: str, nwp: str):
            c = f"{prefix}_{nwp}"
            return df[c] if c in df.columns else None

        # Mean/Std/Range over the 6-NWP wsp ensemble (matches the .NET row's
        # wsp-only spread semantics).
        wsp_cols = [f"WindSpeed10m_{m}" for m in SPREAD_NWPS
                    if f"WindSpeed10m_{m}" in df.columns]
        wsp_mat = df[wsp_cols].to_numpy(dtype=np.float64) if wsp_cols else None

        for i, valid in enumerate(df["ValidTimeUtc"].values):
            row = {
                "LocationName":        location,
                "Element":             "wind",
                "ModelVersion":        version,
                "PredictionMadeAtUtc": prediction_made_at,
                "ValidTimeUtc":        pd.Timestamp(valid).to_pydatetime(),
                "LeadHours":           int(lead),
                "BlendValue":          float(q50[i]),
                # CQR band sidecar (new columns; union_by_name → .NET ignores).
                "BandLoMs":            float(lo[i]),
                "BandHiMs":            float(hi[i]),
                "ConformalQ":          Q,
                "FeatureVectorHash":   _feature_hash(X[i]),
            }
            # Per-NWP wsp slots + run-times.
            for slot in SLOT_ORDER:
                nwp = SLOT_NWP[slot]
                wsp_c = col("WindSpeed10m", nwp)
                run_c = col("RunTime", nwp)
                wv = wsp_c.iloc[i] if wsp_c is not None else float("nan")
                rv = run_c.iloc[i] if run_c is not None else None
                row[f"Model{slot}"] = (None if (wv is None or pd.isna(wv))
                                       else float(wv))
                row[f"RunTime{slot}"] = (pd.Timestamp(rv).to_pydatetime()
                                         if (rv is not None and pd.notna(rv))
                                         else None)
            # Mean/Std/Range over present wsp.
            if wsp_mat is not None:
                w = wsp_mat[i]
                w = w[~np.isnan(w)]
                if w.size:
                    row["Mean"] = float(np.mean(w))
                    row["Std"] = float(np.std(w))
                    row["Range"] = float(np.max(w) - np.min(w))
                else:
                    row["Mean"] = row["Std"] = row["Range"] = None
            else:
                row["Mean"] = row["Std"] = row["Range"] = None
            out_rows.append(row)

        log.info("  lead %dh: %d rows (mean q50 %.2f m/s, Q=%.3f, mean width %.2f)",
                 lead, len(df), float(np.mean(q50)), Q, float(np.mean(hi - lo)))

    if not out_rows:
        log.error("No predictions produced for %s — exit 3.", location)
        return 3

    new_df = pd.DataFrame(out_rows)
    date_str = anchor.strftime("%Y-%m-%d")
    out_path = (predictions_root / TARGET
                / f"model_version={version}" / f"date={date_str}"
                / "predictions.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Element tree has NO location subdir, so multiple locations share the file.
    # Merge: drop any prior rows for THIS location (this cycle supersedes them),
    # keep other locations' rows, append the new ones. Bonehill-only today, but
    # this keeps a future multi-location wind safe. Mirrors the overwrite-per-
    # location semantics of the .NET writer at this granularity.
    if out_path.is_file():
        try:
            prev = pd.read_parquet(out_path)
            prev = prev[prev["LocationName"] != location]
            merged = pd.concat([prev, new_df], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("  could not merge existing parquet (%s) — overwriting.", exc)
            merged = new_df
    else:
        merged = new_df
    merged = merged.sort_values(["ValidTimeUtc", "LeadHours"]).reset_index(drop=True)
    merged.to_parquet(out_path, index=False)
    log.info("Wrote %d rows (%d new for %s) → %s",
             len(merged), len(new_df), location, out_path)
    return 0


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

    log.info("wind_speed_lgb — PREDICT (CQR quantile-LGB + cross-conformal)")
    log.info("  location: %s", args.location)
    log.info("  anchor:   %s", anchor.isoformat())

    rc = predict_for_location(
        args.location, anchor, Path(args.models_root), Path(args.predictions_root))
    sys.exit(rc)


if __name__ == "__main__":
    main()
