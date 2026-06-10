"""Hermetic smoke for predict_wind_speed_pi.py.

Builds a tiny synthetic forecast tree + copies the real latest *_wind_speed_lgb
bundle and the location's orographic JSON into a throwaway WEATHERBLEND_DATA_ROOT,
runs the predictor as a subprocess, and asserts the output parquet has the exact
ElementPredictionRow schema wind_blend consumes plus the CQR band sidecar, with
sane values (band brackets the point, point in a physical wind range, all leads
present). Fast + self-contained — does NOT touch the real 40k-file forecast tree
or the production predictions prefix.

Exit 0 = pass. Run: ``.venv\\Scripts\\python.exe scripts/smoke_wind_speed_pi.py``
(optionally ``--bundle <dir>`` to point at a specific bundle; default = newest on
the real models tree).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REAL_DATA = Path(os.environ.get("WEATHERBLEND_DATA_ROOT",
                                "C:/Projects/Weather/WeatherBlend/data"))
LOCATION = "bonehill_rocks"
LEADS = (24, 48, 72)
NWPS = ("gfs_seamless", "ecmwf_ifs025", "icon_seamless",
        "gem_seamless", "ukmo_seamless", "jma_seamless")
_VERSION_RE = re.compile(r"^v\d{4}-\d{2}-\d{2}_\d{6}_wind_speed_lgb$")

REQUIRED_COLS = {
    "LocationName", "Element", "ModelVersion", "PredictionMadeAtUtc",
    "ValidTimeUtc", "LeadHours", "BlendValue",
    "ModelGfs", "ModelEcmwf", "ModelIcon", "ModelMf", "ModelUkmo",
    "ModelGem", "ModelAifs",
    "RunTimeGfs", "RunTimeEcmwf", "RunTimeIcon", "RunTimeMf", "RunTimeUkmo",
    "RunTimeGem", "RunTimeAifs",
    "Mean", "Std", "Range", "FeatureVectorHash",
    "BandLoMs", "BandHiMs", "ConformalQ",   # CQR sidecar
}


def _newest_bundle() -> Path:
    parent = REAL_DATA / "models" / "wind" / LOCATION
    cands = sorted((d for d in parent.iterdir()
                    if d.is_dir() and _VERSION_RE.match(d.name)
                    and (d / "q_med_lead_24h.txt").is_file()),
                   key=lambda d: d.name, reverse=True)
    if not cands:
        raise SystemExit(f"FAIL: no *_wind_speed_lgb bundle under {parent} "
                         "(train_wind_speed_pi.py must run first).")
    return cands[0]


def _synth_forecasts(root: Path, anchor: datetime) -> None:
    """One parquet with plausible rows for every (NWP, ValidTime) across the
    three lead-day windows the predictor reads (anchor+1/+2/+3 days, 24h each)."""
    rng = np.random.RandomState(7)
    rows = []
    for lead in LEADS:
        day0 = anchor + timedelta(days=lead // 24)
        for h in range(24):
            valid = day0 + timedelta(hours=h)
            for m in NWPS:
                wsp = float(4.0 + rng.rand() * 6.0)        # 4-10 m/s
                rows.append(dict(
                    Model=m, ValidTimeUtc=valid, RunTimeUtc=anchor,
                    RunTimeSource="reported",
                    WindSpeed10m=wsp,
                    WindDirection10m=float(rng.rand() * 360.0),
                    WindGusts10m=wsp + float(rng.rand() * 4.0),
                    Temperature2m=float(8.0 + rng.rand() * 6.0),
                    DewPoint2m=float(4.0 + rng.rand() * 4.0),
                    SurfacePressure=float(1000.0 + rng.rand() * 20.0),
                    CloudCover=float(rng.rand() * 100.0),
                ))
    df = pd.DataFrame(rows)
    # Partition per-NWP so the hive `model=` path value equals the in-file
    # `Model` column. build_live's read_parquet auto-detects hive partitioning
    # and (case-insensitively) the hive `model` shadows the in-file `Model`, so
    # a single `model=synthetic` file would make `Model IN (...)` match
    # 'synthetic' and return nothing (the CLAUDE.md hive-collision gotcha).
    for m, sub in df.groupby("Model"):
        out = (root / "forecasts" / f"location={LOCATION}"
               / f"model={m}" / "date=smoke")
        out.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(out / "run.parquet", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None)
    args = ap.parse_args()
    bundle = Path(args.bundle) if args.bundle else _newest_bundle()
    version = bundle.name
    print(f"smoke: bundle={version}")

    anchor = datetime(2026, 6, 9, 0, 0, 0)  # fixed; synthetic data is anchored to it
    tmp = Path(tempfile.mkdtemp(prefix="wsp_smoke_"))
    try:
        # 1. synthetic forecasts
        _synth_forecasts(tmp, anchor)
        # 2. orographic JSON
        oro_src = REAL_DATA / "static" / "orographic" / f"{LOCATION}.json"
        oro_dst = tmp / "static" / "orographic" / f"{LOCATION}.json"
        oro_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(oro_src, oro_dst)
        # 3. the bundle
        bdst = tmp / "models" / "wind" / LOCATION / version
        shutil.copytree(bundle, bdst)

        # 4. run predictor as a subprocess with the temp root
        env = dict(os.environ)
        env["WEATHERBLEND_DATA_ROOT"] = str(tmp)
        pred_root = tmp / "predictions"
        cmd = [sys.executable, str(ROOT / "scripts" / "predict_wind_speed_pi.py"),
               "--location", LOCATION, "--anchor", "2026-06-09",
               "--models-root", str(tmp / "models"),
               "--predictions-root", str(pred_root)]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
        print(r.stdout[-1500:])
        if r.returncode != 0:
            print(r.stderr[-2000:])
            print(f"FAIL: predictor exited {r.returncode}")
            return 1

        # 5. assertions
        pq = (pred_root / "wind" / f"model_version={version}"
              / "date=2026-06-09" / "predictions.parquet")
        if not pq.is_file():
            print(f"FAIL: no output parquet at {pq}")
            return 1
        out = pd.read_parquet(pq)
        missing = REQUIRED_COLS - set(out.columns)
        if missing:
            print(f"FAIL: output missing columns: {sorted(missing)}")
            return 1
        leads = set(out["LeadHours"].unique())
        if set(LEADS) - leads:
            print(f"FAIL: missing leads {set(LEADS) - leads} (got {sorted(leads)})")
            return 1
        # band brackets point; point physical; band ordered
        bad_band = out[(out.BandLoMs > out.BlendValue + 1e-6)
                       | (out.BandHiMs < out.BlendValue - 1e-6)]
        if len(bad_band):
            print(f"FAIL: {len(bad_band)} rows where band does not bracket point")
            return 1
        if not ((out.BlendValue >= 0).all() and (out.BlendValue < 60).all()):
            print(f"FAIL: BlendValue out of physical range "
                  f"[{out.BlendValue.min():.2f}, {out.BlendValue.max():.2f}]")
            return 1
        if (out.Element != "wind").any():
            print("FAIL: Element != 'wind'")
            return 1
        if (out.ModelVersion != version).any():
            print("FAIL: ModelVersion mismatch")
            return 1
        print(f"PASS: {len(out)} rows, leads={sorted(leads)}, "
              f"q50 mean={out.BlendValue.mean():.2f} m/s, "
              f"mean band width={(out.BandHiMs - out.BandLoMs).mean():.2f} m/s, "
              f"all bands bracket point, schema complete.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
