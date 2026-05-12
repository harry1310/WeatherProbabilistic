"""Phase 4b — per-cycle live predict step.

4b is the arithmetic mean of phase 4a + phase 3e P(wet) — see
``mint_4b_bundle.py`` for the bundle minting and the bake-off
rationale. This script runs at every predict-and-render tick AFTER
predict_4a (Python) and predict_4e via the .NET PrecipPredictCommand
(--feature-set mlp output) have written their parquets, then
inner-joins, averages, and writes a phase-4b parquet at the same
hive path the rest of the prediction tree uses.

For each Bonehill station:
  1. Find the latest 4a + 3e bundle on disk (via MANIFEST.Active
     plus the predict tree's most-recent `date=` partition).
  2. Read the most recent prediction parquet for both phases.
  3. Inner-join on (LocationName, TruthStation, ValidTimeUtc,
     LeadHours, PredictionMadeAtUtc).
  4. Average ProbWet column. Carry the rest of the row from 4a
     (the per-NWP precip columns, climatology, agreement, etc) so
     downstream consumers see the same schema as any other phase.
  5. Write to ``data/predictions/precipitation/{station}/
     model_version={4b_bundle_version}/date={anchor_date}/
     predictions.parquet`` where the 4b bundle version comes from
     MANIFEST.Active.

The 4b bundle version doesn't change every cycle — only when
mint_4b_bundle.py runs (after Sunday retrain). So all 4b
predictions for a week of cycles land under the same
model_version=v..._phase4b/ partition, just different date=*/
sub-partitions, identical to how 4a + 3e work.

If either 4a or 3e is missing for this cycle, the corresponding
station is skipped with a warning rather than emitted with bogus
data — the 2-way is undefined without both components.

Usage (mirrors predict_4a.py)::

    python scripts/predict_4b.py [--for-date YYYY-MM-DD] \
        [--stations slug1,slug2,...] \
        [--predictions-root /path/to/data/predictions] \
        [--models-root /path/to/data/models/precipitation]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, date, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STATIONS = ["ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"]
ACTIVE_LOCATION = os.environ.get("WB_LOCATION", "bonehill_rocks")

# Bundle-name suffix patterns used to identify each component on disk.
SUFFIX_4A = "_phase4a"
SUFFIX_3E = "_phase3e"
SUFFIX_4B = "_phase4b"


def find_4b_bundle_version(models_root: Path, station: str) -> str | None:
    """Return the 4b bundle version this station's MANIFEST.Active
    points at — the version we'll tag the synthesised parquet with.

    The MANIFEST is shared across stations (it's at
    ``models_root/MANIFEST.json``). Each station entry's Active array
    is parsed; we return the only ``*_phase4b`` entry (or None if 4b
    isn't registered for this station yet).
    """
    manifest = models_root / "MANIFEST.json"
    if not manifest.exists():
        return None
    m = json.loads(manifest.read_text())
    entry = m.get("Stations", {}).get(station, {})
    active = entry.get("Active", [])
    for v in active:
        if v.endswith(SUFFIX_4B):
            return v
    return None


def latest_prediction_parquet(
    predictions_root: Path, station: str, suffix: str, for_date: date | None,
) -> Path | None:
    """Find the per-cycle prediction parquet for the latest
    model_version ending with ``suffix`` (e.g. ``_phase4a``) that has
    a ``date={anchor_date}`` partition. When ``for_date`` is None,
    the newest date partition on disk is used.
    """
    station_dir = predictions_root / "precipitation" / station
    if not station_dir.is_dir():
        return None
    # Pick the newest model_version=X dir matching the suffix.
    version_dirs = [
        d for d in station_dir.iterdir()
        if d.is_dir()
        and d.name.startswith("model_version=")
        and d.name[len("model_version="):].endswith(suffix)
    ]
    if not version_dirs:
        return None
    version_dir = max(version_dirs, key=lambda d: d.name)
    # Pick the date partition.
    if for_date is None:
        date_dirs = [d for d in version_dir.iterdir() if d.is_dir() and d.name.startswith("date=")]
        if not date_dirs:
            return None
        date_dir = max(date_dirs, key=lambda d: d.name)
    else:
        date_dir = version_dir / f"date={for_date.isoformat()}"
        if not date_dir.is_dir():
            return None
    parquet = date_dir / "predictions.parquet"
    return parquet if parquet.exists() else None


def synthesise_for_station(
    predictions_root: Path,
    models_root: Path,
    station: str,
    for_date: date | None,
) -> tuple[bool, str]:
    """Produce the 4b prediction parquet for one station.
    Returns ``(ok, note)``."""
    version_4b = find_4b_bundle_version(models_root, station)
    if version_4b is None:
        return False, f"{station}: MANIFEST.Active has no *_phase4b entry; run mint_4b_bundle.py first"

    parquet_4a = latest_prediction_parquet(predictions_root, station, SUFFIX_4A, for_date)
    parquet_3e = latest_prediction_parquet(predictions_root, station, SUFFIX_3E, for_date)
    if parquet_4a is None:
        return False, f"{station}: no 4a predictions parquet for the target date — skipping (4b is undefined without both components)"
    if parquet_3e is None:
        return False, f"{station}: no 3e predictions parquet for the target date — skipping"

    df_4a = pd.read_parquet(parquet_4a)
    df_3e = pd.read_parquet(parquet_3e)

    # Coerce datetime + integer dtypes to a single shape before merge.
    # Pandas refuses to merge `datetime64[us, UTC]` (Python-side
    # writers) against `datetime64[ns]` (Parquet.NET on the .NET side);
    # LeadHours can also come back as int64 vs int32 across writers.
    for df in (df_4a, df_3e):
        for col in ("ValidTimeUtc", "PredictionMadeAtUtc"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
        if "LeadHours" in df.columns:
            df["LeadHours"] = df["LeadHours"].astype("int32")

    # 4a and 3e DON'T share a PredictionMadeAtUtc — 4a is on its own
    # 6-hourly cron, 3e is on the .NET predict-and-render 2-hourly
    # cron. They DO share (station, valid_time, lead), and that's the
    # natural join key for "what does the 2-way mean predict for this
    # forecast hour?". Take the latest cycle's prediction from each
    # parquet per (valid_time, lead) (the parquet may carry multiple
    # cycles' rows for the same valid_time on the in-day merge), then
    # inner-join.
    def latest_per_cell(df):
        return (df.sort_values("PredictionMadeAtUtc")
                  .drop_duplicates(["TruthStation", "ValidTimeUtc", "LeadHours"], keep="last"))
    df_4a_latest = latest_per_cell(df_4a)
    df_3e_latest = latest_per_cell(df_3e)

    join_keys = ["LocationName", "TruthStation", "ValidTimeUtc", "LeadHours"]
    j = df_4a_latest.merge(
        df_3e_latest[join_keys + ["ProbWet", "PredictionMadeAtUtc"]].rename(
            columns={"ProbWet": "ProbWet_3e", "PredictionMadeAtUtc": "PredictionMadeAtUtc_3e"},
        ),
        on=join_keys, how="inner",
    )
    if j.empty:
        return False, (
            f"{station}: inner-join of 4a + 3e parquets produced 0 rows "
            f"(no shared valid_time + lead — possibly stale data, "
            f"check that both phases ran for the target date)"
        )

    # 4b prediction = arithmetic mean. Keep the rest of the columns
    # from 4a (per-NWP precip rates, ensemble stats, climatology,
    # agreement) so downstream render reads identical schema. PMT
    # becomes synthesis-time so the renderer's "freshness" filter
    # treats 4b consistently against the .NET PMT clock.
    j["ProbWet"] = (j["ProbWet"] + j["ProbWet_3e"]) / 2.0
    j["ModelVersion"] = version_4b
    synthesis_time = pd.Timestamp.utcnow().tz_localize(None)
    j["PredictionMadeAtUtc"] = synthesis_time
    j = j.drop(columns=["ProbWet_3e", "PredictionMadeAtUtc_3e"])

    # 4a writes quantile + CI columns (Q05/Q95/Ci80Width/Ci90Width)
    # from its BART posterior. Those don't have a defined meaning for
    # a non-posterior arithmetic mean — null them so downstream
    # readers see "this phase has no band" rather than misleading 4a
    # quantiles. Same logic for ProbWetStd (which 4a writes).
    for col in ("ProbWetStd", "ProbWetQ05", "ProbWetQ95", "Ci80Width", "Ci90Width"):
        if col in j.columns:
            j[col] = np.nan

    # Anchor date from PredictionMadeAtUtc (which is identical across
    # rows in a single cycle; pick the first).
    anchor_dt = pd.to_datetime(j["PredictionMadeAtUtc"].iloc[0])
    anchor_date = anchor_dt.date().isoformat()
    out_dir = (predictions_root / "precipitation" / station
               / f"model_version={version_4b}" / f"date={anchor_date}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.parquet"

    # Merge with existing parquet at this path (mirrors the .NET
    # PrecipPredictCommand convention — multiple cycles per day land
    # in the same file, deduped on (PredictionMadeAtUtc, LeadHours,
    # ValidTimeUtc)). Keeps the in-day history viewable.
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, j], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["PredictionMadeAtUtc", "LeadHours", "ValidTimeUtc"],
            keep="last",
        )
    else:
        combined = j
    combined = combined.sort_values(["ValidTimeUtc", "LeadHours"]).reset_index(drop=True)
    combined.to_parquet(out_path, index=False)

    return True, (
        f"{station}: wrote {len(j):,} new rows to {out_path.name} "
        f"(file now holds {len(combined):,}); model_version={version_4b}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--stations", default=",".join(DEFAULT_STATIONS),
        help="Comma-separated station slugs (default: 3 Bonehill stations).",
    )
    ap.add_argument(
        "--predictions-root", required=True,
        help="Path to WeatherBlend's data/predictions directory.",
    )
    ap.add_argument(
        "--models-root", required=True,
        help="Path to WeatherBlend's data/models/precipitation directory "
             "(MANIFEST.json lives here).",
    )
    ap.add_argument(
        "--for-date", default=None,
        help="Target date (YYYY-MM-DD) — overrides the 'latest on disk' "
             "default. Used by backfill-style runs that re-emit older "
             "cycles.",
    )
    args = ap.parse_args()

    predictions_root = Path(args.predictions_root)
    models_root = Path(args.models_root)
    if not predictions_root.is_dir():
        print(f"::error::predictions-root not a directory: {predictions_root}")
        return 2
    if not models_root.is_dir():
        print(f"::error::models-root not a directory: {models_root}")
        return 2

    stations = [s.strip() for s in args.stations.split(",")]
    if args.for_date:
        for_date = datetime.strptime(args.for_date, "%Y-%m-%d").date()
    else:
        for_date = None

    print(f"Predict Phase 4b: location={ACTIVE_LOCATION}, stations={stations}, "
          f"for_date={for_date or 'latest'}")
    print(f"  predictions_root: {predictions_root}")
    print(f"  models_root:      {models_root}\n")

    n_ok = 0
    n_skip = 0
    for station in stations:
        ok, note = synthesise_for_station(predictions_root, models_root, station, for_date)
        prefix = "OK " if ok else "SKIP"
        print(f"  {prefix} {note}")
        if ok:
            n_ok += 1
        else:
            n_skip += 1

    print(f"\nDone. Wrote: {n_ok}, skipped: {n_skip}.")
    # Soft-skip exit code 3 (matches PrecipPredictCommand's convention)
    # when nothing was written — predict-and-render's outer step treats
    # 3 as non-fatal.
    if n_ok == 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
