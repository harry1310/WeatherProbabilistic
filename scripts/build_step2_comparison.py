"""Step 2 — apples-to-apples comparison: lean-per-cell vs lean-pooled-terrain
vs rich-pooled-terrain, row-aligned on (station, lead, valid_time).

Inputs:
  * Per-cell 4a baseline test_predictions:
      WB/data/models/precipitation/{station}/{latest}_phase4a/test_predictions.parquet
      schema: (valid_time, station, lead, p_wet, observed_wet)
      Cover stations: bellever / bovey / hexworthy from production retrain,
      and ea_princetown from the just-fitted Princetown baseline.

  * Pooled bake-off test_predictions:
      WP/reports/pooled_oro_4a_bakeoff_{date}/test_predictions.parquet
      schema: (valid_time, station, lead, variant, p_wet, observed_wet)
      Covers 4 stations × 5 leads × 2 variants.

Method:
  1. Inner-join per-cell baseline rows with each pooled variant on
     (station, lead, valid_time) — ensures Brier comparison is on identical
     rows.
  2. Per (station, lead, variant): compute Brier on the aligned subset.
     For per-cell baseline, compute Brier on the SAME aligned subset
     (NOT on its full test slice) so the numbers are comparable.
  3. Aggregate per lead (mean across 4 stations) and per station (mean
     across 5 leads).

Output: appends an "Apples-to-apples comparison" section to the latest
pooled_oro_4a_bakeoff_{date}/summary.txt + writes a comparison CSV.

Decision criteria from project_4a_oro_bakeoff_plan.md (3o-style):
  * lean-pooled-terrain or rich-pooled-terrain beats per-cell baseline by
    >=2% Brier at 24h+48h AND >=1% at 72h+96h+120h, averaged across the
    4 Bonehill-pool stations.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402

STATIONS = ["ea_bellever_dartmoor", "ea_dartmoor_nr_hexworthy",
            "ea_bovey_tracey", "ea_princetown"]
LEADS = [24, 48, 72, 96, 120]


def find_latest_4a_bundle(station_slug: str) -> Path | None:
    """Pick the latest `v..._phase4a` bundle on disk for this station that
    has a non-empty `test_predictions.parquet`."""
    base = WEATHERBLEND_DATA_ROOT / "models" / "precipitation" / station_slug
    if not base.exists():
        return None
    cands = sorted([d for d in base.iterdir() if d.is_dir() and d.name.endswith("_phase4a")])
    for d in reversed(cands):  # newest first
        tp = d / "test_predictions.parquet"
        if tp.exists() and tp.stat().st_size > 0:
            return d
    return None


def load_baseline_predictions() -> pd.DataFrame:
    """Concat per-cell 4a test_predictions across all 4 stations. Missing
    bundles are reported but don't fail the comparison — the inner-join later
    just won't have rows for that station."""
    frames = []
    for slug in STATIONS:
        bundle = find_latest_4a_bundle(slug)
        if bundle is None:
            print(f"  WARN: no 4a bundle with test_predictions for {slug}", flush=True)
            continue
        df = pd.read_parquet(bundle / "test_predictions.parquet")
        df["bundle"] = bundle.name
        frames.append(df)
        print(f"  loaded {len(df):,} rows from {bundle.name} ({slug})", flush=True)
    if not frames:
        raise RuntimeError("No baseline predictions loaded — refit per-cell 4a or check paths.")
    return pd.concat(frames, ignore_index=True)


def load_pooled_predictions(bakeoff_dir: Path) -> pd.DataFrame:
    p = bakeoff_dir / "test_predictions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No pooled predictions at {p}")
    df = pd.read_parquet(p)
    print(f"  loaded {len(df):,} pooled rows from {p.name}", flush=True)
    return df


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) -
                          np.asarray(y, dtype=np.float64)) ** 2))


def aligned_comparison(baseline: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    """Inner-join baseline + pooled on (station, lead, valid_time). For each
    pooled variant, compute baseline Brier and variant Brier on the SAME
    aligned subset. Returns a long DataFrame with one row per
    (station, lead, variant)."""
    base = baseline.rename(columns={"p_wet": "p_wet_baseline",
                                    "observed_wet": "observed_wet_baseline"})
    base["valid_time"] = pd.to_datetime(base["valid_time"])
    pooled["valid_time"] = pd.to_datetime(pooled["valid_time"])

    rows = []
    for variant in pooled["variant"].unique():
        sub = pooled[pooled["variant"] == variant]
        merged = sub.merge(base[["valid_time", "station", "lead",
                                 "p_wet_baseline", "observed_wet_baseline"]],
                           on=["valid_time", "station", "lead"], how="inner")
        if merged.empty:
            print(f"  WARN: no aligned rows for variant {variant}", flush=True)
            continue
        # Sanity: observed_wet must match between baseline and pooled for every row.
        mismatch = (merged["observed_wet"] != merged["observed_wet_baseline"]).sum()
        if mismatch:
            print(f"  WARN: {mismatch} truth-label mismatches in {variant} merge — "
                  f"check the dump's Label semantics vs train_4a's", flush=True)
        for (station, lead), g in merged.groupby(["station", "lead"]):
            b_base   = brier(g["p_wet_baseline"].to_numpy(), g["observed_wet"].to_numpy())
            b_pool   = brier(g["p_wet"].to_numpy(),          g["observed_wet"].to_numpy())
            delta    = b_pool - b_base
            delta_pc = (delta / b_base * 100) if b_base > 0 else float("nan")
            rows.append({
                "station": station, "lead": lead, "variant": variant,
                "n_aligned":     int(len(g)),
                "brier_baseline": round(b_base, 4),
                "brier_pooled":   round(b_pool, 4),
                "delta_brier":    round(delta, 4),
                "delta_pct":      round(delta_pc, 2),
            })
    return pd.DataFrame(rows)


def aggregate_per_lead(cmp: pd.DataFrame) -> pd.DataFrame:
    """Mean Δ Brier across stations, per (lead, variant)."""
    agg = cmp.groupby(["lead", "variant"], as_index=False).agg(
        n_cells=("station", "count"),
        mean_brier_baseline=("brier_baseline", "mean"),
        mean_brier_pooled=("brier_pooled", "mean"),
        mean_delta_brier=("delta_brier", "mean"),
        mean_delta_pct=("delta_pct", "mean"),
    )
    return agg.round(4)


def aggregate_overall(cmp: pd.DataFrame) -> pd.DataFrame:
    """Mean Δ across (station, lead) per variant."""
    return cmp.groupby("variant", as_index=False).agg(
        n_cells=("station", "count"),
        mean_brier_baseline=("brier_baseline", "mean"),
        mean_brier_pooled=("brier_pooled", "mean"),
        mean_delta_brier=("delta_brier", "mean"),
        mean_delta_pct=("delta_pct", "mean"),
    ).round(4)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--bakeoff-dir", default=None,
                   help="Pooled bake-off output dir (default: latest pooled_oro_4a_bakeoff_* under WP/reports/).")
    args = p.parse_args()

    if args.bakeoff_dir:
        bakeoff_dir = Path(args.bakeoff_dir)
    else:
        reports = ROOT / "reports"
        cands = sorted([d for d in reports.glob("pooled_oro_4a_bakeoff_*") if d.is_dir()])
        if not cands:
            print("ERROR: no pooled_oro_4a_bakeoff_* directory found under WP/reports/", file=sys.stderr)
            sys.exit(1)
        bakeoff_dir = cands[-1]
    print(f"Comparing pooled bake-off at: {bakeoff_dir}")

    print("\n1. Loading per-cell 4a baseline predictions ...")
    baseline = load_baseline_predictions()

    print("\n2. Loading pooled bake-off predictions ...")
    pooled = load_pooled_predictions(bakeoff_dir)

    print("\n3. Building aligned comparison ...")
    cmp = aligned_comparison(baseline, pooled)
    if cmp.empty:
        print("ERROR: no aligned rows — nothing to compare.", file=sys.stderr)
        sys.exit(1)

    per_lead = aggregate_per_lead(cmp)
    overall  = aggregate_overall(cmp)

    cmp_path = bakeoff_dir / "step2_comparison.csv"
    cmp.to_csv(cmp_path, index=False)
    print(f"\nWrote per-cell comparison: {cmp_path}")

    text_lines = [
        "Step 2 apples-to-apples comparison — pooled vs per-cell baseline",
        "================================================================",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"Bake-off: {bakeoff_dir.name}",
        "",
        "Per-cell rows (baseline vs each pooled variant, row-aligned subset):",
        "",
        cmp.to_string(index=False),
        "",
        "Aggregate per lead (mean across stations):",
        "",
        per_lead.to_string(index=False),
        "",
        "Aggregate overall (mean across stations and leads):",
        "",
        overall.to_string(index=False),
        "",
        "Decision criteria (project_4a_oro_bakeoff_plan.md):",
        "  - >=2% mean Brier improvement at 24h+48h, AND",
        "  - >=1% mean Brier improvement at 72h+96h+120h,",
        "    averaged across the 4 Bonehill-pool stations.",
        "  Negative delta_pct = pooled wins.",
        "",
    ]
    text = "\n".join(text_lines)
    print()
    print(text)

    # Append to summary.txt rather than overwrite — keeps the bake-off log + the comparison together.
    summary_path = bakeoff_dir / "summary.txt"
    existing = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    summary_path.write_text(existing + "\n\n" + text, encoding="utf-8")
    print(f"Appended comparison to: {summary_path}")


if __name__ == "__main__":
    main()
