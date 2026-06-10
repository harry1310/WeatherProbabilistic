"""Merge the 24h backup with the freshly-written 48h+72h artefacts.

Step 1's smoke wrote per_cell_brier.csv / test_predictions.parquet / summary.txt
for lead 24h only. We backed those up as *_24h.* before re-launching the runner
with --leads 48 72. The runner overwrites (doesn't append), so the freshly-
written files contain only 48h+72h. This script stitches the halves together
so step2_comparison can run across all 3 leads.

Idempotent — running twice does the right thing because we only merge if the
backup exists and only ever produce the union (no duplicates by lead key).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BAKEOFF_DIR = ROOT / "reports" / "pooled_oro_4a_bakeoff_2026-05-29"


def merge_csv(new_path: Path, backup_path: Path, out_path: Path) -> None:
    if not backup_path.exists():
        print(f"  no backup at {backup_path.name} — skipping")
        return
    new = pd.read_csv(new_path) if new_path.exists() else pd.DataFrame()
    old = pd.read_csv(backup_path)
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["lead", "variant", "station"], keep="last")
    combined = combined.sort_values(["variant", "lead", "station"]).reset_index(drop=True)
    combined.to_csv(out_path, index=False)
    print(f"  wrote {out_path.name}: {len(old)} backup + {len(new)} new -> {len(combined)} unique")


def merge_parquet(new_path: Path, backup_path: Path, out_path: Path) -> None:
    if not backup_path.exists():
        print(f"  no backup at {backup_path.name} — skipping")
        return
    new = pd.read_parquet(new_path) if new_path.exists() else pd.DataFrame()
    old = pd.read_parquet(backup_path)
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["valid_time", "station", "lead", "variant"], keep="last")
    combined = combined.sort_values(["variant", "lead", "station", "valid_time"]).reset_index(drop=True)
    combined.to_parquet(out_path, index=False)
    print(f"  wrote {out_path.name}: {len(old)} backup + {len(new)} new -> {len(combined)} unique")


def main() -> None:
    print(f"Merging artefacts in {BAKEOFF_DIR}")
    if not BAKEOFF_DIR.exists():
        print(f"ERROR: {BAKEOFF_DIR} not found", file=sys.stderr)
        sys.exit(1)

    merge_csv(BAKEOFF_DIR / "per_cell_brier.csv",
              BAKEOFF_DIR / "per_cell_brier_24h.csv",
              BAKEOFF_DIR / "per_cell_brier.csv")

    merge_parquet(BAKEOFF_DIR / "test_predictions.parquet",
                  BAKEOFF_DIR / "test_predictions_24h.parquet",
                  BAKEOFF_DIR / "test_predictions.parquet")

    print("\nNow re-run scripts/build_step2_comparison.py to refresh "
          "step2_comparison.csv + summary.txt aligned across all 3 leads.")


if __name__ == "__main__":
    main()
