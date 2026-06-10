#!/bin/bash
# Chained overnight tests for 2026-05-29 -> 30:
#   1) Test 2: per-station 3o LightGBM (Python) — fast, ~10 min
#   2) Test 1a: per-station rich+oro+dynlee BART, 4 stations seq — ~40 min
#   3) Test 1b: per-station rich+oro+v3-climatology BART, 4 stations seq — ~40 min
# Then merge + aligned comparison + summary_overnight.md.
#
# Designed to run in background, no foreground IO required.

set -uo pipefail
cd /c/Projects/Weather/WeatherProbabilistic

D=reports/pooled_oro_4a_bakeoff_2026-05-29
PY=.venv/Scripts/python.exe

echo "=== BACKUP CURRENT STATE ==="
cp $D/per_cell_brier.csv     $D/per_cell_brier_pre_overnight.csv
cp $D/test_predictions.parquet $D/test_predictions_pre_overnight.parquet

# ---- TEST 2: per-station 3o LightGBM (Python) -----------------------------
echo "=== TEST 2: per-station 3o LightGBM ==="
date -u +"start: %Y-%m-%d %H:%M:%S UTC"
$PY scripts/run_3o_per_station.py 2>&1 | tee $D/test2_log.txt
date -u +"end:   %Y-%m-%d %H:%M:%S UTC"

# ---- TEST 1a: per-station BART, dynlee variant ----------------------------
echo ""
echo "=== TEST 1a: per-station BART rich+oro+dynlee ==="
for stn in ea_bellever_dartmoor ea_dartmoor_nr_hexworthy ea_bovey_tracey ea_princetown; do
    date -u +"%Y-%m-%d %H:%M:%S UTC — dynlee $stn"
    WB_STATIONS=$stn $PY scripts/run_phase6_dbarts_pooled_oro.py \
        --variants rich-pooled-terrain-dynlee --leads 24 48 72 2>&1 | tail -8
    cp $D/per_cell_brier.csv     $D/per_cell_brier_dynlee_${stn}.csv
    cp $D/test_predictions.parquet $D/test_predictions_dynlee_${stn}.parquet
done

# ---- TEST 1b: per-station BART, v3 climatology variant --------------------
echo ""
echo "=== TEST 1b: per-station BART rich+oro+v3-climatology ==="
for stn in ea_bellever_dartmoor ea_dartmoor_nr_hexworthy ea_bovey_tracey ea_princetown; do
    date -u +"%Y-%m-%d %H:%M:%S UTC — v3 $stn"
    WB_STATIONS=$stn $PY scripts/run_phase6_dbarts_pooled_oro.py \
        --variants rich-pooled-terrain-v3 --leads 24 48 72 2>&1 | tail -8
    cp $D/per_cell_brier.csv     $D/per_cell_brier_v3_${stn}.csv
    cp $D/test_predictions.parquet $D/test_predictions_v3_${stn}.parquet
done

# ---- MERGE + ALIGNED COMPARISON -------------------------------------------
echo ""
echo "=== MERGE + COMPARISON ==="
$PY - <<'PYEOF'
import pandas as pd
from pathlib import Path
d = Path("reports/pooled_oro_4a_bakeoff_2026-05-29")

base = pd.read_csv(d / "per_cell_brier_pre_overnight.csv")
base_pq = pd.read_parquet(d / "test_predictions_pre_overnight.parquet")

STATIONS = ["ea_bellever_dartmoor", "ea_dartmoor_nr_hexworthy",
            "ea_bovey_tracey", "ea_princetown"]
for tag, label in [("dynlee", "rich-per-station-dynlee"),
                   ("v3",     "rich-per-station-v3")]:
    for stn in STATIONS:
        csv_path = d / f"per_cell_brier_{tag}_{stn}.csv"
        pq_path  = d / f"test_predictions_{tag}_{stn}.parquet"
        if csv_path.exists():
            df = pd.read_csv(csv_path); df["variant"] = label
            base = pd.concat([base, df], ignore_index=True)
        if pq_path.exists():
            df = pd.read_parquet(pq_path); df["variant"] = label
            base_pq = pd.concat([base_pq, df], ignore_index=True)

(base.sort_values(["variant","lead","station"]).reset_index(drop=True)
     .to_csv(d / "per_cell_brier.csv", index=False))
base_pq.to_parquet(d / "test_predictions.parquet", index=False)
print(f"final csv: {len(base)} rows ({base.variant.nunique()} variants)")
PYEOF

$PY scripts/build_step2_comparison.py 2>&1 | tee $D/comparison_overnight.txt

# ---- BUILD SUMMARY ---------------------------------------------------------
echo ""
echo "=== WRITE summary_overnight.md ==="
$PY - <<'PYEOF'
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

d = Path("reports/pooled_oro_4a_bakeoff_2026-05-29")
out = d / "summary_overnight.md"

# 1) Test 2 results
test2 = pd.read_csv(d / "test2_3o_per_station_vs_pooled.csv")
piv = test2.pivot_table(index=["station","lead"], columns="mode", values="brier").reset_index()
piv["delta_brier"] = piv["per-station"] - piv["pooled"]
piv["delta_pct"]   = (piv["delta_brier"] / piv["pooled"] * 100).round(2)
test2_agg = piv.groupby("lead", as_index=False).agg(
    mean_pooled=("pooled","mean"), mean_per_station=("per-station","mean"),
    mean_delta_pct=("delta_pct","mean")).round(4)

# 2) BART per-station variants — pull from per_cell_brier.csv
brier = pd.read_csv(d / "per_cell_brier.csv")
focus_variants = ["rich-per-station", "rich-per-station-dynlee", "rich-per-station-v3"]
brier_focus = brier[brier.variant.isin(focus_variants)].copy()

# 3) Aligned vs production 4a — parse from build_step2_comparison output
import subprocess, sys
cmp_out = subprocess.run(
    [".venv/Scripts/python.exe", "scripts/build_step2_comparison.py"],
    capture_output=True, text=True).stdout

lines = ["# Overnight test results — 2026-05-29 -> 30",
         "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC.",
         "",
         "## Test 2 — per-station 3o LightGBM vs pooled 3o LightGBM (Python)",
         "",
         "_Python LightGBM (4.x), NOT ML.NET. Absolute Brier won't match production 3o; "
         "per-station-vs-pooled DELTA is the structural answer._",
         "",
         "### Per-cell",
         "",
         piv.round(4).to_markdown(index=False),
         "",
         "### Aggregate per lead (mean across 4 stations)",
         "",
         test2_agg.to_markdown(index=False),
         "",
         "## Test 1 — per-station BART variants (aligned vs production 4a)",
         "",
         "_All three variants are per-station rich+oro BART. Differs only in the extra "
         "feature block on top of v1's 9 terrain features:_",
         "",
         "- **rich-per-station** = rich (59) + v1 oro (9) = 68 features (today's win)",
         "- **rich-per-station-dynlee** = + 1 dynamic lee_obstr feature = 69",
         "- **rich-per-station-v3** = + 6 dynamic climatology features = 74",
         "",
         "### Raw Brier (4 stations × 3 leads × 3 variants)",
         "",
         brier_focus[brier_focus.lead.isin([24,48,72])][
             ["lead","variant","station","brier","bss","n_test","wall_s"]
         ].sort_values(["station","lead","variant"]).reset_index(drop=True).to_markdown(index=False),
         "",
         "### Aligned vs production 4a — full output from build_step2_comparison.py",
         "",
         "```",
         cmp_out,
         "```",
         ""]
out.write_text("\n".join(lines))
print(f"wrote {out}")
PYEOF

echo ""
echo "=== ALL DONE ==="
date -u +"finished: %Y-%m-%d %H:%M:%S UTC"
