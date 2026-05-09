"""Replicate ComputeRollingMae locally to find why lead 96 has no point at 02/05."""
import sys
import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect(":memory:")

pred_glob = r"C:/Users/rhcsl/AppData/Local/Temp/temp_preds_recent/**/predictions.parquet"
era5_glob = r"C:/Users/rhcsl/AppData/Local/Temp/era5_recent/**/data.parquet"

pred = con.execute(f"""
    SELECT * FROM read_parquet('{pred_glob}', hive_partitioning=true, union_by_name=true)
""").fetch_df()
print(f"Total predictions in v2026-04-28_232613: {len(pred)}")
print(f"  cols: {list(pred.columns)[:8]}")
print(f"  leads: {sorted(pred['LeadHours'].unique().tolist())}")
print(f"  ValidTime: {pred['ValidTimeUtc'].min()} to {pred['ValidTimeUtc'].max()}")
print()

era5 = con.execute(f"""
    SELECT ValidTimeUtc, Temperature2m
    FROM read_parquet('{era5_glob}', hive_partitioning=true, union_by_name=true)
    ORDER BY ValidTimeUtc
""").fetch_df()
print(f"ERA5 truth: {len(era5)} rows; range {era5['ValidTimeUtc'].min()} to {era5['ValidTimeUtc'].max()}")
print()

# Pair by ValidTimeUtc
truth_by_time = dict(zip(era5["ValidTimeUtc"], era5["Temperature2m"]))
pred["truth"] = pred["ValidTimeUtc"].map(truth_by_time)
paired = pred.dropna(subset=["truth", "BlendTemperature"]).copy()
paired["abs_err"] = (paired["BlendTemperature"] - paired["truth"]).abs()
print(f"Paired: {len(paired)} of {len(pred)} predictions matched ERA5")

# Per-lead breakdown of paired rows
print("\nPaired rows by lead:")
print(paired.groupby("LeadHours").size().to_string())

print("\nPaired rows by lead, by valid_time date:")
paired["valid_date"] = pd.to_datetime(paired["ValidTimeUtc"]).dt.date
piv = paired.groupby(["LeadHours", "valid_date"]).size().unstack(fill_value=0)
print(piv.to_string())

# Now simulate the rolling chart for lead 96 vs 120 at WindowEnd = 2026-05-02 23:59
print("\n\nRolling 14d window ending 2026-05-02 23:59 (the chart point user is asking about):")
window_end = pd.Timestamp("2026-05-02 23:59:59.999")
window_start = window_end - pd.Timedelta(days=14)
print(f"  window: {window_start} to {window_end}")
in_window = paired[(paired["ValidTimeUtc"] >= window_start) & (paired["ValidTimeUtc"] <= window_end)]
print(f"\n  paired rows in this window per lead:")
print(in_window.groupby("LeadHours").size().to_string())

# Per (phase placeholder, lead) — we need to know phase but for v2026-04-28_232613 we hardcode 2b
phase = "2b"  # active for this version
print(f"\n  → For phase '{phase}', leads producing rolling-MAE point at WindowEnd 2026-05-02:")
for lead in sorted(in_window["LeadHours"].unique()):
    sub = in_window[in_window["LeadHours"] == lead]
    if len(sub) > 0:
        mae = sub["abs_err"].mean()
        print(f"     Lead {lead:3d}h: N={len(sub):3d}  MAE={mae:.3f}°C")

con.close()
