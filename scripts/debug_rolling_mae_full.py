"""Replicate ComputeRollingMae across ALL temperature versions to find the
96 vs 120 asymmetry."""
import duckdb
import pandas as pd

con = duckdb.connect(":memory:")

pred_glob = r"C:/Users/rhcsl/AppData/Local/Temp/temp_preds_all/**/predictions.parquet"
era5_glob = r"C:/Users/rhcsl/AppData/Local/Temp/era5_recent/**/data.parquet"

pred = con.execute(f"""
    SELECT * FROM read_parquet('{pred_glob}', hive_partitioning=true, union_by_name=true)
""").fetch_df()
print(f"Total predictions across all versions: {len(pred)}")
print(f"Versions: {sorted(pred['ModelVersion'].unique().tolist())}")
print()

era5 = con.execute(f"""
    SELECT ValidTimeUtc, Temperature2m
    FROM read_parquet('{era5_glob}', hive_partitioning=true, union_by_name=true)
""").fetch_df()
truth_by_time = dict(zip(era5["ValidTimeUtc"], era5["Temperature2m"]))

pred["truth"] = pred["ValidTimeUtc"].map(truth_by_time)
paired = pred.dropna(subset=["truth", "BlendTemperature"]).copy()
paired["abs_err"] = (paired["BlendTemperature"] - paired["truth"]).abs()

# Phase mapping — use the in-file Phase if present, otherwise infer from version
def infer_phase(v):
    if "phase2c" in v:  return "2c"
    if "phase2redo" in v: return "2b_redo"
    if "phase2d" in v:  return "2d"
    return "2b"  # default for non-suffixed versions
paired["Phase"] = paired["ModelVersion"].apply(infer_phase)

active_phases = {"2b", "2c", "2d"}
filt = paired[paired["Phase"].isin(active_phases)].copy()
print(f"After active-phase filter (2b/2c/2d): {len(filt)} of {len(paired)} paired rows\n")

# Dedup by (Phase, Lead, ValidTime) preferring freshest PredictionMadeAtUtc
filt = (filt
        .sort_values("PredictionMadeAtUtc", ascending=False)
        .drop_duplicates(["Phase", "LeadHours", "ValidTimeUtc"]))
print(f"After dedup (phase×lead×validtime, freshest wins): {len(filt)}\n")

# Window-end 2026-05-02 23:59
window_end = pd.Timestamp("2026-05-02 23:59:59.999")
window_start = window_end - pd.Timedelta(days=14)
in_window = filt[(filt["ValidTimeUtc"] >= window_start) & (filt["ValidTimeUtc"] <= window_end)]

print(f"Rolling MAE window {window_start} to {window_end}\n")
print(f"Paired rows in window per (Phase, Lead):")
print(in_window.groupby(["Phase", "LeadHours"]).size().to_string())

print("\nMAE per (Phase, Lead):")
mae_per = in_window.groupby(["Phase", "LeadHours"])["abs_err"].agg(["mean", "count"]).rename(columns={"mean": "MAE", "count": "N"})
print(mae_per.to_string())

# Specifically: lead 96 vs 120 details
print("\n\n=== LEAD 96 vs LEAD 120 detail in window ===")
for lead in (96, 120):
    sub = in_window[in_window["LeadHours"] == lead]
    print(f"\nLead {lead}h: {len(sub)} rows")
    if len(sub) > 0:
        print(f"  by Phase: {sub.groupby('Phase').size().to_dict()}")
        print(f"  versions used: {sorted(sub['ModelVersion'].unique().tolist())}")
        print(f"  valid_time range: {sub['ValidTimeUtc'].min()} to {sub['ValidTimeUtc'].max()}")

con.close()
