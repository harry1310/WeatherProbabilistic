"""Find every lead 96 prediction in the dataset to see what valid_times exist."""
import duckdb
import pandas as pd

con = duckdb.connect(":memory:")
pred = con.execute(f"""
    SELECT ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours, BlendTemperature
    FROM read_parquet('C:/Users/rhcsl/AppData/Local/Temp/temp_preds_all/**/predictions.parquet',
                      hive_partitioning=true, union_by_name=true)
    WHERE LeadHours = 96
    ORDER BY ValidTimeUtc
""").fetch_df()

print(f"All lead-96 predictions: {len(pred)}")
if len(pred) > 0:
    print(f"  valid_time range: {pred['ValidTimeUtc'].min()} → {pred['ValidTimeUtc'].max()}")
    print(f"  unique versions:  {sorted(pred['ModelVersion'].unique().tolist())}")
    print(f"\n  by version (count + valid range):")
    for v, sub in pred.groupby("ModelVersion"):
        print(f"    {v}: n={len(sub)}, valid {sub['ValidTimeUtc'].min()} → {sub['ValidTimeUtc'].max()}")

print()
print("=" * 60)

pred120 = con.execute(f"""
    SELECT ModelVersion, PredictionMadeAtUtc, ValidTimeUtc, LeadHours, BlendTemperature
    FROM read_parquet('C:/Users/rhcsl/AppData/Local/Temp/temp_preds_all/**/predictions.parquet',
                      hive_partitioning=true, union_by_name=true)
    WHERE LeadHours = 120
    ORDER BY ValidTimeUtc
""").fetch_df()
print(f"\nAll lead-120 predictions: {len(pred120)}")
if len(pred120) > 0:
    print(f"  valid_time range: {pred120['ValidTimeUtc'].min()} → {pred120['ValidTimeUtc'].max()}")
    print(f"  unique versions:  {sorted(pred120['ModelVersion'].unique().tolist())}")
    print(f"\n  by version (count + valid range):")
    for v, sub in pred120.groupby("ModelVersion"):
        print(f"    {v}: n={len(sub)}, valid {sub['ValidTimeUtc'].min()} → {sub['ValidTimeUtc'].max()}")

# Specifically: what's the EARLIEST valid_time at lead 96 from any version?
print("\n\n=== Earliest predictions per lead, by version ===")
combined = con.execute(f"""
    SELECT ModelVersion, LeadHours,
           MIN(ValidTimeUtc) AS first_valid,
           MAX(ValidTimeUtc) AS last_valid,
           COUNT(*) AS n
    FROM read_parquet('C:/Users/rhcsl/AppData/Local/Temp/temp_preds_all/**/predictions.parquet',
                      hive_partitioning=true, union_by_name=true)
    WHERE LeadHours IN (96, 120)
    GROUP BY ModelVersion, LeadHours
    ORDER BY ModelVersion, LeadHours
""").fetch_df()
print(combined.to_string(index=False))

con.close()
