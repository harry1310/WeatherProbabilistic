"""Inspect a temperature prediction parquet — what leads + valid_times?"""
import sys
import duckdb
import pandas as pd

paths = sys.argv[1:]
con = duckdb.connect(":memory:")
for p in paths:
    print(f"\n=== {p} ===")
    df = con.execute(f"SELECT * FROM read_parquet('{p}')").fetch_df()
    print(f"  rows: {len(df)}")
    print(f"  cols: {list(df.columns)[:10]}…")
    if "LeadHours" in df.columns:
        leads = sorted(df["LeadHours"].unique().tolist())
        print(f"  leads: {leads}")
    if "ValidTimeUtc" in df.columns:
        print(f"  valid range: {df['ValidTimeUtc'].min()} → {df['ValidTimeUtc'].max()}")
    if "RunTimeUtc" in df.columns:
        print(f"  run_time:    {df['RunTimeUtc'].min()} → {df['RunTimeUtc'].max()}")
    if "ModelVersion" in df.columns:
        print(f"  versions:    {df['ModelVersion'].unique().tolist()}")
con.close()
