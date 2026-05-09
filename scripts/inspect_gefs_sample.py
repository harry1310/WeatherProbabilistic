"""Inspect one GEFS parquet pulled from R2 — verify schema, lead range,
RunTimeSource, and Temperature2m availability."""
import duckdb

con = duckdb.connect(":memory:")
schema = con.execute(
    "DESCRIBE SELECT * FROM read_parquet('C:/Users/rhcsl/AppData/Local/Temp/gefs_sample/run=00.parquet') LIMIT 1"
).fetch_df()
print("=== Schema ===")
for _, r in schema.iterrows():
    print(f'  {r["column_name"]:30s} {r["column_type"]}')
print()

print("=== Row count + meta ===")
m = con.execute("""
    SELECT COUNT(*) AS rows,
           MIN(LeadHours) AS lead_min, MAX(LeadHours) AS lead_max,
           MIN(ValidTimeUtc) AS valid_min, MAX(ValidTimeUtc) AS valid_max,
           MIN(RunTimeUtc) AS run_min, MAX(RunTimeUtc) AS run_max,
           ANY_VALUE(Model) AS model, ANY_VALUE(RunTimeSource) AS rts,
           ANY_VALUE(LocationName) AS loc
    FROM read_parquet('C:/Users/rhcsl/AppData/Local/Temp/gefs_sample/run=00.parquet')
""").fetch_df()
for col in m.columns:
    print(f'  {col:15s} {m[col].iloc[0]}')

print()
print("=== Temperature2m sample at first 5 leads ===")
t = con.execute("""
    SELECT LeadHours, Temperature2m, Precipitation, SurfacePressure, WindSpeed10m, WindDirection10m
    FROM read_parquet('C:/Users/rhcsl/AppData/Local/Temp/gefs_sample/run=00.parquet')
    ORDER BY LeadHours LIMIT 5
""").fetch_df()
print(t.to_string(index=False))
con.close()
