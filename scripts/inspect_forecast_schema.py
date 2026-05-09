"""Quick schema dump for the forecast parquet tree."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb
from src.data import WEATHERBLEND_DATA_ROOT

fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
con = duckdb.connect(":memory:")
cols = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{fc_glob}', hive_partitioning=false, union_by_name=true) LIMIT 1"
).fetch_df()
con.close()
for _, r in cols.iterrows():
    print(f'{r["column_name"]:30s} {r["column_type"]}')
