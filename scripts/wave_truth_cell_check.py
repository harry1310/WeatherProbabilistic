"""Truth-of-truths: which era5_ocean cell best represents exposed Sennen swell?

The era5-waves backfill snapped to the 0.5-degree cell (50.0, -5.5) - SE of
Land's End, partially SHELTERED from NW swell. The cliff faces W/NW. Compare
both candidate cells against the fully-exposed Sevenstones buoy on matched
hours, overall and split by wave direction, to decide the truth coordinate
before production wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

WB_DATA = Path(r"C:\Projects\Weather\WeatherBlend\data")
CELLS = {"cell_SE_50.0_-5.5": (50.0, -5.5), "cell_W_50.0_-6.0": (50.0, -6.0)}
START, END = "2025-06-01", "2026-06-05"


def fetch_era5(lat: float, lon: float) -> pd.Series:
    url = ("https://marine-api.open-meteo.com/v1/marine"
           f"?latitude={lat}&longitude={lon}&hourly=wave_height,wave_direction"
           f"&models=era5_ocean&timezone=UTC&start_date={START}&end_date={END}")
    h = requests.get(url, timeout=120).json()["hourly"]
    df = pd.DataFrame({"hs": h["wave_height"], "dir": h["wave_direction"]},
                      index=pd.to_datetime(h["time"]))
    print(f"  fetched ({lat},{lon}) -> {len(df.dropna())} rows")
    return df


def main() -> int:
    con = duckdb.connect()
    buoy = con.sql(f"""
        SELECT ValidTimeUtc, WaveHeight AS hs_buoy
        FROM read_parquet('{WB_DATA.as_posix()}/truth/waves/location=sennen_cove/source=sevenstones_62107/*/data.parquet',
                          hive_partitioning=false)
        WHERE WaveHeight IS NOT NULL
    """).df().set_index("ValidTimeUtc")["hs_buoy"]
    print(f"Sevenstones rows {START}..{END}: "
          f"{len(buoy[(buoy.index >= START) & (buoy.index <= END)])}")

    for name, (lat, lon) in CELLS.items():
        cell = fetch_era5(lat, lon)
        j = pd.concat([cell, buoy], axis=1).dropna()
        err = j["hs"] - j["hs_buoy"]
        print(f"\n{name}: n={len(j)}  MAE {err.abs().mean():.3f} m  "
              f"bias {err.mean():+.3f} m  r {j['hs'].corr(j['hs_buoy']):.4f}")
        j["bin"] = pd.cut(j["dir"], bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                          labels=["N-NE", "NE-E", "E-SE", "SE-S", "S-SW", "SW-W", "W-NW", "NW-N"])
        by = j.groupby("bin", observed=True).apply(
            lambda g: pd.Series({"n": len(g), "mae": (g.hs - g.hs_buoy).abs().mean(),
                                 "bias": (g.hs - g.hs_buoy).mean()}), include_groups=False)
        print(by.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
