"""4b blend re-evaluation with the new rich-per-station 4a candidate.

Production 4b = arithmetic mean of (production 4a, 3o) per (station, lead, valid_time).
Now we have a candidate replacement for the 4a input: rich-per-station BART. Does
the blend still beat each model alone? Does swapping 4a improve it? Are there
better blend formulas than the arithmetic mean?

Inputs:
  * Production 4a per-row predictions — WB/data/models/precipitation/{slug}/{ver}_phase4a/test_predictions.parquet
  * Rich-per-station 4a per-row predictions — WP/reports/.../test_predictions.parquet, variant=='rich-per-station'
  * Pooled 3o LightGBM predictions — re-computed here on the same rich-oro dumps (Python LightGBM, same
    hypers as run_3o_per_station.py — close-enough proxy to production ML.NET 3o for the structural question)

Caveat: Python LightGBM 3o won't be bit-identical to production ML.NET 3o, but the BLEND-SHAPE
conclusion (does new 4a help? is arithmetic mean still best?) is robust to the small absolute drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import WEATHERBLEND_DATA_ROOT  # noqa: E402
from run_phase6_dbarts_pooled_oro import load_dump_wide, STATIONS  # noqa: E402
from _shared import time_split  # noqa: E402
import lightgbm as lgb  # noqa: E402

LEADS = [24, 48, 72]
WP_BAKEOFF_DIR = ROOT / "reports" / "pooled_oro_4a_bakeoff_2026-05-29"
PROD_4A_DIRS = {
    "ea_bellever_dartmoor": WEATHERBLEND_DATA_ROOT / "models/precipitation/ea_bellever_dartmoor/v2026-05-17_165023_phase4a",
    "ea_dartmoor_nr_hexworthy": WEATHERBLEND_DATA_ROOT / "models/precipitation/ea_dartmoor_nr_hexworthy/v2026-05-10_173955_phase4a",
    "ea_bovey_tracey": WEATHERBLEND_DATA_ROOT / "models/precipitation/ea_bovey_tracey/v2026-05-10_173955_phase4a",
}

LGB_PARAMS = dict(objective="binary", metric="binary_logloss", num_leaves=31,
                  learning_rate=0.2, min_data_in_leaf=20, verbosity=-1)
N_EST = 100


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, np.float64) - np.asarray(y, np.float64)) ** 2))


def fit_pooled_3o_lgb_and_predict() -> pd.DataFrame:
    """Return long-form DF: (station, lead, valid_time, p_3o, observed_wet)."""
    out = []
    for lead in LEADS:
        per_station = {}
        feat_names = None
        for slug in STATIONS:
            wide, names, _ = load_dump_wide(slug, lead, "rich-oro")
            if feat_names is None:
                feat_names = names
            tr, va, te = time_split(wide)
            per_station[slug] = (tr, va, te)
        # Pool train+val, fit one LGB
        tr_p = pd.concat([per_station[s][0] for s in STATIONS], ignore_index=True)
        va_p = pd.concat([per_station[s][1] for s in STATIONS], ignore_index=True)
        X_tr = tr_p[feat_names].to_numpy(np.float64); y_tr = tr_p["wet"].to_numpy(np.float64)
        X_va = va_p[feat_names].to_numpy(np.float64); y_va = va_p["wet"].to_numpy(np.float64)
        ds_tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr, free_raw_data=False)
        model = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=N_EST,
                          valid_sets=[ds_va], callbacks=[lgb.early_stopping(10, verbose=False)])
        best = int(model.best_iteration or N_EST)
        print(f"  pooled 3o lead {lead}h fit (best_iter={best})", flush=True)
        # Predict per-station test
        for slug in STATIONS:
            te = per_station[slug][2]
            X_te = te[feat_names].to_numpy(np.float64)
            p = model.predict(X_te, num_iteration=best)
            out.append(pd.DataFrame({
                "station": slug, "lead": lead,
                "valid_time": pd.to_datetime(te["ValidTimeUtc"].values),
                "p_3o": p, "observed_wet": te["wet"].to_numpy(np.int8),
            }))
    return pd.concat(out, ignore_index=True)


def load_prod_4a() -> pd.DataFrame:
    frames = []
    for slug, d in PROD_4A_DIRS.items():
        df = pd.read_parquet(d / "test_predictions.parquet")
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df = df.rename(columns={"p_wet": "p_prod_4a"})
        df = df[["valid_time", "station", "lead", "p_prod_4a", "observed_wet"]]
        # Some bundles store station as friendly name; canonicalise to slug
        df["station"] = slug
        frames.append(df[df["lead"].isin(LEADS)])
    return pd.concat(frames, ignore_index=True)


def load_new_4a() -> pd.DataFrame:
    pq = pd.read_parquet(WP_BAKEOFF_DIR / "test_predictions.parquet")
    pq["valid_time"] = pd.to_datetime(pq["valid_time"])
    pq = pq[(pq["variant"] == "rich-per-station") & (pq["lead"].isin(LEADS))]
    pq = pq.rename(columns={"p_wet": "p_new_4a"})
    return pq[["valid_time", "station", "lead", "p_new_4a", "observed_wet"]].copy()


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def blend_mean(p_a, p_b):    return (p_a + p_b) / 2
def blend_geom(p_a, p_b):    return np.sqrt(np.clip(p_a, 1e-6, 1) * np.clip(p_b, 1e-6, 1))
def blend_logit_mean(p_a, p_b): return sigmoid((logit(p_a) + logit(p_b)) / 2)


def best_weighted_blend(p_a, p_b, y):
    """Grid-search weight w in [0, 1] for p_blend = w*p_a + (1-w)*p_b minimising Brier."""
    ws = np.linspace(0, 1, 51)
    briers = [brier(w*p_a + (1-w)*p_b, y) for w in ws]
    j = int(np.argmin(briers))
    return float(ws[j]), float(briers[j])


def main() -> None:
    print("Loading inputs ...", flush=True)
    print("  re-fitting pooled 3o LightGBM (4 stations × 3 leads = 3 fits)")
    df_3o = fit_pooled_3o_lgb_and_predict()
    print(f"  3o rows: {len(df_3o)}")
    df_prod = load_prod_4a();   print(f"  prod 4a rows: {len(df_prod)}")
    df_new  = load_new_4a();    print(f"  new 4a rows:  {len(df_new)}")

    # Inner-join all three on (station, lead, valid_time)
    j = df_3o.merge(df_prod, on=["station", "lead", "valid_time"], how="inner",
                    suffixes=("", "_prod"))
    # Sanity: observed_wet should match
    if (j["observed_wet"] != j["observed_wet_prod"]).any():
        print("  WARN: observed_wet mismatch between 3o and prod 4a — check joins")
    j = j.drop(columns=["observed_wet_prod"])
    j = j.merge(df_new, on=["station", "lead", "valid_time"], how="inner",
                suffixes=("", "_new"))
    if (j["observed_wet"] != j["observed_wet_new"]).any():
        print("  WARN: observed_wet mismatch between 3o and new 4a")
    j = j.drop(columns=["observed_wet_new"])
    print(f"  aligned rows (3-way): {len(j)}")
    print(f"  per-station counts:")
    print(j.groupby(["station", "lead"]).size().to_string())

    # Compute Brier for each model alone + each blend variant
    rows = []
    for (stn, lead), g in j.groupby(["station", "lead"]):
        y = g["observed_wet"].to_numpy(np.float64)
        p_prod = g["p_prod_4a"].to_numpy(np.float64)
        p_new  = g["p_new_4a"].to_numpy(np.float64)
        p_3o   = g["p_3o"].to_numpy(np.float64)

        row = {"station": stn, "lead": lead, "n": int(len(g))}
        # Singles
        row["prod_4a"] = brier(p_prod, y)
        row["new_4a"]  = brier(p_new, y)
        row["3o"]      = brier(p_3o, y)
        # Current 4b (mean of prod 4a + 3o)
        row["4b_current"]     = brier(blend_mean(p_prod, p_3o), y)
        # New 4b candidate (mean of new 4a + 3o)
        row["4b_new_4a"]      = brier(blend_mean(p_new,  p_3o), y)
        # Alternative blend formulas for the NEW 4a
        row["4b_new_geom"]    = brier(blend_geom(p_new,  p_3o), y)
        row["4b_new_logit"]   = brier(blend_logit_mean(p_new, p_3o), y)
        # Best weighted blend (new 4a + 3o)
        w_opt, b_opt = best_weighted_blend(p_new, p_3o, y)
        row["4b_new_w_opt"]   = b_opt
        row["w_opt"]          = w_opt
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["station", "lead"]).reset_index(drop=True)
    out_csv = WP_BAKEOFF_DIR / "blend_4b_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    print("\nPer-cell Brier:")
    for col in ["prod_4a", "new_4a", "3o", "4b_current", "4b_new_4a",
                "4b_new_geom", "4b_new_logit", "4b_new_w_opt"]:
        df[col] = df[col].round(4)
    print(df[["station","lead","n","prod_4a","new_4a","3o",
              "4b_current","4b_new_4a","4b_new_geom","4b_new_logit",
              "4b_new_w_opt","w_opt"]].to_string(index=False))

    print("\nAggregate (mean across 3 stations × 3 leads):")
    agg = df[["prod_4a","new_4a","3o",
              "4b_current","4b_new_4a","4b_new_geom","4b_new_logit",
              "4b_new_w_opt"]].mean().round(4)
    print(agg.to_string())

    # Delta vs current 4b
    base = agg["4b_current"]
    print("\nΔ% vs current 4b (mean(prod_4a, 3o)) — negative = improvement:")
    for k in ["new_4a","3o","4b_new_4a","4b_new_geom","4b_new_logit","4b_new_w_opt"]:
        d = (agg[k] - base) / base * 100
        print(f"  {k:22s}  {agg[k]:.4f}   Δ {d:+.2f}%")

    print(f"\nMean optimal weight (new_4a vs 3o) across cells: {df['w_opt'].mean():.2f}")
    print(f"  (1.0 = ignore 3o entirely; 0.0 = ignore new 4a; 0.5 = arithmetic mean)")


if __name__ == "__main__":
    main()
