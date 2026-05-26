"""Membury mm/h via *member-weighted* bias correction.

Fits a constrained linear blend of the 7 NWP precip predictions with
non-negative weights summing to 1, plus a per-station intercept (pooled
weights across stations; per-station mean offset). Trained per lead
{24, 48, 72}h, chronologically split 70/15/15 within each station then
concatenated so test always lives in the future.

Two losses, same parameter space:
  * MSE     — squared error
  * Tweedie — deviance with variance_power=1.5 (heavy-tailed, zero-inflated)

Baselines for context:
  * equal_mean — fixed 1/7 weights, zero intercept (the same NWP-mean that
    beat LightGBM in the earlier sweep, recomputed on the same NaN
    imputation as the fitted blend so the comparison is apples to apples)
  * intercept_only_mean — equal_mean + per-station intercept (isolates the
    "are stations systematically biased?" question from the weight-tuning)
  * unconstrained_ols  — same parameter shape but no weight constraints
    (sanity check: how much does the constraint cost?)

NaN handling: per row, missing NWP predictions are imputed with the row's
mean of present NWPs. This preserves the row count and is what the
equal-weight mean does implicitly.

Run:
    .venv/Scripts/python.exe -u scripts/run_membury_member_weighted.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import WEATHERBLEND_DATA_ROOT, WET_THRESHOLD_MM  # noqa: E402

LOCATION = "membury_devon"
STATIONS = ("Chards Snowdon Hill", "Goren", "Raymonds Hill")
LEADS = (24, 48, 72)

MODELS_LEAN = [
    ("gfs_seamless",         "gfs"),
    ("ecmwf_ifs025",         "ecmwf"),
    ("icon_seamless",        "icon"),
    ("meteofrance_seamless", "mf"),
    ("gem_seamless",         "gem"),
    ("ecmwf_aifs025_single", "aifs"),
    ("jma_seamless",         "jma"),
]
PRECIP_COLS = [f"precip_{s}" for _, s in MODELS_LEAN]
K = len(MODELS_LEAN)

OUT_DIR = ROOT / "reports" / "membury_intensity_lgbm"
CACHE = OUT_DIR / "_precip_cache.parquet"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_all() -> pd.DataFrame:
    """One DuckDB scan for all (station, lead, valid_time) rows we need:
    7 NWP precip values + truth mm/h. Cached to parquet."""
    if CACHE.exists():
        print(f"[cache] loading {CACHE}")
        return pd.read_parquet(CACHE)

    fc_glob = str((WEATHERBLEND_DATA_ROOT / "forecasts" / "**" / "*.parquet")).replace("\\", "/")
    rn_glob = str((WEATHERBLEND_DATA_ROOT / "truth" / "rainfall" / "**" / "*.parquet")).replace("\\", "/")
    model_in = "(" + ",".join(f"'{full}'" for full, _ in MODELS_LEAN) + ")"
    station_in = "(" + ",".join(f"'{s}'" for s in STATIONS) + ")"
    lead_in = "(" + ",".join(str(l) for l in LEADS) + ")"
    pivot_lines = ",\n            ".join(
        f"MAX(CASE WHEN Model = '{full}' THEN Precipitation END) AS precip_{short}"
        for full, short in MODELS_LEAN
    )

    sql = f"""
    WITH hourly_truth AS (
        SELECT
            date_trunc('hour', ObservedTimeUtc) AS valid_time,
            StationName,
            SUM(Value15MinMm) AS precip_mm_hour
        FROM read_parquet('{rn_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND StationName  IN {station_in}
          AND Value15MinMm IS NOT NULL
        GROUP BY 1, 2 HAVING COUNT(*) = 4
    ),
    latest AS (
        SELECT
            ValidTimeUtc, Model, LeadHours, Precipitation,
            ROW_NUMBER() OVER (
                PARTITION BY ValidTimeUtc, Model, LeadHours
                ORDER BY RunTimeUtc DESC
            ) AS rn
        FROM read_parquet('{fc_glob}', hive_partitioning = false, union_by_name = true)
        WHERE LocationName = '{LOCATION}'
          AND RunTimeSource = 'offset_day'
          AND LeadHours IN {lead_in}
          AND Model IN {model_in}
    ),
    pivoted AS (
        SELECT
            ValidTimeUtc, LeadHours,
            {pivot_lines}
        FROM latest WHERE rn = 1
        GROUP BY ValidTimeUtc, LeadHours
    )
    SELECT t.StationName AS station, p.LeadHours AS lead, p.ValidTimeUtc,
           {", ".join(f"p.precip_{s}" for _, s in MODELS_LEAN)},
           t.precip_mm_hour
    FROM pivoted p
    JOIN hourly_truth t ON p.ValidTimeUtc = t.valid_time
    ORDER BY t.StationName, p.LeadHours, p.ValidTimeUtc
    """
    print("[build] running combined DuckDB query (this is the slow step) ...")
    t0 = time.time()
    con = duckdb.connect(":memory:")
    df = con.execute(sql).fetch_df()
    con.close()
    print(f"[build] {len(df):,} rows in {time.time()-t0:.1f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"[cache] wrote {CACHE}")
    return df


def impute_rowmean(P: np.ndarray) -> np.ndarray:
    """For each row, fill NaN cells with the row's mean of present cells.
    If all cells are NaN, leave as 0 (and caller should drop those rows).
    """
    P = P.copy()
    row_mean = np.nanmean(P, axis=1)
    row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
    inds = np.where(np.isnan(P))
    P[inds] = row_mean[inds[0]]
    return P


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

def fit_constrained(P_tr, S_tr, y_tr, *, loss="mse"):
    """Fit weights w (>=0, sum=1) and per-station intercepts b.

    Prediction:  yhat = S @ b + P @ w
    """
    n, k = P_tr.shape
    s_dim = S_tr.shape[1]

    def predict(theta):
        b = theta[:s_dim]
        w = theta[s_dim:]
        return S_tr @ b + P_tr @ w

    if loss == "mse":
        def obj(theta):
            err = predict(theta) - y_tr
            return float(np.mean(err ** 2))
    elif loss == "tweedie":
        # Unit deviance for Tweedie with variance_power p in (1,2):
        #   d = 2*( y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) )
        # The y-only terms drop to 0 at y=0 (limit).
        p = 1.5
        eps = 1e-6
        def obj(theta):
            mu = np.clip(predict(theta), eps, None)
            y = y_tr
            d = 2.0 * (
                np.where(y > 0, y ** (2.0 - p) / ((1.0 - p) * (2.0 - p)), 0.0)
                - np.where(y > 0, y * mu ** (1.0 - p) / (1.0 - p), 0.0)
                + mu ** (2.0 - p) / (2.0 - p)
            )
            return float(np.mean(d))
    else:
        raise ValueError(loss)

    x0 = np.concatenate([np.full(s_dim, float(y_tr.mean())), np.full(k, 1.0 / k)])
    bounds = [(-2.0, 5.0)] * s_dim + [(0.0, 1.0)] * k
    cons = [{"type": "eq", "fun": lambda t, sd=s_dim: t[sd:].sum() - 1.0}]
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    return res


def fit_ols(P_tr, S_tr, y_tr):
    """Per-station intercepts + per-model weights, unconstrained OLS."""
    X = np.hstack([S_tr, P_tr])
    coef, *_ = np.linalg.lstsq(X, y_tr, rcond=None)
    return coef


def predict_linear(P, S, b, w):
    return S @ b + P @ w


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    wet = y >= WET_THRESHOLD_MM
    mae_wet = float(np.mean(np.abs(err[wet]))) if wet.any() else float("nan")
    corr = float(np.corrcoef(p, y)[0, 1]) if y.std() > 0 and p.std() > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mae_wet": mae_wet, "corr": corr,
            "n": int(len(y)), "n_wet": int(wet.sum())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def split_by_station(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """Per-station chronological 70/15/15, then concatenate. Ensures every
    station contributes to both train and test."""
    tr, va, te = [], [], []
    for st, sub in df.groupby("station", sort=False):
        sub = sub.sort_values("ValidTimeUtc").reset_index(drop=True)
        n = len(sub)
        a = int(n * train_frac); b = a + int(n * val_frac)
        tr.append(sub.iloc[:a]); va.append(sub.iloc[a:b]); te.append(sub.iloc[b:])
    return (pd.concat(tr, ignore_index=True),
            pd.concat(va, ignore_index=True),
            pd.concat(te, ignore_index=True))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_all = load_all()
    print(f"[data] {len(df_all):,} total rows  "
          f"stations: {df_all['station'].nunique()}  "
          f"leads: {sorted(df_all['lead'].unique().tolist())}")

    rows = []
    weights_log = []

    for lead in LEADS:
        sub = df_all[df_all["lead"] == lead].copy()
        tr, va, te = split_by_station(sub)
        # Pooled stations, per-station one-hot intercept
        stations_in_order = list(STATIONS)
        s_to_idx = {s: i for i, s in enumerate(stations_in_order)}

        def to_arrays(d):
            P = d[PRECIP_COLS].to_numpy(dtype="float64")
            P = impute_rowmean(P)
            S = np.zeros((len(d), len(stations_in_order)), dtype="float64")
            for j, st in enumerate(d["station"].tolist()):
                S[j, s_to_idx[st]] = 1.0
            y = d["precip_mm_hour"].to_numpy(dtype="float64")
            return P, S, y, d

        P_tr, S_tr, y_tr, d_tr = to_arrays(tr)
        P_te, S_te, y_te, d_te = to_arrays(te)
        print(f"\n=== lead {lead}h ===  train n={len(y_tr):,}  test n={len(y_te):,}  "
              f"train wet rate={(y_tr>=WET_THRESHOLD_MM).mean()*100:.1f}%")

        # Equal-weight mean, zero intercept (apples-to-apples vs the C# nwp_mean)
        w_eq = np.full(K, 1.0 / K)
        b_zero = np.zeros(len(stations_in_order))
        p = predict_linear(P_te, S_te, b_zero, w_eq).clip(0.0)
        m = metrics(p, y_te); m.update({"lead": lead, "model": "equal_mean (no intercept)"}); rows.append(m)
        print(f"  equal_mean (no intercept)     MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
              f"R2={m['r2']:+.3f}  MAE_wet={m['mae_wet']:.3f}  r={m['corr']:+.3f}")

        # Equal weights + per-station intercept (just learn the bias)
        # Closed-form: per-station intercept = y_train_mean(station) - row_mean_pred_train(station)
        b_int = np.zeros(len(stations_in_order))
        for j, st in enumerate(stations_in_order):
            mask_tr = (d_tr["station"] == st).to_numpy()
            if mask_tr.sum() == 0: continue
            pred_eq_tr = P_tr[mask_tr] @ w_eq
            b_int[j] = y_tr[mask_tr].mean() - pred_eq_tr.mean()
        p = predict_linear(P_te, S_te, b_int, w_eq).clip(0.0)
        m = metrics(p, y_te); m.update({"lead": lead, "model": "equal_mean + per-station bias"}); rows.append(m)
        print(f"  equal_mean + per-station bias MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
              f"R2={m['r2']:+.3f}  MAE_wet={m['mae_wet']:.3f}  r={m['corr']:+.3f}  "
              f"(bias mm/h: " + ", ".join(f"{stations_in_order[j][:6]}={b_int[j]:+.3f}" for j in range(len(stations_in_order))) + ")")

        # Constrained MSE
        t0 = time.time()
        res = fit_constrained(P_tr, S_tr, y_tr, loss="mse")
        b_mse = res.x[:len(stations_in_order)]
        w_mse = res.x[len(stations_in_order):]
        p = predict_linear(P_te, S_te, b_mse, w_mse).clip(0.0)
        m = metrics(p, y_te); m.update({"lead": lead, "model": "constrained_mse"}); rows.append(m)
        print(f"  constrained_mse              MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
              f"R2={m['r2']:+.3f}  MAE_wet={m['mae_wet']:.3f}  r={m['corr']:+.3f}  "
              f"(fit {time.time()-t0:.1f}s)")
        weights_log.append({"lead": lead, "loss": "mse",
                            **{f"w_{short}": float(w_mse[i]) for i, (_, short) in enumerate(MODELS_LEAN)},
                            **{f"b_{stations_in_order[j][:6]}": float(b_mse[j]) for j in range(len(stations_in_order))}})

        # Constrained Tweedie
        t0 = time.time()
        res = fit_constrained(P_tr, S_tr, y_tr, loss="tweedie")
        b_tw = res.x[:len(stations_in_order)]
        w_tw = res.x[len(stations_in_order):]
        p = predict_linear(P_te, S_te, b_tw, w_tw).clip(0.0)
        m = metrics(p, y_te); m.update({"lead": lead, "model": "constrained_tweedie"}); rows.append(m)
        print(f"  constrained_tweedie          MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
              f"R2={m['r2']:+.3f}  MAE_wet={m['mae_wet']:.3f}  r={m['corr']:+.3f}  "
              f"(fit {time.time()-t0:.1f}s)")
        weights_log.append({"lead": lead, "loss": "tweedie",
                            **{f"w_{short}": float(w_tw[i]) for i, (_, short) in enumerate(MODELS_LEAN)},
                            **{f"b_{stations_in_order[j][:6]}": float(b_tw[j]) for j in range(len(stations_in_order))}})

        # Unconstrained OLS (sanity: how much does the constraint cost?)
        coef = fit_ols(P_tr, S_tr, y_tr)
        b_ols = coef[:len(stations_in_order)]
        w_ols = coef[len(stations_in_order):]
        p = predict_linear(P_te, S_te, b_ols, w_ols).clip(0.0)
        m = metrics(p, y_te); m.update({"lead": lead, "model": "unconstrained_ols"}); rows.append(m)
        print(f"  unconstrained_ols            MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
              f"R2={m['r2']:+.3f}  MAE_wet={m['mae_wet']:.3f}  r={m['corr']:+.3f}  "
              f"(weights sum={w_ols.sum():.3f}, range [{w_ols.min():+.3f}, {w_ols.max():+.3f}])")

        # Per-station test breakdown for the best constrained models
        print(f"  -- per-station test MAE (lead {lead}h):")
        for st in stations_in_order:
            mask = (d_te["station"] == st).to_numpy()
            if mask.sum() == 0: continue
            p_eq = predict_linear(P_te[mask], S_te[mask], b_zero, w_eq).clip(0.0)
            p_in = predict_linear(P_te[mask], S_te[mask], b_int, w_eq).clip(0.0)
            p_mse = predict_linear(P_te[mask], S_te[mask], b_mse, w_mse).clip(0.0)
            p_tw  = predict_linear(P_te[mask], S_te[mask], b_tw,  w_tw).clip(0.0)
            ys = y_te[mask]
            print(f"     {st:22s} n={mask.sum():>5d}  "
                  f"equal={np.mean(np.abs(p_eq-ys)):.4f}  "
                  f"+bias={np.mean(np.abs(p_in-ys)):.4f}  "
                  f"cmse={np.mean(np.abs(p_mse-ys)):.4f}  "
                  f"ctwd={np.mean(np.abs(p_tw-ys)):.4f}")

    # Aggregate
    df_r = pd.DataFrame(rows)
    print("\n\n========== SUMMARY ==========")
    print(df_r[["lead", "model", "n", "mae", "rmse", "r2", "mae_wet", "corr"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df_r.to_csv(OUT_DIR / "member_weighted_summary.csv", index=False)

    df_w = pd.DataFrame(weights_log)
    print("\n========== LEARNED WEIGHTS ==========")
    print(df_w.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    df_w.to_csv(OUT_DIR / "member_weighted_weights.csv", index=False)

    print(f"\nWrote {OUT_DIR/'member_weighted_summary.csv'}")
    print(f"Wrote {OUT_DIR/'member_weighted_weights.csv'}")


if __name__ == "__main__":
    main()
