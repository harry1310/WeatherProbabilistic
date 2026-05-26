"""3f stage-1 bake-off (Membury). Re-fits NGBoost-LogNormal stage 2 on the
existing precip cache for each candidate stage-1 (3a / 3c / 3c-oro), scores
CRPS on the test slice, writes a Brier+CRPS comparison report.

Stage-1 predictions are produced by the .NET command
`precip-3f-stage1-bakeoff` and saved as
  data/reports/3f_stage1_bakeoff_<date>/{3a,3c,3c_oro}/<station>.parquet.

Stage-2 features mirror run_membury_two_stage_ngboost.py (lean 15-feat
NGBoost-LogNormal champion from the 2026-05-22/23 exploration). The only
swap is the stage-1 source.

Usage:
    .venv/Scripts/python.exe scripts/run_membury_3f_stage1_bakeoff.py \
        --bakeoff-dir C:/Projects/Weather/WeatherBlend/data/reports/3f_stage1_bakeoff_2026-05-25
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import lognorm

from ngboost import NGBRegressor
from ngboost.distns import LogNormal as NGBLogNormal
from ngboost.scores import LogScore

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import WET_THRESHOLD_MM  # noqa: E402

LOCATION = "membury_devon"
STATION_FRIENDLY = {
    "ea_chards_snowdon_hill": "Chards Snowdon Hill",
    "ea_goren":               "Goren",
    "ea_raymonds_hill":       "Raymonds Hill",
}
STATIONS_FRIENDLY = ("Chards Snowdon Hill", "Goren", "Raymonds Hill")
LEADS = (24, 48, 72)
STAGE1_VARIANTS = ("3a", "3c", "3c_oro")

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

QUANTILE_ALPHAS_EVAL = tuple(np.round(np.linspace(1/41, 40/41, 20), 4).tolist())

NGB_BASE = dict(
    n_estimators=500, learning_rate=0.01,
    minibatch_frac=1.0, col_sample=1.0, verbose=False, random_state=42,
)
NGB_EARLY_STOP = 30


def build_stage2_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Same 15-feat lean spec as run_membury_two_stage_ngboost.py's stage 2."""
    pm = df[PRECIP_COLS].to_numpy(dtype="float64")
    present = (~np.isnan(pm)).sum(axis=1)
    sumv = np.nansum(pm, axis=1); sumsq = np.nansum(pm ** 2, axis=1)
    mean_safe = np.where(present > 0, sumv / np.maximum(present, 1), np.nan)
    var = np.maximum(0.0, sumsq / np.maximum(present, 1) - mean_safe ** 2)
    wet_count = (pm >= WET_THRESHOLD_MM).sum(axis=1)
    df = df.copy()
    df["precip_mean"] = mean_safe
    df["precip_std"]  = np.where(present > 1, np.sqrt(var), 0.0)
    df["precip_max"]  = np.where(present > 0, np.nanmax(pm, axis=1), np.nan)
    df["precip_agreement_wet_01"] = np.where(present > 0, wet_count / np.maximum(present, 1), np.nan)
    vt = pd.to_datetime(df["ValidTimeUtc"])
    df["hour_sin"] = np.sin(2.0 * np.pi * vt.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * vt.dt.hour / 24.0)
    df["doy_sin"]  = np.sin(2.0 * np.pi * (vt.dt.dayofyear - 1) / 365.0)
    df["doy_cos"]  = np.cos(2.0 * np.pi * (vt.dt.dayofyear - 1) / 365.0)
    feats = PRECIP_COLS + ["precip_mean", "precip_std", "precip_max",
                           "precip_agreement_wet_01",
                           "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    return df, feats


def emos_impute(X: np.ndarray) -> np.ndarray:
    """Same impute as the original NGBoost script."""
    X = X.copy()
    P = X[:, :7]
    row_mean = np.nanmean(P, axis=1)
    row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
    inds = np.where(np.isnan(P))
    P[inds] = row_mean[inds[0]]
    X[:, :7] = P
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = med[inds[1]]
    return X


def crps_mixed(pi: np.ndarray, quantiles: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = quantiles.shape[1]
    w_dry = 1.0 - pi
    w_wet = pi / K
    term1 = w_dry * y + w_wet * np.abs(quantiles - y[:, None]).sum(axis=1)
    cross_0k = 2.0 * w_dry * w_wet * quantiles.sum(axis=1)
    pairwise = np.abs(quantiles[:, :, None] - quantiles[:, None, :]).sum(axis=(1, 2))
    cross_kl = (w_wet ** 2) * pairwise
    return term1 - 0.5 * (cross_0k + cross_kl)


def brier(p, y):
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


def fit_ngboost_lognormal(X_tr_w, y_tr_w, X_va_w, y_va_w):
    ngb = NGBRegressor(Dist=NGBLogNormal, Score=LogScore, **NGB_BASE)
    ngb.fit(X_tr_w, y_tr_w, X_val=X_va_w, Y_val=y_va_w,
            early_stopping_rounds=NGB_EARLY_STOP)
    return ngb


def predict_ngboost_lognormal(ngb, X_te) -> tuple[np.ndarray, np.ndarray]:
    dist = ngb.pred_dist(X_te, max_iter=ngb.best_val_loss_itr)
    s = np.asarray(dist.params["s"])
    scale = np.asarray(dist.params["scale"])
    qs = np.empty((len(X_te), len(QUANTILE_ALPHAS_EVAL)), dtype="float64")
    for k, a in enumerate(QUANTILE_ALPHAS_EVAL):
        qs[:, k] = lognorm.ppf(a, s=s, scale=scale)
    mu = scale * np.exp(0.5 * s ** 2)
    return np.clip(qs, WET_THRESHOLD_MM, None), mu


def time_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    a = int(n * train_frac); b = a + int(n * val_frac)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def load_stage1_pwet(bakeoff_dir: Path, variant: str) -> pd.DataFrame:
    """Read all per-station parquet files for a stage-1 variant + concat.
    Returns columns: station, valid_time, lead, p_wet."""
    vdir = bakeoff_dir / variant
    frames = []
    for f in sorted(vdir.glob("*.parquet")):
        df = pd.read_parquet(f)
        df.columns = [c.lower() for c in df.columns]
        df["valid_time"] = pd.to_datetime(df["valid_time"]).dt.tz_localize(None)
        # Convert slug -> friendly name so we can join with the cache
        df["station_friendly"] = df["station"].map(STATION_FRIENDLY)
        if df["station_friendly"].isnull().any():
            unknown = df.loc[df["station_friendly"].isnull(), "station"].unique()
            raise SystemExit(f"Unknown station slug(s) in {f}: {unknown}")
        frames.append(df[["station_friendly", "valid_time", "lead", "p_wet"]])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bakeoff-dir", required=True, type=Path)
    parser.add_argument("--cache",
                        default=ROOT / "reports" / "membury_intensity_lgbm" / "_precip_cache.parquet",
                        type=Path)
    args = parser.parse_args()
    bakeoff_dir = args.bakeoff_dir
    if not bakeoff_dir.exists():
        raise SystemExit(f"Bake-off dir not found: {bakeoff_dir}")

    print(f"[cache] loading {args.cache}", flush=True)
    df_all = pd.read_parquet(args.cache)
    df_all, FEATS = build_stage2_features(df_all)
    df_all["ValidTimeUtc"] = pd.to_datetime(df_all["ValidTimeUtc"]).dt.tz_localize(None)
    print(f"[data] cache: {len(df_all):,} rows", flush=True)

    # Load all three stage-1 variants
    stage1: dict[str, pd.DataFrame] = {}
    for variant in STAGE1_VARIANTS:
        stage1[variant] = load_stage1_pwet(bakeoff_dir, variant)
        print(f"[stage1] {variant}: {len(stage1[variant]):,} rows", flush=True)

    rows = []
    for station in STATIONS_FRIENDLY:
        for lead in LEADS:
            sub = (df_all[(df_all["station"] == station) & (df_all["lead"] == lead)]
                   .sort_values("ValidTimeUtc").reset_index(drop=True))
            tr, va, te = time_split(sub)
            # Stage 2 features
            X_tr = tr[FEATS].to_numpy(dtype="float64")
            X_va = va[FEATS].to_numpy(dtype="float64")
            X_te = te[FEATS].to_numpy(dtype="float64")
            y_tr = tr["precip_mm_hour"].to_numpy(dtype="float64")
            y_va = va["precip_mm_hour"].to_numpy(dtype="float64")
            y_te = te["precip_mm_hour"].to_numpy(dtype="float64")
            wmask_tr = (y_tr >= WET_THRESHOLD_MM).astype(bool)
            wmask_va = (y_va >= WET_THRESHOLD_MM).astype(bool)
            X_tr_im = emos_impute(X_tr); X_va_im = emos_impute(X_va); X_te_im = emos_impute(X_te)

            print(f"\n=== {station} | lead {lead}h ===  wet_tr={wmask_tr.sum()}  wet_va={wmask_va.sum()}  n_te={len(te)}", flush=True)
            te_vt = te["ValidTimeUtc"].values

            # Fit stage-2 ONCE per (station, lead) — same model, three stage-1 variants
            t0 = time.time()
            ngb = fit_ngboost_lognormal(X_tr_im[wmask_tr], y_tr[wmask_tr],
                                        X_va_im[wmask_va], y_va[wmask_va])
            qs, mu = predict_ngboost_lognormal(ngb, X_te_im)
            stage2_fit_s = time.time() - t0
            print(f"   stage-2 fit {stage2_fit_s:.1f}s (best_iter={ngb.best_val_loss_itr})", flush=True)

            # For each stage-1 variant: join its pi values, compute CRPS
            for variant in STAGE1_VARIANTS:
                stage1_df = stage1[variant]
                merged = pd.DataFrame({"ValidTimeUtc": te_vt}).merge(
                    stage1_df[(stage1_df["station_friendly"] == station) & (stage1_df["lead"] == lead)]
                        [["valid_time", "p_wet"]].rename(columns={"valid_time": "ValidTimeUtc"}),
                    on="ValidTimeUtc", how="left")
                pi_te = merged["p_wet"].to_numpy(dtype="float64")
                covered = np.isfinite(pi_te)
                if covered.sum() < len(pi_te) * 0.9:
                    print(f"   WARN {variant}: only {covered.sum()}/{len(pi_te)} rows have stage-1 pi", flush=True)

                # Brier of stage-1 alone (where covered)
                wet_te = (y_te >= WET_THRESHOLD_MM).astype("float64")
                bri = brier(pi_te[covered], wet_te[covered])

                # CRPS using only covered rows (mask both pi and qs)
                pi_c = pi_te[covered]; qs_c = qs[covered]; y_c = y_te[covered]
                crps = float(crps_mixed(pi_c, qs_c, y_c).mean())

                yhat = pi_c * mu[covered]
                err = yhat - y_c
                rows.append({
                    "station": station, "lead": lead, "stage1": variant,
                    "n_test": int(covered.sum()), "wet_rate_te": float(wet_te[covered].mean()),
                    "brier_stage1": bri, "crps": crps,
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err ** 2))),
                    "stage2_fit_s": round(stage2_fit_s, 1),
                })
                print(f"   {variant:8s} Brier(stage1)={bri:.4f}  CRPS={crps:.4f}  MAE={rows[-1]['mae']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    out_root = bakeoff_dir
    csv_path = out_root / "stage1_bakeoff_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}", flush=True)

    # Aggregate
    print("\n========== AGGREGATE per (lead, stage1) — mean across 3 stations ==========", flush=True)
    agg = (df.groupby(["lead", "stage1"])
             .agg(brier_stage1=("brier_stage1", "mean"),
                  crps=("crps", "mean"),
                  mae=("mae", "mean"),
                  rmse=("rmse", "mean"))
             .reset_index().sort_values(["lead", "crps"]))
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    print("\n========== OVERALL per stage1 (weighted by n_test) ==========", flush=True)
    overall = (df.groupby("stage1")
                 .apply(lambda g: pd.Series({
                     "brier_stage1": float((g["brier_stage1"] * g["n_test"]).sum() / g["n_test"].sum()),
                     "crps": float((g["crps"] * g["n_test"]).sum() / g["n_test"].sum()),
                     "mae":  float((g["mae"]  * g["n_test"]).sum() / g["n_test"].sum()),
                     "n_test": int(g["n_test"].sum()),
                 }))
                 .reset_index().sort_values("crps"))
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    # Markdown report
    md_path = out_root / "stage1_bakeoff_report.md"
    lines = ["# Phase 3f stage-1 bake-off (Membury)", "",
             "Stage-2: NGBoost-LogNormal (champion from 2026-05-22/23). Only the stage-1 P(wet) source varies.",
             "", "## Aggregate per (lead, stage1) — mean across 3 stations", ""]
    lines.append("| lead | stage1 | Brier(stage1) | CRPS | MAE | RMSE |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        lines.append(f"| {int(r['lead'])} | {r['stage1']} | {r['brier_stage1']:.4f} | {r['crps']:.4f} | {r['mae']:.4f} | {r['rmse']:.4f} |")
    lines.append("")
    lines.append("## Overall per stage1 (weighted by n_test)")
    lines.append("")
    lines.append("| stage1 | Brier(stage1) | CRPS | MAE | n_test |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in overall.iterrows():
        lines.append(f"| {r['stage1']} | {r['brier_stage1']:.4f} | {r['crps']:.4f} | {r['mae']:.4f} | {int(r['n_test'])} |")
    lines.append("")
    # Δ vs 3a baseline
    base_crps = float(overall[overall["stage1"] == "3a"]["crps"].iloc[0]) if "3a" in set(overall["stage1"]) else float("nan")
    lines.append("## Δ vs 3a baseline (overall)")
    lines.append("")
    lines.append("| stage1 | overall CRPS | Δ vs 3a |")
    lines.append("|---|---:|---:|")
    for _, r in overall.sort_values("crps").iterrows():
        d = (r["crps"] - base_crps) / base_crps * 100
        lines.append(f"| {r['stage1']} | {r['crps']:.4f} | {d:+.2f}% |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
