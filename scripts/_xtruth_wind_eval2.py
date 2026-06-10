"""Cross-truth wind-speed eval + champion/lgb BLEND (2026-06-09, throwaway).

Same controlled setup as before: one feature matrix per lead (floor-free
29-feature set), both truths attached (Dunkeswell SYNOP speed + ERA5
WindSpeed10m at Bonehill), chronological 70/15/15, two LightGBM regressors
trained on identical rows differing only in label (D=Dunkeswell, E=ERA5).

Adds blends of the two model POINT predictions:
  - blend50: equal-weight mean
  - blendOptD: weight w on D fit on VAL to minimise MAE vs Dunkeswell (real wind)

All scored on the SAME OOS test rows vs BOTH truths. Per-lead feature frames
are cached to parquet so re-runs are instant.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb

sys.path.insert(0, "scripts")
import train_wind_mvn as T  # noqa: E402

LOCATION = "bonehill_rocks"
LEADS = (24, 48, 72)
ERA5_GLOB = "../WeatherBlend/data/truth/era5/location=bonehill_rocks/**/*.parquet"
CACHE = Path("scripts/_xtruth_cache")
CACHE.mkdir(exist_ok=True)


def mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m])))


def fit(Xtr, ytr, Xva, yva):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                          random_state=42, verbose=-1)
    ok = ~np.isnan(ytr); okv = ~np.isnan(yva)
    m.fit(Xtr[ok], ytr[ok], eval_set=[(Xva[okv], yva[okv])],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def build_or_load(lead, dunk, oro, era5):
    cpath = CACHE / f"lead{lead}.parquet"
    if cpath.exists():
        return pd.read_parquet(cpath)
    df = T.build_features(LOCATION, lead, dunk, oro)
    if df.empty:
        return df
    feats = T.feature_column_order(df)
    df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"])
    df = df.merge(era5, on="ValidTimeUtc", how="inner").dropna(subset=["wsp_ms", "era5_spd"])
    df = df.sort_values("ValidTimeUtc").reset_index(drop=True)
    keep = feats + ["ValidTimeUtc", "wsp_ms", "era5_spd"]
    out = df[keep].copy()
    out.attrs = {}
    out.to_parquet(cpath)
    # stash feature list alongside
    (CACHE / f"lead{lead}.feats").write_text("\n".join(feats), encoding="utf-8")
    return out


def main():
    dunk = T.load_dunkeswell()
    oro = T.load_oro_static(LOCATION)
    con = duckdb.connect()
    era5 = con.execute(
        f"SELECT ValidTimeUtc, WindSpeed10m AS era5_spd FROM read_parquet('{ERA5_GLOB}') "
        f"WHERE WindSpeed10m IS NOT NULL").df()
    era5["ValidTimeUtc"] = pd.to_datetime(era5["ValidTimeUtc"])

    for lead in LEADS:
        df = build_or_load(lead, dunk, oro, era5)
        if df.empty:
            print(f"lead {lead}: no rows"); continue
        feats = (CACHE / f"lead{lead}.feats").read_text(encoding="utf-8").splitlines()
        n = len(df); i_tr, i_va = int(n * 0.70), int(n * 0.85)
        X = df[feats].to_numpy(dtype=np.float64)
        yd = df["wsp_ms"].to_numpy(dtype=np.float64)
        ye = df["era5_spd"].to_numpy(dtype=np.float64)
        Xtr, Xva, Xte = X[:i_tr], X[i_tr:i_va], X[i_va:]
        yd_tr, yd_va, yd_te = yd[:i_tr], yd[i_tr:i_va], yd[i_va:]
        ye_tr, ye_va, ye_te = ye[:i_tr], ye[i_tr:i_va], ye[i_va:]

        m_d = fit(Xtr, yd_tr, Xva, yd_va)
        m_e = fit(Xtr, ye_tr, Xva, ye_va)
        pd_te, pe_te = m_d.predict(Xte), m_e.predict(Xte)
        pd_va, pe_va = m_d.predict(Xva), m_e.predict(Xva)

        # optimal blend weight on D (real wind) fit on val
        ws = np.linspace(0, 1, 21)
        best_w = min(ws, key=lambda w: mae(w * pd_va + (1 - w) * pe_va, yd_va))
        b50 = 0.5 * (pd_te + pe_te)
        bopt = best_w * pd_te + (1 - best_w) * pe_te

        win = f"{df['ValidTimeUtc'].iloc[i_va]:%Y-%m-%d}..{df['ValidTimeUtc'].iloc[-1]:%Y-%m-%d}"
        print(f"\n=== lead {lead}h  (n_test={len(Xte)}, window {win}) ===")
        print(f"  {'model':<22}{'vs Dunkeswell':>15}{'vs ERA5':>12}")
        print(f"  {'D  (lgb / real-obs)':<22}{mae(pd_te, yd_te):>15.3f}{mae(pd_te, ye_te):>12.3f}")
        print(f"  {'E  (champ / ERA5)':<22}{mae(pe_te, yd_te):>15.3f}{mae(pe_te, ye_te):>12.3f}")
        print(f"  {'blend 50/50':<22}{mae(b50, yd_te):>15.3f}{mae(b50, ye_te):>12.3f}")
        print(f"  {f'blend opt (w_D={best_w:.2f})':<22}{mae(bopt, yd_te):>15.3f}{mae(bopt, ye_te):>12.3f}")
    print("\nLower=better. w_D fit on val to minimise MAE vs Dunkeswell (real wind).")


if __name__ == "__main__":
    main()
