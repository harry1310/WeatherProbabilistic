"""EVAL-ONLY: walk-forward backtest of wind-speed CQR cross-conformal coverage.

Measures the PRODUCTION-REGIME coverage of the cross-conformal band. Production
retrains weekly on a rolling window that ALWAYS includes the current season and
predicts the seasonally-matched near future. The existing held-out-tail number
(82-84%) is pessimistic because it calibrates on calmer data and tests on an
excluded windy season. This walk-forward mimics the production regime instead.

For each lead and each rolling cutoff c (over the dense / required-filter data):
  * train = dense rows with ValidTimeUtc <= c
  * fit q_lo / q_med / q_hi on `train` EXACTLY as the crossconf production path:
      - deployed q_lo/q_hi: _fit_quantile_fixed on all of `train` at the fixed
        best_iteration_ from a full-`train` early-stopped fit (_best_iter_for_alpha)
      - q_med: _fit_point with an internal early-stop val carved from `train`
  * conformal Q: K-fold cross-conformal (_crossconf_oof_scores) over `train` ONLY
  * test = dense rows with c < ValidTimeUtc <= c + 28 days
  * coverage on test = fraction with q_lo(x)-Q <= y <= q_hi(x)+Q

NO LEAKAGE: models + Q at cutoff c use ONLY rows <= c; test is rows > c.

This script does NOT train, promote, or write any production artefact. It imports
and REUSES the production helpers from train_wind_speed_pi; it does not duplicate
or modify them. It prints numbers only.
"""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import train_wind_mvn as T          # noqa: E402
import train_wind_speed_pi as P     # noqa: E402

LOCATION = "bonehill_rocks"
TEST_WINDOW_DAYS = 28
MIN_TEST_ROWS = 50
START_FRAC = 0.55        # first cutoff at ~55% of dense rows accrued
END_BUFFER_DAYS = 28     # last cutoff at least one test window before dense end
STEP_DAYS = 16           # ~2-3 week spacing; dense span is only ~11 months so a
                         # smaller step is needed to land >=6-8 cutoffs (documented)
TARGET_CUTOFFS = 8       # aim for at least 6-8 per lead


def _pick_cutoffs(times: pd.Series) -> list[pd.Timestamp]:
    """A sequence of rolling cutoff dates mimicking weekly-ish retrains: step
    every ~STEP_DAYS from where ~START_FRAC of dense data has accrued up to
    END_BUFFER_DAYS before the dense end (so each test window has data)."""
    t0 = times.iloc[int(len(times) * START_FRAC)]
    t_end = times.iloc[-1]
    last = t_end - timedelta(days=END_BUFFER_DAYS)
    if last <= t0:
        return []
    cutoffs = []
    c = t0
    while c <= last:
        cutoffs.append(c)
        c = c + timedelta(days=STEP_DAYS)
    return cutoffs


def _fit_at_cutoff(Xtr_all, ytr_all):
    """Fit deployed q_lo/q_med/q_hi + cross-conformal Q on `train` ONLY, exactly
    as train_wind_speed_pi's crossconf path does. Returns (m_lo, m_hi, m_med, Q)."""
    n = len(Xtr_all)
    # q_med: 70/15/15 internal split with early-stop val (mirrors production point head).
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    m_med = P._fit_point(Xtr_all[:i_tr], ytr_all[:i_tr],
                         Xtr_all[i_tr:i_va], ytr_all[i_tr:i_va])

    # Fixed fold-iteration counts from a full-`train` early-stopped quantile fit.
    lo_iters = P._best_iter_for_alpha(Xtr_all, ytr_all, P.ALPHA_LO)
    hi_iters = P._best_iter_for_alpha(Xtr_all, ytr_all, P.ALPHA_HI)

    # K-fold cross-conformal Q over `train` ONLY.
    k = P.K_FOLDS if n >= 200 else max(3, min(P.K_FOLDS, n // 20))
    oof_e, _, _, _ = P._crossconf_oof_scores(
        Xtr_all, ytr_all, P.COVERAGE, k, lo_iters, hi_iters)
    Q, _ = P._q_from_scores(oof_e, P.COVERAGE)

    # Deployed q_lo/q_hi: fixed-iter fit on ALL of `train`.
    m_lo = P._fit_quantile_fixed(Xtr_all, ytr_all, P.ALPHA_LO, lo_iters)
    m_hi = P._fit_quantile_fixed(Xtr_all, ytr_all, P.ALPHA_HI, hi_iters)
    return m_lo, m_hi, m_med, Q, k, lo_iters, hi_iters


def eval_lead(location, lead, dunk, oro):
    df_raw = T.build_features(location, lead, dunk, oro)
    if df_raw.empty:
        print(f"[lead {lead}h] no rows — skipping lead", flush=True)
        return None
    feats = T.feature_column_order(df_raw)
    df = P._required_filter(df_raw, feats)   # dense / required-filter, sorted by ValidTimeUtc
    n_dense = len(df)
    times = df["ValidTimeUtc"]
    print(f"\n[lead {lead}h] dense n={n_dense}, span "
          f"{times.iloc[0]:%Y-%m-%d} .. {times.iloc[-1]:%Y-%m-%d}", flush=True)

    X = df[feats].to_numpy(dtype=np.float64)
    y = df["wsp_ms"].to_numpy(dtype=np.float64)

    cutoffs = _pick_cutoffs(times)
    if len(cutoffs) < 6:
        print(f"[lead {lead}h] WARNING: only {len(cutoffs)} cutoffs fit in the "
              f"dense span (target {TARGET_CUTOFFS}) — using as many as fit.", flush=True)
    print(f"[lead {lead}h] {len(cutoffs)} candidate cutoffs "
          f"({cutoffs[0]:%Y-%m-%d} .. {cutoffs[-1]:%Y-%m-%d}), step ~{STEP_DAYS}d, "
          f"test window {TEST_WINDOW_DAYS}d", flush=True)

    tv = times.values  # for boolean masks
    rows = []
    pooled_y, pooled_lo, pooled_hi = [], [], []
    n_skipped = 0
    for ci, c in enumerate(cutoffs):
        c64 = np.datetime64(c)
        test_hi = np.datetime64(c + timedelta(days=TEST_WINDOW_DAYS))
        tr_mask = tv <= c64
        te_mask = (tv > c64) & (tv <= test_hi)
        n_tr = int(tr_mask.sum())
        n_te = int(te_mask.sum())
        if n_te < MIN_TEST_ROWS:
            print(f"[lead {lead}h] cutoff {ci+1}/{len(cutoffs)} {c:%Y-%m-%d}: "
                  f"n_test={n_te} < {MIN_TEST_ROWS} — SKIP", flush=True)
            n_skipped += 1
            continue
        if n_tr < 200:
            print(f"[lead {lead}h] cutoff {ci+1}/{len(cutoffs)} {c:%Y-%m-%d}: "
                  f"n_train={n_tr} < 200 — SKIP", flush=True)
            n_skipped += 1
            continue

        print(f"[lead {lead}h] cutoff {ci+1}/{len(cutoffs)} {c:%Y-%m-%d}: "
              f"train n={n_tr}, test n={n_te} — fitting...", flush=True)
        Xtr, ytr = X[tr_mask], y[tr_mask]
        m_lo, m_hi, m_med, Q, k_used, lo_it, hi_it = _fit_at_cutoff(Xtr, ytr)

        Xte, yte = X[te_mask], y[te_mask]
        ok = ~np.isnan(yte)
        lo = m_lo.predict(Xte) - Q
        hi = m_hi.predict(Xte) + Q
        p_med = m_med.predict(Xte)
        cover = float(np.mean((yte[ok] >= lo[ok]) & (yte[ok] <= hi[ok])))
        q50_mae = float(np.mean(np.abs(p_med[ok] - yte[ok])))
        width = float(np.median((hi - lo)[ok]))

        pooled_y.append(yte[ok]); pooled_lo.append(lo[ok]); pooled_hi.append(hi[ok])
        rows.append(dict(cutoff=c, n_test=int(ok.sum()), coverage=cover,
                         q50_mae=q50_mae, width=width, Q=Q, k=k_used,
                         lo_it=lo_it, hi_it=hi_it))
        print(f"[lead {lead}h] cutoff {ci+1}/{len(cutoffs)} {c:%Y-%m-%d}: "
              f"coverage={cover:.1%}  q50_MAE={q50_mae:.4f}  width={width:.3f}  "
              f"Q={Q:+.4f}  K={k_used} (lo_it={lo_it}/hi_it={hi_it})", flush=True)

    if not rows:
        print(f"[lead {lead}h] no usable cutoffs (all skipped: {n_skipped})", flush=True)
        return dict(lead=lead, rows=[], n_skipped=n_skipped, n_candidates=len(cutoffs))

    covs = np.array([r["coverage"] for r in rows])
    py = np.concatenate(pooled_y); plo = np.concatenate(pooled_lo); phi = np.concatenate(pooled_hi)
    pooled_cover = float(np.mean((py >= plo) & (py <= phi)))
    return dict(lead=lead, rows=rows, n_skipped=n_skipped, n_candidates=len(cutoffs),
                avg_cover=float(covs.mean()), min_cover=float(covs.min()),
                max_cover=float(covs.max()), pooled_cover=pooled_cover,
                avg_width=float(np.mean([r["width"] for r in rows])),
                avg_q50_mae=float(np.mean([r["q50_mae"] for r in rows])),
                pooled_n=int(len(py)))


def main():
    print(f"Walk-forward CQR coverage eval — location={LOCATION}, leads={T.LEADS}", flush=True)
    print(f"test window={TEST_WINDOW_DAYS}d, min test rows={MIN_TEST_ROWS}, "
          f"step~{STEP_DAYS}d, K={P.K_FOLDS}, coverage target={P.COVERAGE:.0%}", flush=True)
    dunk = T.load_dunkeswell()
    oro = T.load_oro_static(LOCATION)

    results = []
    for lead in T.LEADS:
        r = eval_lead(LOCATION, lead, dunk, oro)
        if r is not None:
            results.append(r)

    print("\n" + "=" * 78, flush=True)
    print("WALK-FORWARD CQR COVERAGE SUMMARY (production regime, no leakage)", flush=True)
    print("=" * 78, flush=True)
    for r in results:
        lead = r["lead"]
        print(f"\n--- lead {lead}h ---", flush=True)
        print(f"  cutoffs used: {len(r['rows'])}  (candidates {r['n_candidates']}, "
              f"skipped {r['n_skipped']})", flush=True)
        if not r["rows"]:
            continue
        print(f"  {'cutoff':<12} {'n_test':>7} {'coverage':>9} {'q50_MAE':>9} {'width':>8} {'Q':>9}", flush=True)
        for row in r["rows"]:
            print(f"  {row['cutoff']:%Y-%m-%d} {row['n_test']:>7d} "
                  f"{row['coverage']:>8.1%} {row['q50_mae']:>9.4f} "
                  f"{row['width']:>8.3f} {row['Q']:>+9.4f}", flush=True)
        print(f"  avg coverage = {r['avg_cover']:.1%}  "
              f"(min {r['min_cover']:.1%}, max {r['max_cover']:.1%})", flush=True)
        print(f"  pooled coverage = {r['pooled_cover']:.1%}  (n={r['pooled_n']})", flush=True)
        print(f"  avg q50 MAE across windows = {r['avg_q50_mae']:.4f} m/s", flush=True)
        print(f"  avg band width = {r['avg_width']:.3f} m/s", flush=True)
        near = abs(r['avg_cover'] - P.COVERAGE) <= 0.03
        verdict = ("YES — production-regime coverage lands near 90%, materially "
                   "above the 82-84% worst-case tail" if near else
                   "NO — production-regime coverage does NOT land near 90%")
        print(f"  VERDICT: {verdict} "
              f"(avg {r['avg_cover']:.1%}, pooled {r['pooled_cover']:.1%})", flush=True)


if __name__ == "__main__":
    main()
