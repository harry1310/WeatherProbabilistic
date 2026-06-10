"""Train wind_speed_lgb as a Python quantile-LGB + conformal (CQR) model.

Option B of the wind CQR plan (2026-06-09): the wind-speed model moves from the
.NET WindSpeedLgbBlender to Python so it can produce calibrated PREDICTION INTERVALS
(the spike confirmed ML.NET LightGbm has no quantile objective). One model now owns
both the point (q50) and the band (q_lo/q_hi), built from a single feature recipe.

Per lead:
  * train q_lo (alpha=0.05), q_med (0.50), q_hi (0.95) LightGBM quantile heads on the
    29-feature set (reuses train_wind_mvn.build_features) vs Dunkeswell SYNOP truth;
  * CQR conformal correction on a held-out CALIBRATION slice:
        E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i));  Q = ceil((n+1)*0.9)/n quantile of E
    so the deployed 90% band is [q_lo - Q, q_hi + Q] with a coverage guarantee;
  * q_med is the point — emitted exactly where the old .NET wind_speed_lgb did
    (predictions/wind/model_version=v{ts}_wind_speed_lgb), so wind_blend / Models / verify
    keep consuming it unchanged. The band (q_lo/q_hi/Q) is the new sidecar.

Bundle layout (mirrors train_4a/train_wind_mvn so the .NET Models page reads it):
  q_lo_lead_{L}h.txt / q_med_lead_{L}h.txt / q_hi_lead_{L}h.txt  (LightGBM boosters)
  feature_schema.json, calibration.json (per-lead Q + coverage), training_metadata.json
Writes to data/models/wind/{location}/v{ts}_wind_speed_lgb/. Manifest promotion +
R2 push are handled by the retrain-python workflow (same as 4a).
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, "scripts")
import train_wind_mvn as T  # noqa: E402  (build_features / load_dunkeswell / load_oro_static / time_split / LEADS / NWPS)

TARGET = "wind-speed-lgb"
PHASE = "wind_speed_lgb"
COVERAGE = 0.90                # nominal band coverage
ALPHA_LO, ALPHA_HI = 0.05, 0.95
K_FOLDS = 10                   # cross-conformal fold count (contiguous chronological blocks)
EARLY_STOP = 30               # matches the .NET ElementTrainerHarness esr
# Required NWPs — MUST match WindSpeedLgbFeatureBuilder's requiredModels. Rows where
# any of these lacks wsp OR wdir are dropped, exactly like the .NET requiredNotNull.
# Without this the permissive build_features keeps the sparse pre-2024 regime (ecmwf/
# icon/jma mostly NaN) the .NET model never sees, which degraded q50 at L48/L72.
REQUIRED_MODELS = ("gfs_seamless", "ecmwf_ifs025", "icon_seamless", "jma_seamless")
# n_estimators is the cap; early-stopping on the val slice picks the actual count
# (matches the .NET 500-iter + esr=30 setup).
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                  min_child_samples=20, subsample=1.0, random_state=42, verbose=-1)


def _fit_quantile(Xtr, ytr, Xval, yval, alpha):
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **LGB_PARAMS)
    okt, okv = ~np.isnan(ytr), ~np.isnan(yval)
    m.fit(Xtr[okt], ytr[okt], eval_set=[(Xval[okv], yval[okv])],
          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return m


def _fit_quantile_fixed(Xtr, ytr, alpha, n_estimators):
    """Quantile-LGB with a FIXED iteration count (no early stopping / no val
    slice). Used for the K-fold cross-conformal fold models so each fold can use
    all of its K-1 training blocks without carving out a nested val slice. The
    iteration count is taken from a full-dense early-stopped quantile fit (see
    _best_iter_for_alpha) so it tracks the same complexity the deployed models
    settle on."""
    p = {**LGB_PARAMS, "n_estimators": int(n_estimators)}
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **p)
    okt = ~np.isnan(ytr)
    m.fit(Xtr[okt], ytr[okt])
    return m


def _best_iter_for_alpha(X, y, alpha):
    """Fit one early-stopped quantile-LGB on a 85/15 chronological split of the
    full dense data and return its best_iteration_. This becomes the fixed
    n_estimators for the cross-conformal fold models, so the fold trees match
    the complexity an early-stopped fit picks on the same data distribution."""
    n = len(X)
    i = int(n * 0.85)
    m = _fit_quantile(X[:i], y[:i], X[i:], y[i:], alpha)
    bi = getattr(m, "best_iteration_", None)
    if not bi or bi <= 0:
        bi = LGB_PARAMS["n_estimators"]
    return int(bi)


def _fit_point(Xtr, ytr, Xval, yval):
    # The POINT mirrors the .NET wind_speed_lgb exactly: L2 objective with MAE as the
    # early-stopping metric (ML.NET LightGbm 4.0 is L2-only). A pinball-0.5 head is NOT
    # the same and was a touch worse — this is the like-for-like point estimator.
    m = lgb.LGBMRegressor(objective="regression", metric="l1", **LGB_PARAMS)
    okt, okv = ~np.isnan(ytr), ~np.isnan(yval)
    m.fit(Xtr[okt], ytr[okt], eval_set=[(Xval[okv], yval[okv])],
          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return m


def _conformal_q(q_lo_cal, q_hi_cal, y_cal, coverage):
    """Split-CQR correction: the ceil((n+1)*coverage)/n empirical quantile of the
    nonconformity scores E_i = max(q_lo - y, y - q_hi). Widens (or, if negative,
    tightens) the quantile band to the guaranteed coverage level."""
    e = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    e = e[~np.isnan(e)]
    n = len(e)
    if n == 0:
        return 0.0
    k = math.ceil((n + 1) * coverage)
    k = min(k, n)                       # clamp: with small n the rank can exceed n
    return float(np.sort(e)[k - 1])


def _q_from_scores(e, coverage):
    """The ceil((n+1)*coverage)/n empirical quantile of a 1-D score array — the
    SAME formula as _conformal_q, factored out so cross-conformal can reuse it on
    the pooled out-of-fold scores."""
    e = e[~np.isnan(e)]
    n = len(e)
    if n == 0:
        return 0.0, 0
    k = math.ceil((n + 1) * coverage)
    k = min(k, n)
    return float(np.sort(e)[k - 1]), n


def _crossconf_oof_scores(X, y, coverage, k_folds, lo_iters, hi_iters):
    """K-fold CROSS-CONFORMAL out-of-fold nonconformity scores.

    Partition the dense rows into K CONTIGUOUS chronological blocks (np.array_split
    over the time-sorted index). Contiguous blocks (rather than interleaved/random
    folds) keep each fold's held-out scores from leaking serially-autocorrelated
    neighbours into its own training blocks; POOLING the out-of-fold scores across
    all K blocks then makes the calibration distribution span the entire dense
    period / every season present, which is exactly the seasonal coverage span that
    a single split-conformal calib block cannot capture.

    For each fold k: train q_lo(0.05) and q_hi(0.95) on the other K-1 blocks (fixed
    iteration counts — no nested val), predict block k, score
    E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)). Returns the pooled OOF scores AND a
    parallel array of the OOF (lo, hi) raw-quantile predictions so caller can build
    the OOF band [lo - Q, hi + Q] for the pooled-OOF coverage diagnostic."""
    n = len(X)
    idx = np.arange(n)
    blocks = np.array_split(idx, k_folds)
    oof_e = np.full(n, np.nan)
    oof_lo = np.full(n, np.nan)
    oof_hi = np.full(n, np.nan)
    for kk, blk in enumerate(blocks):
        train_idx = np.setdiff1d(idx, blk, assume_unique=True)
        Xtr, ytr = X[train_idx], y[train_idx]
        Xbk, ybk = X[blk], y[blk]
        m_lo = _fit_quantile_fixed(Xtr, ytr, ALPHA_LO, lo_iters)
        m_hi = _fit_quantile_fixed(Xtr, ytr, ALPHA_HI, hi_iters)
        plo, phi = m_lo.predict(Xbk), m_hi.predict(Xbk)
        oof_lo[blk] = plo
        oof_hi[blk] = phi
        oof_e[blk] = np.maximum(plo - ybk, ybk - phi)
        print(f"      fold {kk+1}/{k_folds}: train n={len(train_idx)} block n={len(blk)} "
              f"(rows {int(blk[0])}..{int(blk[-1])})", flush=True)
    return oof_e, oof_lo, oof_hi, y


def _required_filter(df, feats=None):
    """Apply the .NET WindSpeedLgbFeatureBuilder required-models dropna: drop any
    row where any required NWP lacks wsp or wdir. Returns a sorted, reindexed copy."""
    req_cols = [f"WindSpeed10m_{m}" for m in REQUIRED_MODELS] \
             + [f"WindDirection10m_{m}" for m in REQUIRED_MODELS]
    have = [c for c in req_cols if c in df.columns]
    out = df.dropna(subset=have)
    return out.sort_values("ValidTimeUtc").reset_index(drop=True)


def train_location(location: str, models_root: Path, leads=T.LEADS,
                   build: str = "required", conformal: str = "split") -> int:
    if conformal == "crossconf" and build != "required":
        print(f"crossconf is only implemented for build=required (got {build!r}) — aborting.",
              flush=True)
        return 2
    dunk = T.load_dunkeswell()
    oro = T.load_oro_static(location)
    version = "v" + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S") + "_wind_speed_lgb"
    out_dir = models_root / "wind" / location / version
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = None
    per_lead = {}
    for lead in leads:
        df_raw = T.build_features(location, lead, dunk, oro)
        if df_raw.empty:
            print(f"lead {lead}h: no rows — skipping", flush=True)
            continue
        feats = T.feature_column_order(df_raw)
        feature_names = feature_names or feats

        if build == "required":
            # Required-models filter (match the .NET WindSpeedLgbFeatureBuilder): drop
            # rows where any required NWP lacks wsp or wdir.
            df = _required_filter(df_raw, feats)
            if len(df) < 500:
                print(f"lead {lead}h: only {len(df)} rows after required-models filter — skipping", flush=True)
                continue
            n = len(df)
            # EXACT match to the .NET ElementTrainerHarness split: chronological 70/15/15
            # (train / val / test). Train on 70% so q50 sees the same data as the .NET
            # point; test on the last 15% so the q50 test MAE is directly comparable to
            # the .NET bundle's reported test MAE (same rows, same window).
            i_tr, i_te = int(n * 0.70), int(n * 0.85)
            X = df[feats].to_numpy(dtype=np.float64)
            y = df["wsp_ms"].to_numpy(dtype=np.float64)
            Xtr, Xval, Xcal, ycal = X[:i_tr], X[i_tr:i_te], X[i_tr:i_te], y[i_tr:i_te]
            ytr, yval = y[:i_tr], y[i_tr:i_te]
            Xte, yte = X[i_te:], y[i_te:]
            split_desc = "70/15/15 train/val/test; conformal Q on val (=early-stop slice)"
            n_train, n_val, n_calib, n_test = i_tr, i_te - i_tr, i_te - i_tr, n - i_te
            range_train = f"{df.iloc[0]['ValidTimeUtc']} -> {df.iloc[i_tr-1]['ValidTimeUtc']}"
            range_test = f"{df.iloc[i_te]['ValidTimeUtc']} -> {df.iloc[n-1]['ValidTimeUtc']}"
            test_months = int(df.iloc[i_te:]["ValidTimeUtc"].dt.to_period("M").nunique())
            test_lo_hi = df.iloc[i_te], df.iloc[n-1]
        else:  # permissive
            # Keep EVERY valid-time build_features returned (>=1 NWP present), incl.
            # the feature-sparse 2022-23 rows (missing models stay NaN; LightGBM
            # handles NaN natively). Chronological 60/15/15/10 train/val/calib/test.
            # The conformal Q comes from the dedicated CALIB slice — NOT the
            # early-stopping val set and NOT the test set — so coverage is honest.
            df = df_raw.sort_values("ValidTimeUtc").reset_index(drop=True)
            if len(df) < 500:
                print(f"lead {lead}h: only {len(df)} permissive rows — skipping", flush=True)
                continue
            n = len(df)
            i_tr = int(n * 0.60)
            i_va = int(n * 0.75)
            i_ca = int(n * 0.90)
            X = df[feats].to_numpy(dtype=np.float64)
            y = df["wsp_ms"].to_numpy(dtype=np.float64)
            Xtr, ytr = X[:i_tr], y[:i_tr]
            Xval, yval = X[i_tr:i_va], y[i_tr:i_va]
            Xcal, ycal = X[i_va:i_ca], y[i_va:i_ca]
            Xte, yte = X[i_ca:], y[i_ca:]
            split_desc = "60/15/15/10 train/val/calib/test; conformal Q on dedicated calib slice"
            n_train, n_val, n_calib, n_test = i_tr, i_va - i_tr, i_ca - i_va, n - i_ca
            range_train = f"{df.iloc[0]['ValidTimeUtc']} -> {df.iloc[i_tr-1]['ValidTimeUtc']}"
            range_test = f"{df.iloc[i_ca]['ValidTimeUtc']} -> {df.iloc[n-1]['ValidTimeUtc']}"
            test_months = int(df.iloc[i_ca:]["ValidTimeUtc"].dt.to_period("M").nunique())
            test_lo_hi = df.iloc[i_ca], df.iloc[n-1]

        # POINT (q_med): UNCHANGED in both conformal modes — 70/15/15 train slice,
        # L2+l1 early stopping. This is the gated .NET-comparable point estimator and
        # must keep printing 0.969/1.076/1.248 at L24/48/72.
        m_med = _fit_point(Xtr, ytr, Xval, yval)       # L2 point — like-for-like with .NET

        # Extra crossconf diagnostics (filled below) — None in split mode.
        oof_cover = None
        holdout_cover = None
        k_used = None
        fold_iters = None

        if conformal == "split":
            m_lo = _fit_quantile(Xtr, ytr, Xval, yval, ALPHA_LO)
            m_hi = _fit_quantile(Xtr, ytr, Xval, yval, ALPHA_HI)
            # Conformal Q on the CALIB slice (in permissive this is a dedicated held-out
            # slice; in required it falls back to the val slice — see split_desc).
            Q = _conformal_q(m_lo.predict(Xcal), m_hi.predict(Xcal), ycal, COVERAGE)
            split_desc_use = split_desc
        else:  # crossconf (required build only — guarded in train_location)
            # BAND via K-fold CROSS-CONFORMAL on the FULL dense rows (X, y here are the
            # required-filtered dense arrays). q_med stays the 70/15/15 fit above; only
            # the band changes.
            # Fixed fold-iteration counts: take best_iteration_ from a full-dense
            # early-stopped quantile fit per side, so the no-early-stop fold models match
            # the complexity an early-stopped fit settles on (documented choice).
            lo_iters = _best_iter_for_alpha(X, y, ALPHA_LO)
            hi_iters = _best_iter_for_alpha(X, y, ALPHA_HI)
            fold_iters = {"q_lo": lo_iters, "q_hi": hi_iters}
            print(f"    lead {lead}h crossconf: fold iters q_lo={lo_iters} q_hi={hi_iters}, "
                  f"K={K_FOLDS}, dense n={n}", flush=True)
            oof_e, oof_lo, oof_hi, oof_y = _crossconf_oof_scores(
                X, y, COVERAGE, K_FOLDS, lo_iters, hi_iters)
            Q, q_n = _q_from_scores(oof_e, COVERAGE)
            k_used = K_FOLDS
            # (i) Pooled-OOF coverage: each OOF point covered by its own OOF band at the
            # pooled Q. In-distribution, all-seasons estimate — ~90% by construction.
            okf = ~np.isnan(oof_e)
            lo_oof = oof_lo[okf] - Q
            hi_oof = oof_hi[okf] + Q
            y_oof = oof_y[okf]
            oof_cover = float(np.mean((y_oof >= lo_oof) & (y_oof <= hi_oof)))
            # (ii) Held-out chronological backtest: cross-conformal Q from the first 85%
            # of dense rows, empirical coverage on the held-out last 15% (the Nov-Dec
            # windy tail, the same window q50 uses). Pessimistic by design.
            h = int(n * 0.85)
            X_cal, y_cal = X[:h], y[:h]
            X_hd, y_hd = X[h:], y[h:]
            li2 = _best_iter_for_alpha(X_cal, y_cal, ALPHA_LO)
            hi2 = _best_iter_for_alpha(X_cal, y_cal, ALPHA_HI)
            e_hd, _, _, _ = _crossconf_oof_scores(X_cal, y_cal, COVERAGE, K_FOLDS, li2, hi2)
            Q_hd, _ = _q_from_scores(e_hd, COVERAGE)
            # Deployed-style q_lo/q_hi for the held-out band: trained on the first-85%
            # calib pool (mirrors the production all-dense fit, but only on calib rows).
            mlo_hd = _fit_quantile_fixed(X_cal, y_cal, ALPHA_LO, li2)
            mhi_hd = _fit_quantile_fixed(X_cal, y_cal, ALPHA_HI, hi2)
            ok_hd = ~np.isnan(y_hd)
            lo_hd = mlo_hd.predict(X_hd) - Q_hd
            hi_hd = mhi_hd.predict(X_hd) + Q_hd
            holdout_cover = float(np.mean((y_hd[ok_hd] >= lo_hd[ok_hd]) &
                                          (y_hd[ok_hd] <= hi_hd[ok_hd])))
            # DEPLOYED q_lo/q_hi: trained on ALL dense rows (fixed iters), per brief.
            m_lo = _fit_quantile_fixed(X, y, ALPHA_LO, lo_iters)
            m_hi = _fit_quantile_fixed(X, y, ALPHA_HI, hi_iters)
            split_desc_use = (f"q_med 70/15/15; band = {K_FOLDS}-fold cross-conformal Q on "
                              f"pooled OOF scores over contiguous chronological blocks "
                              f"(deployed q_lo/q_hi on all {n} dense rows, fixed iters "
                              f"q_lo={lo_iters}/q_hi={hi_iters})")

        # Test-set diagnostics: q50 MAE (the point metric, matches the old .NET lgb),
        # plus empirical coverage + median width of the conformalised band. NOTE: in
        # crossconf the OOF rows include this test slice (cross-conformal uses all dense
        # rows), so the headline 'cover' here is the test-slice subset of the OOF
        # coverage — the honest backtest is holdout_cover (ii), reported separately.
        p_med = m_med.predict(Xte)
        lo, hi = m_lo.predict(Xte) - Q, m_hi.predict(Xte) + Q
        ok = ~np.isnan(yte)
        q50_mae = float(np.mean(np.abs(p_med[ok] - yte[ok])))
        cover = float(np.mean((yte[ok] >= lo[ok]) & (yte[ok] <= hi[ok])))
        width = float(np.median((hi - lo)[ok]))
        split_desc = split_desc_use

        # GATE re-score (permissive only): take the permissive-trained q_med head and
        # score it on the SAME required-filter Nov-Dec-2024 test window the .NET
        # baseline reports (REQUIRED_MODELS dropna, 70/15/15, last-15% slice). This
        # confirms permissive training didn't cost q50 on the .NET-comparable window.
        rf_mae = None
        rf_n = 0
        rf_range = None
        if build == "permissive":
            df_rf = _required_filter(df_raw, feats)
            if len(df_rf) >= 20:
                n_rf = len(df_rf)
                j_te = int(n_rf * 0.85)
                X_rf = df_rf[feats].to_numpy(dtype=np.float64)
                y_rf = df_rf["wsp_ms"].to_numpy(dtype=np.float64)
                Xte_rf, yte_rf = X_rf[j_te:], y_rf[j_te:]
                ok_rf = ~np.isnan(yte_rf)
                p_rf = m_med.predict(Xte_rf)
                rf_mae = float(np.mean(np.abs(p_rf[ok_rf] - yte_rf[ok_rf])))
                rf_n = int(ok_rf.sum())
                rf_range = (f"{df_rf.iloc[j_te]['ValidTimeUtc']} -> "
                            f"{df_rf.iloc[n_rf-1]['ValidTimeUtc']}")

        for tag, m in (("q_lo", m_lo), ("q_med", m_med), ("q_hi", m_hi)):
            m.booster_.save_model(str(out_dir / f"{tag}_lead_{lead}h.txt"))

        per_lead[lead] = dict(
            n_train=n_train, n_val=n_val, n_calib=n_calib, n_test=n_test,
            q50_mae=q50_mae, conformal_q=Q, coverage=cover, width=width,
            build=build, conformal=conformal, split_desc=split_desc,
            range_train=range_train, range_test=range_test, test_months=test_months,
            rf_q50_mae=rf_mae, rf_n_test=rf_n, rf_range=rf_range,
            oof_cover=oof_cover, holdout_cover=holdout_cover,
            k_folds=k_used, fold_iters=fold_iters,
        )
        print(f"lead {lead}h [{build}/{conformal}]: q50 MAE={q50_mae:.4f} m/s, Q={Q:+.4f}, "
              f"width={width:.3f} m/s, "
              f"n_tr={n_train} n_val={n_val} n_cal={n_calib} n_te={n_test} "
              f"(test {test_lo_hi[0]['ValidTimeUtc']:%Y-%m-%d}..{test_lo_hi[1]['ValidTimeUtc']:%Y-%m-%d})", flush=True)
        if conformal == "crossconf":
            print(f"           (i) pooled-OOF coverage = {oof_cover:.1%} (in-distribution, ~{COVERAGE:.0%} by construction)", flush=True)
            print(f"           (ii) held-out-tail coverage = {holdout_cover:.1%} (calib first 85%, test windy last 15% — pessimistic backtest)", flush=True)
            print(f"           (iii) median band width = {width:.3f} m/s, pooled Q = {Q:+.4f}, K = {k_used}, fold iters = {fold_iters}", flush=True)
        else:
            print(f"           split-conformal test-slice coverage = {cover:.1%} (target {COVERAGE:.0%})", flush=True)
        if rf_mae is not None:
            print(f"           required-filter re-score: q50 MAE={rf_mae:.4f} m/s "
                  f"on {rf_n} test rows ({rf_range})", flush=True)

    if not per_lead:
        print("No leads trained — aborting.", flush=True)
        return 3

    _write_bundle(out_dir, location, version, per_lead, feature_names)
    print(f"Wrote bundle → {out_dir}", flush=True)
    return 0


def _write_bundle(out_dir, location, version, per_lead, feature_names):
    (out_dir / "feature_schema.json").write_text(json.dumps({
        "Target": TARGET, "Phase": PHASE, "Location": location,
        "FeatureNames": feature_names, "NwpModels": list(T.NWPS),
        "PerLead": {str(l): {"LeadHours": l, "RequiredModels": [], "OptionalModels": list(T.NWPS),
                             "Models": list(T.NWPS), "Tier": "quantile+cqr"} for l in per_lead},
    }, indent=2, allow_nan=False))

    any_cc = any(c.get("conformal") == "crossconf" for c in per_lead.values())
    (out_dir / "calibration.json").write_text(json.dumps({
        "method": ("CQR (K-fold cross-conformal quantile regression)" if any_cc
                   else "CQR (split-conformal quantile regression)"),
        "coverage": COVERAGE, "alpha_lo": ALPHA_LO, "alpha_hi": ALPHA_HI,
        "per_lead": {str(l): {
            "conformal_q": c["conformal_q"], "test_coverage": c["coverage"],
            "test_median_width_ms": c["width"],
            "conformal_mode": c.get("conformal"),
            "k_folds": c.get("k_folds"),
            "fold_iters": c.get("fold_iters"),
            "pooled_oof_coverage": c.get("oof_cover"),
            "holdout_tail_coverage": c.get("holdout_cover"),
        } for l, c in per_lead.items()},
    }, indent=2, allow_nan=False))

    # training_metadata.json — BlendTestMae carries q50 MAE (m/s) so the .NET Models
    # page renders the wind_speed_lgb card exactly as before.
    (out_dir / "training_metadata.json").write_text(json.dumps({
        "Version": version, "Target": TARGET, "Phase": PHASE, "LocationName": location,
        "DataSource": "open_meteo_previous_runs+dunkeswell_midas_synop",
        "TrainedAtUtc": datetime.now(timezone.utc).isoformat(),
        "Hyperparameters": {"library": f"lightgbm=={lgb.__version__}",
                            "objective": "quantile", "quantiles": [ALPHA_LO, 0.5, ALPHA_HI],
                            "conformal": "split-CQR", **{k: v for k, v in LGB_PARAMS.items()}},
        "PerLead": {str(l): {
            "LeadHours": l, "TrainRows": c["n_train"], "ValRows": c["n_val"],
            "CalibRows": c.get("n_calib"), "TestRows": c["n_test"],
            "BuildMode": c.get("build"), "SplitDescription": c.get("split_desc"),
            "BestSingle": "wind_speed_lgb_qmed",
            "BlendTestMae": c["q50_mae"], "BlendTestRmse": c["width"], "BlendTestBias": None,
            "BestSingleValMae": c["q50_mae"], "BestSingleTestMae": c["q50_mae"],
            "RequiredFilterTestMae": c.get("rf_q50_mae"),
            "RequiredFilterTestRows": c.get("rf_n_test"),
            "RequiredFilterTestRange": c.get("rf_range"),
            "DataRangeTrain": c["range_train"], "DataRangeTest": c["range_test"],
            "TestCalendarMonths": c["test_months"],
        } for l, c in per_lead.items()},
        "DeviationsFromBrief": [
            "Python quantile-LGB + split-CQR (Option B): one model owns q50 (point) + the "
            "[q_lo-Q, q_hi+Q] 90% band. Replaces the .NET WindSpeedLgbBlender.",
            "BlendTestRmse repurposed to carry the conformalised median band width (m/s).",
        ],
    }, indent=2, allow_nan=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--location", default="bonehill_rocks")
    ap.add_argument("--models-root", default=str(T.WEATHERBLEND_DATA_ROOT / "models"))
    ap.add_argument("--build", choices=("required", "permissive"), default="required",
                    help="required: .NET-matching required-models dropna (~1yr); "
                         "permissive: keep all build_features rows (~3yr) with a "
                         "dedicated conformal calib slice. Default: required.")
    ap.add_argument("--conformal", choices=("split", "crossconf"), default="split",
                    help="split: single held-out calib slice (existing behaviour); "
                         "crossconf: K-fold cross-conformal Q on pooled out-of-fold "
                         "scores (required build only). Default: split.")
    args = ap.parse_args()
    sys.exit(train_location(args.location, Path(args.models_root),
                            build=args.build, conformal=args.conformal))


if __name__ == "__main__":
    main()
