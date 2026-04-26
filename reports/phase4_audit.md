# Phase 4 — Audit findings

## UKMO missingness handling per pipeline

| Pipeline | UKMO handling in training | Effective pattern | Source |
|---|---|---|---|
| **WeatherBlend 3a-lean (current production champion)** | UKMO removed from feature set; `Model IN (...)` SQL filter excludes UKMO; `precip_ukmo`/`prob_ukmo` are `CAST(NULL AS DOUBLE)` in the pivot. All 105k training rows kept. | **Pattern 1 — drop UKMO entirely** | `WeatherBlend/src/WeatherBlend/Train/PrecipFeatureBuilder.cs` (commit `1deb190`); per-station champion versions `v2026-04-26_085126/085144/085202` |
| **WeatherBlend 3a-lean (prior champion, still A/B-deployed)** | UKMO present as feature; `WHERE COALESCE(...precip_ukmo...) IS NOT NULL` accepts any 1+ model present per row → UKMO can be NaN. | **Pattern 3 — NaN-tolerant** | Prior 3a versions `v2026-04-23_071842/071934/163848`, still in MANIFEST.Active for A/B coexistence |
| **weather-probabilistic Phase 3 Model A (current Bayesian baseline)** | `_select_features` (data.py:300) calls `df.dropna(subset=precip_cols)` → drops every row where any precip_* including UKMO is null. `prob_*` are 100% null and dropped. | **Pattern 2 — require UKMO non-null** (drops ~26% of rows) | `weather-probabilistic/src/data.py` `_select_features` |

This is significant: **the current published Phase 3 Bayesian numbers and current production Pattern 3 LightGBM numbers are scored on different test sets** because they handle UKMO nulls differently. The original Phase 4 brief assumed pipelines were already comparable; they aren't.

## Feature parity audit

WeatherBlend 3a-lean (`PrecipFeatureBuilder.OccurrenceFeatureNames`, 27 features):

```
precip_gfs, precip_ecmwf, precip_icon, precip_mf, precip_ukmo, precip_gem,         (6 per-model precip)
prob_gfs, prob_ecmwf, prob_icon, prob_mf, prob_ukmo, prob_gem,                     (6 per-model prob — all 100% null in training, harmlessly carried)
precip_mean, precip_std, precip_max, precip_agreement_wet_01,                      (4 ensemble spread)
rh_mean, dew_depression_mean, cloud_low_mean, cloud_mid_mean, cloud_high_mean,
cape_mean, wind_speed_mean,                                                         (7 meteo covariates)
hour_sin, hour_cos, doy_sin, doy_cos                                                (4 calendar)
```

weather-probabilistic Phase 3 Model A (after `_select_features`, 8 features):

```
precip_ecmwf_ifs025, precip_gem_seamless, precip_gfs_seamless, precip_icon_seamless,
precip_meteofrance_seamless, precip_ukmo_seamless,                                  (6 per-model precip)
hour_sin, hour_cos                                                                   (2 calendar)
```

(Phase 3 loads `prob_*` columns then immediately drops them: every prob_* is 100% null in WeatherBlend's offset_day parquet.)

**The two methods are not feature-identical and never have been.** WeatherBlend has 20 more features baked into its pipeline: ensemble spread (mean/std/max/agreement), meteo covariates (RH, dew depression, cloud layers, CAPE, wind), and day-of-year cyclical features. The Bayesian Phase 3 Model A uses only the per-model precip values plus hour-of-day.

This is a meaningful confound for any "LightGBM vs Bayesian" comparison.

## Phase 4 decision: do both (a) and (b)

Per user direction:
- **Headline = (b)** — train a stripped LightGBM with the same 7 features as the Bayesian 5-model variant (5 precip + hour sin/cos). Pure-algorithm comparison on identical inputs.
- **Supporting context = (a)** — extract the native 27-feature production LightGBM's predictions on the same test rows. Frame as "LightGBM in production with its richer feature engineering" — system comparison, not algorithm comparison.

The Bayesian side is a single 5-model variant: Phase 3 Model A re-fit with `MODELS_NO_UKMO` (the new constant added in this session). 7 features matching the stripped LightGBM exactly.

## 5-model dataset confirmation

Loading `prepare_phase3_dataset(models=MODELS_NO_UKMO)`:
- Train rows: **105,360** (vs 78,479 with 6-model — recovers 26% by dropping UKMO requirement)
- Test rows: **26,397**
- Features: 7 (5 per-model precip + hour_sin + hour_cos)
- Stations: Bellever / Princetown / Hexworthy
- Leads: 24h / 48h / 72h
- Split: chronological per-station on unique valid_times, 80/20

This is the canonical Phase 4 split. Both LightGBM stripped and Bayesian 5-model train and test on these exact rows.

## Cross-repo data flow

Stripped LightGBM is implemented **in-repo** as a Python script using the `lightgbm` package against `Phase3Dataset` directly. Avoids cross-repo coordination entirely; both methods consume identical inputs from `prepare_phase3_dataset(models=MODELS_NO_UKMO)`.

Native 27-feature LightGBM (supporting context) requires either:
- Loading WeatherBlend's saved `model.zip` (ML.NET format) — non-trivial without ML.NET runtime in Python; needs ONNX export or shell-out
- Re-implementing WeatherBlend's 27-feature engineering in Python — significant work to replicate ensemble spread + meteo covariates accurately

Defer the native 27-feature comparison until headline (b) lands. Document as Phase 4 stretch.

## Sampler-side: nutpie progress visibility

Phase 3 Model B was killed at 6h 41min sampling because we couldn't tell stuck-vs-slow without per-draw progress. Phase 3 Model A (per-lead independent partial pool — what Phase 4 reuses) finished cleanly per lead at ~50 min each, with the heartbeat-and-summary pattern providing adequate visibility. Phase 4 reuses Model A's structure exactly, so the heartbeat-only approach is sufficient. Progress-bar refactor (blocking=False + polling thread) is documented as a known follow-up but **not blocking Phase 4** — Model A's per-lead durations are bounded.
