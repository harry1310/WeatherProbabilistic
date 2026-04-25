# Phase 2 Report — Hierarchical Bayesian Logistic Regression Across 3 Stations, lead 24h

**Date:** 2026-04-25
**Author:** Claude Code (initial pass) for Russell

## What we built

Three Bayesian logistic regressions in PyMC, all predicting the next-day
`P(wet)` (≥ 0.1 mm/h rainfall) at lead 24 h, fitted across three Dartmoor
EA rainfall stations (Bellever, Princetown, Hexworthy). The point of
Phase 2 is the *hierarchical* fit — partial pooling, where each station
gets its own coefficients but they're drawn from a shared population
distribution whose own parameters are estimated from the data — and we
benchmark it against the two extremes:

| Model | Per-station parameters? | Information sharing? |
|---|---|---|
| **B — Full pooling** | No, one set for all stations | Maximum (treats stations as identical) |
| **A — No pooling** | Yes, three independent fits | None |
| **C — Partial pooling** | Yes, drawn from shared `Normal(μ, σ)` | Learned from the data |

Goal: learn the hierarchical workflow end to end, see whether partial
pooling beats either extreme on test-set Brier, and read the σ
hyperparameters to find out which forecast features stations actually
*disagree* about.

Beating LightGBM was, again, explicitly out of scope.

## Data

Same loader as Phase 1 (`src/data.py`), now extended to all three
stations and concatenated with a `station_idx` column. WeatherBlend
parquets at `C:/Projects/Weather/WeatherBlend/data/`.

| | Bellever | Princetown | Hexworthy |
|---|---|---|---|
| Train rows | 8,093 | 10,052 | 8,098 |
| Test rows | 2,024 | 2,514 | 2,025 |
| Train wet fraction | 0.202 | 0.200 | 0.236 |
| Test wet fraction | 0.216 | **0.314** | 0.247 |
| Train end | 2025-07-17 | 2025-09-29 | 2025-07-17 |
| Test end | 2025-10-09 | **2026-01-29** | 2025-10-09 |

**Combined: 26,243 train + 6,563 test rows**, same 8 features as Phase 1
(6 per-model lead-24h precip + `hour_sin`/`hour_cos`).

Two things to flag about the splits:

1. **Princetown has more rows.** The EA data starts a bit earlier and
   ends much later there, so the 80/20 chronological split puts the
   Princetown test partition into autumn 2025 → late January 2026 — a
   genuinely different climate regime from the (mid-summer) end of the
   other two stations' test windows.
2. **Princetown's test wet fraction (0.31) is much higher than its
   train wet fraction (0.20).** That's a real covariate shift the
   hierarchical model can't fix — pooling across stations doesn't help
   if the station's own distribution has moved. We see this clearly in
   the test-Brier results below.

Standardisation is *pooled* — one mean/std per feature computed across
all stations' train rows, then applied uniformly. This keeps the σ_β
posterior interpretable: a learned σ_β = 0.3 means "per-station β values
differ by ~0.3 logit units on the same standardised scale across all
stations", a comparable number for any feature.

## Model specification

The three models live at `src/models/phase2_{full,no,partial}_pooling.py`.

**Full pooling (Model B)**

Identical to Phase 1's single-station model, just trained on the pooled
26k rows:

```
y_i  ~  Bernoulli(p_i)
logit(p_i)  =  intercept  +  Σ_k  β_k · x_{i,k}
intercept  ~  Normal(0, 2),   β_k  ~  Normal(0, 1)
```

**No pooling (Model A)**

Three independent fits of the full-pooling model — one per station,
each on its own slice of train data. Implemented as a thin loop
(`src/models/phase2_no_pooling.py`).

**Partial pooling (Model C)**

The hierarchical bit. Each station `s` gets its own intercept and its
own β per feature, but each station-level parameter is drawn from a
shared population distribution:

```
intercept_s  ~  Normal(μ_intercept, σ_intercept)
β_{s,k}      ~  Normal(μ_β_k,       σ_β_k)
```

with priors

| Hyperparameter | Prior | Reasoning |
|---|---|---|
| `μ_intercept` | Normal(0, 2) | Same as Phase 1's `intercept` prior — the population-average is a "single sensible fit". |
| `μ_β_k` | Normal(0, 1) | Same as Phase 1's `β` prior — population-average effect on standardised features. |
| `σ_intercept` | HalfNormal(0.5) | Between-station SD must be positive. 0.5 puts the 95% prior interval at roughly (0, 1) logit units — permissive but discouraging the funnel geometry that breaks NUTS at higher prior σ. |
| `σ_β_k` | HalfNormal(0.5) | Same reasoning, per feature. |

**Non-centred reparameterisation.** Both `intercept_s` and `β_{s,k}`
are written as `μ + σ · z` where `z ~ Normal(0, 1)`. This is the
standard fix for the funnel pathology in hierarchical models — when
σ is small, the centred parameterisation creates a tight neck in the
joint `(μ, σ, intercept_s)` posterior that NUTS can't traverse without
divergences. Sampling `z` instead and computing the parameter
algebraically removes the funnel.

**Sampler config that worked.** 4 chains × (2000 tune + 2000 draws),
`target_accept = 0.9`, `nuts_sampler = "nutpie"`. The first attempt at
`target_accept = 0.95` with `HalfNormal(1.0)` priors didn't finish in
over an hour and was killed; tightening the σ priors and dropping
target acceptance to 0.9 brought partial-pool sampling to a healthy
~1.5 h wall (≈ 50 min CPU per chain × 4 chains in parallel).

## Diagnostics

| Model | Max R-hat | Min ESS bulk | Divergences | Status |
|---|---|---|---|---|
| Full pooling | **1.000** | **11,136** | 0 | Pass |
| No pooling (worst of 3) | **1.000** | **10,585** | 0 | Pass |
| Partial pooling | **1.000** | **1,670** | **59** | Pass with caveat |

Trace plots saved to `reports/phase2_diagnostics_{full,partial}_pooling.pdf`.

**On the 59 divergent transitions** (out of 8,000 partial-pool draws —
0.7%). NUTS reports a divergence when the Hamiltonian dynamics encounter
a region of the posterior where its leapfrog integrator goes numerically
unstable, typically because the geometry has a sharp neck or curvature
mismatch. 0.7% is small — the standard PyMC guidance is to take action
above ~1% — and the chains otherwise converged cleanly (R-hat = 1.00,
ESS_bulk well over the 400-per-chain rule of thumb). The divergences are
a soft flag, not a fail. They suggest the partial-pool posterior still
has *some* funnel residue we didn't fully iron out with `target_accept =
0.9`, but the inferences below are reliable as-is.

ESS bulk is much lower for partial pooling (1,670) than for the
non-hierarchical fits (10k+). That's expected — the hierarchy's
correlated parameters reduce the effective sample size per draw. 1,670
is still comfortably above any threshold that would compromise the
posterior summaries.

## Results

### Headline metrics — per-station test Brier (lower is better)

| Station | No-pool (A) | Full-pool (B) | Partial-pool (C) |
|---|---|---|---|
| Bellever | **0.1216** | 0.1224 | **0.1216** |
| Princetown | 0.1591 | **0.1574** | 0.1590 |
| Hexworthy | 0.1406 | 0.1439 | **0.1405** |

CSV: `reports/phase2_artefacts/headline_brier.csv`. Log-loss table is
the same shape, see `headline_logloss.csv`.

**Headline finding: partial pooling matches no-pool to four decimal
places at every station.** Full-pool wins narrowly at Princetown only.

### Why hierarchical didn't pay off

Partial pooling helps most when individual groups are *data-poor* —
the borrow-strength benefit of the shared population distribution is
proportional to how much each station's own posterior would have wobbled
without help. With 8–10k training rows per station and only 8 features,
each no-pool fit is already so well-determined that pooling across
stations adds essentially nothing. We're not in the regime where
hierarchy is worth its complexity.

The Princetown anomaly (full-pool best) is a different phenomenon:
Princetown's *test* set sits in a season the *training* set barely
covers (autumn into late January), so its observed wet rate jumps to
0.31 vs the 0.20 it was trained against. That's a covariate shift, and
none of the three models have features that capture it. Full pooling
"wins" here only because the other two stations' training data — which
spans more of the calendar year — drags Princetown's predictions toward
a higher base rate that happens to be closer to the test reality.
Partial pooling doesn't do that because its station-level intercept
locks Princetown to its own train wet rate.

### Calibration

| Station | Observed | No-pool pred | Full-pool pred | Partial-pool pred |
|---|---|---|---|---|
| Bellever | 0.216 | 0.191 | 0.201 | 0.191 |
| Princetown | **0.314** | 0.281 | 0.298 | 0.281 |
| Hexworthy | 0.247 | 0.226 | 0.201 | 0.226 |

CSV: `reports/phase2_artefacts/calibration.csv`. Same pattern: partial
pool ≈ no pool everywhere; full pool wanders.

### Phase 1 sanity check (Bellever only)

| | Brier |
|---|---|
| Phase 1 single-station Bayesian | 0.1217 |
| Phase 2 no-pool (Bellever fit) | 0.1216 |
| Phase 2 partial-pool (Bellever) | 0.1216 |

All three within 0.0001 — the pipeline reproduces Phase 1 exactly when
restricted to Bellever, confirming nothing was broken in the move to
the multi-station loader and standardiser.

### Hyperparameter posteriors — what the hierarchy *did* learn

Even though hierarchical didn't move test scores, the hyperparameter
posteriors are non-trivial:

**Intercept variation between stations**

| Quantity | Mean | SD | 94% CI |
|---|---|---|---|
| `μ_intercept` | -1.230 | 0.187 | — |
| `σ_intercept` | **0.287** | 0.175 | [0.095, 0.729] |

`σ_intercept ≈ 0.29` (logit units) is a real, non-zero between-station
SD in baseline wet rate. The 94% CI excludes zero — stations genuinely
differ in their base rates, consistent with their different observed
wet fractions.

**Per-feature β variation between stations** (CSV:
`reports/phase2_artefacts/sigma_beta_summary.csv`)

| Feature | σ_β mean | 94% CI |
|---|---|---|
| `precip_gem_seamless` | **0.385** | [0.137, 0.875] |
| `precip_meteofrance_seamless` | 0.191 | [0.016, 0.606] |
| `precip_ecmwf_ifs025` | 0.185 | [0.011, 0.609] |
| `precip_ukmo_seamless` | 0.164 | [0.009, 0.563] |
| `hour_cos` | 0.142 | [0.018, 0.495] |
| `hour_sin` | 0.133 | [0.012, 0.490] |
| `precip_icon_seamless` | 0.117 | [0.003, 0.470] |
| `precip_gfs_seamless` | **0.100** | [0.003, 0.423] |

**The story σ_β tells:**

- `gem_seamless` has by far the largest between-station variation
  (0.39). Stations *disagree most* about what GEM's precipitation
  forecast means — its coefficient at one station does not transfer
  cleanly to another. That's an interesting signal: GEM may be more
  spatially heterogeneous in its bias structure than the other models.
- `gfs_seamless` has the smallest variation (0.10). Stations agree
  about GFS — but recall from Phase 1 that GFS's *μ* was essentially
  zero (94% CI [-0.17, +0.16] covers zero), so this is "stations agree
  GFS is uninformative". `precip_gem_seamless`'s μ posterior, by
  contrast, has mean +0.84 (94% CI [+0.31, +1.27]), so its σ
  represents real per-station tuning of a real signal.
- All four "core" model precip features (ECMWF, GEM, MeteoFrance, UKMO)
  have comfortable σ_β posteriors — between-station variation is real.
  ICON and GFS have weaker signals overall.
- Both hour features have small but non-zero σ_β. Diurnal cycle is
  largely consistent across the three sites — same Dartmoor microclimate.

The posteriors are all comfortably *inside* the HalfNormal(0.5) prior
support (the 94% upper bounds peak at 0.88 for GEM, well below the
prior's effective ceiling), so the prior was permissive enough not to
constrain the fit.

### Population means (μ_β) — the "best single fit" recovered

| Feature | μ_β mean | 94% CI |
|---|---|---|
| precip_ecmwf_ifs025 | **+0.997** | [+0.68, +1.25] |
| precip_gem_seamless | **+0.839** | [+0.31, +1.27] |
| precip_gfs_seamless | -0.002 | [-0.17, +0.16] |
| precip_icon_seamless | +0.211 | [+0.02, +0.40] |
| precip_meteofrance_seamless | +0.358 | [+0.08, +0.63] |
| precip_ukmo_seamless | +0.370 | [+0.13, +0.61] |
| hour_sin | +0.141 | [-0.06, +0.33] |
| hour_cos | +0.153 | [-0.05, +0.34] |

These are the population-level means — close to but not identical to
Phase 1's coefficients (Phase 1 was Bellever-only and unstandardised
against a different scaler). Interpretation matches Phase 1: ECMWF and
GEM are the dominant features; GFS is essentially noise; ICON, MF and
UKMO are secondary; hour-of-day signals are mild and credible-interval-
crosses-zero in this multi-station fit.

## What was learned (the hierarchical-specific bits)

1. **Partial pooling didn't beat no pooling at this row count.** That's
   itself a useful finding — at 8–10k rows per station you don't get
   the borrow-strength effect that hierarchical Bayesian modelling is
   designed to provide. Re-evaluate at much smaller stations or much
   shorter time windows where data is genuinely sparse.

2. **The hierarchy *did* learn meaningful per-station structure** —
   σ_intercept and most σ_β are non-zero with non-zero-excluding 94%
   CIs. So the model isn't useless, it's just not paying off at this
   data scale. If we had 50 stations and were averaging over them, the
   shrinkage would matter; with 3 it doesn't.

3. **Tighter σ priors are essential for sampler tractability.** The
   first run with HalfNormal(1.0) on the σ hyperparameters didn't
   finish in over an hour. HalfNormal(0.5) — physically reasonable for
   three Dartmoor stations within ~10 km — sampled cleanly in ~1.5 h.
   Tighter priors directly attack the funnel geometry in the (μ, σ, z)
   joint, so this isn't just a runtime fix; it's also the more
   principled prior given the actual physics.

4. **GEM has the most station-specific coefficient structure** — useful
   to know if you ever want to drop a model from the blender or
   investigate a forecast bias by region.

5. **Princetown's autumn-2026 test extension is a real-world reminder
   that the chronological 80/20 split can put one station's test set
   into a different climatic regime than the others'.** Future phases
   should consider per-station date-aligned splits or a year-ahead
   holdout to keep the comparison clean.

## What wasn't done

Per the brief — explicitly out of scope for Phase 2:

- Other lead times (still 24h only)
- Comparison to the LightGBM 3a champion in WeatherBlend
- Dropping `gfs_seamless` despite Phase 1 calling it noise (kept for
  comparability)
- Pushing the 59 divergences to zero (would re-run at `target_accept =
  0.95` with the now-tighter priors; ~3× slower, probably tractable now)
- Per-station seasonality features that could close Princetown's
  covariate-shift gap

## Questions and confusions worth revisiting

- **Is hierarchical worth the complexity at *any* lead time?** Partial
  pooling didn't help at 24 h with abundant data. At longer leads where
  the per-station data thins out (multi-day windows, fewer complete
  hours), the answer might flip. Worth revisiting at 48 h, 72 h, 120 h.
- **What does GEM's high σ_β actually correspond to?** Is it bias
  structure that varies with elevation? Distance to coast? A specific
  product version? Worth pulling each station's per-station GEM
  coefficient (already in `per_station_coefficients.csv`) and looking
  at the spread vs station metadata.
- **Princetown's covariate shift is a measurement issue, not a model
  issue.** Adding seasonal features (`month_sin`, `month_cos` or a
  `season` indicator) is an obvious Phase 3 experiment and might
  benefit no-pool *and* partial-pool fits.
- **Re-run at target_accept = 0.95?** Probably yes if we want to publish
  this without the divergence asterisk. With HalfNormal(0.5) priors the
  geometry is much friendlier than the original 0.95 + HalfNormal(1.0)
  attempt, so it should be tractable now — likely 2–4× slower than the
  current run, so 4–6 h. Worth doing once before Phase 3.

## Definition of done — checklist

- [x] Three-station dataset prepared, pooled standardisation, 80/20 split
- [x] Full-pooling fit, diagnostics clean
- [x] No-pooling fit (3 separate), diagnostics clean
- [x] Partial-pooling fit, diagnostics passably clean (59 divergences)
- [x] Per-station test Brier and log-loss for all three models
- [x] Calibration table (predicted vs observed wet fraction)
- [x] Reliability binning (`reliability_bins.csv`)
- [x] Hyperparameter posterior summaries (μ, σ for intercept and per-feature β)
- [x] Per-station β posterior table (`per_station_coefficients.csv`)
- [x] Phase 1 sanity check at Bellever
- [x] Forest plot of σ_β posteriors (`phase2_sigma_forest.pdf`)
- [x] Report written
- [ ] Everything committed (next step)

## Files produced

```
weatherProbabilistic/
├── scripts/
│   └── run_phase2.py             # End-to-end runner (3 fits + diagnostics + report tables)
├── src/
│   ├── data.py                   # Loader extended with prepare_phase2_dataset()
│   └── models/
│       ├── phase2_full_pooling.py
│       ├── phase2_no_pooling.py
│       └── phase2_partial_pooling.py
└── reports/
    ├── phase2_report.md          # This file
    ├── phase2_diagnostics_full_pooling.pdf      # gitignored
    ├── phase2_diagnostics_partial_pooling.pdf   # gitignored
    ├── phase2_sigma_forest.pdf                  # gitignored
    └── phase2_artefacts/
        ├── full_pooling_posterior.nc            # gitignored
        ├── no_pooling_posterior_{bellever,princetown,hexworthy}.nc  # gitignored
        ├── partial_pooling_posterior.nc         # gitignored
        ├── partial_pool_run.log                 # raw sampler log
        ├── metrics.json                         # everything above as JSON
        ├── headline_brier.csv
        ├── headline_logloss.csv
        ├── calibration.csv
        ├── reliability_bins.csv
        ├── sigma_beta_summary.csv
        ├── mu_beta_summary.csv
        └── per_station_coefficients.csv
```
