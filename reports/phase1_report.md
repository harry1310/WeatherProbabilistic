# Phase 1 Report — Bayesian Logistic Regression for P(wet) at Bellever, lead 24h

**Date:** 2026-04-24
**Author:** Claude Code (initial pass) for Harry

## What we built

A Bayesian logistic regression in PyMC predicting whether the next-day
forecast for Bellever Dartmoor will see a "wet" hour (≥ 0.1 mm rainfall),
using six numerical-weather-prediction (NWP) model precipitation forecasts
plus a cyclical hour-of-day encoding. We fit the model with NUTS, ran the
standard MCMC diagnostics, drew posterior predictive samples on a held-out
test set, and compared everything to a vanilla scikit-learn logistic
regression on the same features.

The goal was to learn the Bayesian workflow end-to-end on the simplest
meaningful problem. Beating LightGBM was explicitly out of scope — and
indeed we don't try to.

## Data

We reuse the WeatherBlend parquet tree at
`C:/Projects/Weather/WeatherBlend/data/`. Loader at `src/data.py`.

| | |
|---|---|
| Forecast location | `bonehill_rocks` (the Open-Meteo grid point used by WeatherBlend) |
| Truth station | `Bellever Dartmoor` (Environment Agency 15-minute rainfall) |
| Wet threshold | ≥ 0.1 mm in the hour |
| Lead time | 24 h only |
| Models used | `ecmwf_ifs025`, `gem_seamless`, `gfs_seamless`, `icon_seamless`, `meteofrance_seamless`, `ukmo_seamless` |
| Models excluded | `ecmwf_hres_wb2` (7 files), `gfs_ncep` (4 files), `met_office_spot` (1 file) — too sparse for Phase 1 |

After joining all six models' lead-24h forecasts to the EA hourly truth
(only "Good" quality 15-min readings, all four 15-min slots in the hour
present), we end up with:

| | |
|---|---|
| Total rows | 10,117 |
| Train (earliest 80%) | 8,093 rows, 2024-08-06 → 2025-07-17 |
| Test (latest 20%) | 2,024 rows, 2025-07-17 → 2025-10-09 |
| Wet fraction (train) | 0.202 |
| Wet fraction (test) | 0.216 |

**Feature set (8 features after null-handling)**

- 6 × per-model lead-24h precipitation (mm/h)
- `hour_sin`, `hour_cos` (cyclical hour-of-day encoding)

The brief allowed up to 6 additional `precipitation_probability` features.
**All six were 100% null** in the WeatherBlend parquet tree — this is a
real WeatherBlend ingestion gap (none of the per-model probability fields
are populated) rather than a join issue. They were therefore all dropped
per the >50%-null rule. The Phase 1 SHAP findings memory (`prob_*` features
"100% dead") is consistent with this: they're literally absent.

Continuous features were standardised on the train split (mean 0, std 1)
and the same scaler applied to test. The cyclical hour features are
already on a [-1, 1] scale but were standardised too for consistency;
this has no effect on a logistic-regression fit.

## Model specification

The model lives at `src/models/phase1.py`.

**Likelihood**

```
y_i  ~  Bernoulli(p_i)
logit(p_i)  =  intercept  +  Σ_k  β_k · x_{i,k}
```

**Priors**

| Parameter | Prior | Reasoning |
|---|---|---|
| `intercept` | Normal(0, 2) | Centred on `logit(0.5) = 0`. SD = 2 spans roughly logit(0.02) to logit(0.98), so it allows the data to push the base rate anywhere reasonable but lightly discourages extremes. |
| `β_k` per feature | Normal(0, 1) | With *standardised* features, β = 1 means "a one-SD bump in this feature shifts logit(p) by 1" — a substantial effect. SD = 1 is *weakly informative*: the prior says we don't know the sign, doubt the magnitude is huge, and lets the data dominate. |

**What "weakly informative" actually does here**

Bayes' theorem says
`posterior ∝ prior × likelihood`. With 8,093 training points the
likelihood term is overwhelmingly large compared to a Normal(0, 1) prior,
so the posterior is essentially driven by the data — the prior just gates
out absurd values. We expect (and confirm below) that the Bayesian
posterior means therefore line up almost exactly with the maximum-
likelihood (frequentist) estimates.

If we'd used Normal(0, 0.01) priors instead, that would be *strongly
informative*: a prior strong enough to drag the posterior toward zero
even against the data. With Normal(0, 1000) the prior would be effectively
flat — also fine here, but with worse sampler geometry.

**Sampler**

NUTS (the No-U-Turn Sampler) is PyMC's default MCMC algorithm. It uses
gradient information to take long, well-aimed steps through the posterior
rather than shuffling around blindly. We ran 2 chains × (2000 tune + 2000
draws) = 4,000 posterior samples per parameter.

Two chains let us *diagnose convergence*: if both chains independently
explore the same region of parameter space, that's evidence the sampler
has actually located the posterior rather than getting stuck in a corner.
With one chain you can't tell.

**Implementation note: `nutpie` instead of PyTensor's C backend**

PyMC's default NUTS sampler compiles its log-density to C via PyTensor.
This Windows machine doesn't have a C++ compiler installed, so PyTensor
would fall back to pure Python — orders of magnitude slower, infeasible
for a real run. We installed **`nutpie`** instead, which is a Rust
re-implementation of the same NUTS algorithm. It ships as a pre-built
wheel, requires no compiler, and is invoked simply with
`pm.sample(nuts_sampler="nutpie")`. The algorithm is identical — only
the language the gradient is computed in differs.

Sampling 4,000 posterior draws over 8 features × 8,093 rows took roughly
30–60 seconds end to end (the bulk of which was nutpie warming up its
JIT compiler).

## Diagnostics

All four checks passed cleanly. Trace plot saved to
`reports/phase1_diagnostics.pdf`.

| Check | Threshold | Observed | Status |
|---|---|---|---|
| Max R-hat | < 1.01 | **1.00** | Pass |
| Min ESS bulk | > 1000 | **5,090** | Pass |
| Divergent transitions | 0 | **0** | Pass |
| Trace plot mixing | "hairy caterpillar" | Both chains overlap tightly | Pass |

**What each diagnostic told us**

- **R-hat = 1.00** — the within-chain and between-chain variance of the
  posterior samples are indistinguishable for every parameter. The two
  chains have converged to the same distribution.
- **ESS bulk ≥ 5,090** out of 4,000 raw draws (per chain × 2 chains =
  4,000 nominal). Higher than nominal is possible because NUTS proposals
  for this geometry are slightly anti-correlated, increasing effective
  sample size. We have plenty of independent information about each
  parameter.
- **Zero divergences** — the Hamiltonian dynamics underlying NUTS never
  hit numerical trouble. The posterior geometry is benign (no funnels,
  no thin necks), which is what you'd expect from a well-conditioned
  logistic regression on standardised features.

This is the easy case. Future phases (hierarchical models across stations
and lead times) are far more likely to throw R-hat warnings or
divergences, and we'll learn how to debug those when they appear.

## Results

### Headline metrics

| Metric | Frequentist LR | Bayesian LR (posterior mean) |
|---|---|---|
| **Brier score (test)** | 0.1217 | **0.1217** |
| **Log loss (test)** | 0.4110 | **0.4101** |

The Bayesian Brier is identical to four decimal places. Log loss is
0.2% lower, well within sampling noise. **This is exactly what we
expected**: with 8,000+ training points and weakly informative priors,
Bayesian logistic regression should reproduce the maximum-likelihood
estimate to numerical precision. The fact that it does is a strong
positive signal that the implementation is correct.

### Coefficient comparison

All values for *standardised* features.

| Parameter | Frequentist | Bayesian mean | Bayesian SD | 94% CI |
|---|---|---|---|---|
| intercept | -1.292 | -1.291 | 0.037 | [-1.36, -1.22] |
| precip_ecmwf_ifs025 | **+1.102** | **+1.100** | 0.095 | [+0.93, +1.29] |
| precip_gem_seamless | **+1.138** | **+1.134** | 0.094 | [+0.96, +1.31] |
| precip_gfs_seamless | +0.022 | +0.026 | 0.064 | **[-0.10, +0.15]** |
| precip_icon_seamless | +0.139 | +0.143 | 0.079 | [+0.00, +0.29] |
| precip_meteofrance_seamless | +0.497 | +0.502 | 0.075 | [+0.36, +0.65] |
| precip_ukmo_seamless | +0.477 | +0.478 | 0.081 | [+0.33, +0.63] |
| hour_sin | +0.190 | +0.190 | 0.034 | [+0.13, +0.25] |
| hour_cos | +0.236 | +0.236 | 0.035 | [+0.17, +0.30] |

CSV: `reports/phase1_artefacts/coefficient_comparison.csv`.

### Posterior predictive check

| | |
|---|---|
| Observed wet fraction (test) | 0.216 |
| Posterior-predictive wet fraction | 0.191 |

The model very slightly *under*-predicts wet hours overall (≈ 2.5
percentage points). This is not catastrophic — the test period (mid-July
to early October 2025) is a different season from much of the training
data and the absolute miss is small — but it is the kind of mild
miscalibration that hierarchical structure or seasonal features could
help fix in later phases.

## What was learned (the Bayesian-specific bits)

The Bayesian fit returns the *same point estimates* as sklearn, but it
also tells us things sklearn doesn't:

1. **`gfs_seamless` is genuinely uninformative on this dataset.** Its
   posterior mean is 0.026 with SD 0.064; the 94% credible interval
   [-0.10, +0.15] **includes zero**. The frequentist coefficient (0.022)
   doesn't tell us whether that's "really zero" or "small-but-positive
   and the data nailed it down" — the Bayesian posterior makes clear it's
   the former. Removing GFS from the feature set should have essentially
   no effect.

2. **`icon_seamless` is borderline.** 94% CI [+0.00, +0.29] just barely
   excludes zero. Useful, but on the margin.

3. **The two strongest features (ECMWF, GEM) are not just bigger — their
   posteriors are also fairly narrow** (SD ≈ 0.095 on means ≈ 1.1, so
   ~9% relative uncertainty). We're confident about both their sign and
   their magnitude.

4. **The intercept is extremely tightly constrained** (SD 0.037), as we'd
   expect from a calibration parameter learned from 8k binary outcomes.

5. **Posterior predictive samples expose the wet-fraction shortfall**
   above (0.191 predicted vs 0.216 observed). A frequentist fit gives
   only point predictions; you'd have to run it *and then* compute
   calibration separately. The Bayesian PPC bakes that diagnostic into
   the same workflow.

The Bayesian "richer output" is the full distribution rather than a
single number per parameter. That's most valuable when:

- The data is sparse (here it isn't, so the priors barely matter)
- We genuinely care about the uncertainty in a coefficient (e.g. for
  decision-making about whether to keep or drop a feature — see GFS above)
- We want propagation of uncertainty to predictions (the per-row
  posterior over `p` instead of a single `p`)

## What wasn't done

Per the brief — explicitly out of scope for Phase 1:

- Single station only (Bellever); no hierarchy across stations
- Single lead time only (24h)
- No comparison to the LightGBM 3a champion in WeatherBlend
- No hyperparameter tuning of the sampler
- No partial-pooling, no informative priors, no model averaging
- `precipitation_probability` features all dropped because the
  WeatherBlend tree has them 100% null — re-collecting them is a
  separate question

## Questions and confusions worth revisiting

- **Wet-fraction shortfall in PPC.** A 0.191 vs 0.216 gap may reflect
  seasonal mismatch (test set is summer/early-autumn, train set spans
  late summer 2024 through mid-2025). Adding `month_sin/cos` or
  `season` is a natural Phase 2 experiment.
- **GFS is essentially noise here.** Worth testing whether dropping it
  changes anything — the brief asked for a fixed feature set so we
  didn't, but it's a good first ablation.
- **Truth `Quality == "Good"` filter is conservative.** We discard rows
  with anything other than "Good" — and any hour where fewer than four
  15-min slots survive. That's why train starts 2024-08-06 rather than
  2024-02-03 (when ECMWF backfill begins): earlier dates don't have
  enough complete-quality hours to join. Worth profiling the
  completeness curve to see if we're being too strict.
- **`precipitation_probability` is universally null** in the WeatherBlend
  parquets. Whether this is an Open-Meteo API gap or a WeatherBlend
  ingestion gap should be checked — those features were a key part of
  the original Phase 1 plan and their absence isn't this project's bug.
- **NUTS via `nutpie` vs PyTensor C backend.** Algorithmically identical
  but worth knowing: nutpie's gradient compilation runs JIT at first
  use, so the first model fit on a fresh process feels slow even though
  steady-state sampling is fast. Probably worth installing the MSVC
  Build Tools or m2w64 toolchain at some point so we can also use the
  PyMC default backend for comparison.

## Definition of done — checklist

- [x] Environment works, PyMC + nutpie sampling succeeds (`scripts/env_check.py`)
- [x] Data pipeline produces 10,117 rows × 8 features
- [x] Frequentist baseline recorded
- [x] Bayesian model specified, sampled, diagnostics run
- [x] All R-hat < 1.05 (max 1.00), no divergences (0)
- [x] Posterior predictive check performed
- [x] Bayesian Brier within 10% of frequentist (matches to 4dp)
- [x] Report written
- [ ] Everything committed (next step)

## Files produced

```
weatherProbabilistic/
├── requirements.txt
├── scripts/
│   ├── env_check.py          # PyMC stack smoke test
│   └── run_phase1.py         # End-to-end runner
├── src/
│   ├── data.py               # Loader from WeatherBlend parquets
│   ├── baseline.py           # sklearn frequentist baseline
│   ├── diagnostics.py        # R-hat / ESS / divergences / trace plots
│   └── models/
│       └── phase1.py         # Bayesian PyMC model + posterior predictive
└── reports/
    ├── env_check_trace.pdf
    ├── phase1_diagnostics.pdf
    ├── phase1_report.md      # This file
    └── phase1_artefacts/
        ├── posterior.nc      # Saved InferenceData (NetCDF)
        ├── coefficient_comparison.csv
        └── metrics.json
```
