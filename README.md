# weatherProbabilistic

Learning-focused project rebuilding precipitation forecasting in a
Bayesian / probabilistic-programming style with PyMC. Sister project to
[WeatherBlend](../WeatherBlend) (frequentist / gradient-boosted), but
deliberately separate.

## Phases

- **Phase 1** — Bayesian logistic regression for P(precip ≥ 0.1mm) at
  Bellever Dartmoor, lead 24h, ~8 features. End-to-end Bayesian workflow
  on the simplest meaningful problem. See `reports/phase1_report.md`.
- **Phase 2** — Hierarchical partial pooling across 3 stations.
  See `reports/phase2_report.md`.
- **Phase 3** — Per-(station, lead) Model A (independent partial pool
  per lead, 9 cells via 3 fits) + Model B (joint hierarchy, DNF'd).
  Model A is the baseline carried forward. See `reports/phase3_artefacts/`.
- **Phase 4** — Bayesian 5-model vs LightGBM benchmark on identical
  test rows (stripped 7-feature + native 25-feature). LightGBM dominates
  on Brier; Bayesian earns its keep on uncertainty quality (4-5× lower
  Brier on narrow-CI rows than wide-CI rows). See `reports/phase4_report.md`
  + `reports/phase4_audit.md`.
- **Phase 4.5** — Post-hoc isotonic calibration per (station, lead) to
  fix Bayesian's miscalibration vs LightGBM. ECE drops -34.5% (0.079 →
  0.052), now better-calibrated than either LightGBM variant. See
  `reports/phase4_isotonic_report.md`.
## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`nutpie` is used as the NUTS sampler because PyTensor's C backend needs
g++, which isn't installed on the dev box. To use PyTensor's C backend
instead, install MSVC Build Tools or the conda `m2w64-toolchain`.

## Running

```bash
.venv/Scripts/python.exe scripts/env_check.py        # smoke test
.venv/Scripts/python.exe scripts/run_phase1.py       # phase 1
.venv/Scripts/python.exe scripts/run_phase2.py       # phase 2
.venv/Scripts/python.exe scripts/run_phase3.py       # phase 3 (4hr — Bayesian sample)
.venv/Scripts/python.exe scripts/run_phase4_bayesian.py        # phase 4
.venv/Scripts/python.exe scripts/run_phase4_lightgbm.py        # phase 4 LGB stripped
.venv/Scripts/python.exe scripts/run_phase4_lightgbm_native.py # phase 4 LGB native
.venv/Scripts/python.exe scripts/run_phase4_compare.py         # phase 4 comparison
.venv/Scripts/python.exe scripts/run_phase4_isotonic.py        # phase 4.5 calibration
```

Outputs land in `reports/`.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/
```

59 tests covering the isotonic calibration module + the Phase 5
simulation core + aggregations.

## Data

Reads from the existing WeatherBlend parquet tree at
`C:/Projects/Weather/WeatherBlend/data/`. No new ingestion pipeline.
