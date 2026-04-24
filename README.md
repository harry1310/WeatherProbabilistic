# weatherProbabilistic

Learning-focused project rebuilding precipitation forecasting in a
Bayesian / probabilistic-programming style with PyMC. Sister project to
[WeatherBlend](../WeatherBlend) (frequentist / gradient-boosted), but
deliberately separate.

## Phases

- **Phase 1 — Bayesian logistic regression for P(precip ≥ 0.1mm)** at
  Bellever Dartmoor, lead 24h, ~8 features. End-to-end Bayesian workflow
  on the simplest meaningful problem. See `reports/phase1_report.md`.
- Phase 2+ — partial pooling across stations / leads, hierarchical priors,
  comparison to LightGBM 3a (planned, not yet started).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`nutpie` is used as the NUTS sampler because PyTensor's C backend needs
g++, which isn't installed on the dev box. To use PyTensor's C backend
instead, install MSVC Build Tools or the conda `m2w64-toolchain`.

## Running Phase 1

```bash
.venv/Scripts/python.exe scripts/env_check.py     # smoke test
.venv/Scripts/python.exe scripts/run_phase1.py    # full run
```

Outputs land in `reports/`.

## Data

Phase 1 reads from the existing WeatherBlend parquet tree at
`C:/Projects/Weather/WeatherBlend/data/`. No new ingestion pipeline.
