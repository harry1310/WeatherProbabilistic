# Phase 3f orographic-features bake-off (Membury)

Run 2026-05-30 23:17 — previous-runs Open-Meteo (`offset_day`), valid_time ≥ 2024-01-01. Stage-2 NGBoost-LogNormal; stage-1 π = LightGBM P(wet) on lean-15 (identical across arms per cell). Score = mixed-distribution test CRPS (negative Δ = variant beats baseline A).

## Arms

| Arm | Architecture | Features |
|---|---|---|
| A | per-station | baseline lean15 (15 feat) |
| B | per-station | lean15 + dyn-oro (20 feat) |
| C | per-station | rich59 (59 feat) |
| D | per-station | rich59 + dyn-oro (64 feat) |
| E | pooled (3 gauges) | pooled lean15 + 9-terrain (24 feat) |
| F | pooled (3 gauges) | pooled rich59 + 9-terrain (68 feat) |
| G | pooled (3 gauges) | pooled rich59 (no terrain) (59 feat) |

## Per-lead CRPS (mean across 3 gauges)

| Arm | Lead | CRPS | Δ vs A |
|---|---:|---:|---:|
| F | 24 | 0.1109 | -1.74% |
| G | 24 | 0.1113 | -1.35% |
| E | 24 | 0.1115 | -1.21% |
| D | 24 | 0.1115 | -1.21% |
| C | 24 | 0.1118 | -0.92% |
| B | 24 | 0.1121 | -0.65% |
| A | 24 | 0.1129 | +0.00% |
| E | 48 | 0.1232 | -0.75% |
| B | 48 | 0.1233 | -0.70% |
| D | 48 | 0.1239 | -0.19% |
| F | 48 | 0.1240 | -0.13% |
| A | 48 | 0.1242 | +0.00% |
| C | 48 | 0.1242 | +0.04% |
| G | 48 | 0.1245 | +0.26% |
| F | 72 | 0.1326 | -0.89% |
| D | 72 | 0.1326 | -0.86% |
| B | 72 | 0.1329 | -0.62% |
| C | 72 | 0.1331 | -0.52% |
| E | 72 | 0.1331 | -0.48% |
| G | 72 | 0.1332 | -0.46% |
| A | 72 | 0.1338 | +0.00% |

## Overall (n_test-weighted across leads 24/48/72)

| Arm | CRPS | Δ vs A | n_test |
|---|---:|---:|---:|
| F | 0.1225 | -0.89% | 27924 |
| E | 0.1226 | -0.79% | 27924 |
| D | 0.1227 | -0.74% | 27924 |
| B | 0.1228 | -0.66% | 27924 |
| G | 0.1230 | -0.49% | 27924 |
| C | 0.1230 | -0.46% | 27924 |
| A | 0.1236 | +0.00% | 27924 |

## Attribution reads
- **A → C:** rich-feature contribution (per-station).
- **C → D:** dynamic-oro on top of rich (per-station).
- **A → B:** dynamic-oro on lean (intensity-vs-occurrence check).
- **{A,C} → {E,F}:** does pooling + static terrain help?
- **F → G:** isolates pooling-data gain from orography (F≈G ⇒ gain is pooling, not oro).

## Ship-bar
Best arm vs A: ≥2% aggregate CRPS at 24+48, ≥1% at 72. If unmet, record the negative result and stop.