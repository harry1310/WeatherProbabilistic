# 3f orographic-features bake-off — plan

> **OUTCOME (2026-05-30): NEGATIVE — Phase 2 rejected, nothing shipped.**
> All 6 arms (A–G) ran on previous-runs `offset_day` data, valid_time ≥
> 2024-01-01 (n=27,924 test rows). Best arm **F** (pooled rich + 9-terrain)
> reached only **−0.89%** aggregate CRPS vs baseline A (0.1225 vs 0.1236);
> best per-lead was 24h −1.74%, 48h −0.75%, 72h −0.89%. The ship-bar
> (≥2% at 24+48, ≥1% at 72) was **missed on every lead**.
> Attribution: rich barely helps lowland intensity (A→C −0.46%, vs ~5% on
> Bonehill 4a occurrence); orography is directionally real but tiny and
> only in the pooled+rich setting (F beats no-terrain G by 0.41%, as
> hypothesised) — far below shippable. Consistent with the 2026-05-25
> 3c-oro Membury rejection (terrain pool too homogeneous). **3f stays
> per-station lean-15, stage-1 = 3a.** Re-test only if a Bonehill+Membury
> cross-terrain pool ever ships (the F-vs-G margin is the thing to watch).
> Script: `scripts/run_membury_3f_oro_bakeoff.py`. Report:
> `reports/membury_3f_oro_bakeoff_2026-05-30/` (summary.md + per_cell_crps.csv).



Investigate whether orographic features improve **Phase 3f** (Membury
`rainfall_amount` — hourly rain intensity). Bake-off first; no production
change until a variant clears the bar.

## Where 3f is today (ground truth)

- **Model:** NGBoost-LogNormal stage-2, fit per `(station, lead)`, wet-only rows.
  π = P(wet) supplied by the bound 3a champion at predict time. Mixed predictive
  distribution `F(x) = (1−π)·δ₀ + π·LogNormal(μ_log, σ_log)`.
- **Features (LEAN, 15):** 7 NWP `precip_*` + 4 spread (`precip_mean/std/max/
  agreement_wet_01`) + 4 calendar (`hour_sin/cos`, `doy_sin/cos`), built via
  `_shared.build_features_via_duckdb`. (`scripts/train_3f.py:FEATURE_NAMES`.)
- **Scope:** Membury-only champion. 3 EA gauges — Chards Snowdon Hill, Goren,
  Raymonds Hill — × leads {24, 48, 72, 96, 120}.
- **Trainer:** `scripts/train_3f.py`. Predict: `scripts/predict_3f.py`.

## What we already have to reuse (committed)

- `_shared.build_rich_features_via_duckdb` — 8-NWP rich spec (precip, temp, dew,
  RH, dew-depression, pressure per model + cross-NWP aggregates), the same spec
  4a ships.
- `_shared.compose_v1_terrain_block` — Python port of C#
  `PrecipRichOroFeatureBuilder.ComposeTerrainBlock`, returns the 9 terrain
  features. Reads `WeatherBlend/data/static/orographic/{slug}.json`.
- C#-parity guard: `tests/test_rich_oro_python_vs_csharp.py`.
- All 3 Membury gauges have orographic JSONs (`ea_chards_snowdon_hill.json`,
  `ea_goren.json`, `ea_raymonds_hill.json`).
- `compute_test_crps_per_lead` (in `train_3f.py`) — mixed-distribution test CRPS
  via inner-join with 3a's `p_wet`. Reused verbatim for scoring.

The 9 terrain features split into:
- **Static (per-site):** `oro_elevation_vs_cell_m`, `oro_relief_5km_m`,
  `oro_ruggedness_5km_m`, `oro_station_id`.
- **Dynamic (flow-dependent):** `oro_wind_sin`, `oro_wind_cos`,
  `oro_upwind_gain_per_wind_5km_m`, `oro_uplift_m_per_s`, `oro_uplift_x_q_g_per_kg`.

## Hypothesis & the lesson from 4a

The 2026-05-29 4a bake-off found **rich + oro (68 feat) beat lean 4a by ~5%
aggregate Brier**, while **lean + oro was a wash** — oro only pays off layered on
rich features. 4a is *occurrence* at high-relief Bonehill. 3f is *intensity* at
lowland Membury:

- **For:** orographic uplift plausibly modulates rain *amount* more than
  *occurrence* — the regime 4a couldn't test.
- **Against:** Membury is lowland (small terrain gradient / upwind-gain ⇒ the
  dynamic features have little dynamic range), and the dotnet **Membury 3o (oro)
  was rejected** at the 2026-05-25 stage-1 bake-off (+0.36% CRPS vs 3a, "terrain
  pool too homogeneous"). Sober prior: rich likely helps; terrain marginal.

The bake-off is designed to **attribute** any movement to *rich features* vs
*orography* vs *pooling* rather than just report a single number.

## Architecture note: per-station vs pooled

- **Per-station** (3f's current architecture): one fit per `(gauge, lead)`.
  Static terrain features are **constant within a cell ⇒ zero signal**, so they
  are dropped; only the 5 dynamic features can carry within-cell information.
- **Pooled** (3o's architecture): one fit per `lead` over the stacked rows of all
  3 gauges. Static terrain + `oro_station_id` now **vary across gauges**, so the
  full 9-feature block is informative. Pooling also ~3×'s the wet-row count,
  which independently stabilises NGBoost — see the attribution note below.

## Variants

Stage-1 = current bound 3a (unchanged). Score = test-set **CRPS** of the mixed
distribution (existing `compute_test_crps_per_lead` join with 3a `p_wet`),
aggregated across 3 gauges × leads {24, 48, 72} (96/120 secondary).

| Arm | Architecture | Features | Terrain block | Purpose |
|---|---|---|---|---|
| **A — baseline** | per-station | lean 15 | — | control (reproduce production CRPS) |
| **B — lean + dyn-oro** | per-station | lean 15 + 5 dynamic | dynamic only | "oro on lean" for *intensity* (4a said wash for occurrence) |
| **C — rich** | per-station | rich (~59) | — | isolate the rich-feature contribution |
| **D — rich + dyn-oro** | per-station | rich + 5 dynamic | dynamic only | the 4a-winning recipe, per-station |
| **E — pooled lean + oro** | pooled (3 gauges) | lean 15 + 9 terrain | full (static+dynamic) | does pooling activate static terrain on lean? |
| **F — pooled rich + oro** | pooled (3 gauges) | rich + 9 terrain | full (static+dynamic) | 3o-style recipe at Membury, on intensity |

Static block dropped in A–D (constant per cell); full 9-block in E–F (pooling
gives it cross-gauge variance — the whole reason to pool).

### Attribution reads
- **A → C:** rich-feature contribution (per-station).
- **C → D:** dynamic-oro contribution *on top of* rich (per-station).
- **A → B:** dynamic-oro on lean — the intensity-vs-occurrence check against the
  4a wash.
- **{A,C} per-station → {E,F} pooled:** does pooling + static terrain help?
- **E → F:** rich within the pooled+terrain setting.

**Attribution caveat (recommended optional control):** E/F confound two effects —
static terrain *and* 3× training data from pooling. To separate them, add an
optional **arm G — pooled rich, NO terrain**. If F ≈ G, the gain (if any) is the
data-pooling, not the orography. Worth running if E or F clears the bar;
flagged here rather than added to the required set.

## Implementation (no production edits in this phase)

One new bake-off script — `scripts/run_membury_3f_oro_bakeoff.py` — mirroring
`run_membury_3f_stage1_bakeoff.py` / `train_3f.py`'s fit+score, swapping only the
feature assembly per arm. Reuses committed `build_features_via_duckdb`,
`build_rich_features_via_duckdb`, `compose_v1_terrain_block`, `emos_impute`,
`fit_one_lead`-equivalent, and `compute_test_crps_per_lead`.

- **Per-station arms (A–D):** existing per-`(gauge, lead)` loop; for B/D join the
  5 dynamic columns from `compose_v1_terrain_block(slug, …)` (drop static + id).
- **Pooled arms (E–F):** per `lead`, build features for each gauge (with its own
  oro JSON → its static block + `station_id`), `concat` the 3 gauges' train rows,
  fit one NGBoost-LogNormal, then score on **each gauge's own test slice** so the
  aggregate is comparable to the per-station arms.
- **Output:** `reports/membury_3f_oro_bakeoff_<date>/` — per-`(gauge, lead, arm)`
  CRPS CSV + `summary.txt` ("negative delta = variant wins", as in the other
  reports).

## Plan-bar to justify shipping
Best arm vs A: **≥2% aggregate CRPS improvement at 24+48, ≥1% at 72** (same bar as
`RICH_PER_STATION_4A_SHIP_PLAN.md`). If no arm clears it, record the negative
result (mirrors the 3o-Membury rejection) and stop — do not ship.

## Risks / honest priors
1. **Lowland Membury** → weak terrain signal; expect rich to carry most of any
   gain and oro to be marginal. The 3o-Membury rejection is the closest precedent.
2. **Thin wet-only data per cell** (`MIN_WET_TRAIN_ROWS = 100`) + rich (more
   features) ⇒ overfit risk; lean on NGBoost early-stopping, watch `n_train_wet`.
   Pooling (E/F) mitigates this — which is also why a pooled-no-terrain control
   (arm G) matters for clean attribution.
3. **Pressure-level NWP (the `hist_forecast` backfill) is OUT of scope** — not in
   the rich-oro spec; adding it is scope creep (same call the 4a plan made).
4. **Train/serve consistency:** `compose_v1_terrain_block` /
   `build_rich_features_via_duckdb` take `run_time_source` (`offset_day` train /
   `reported` predict) and are the same functions 4a's predict path uses — so a
   shipped variant's `predict_3f.py` swap stays spec-identical. Only relevant in
   Phase 2.
5. **Pooled architecture is a bigger production change** than a feature swap
   (per-`(station,lead)` → per-`lead` pooled bundle + predict plumbing). If a
   pooled arm wins, the Phase-2 ship plan is larger than a per-station feature
   swap — call that out then.

## Phasing
- **Phase 1 (~1 day):** build `run_membury_3f_oro_bakeoff.py`, run A–F on 3 gauges
  × {24, 48, 72}; add arm G if E/F look interesting. Analyse with the attribution
  reads above.
- **Phase 2 (only if bar cleared):** write a 3f ship plan mirroring
  `RICH_PER_STATION_4A_SHIP_PLAN.md` — swap `train_3f.py` + `predict_3f.py` to the
  winning builder (and to pooled bundle layout if a pooled arm won), RetrainGuard
  feature-count reset, verify cycle.
- **Stop** if nothing clears the bar; record the negative result.
