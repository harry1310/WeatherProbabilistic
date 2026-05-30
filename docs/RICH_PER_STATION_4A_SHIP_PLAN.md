# Productionise rich-per-station 4a — ship plan

## What we're shipping

The 2026-05-29 bake-off found that **per-station BART with rich + v1 oro
features (68 features) beats production 4a (lean, 25 features) by ~5%
aggregate Brier across 3 stations × 3 leads**. Every (station × lead) cell
improves or ties; Bellever's long-lead regression that plagued every pooled
variant is largely resolved. Plan-bar (≥2% at 24+48, ≥1% at 72+) cleared at
every lead aggregate. 4b blend re-evaluation confirmed
`mean(new_4a, 3o)` still wins (−1.38% vs current 4b), so the blend formula
stays arithmetic mean — only the 4a input changes.

This document captures what needs to change to move that lab result into the
Sunday auto-retrain pipeline.

## What we have today vs what we need

| | Production 4a today | Rich-per-station target |
|---|---|---|
| Trainer | `train_4a.py`, lean features (22 NWP + 3 syn ≈ 25 effective) | Rich + v1 oro (68 features) |
| Feature source | Pulled in-process via `_shared.build_features_via_duckdb` + `add_synoptic_features` (Python DuckDB SQL) | C# `PrecipRichOroFeatureBuilder.BuildForLead` already produces this for 3o; bake-off ran it via the C# `dump-oro-features` command and read parquets in Python |
| Architecture | per-cell BART, one fit per (station, lead), 5 leads (24/48/72/96/120) | Same per-cell BART, same 5 leads (96/120 yet to be bake-off-validated, see risks) |
| Predictor | `predict_4a.py`, builds lean live features via DuckDB | Needs equivalent rich + dynamic-terrain live pivot |
| Bundle layout | `state_lead_Nh.rds` + `arrays_lead_Nh.npz` + `preprocess.json` + `test_predictions.parquet` + metadata | Identical — just larger preprocess.json (more feature names) and larger arrays.npz |
| Manifest | Auto-promoted per station post-RetrainGuard pass | Same |
| Workflow | `retrain-python.yml` (Sunday) → `predict-4a.yml` (4×/day) | Same workflows, just updated scripts |

## Architectural choice

Three viable paths. **Recommendation: Path A.**

### Path A — port the rich+oro feature builder to Python ← **recommended**

Add Python equivalents of `PrecipRichFeatureBuilder.cs` + the v1 terrain
compose in `_shared.py`. Both `train_4a.py` and `predict_4a.py` call them
directly — same in-process Python function, train and predict use identical
code paths.

- **Pros:** self-contained Python; predict_4a.py stays single-runtime; no cross-language coupling at train OR predict time; train_4a.py is no longer downstream of a C# artefact snapshot, so it can't drift relative to a stale dump.
- **Cons:** ~280 LOC of Python that mirrors well-tested C#; two implementations of the rich-oro spec exist (3o in C#, 4a in Python); drift risk over time if someone changes one without the other.
- **Mitigations for drift:**
  - **Bit-equivalence smoke test** in `tests/` — compare Python builder output against a fresh C# dump-oro-features run on overlapping `(station, valid_time)` rows, assert all 68 columns match within float epsilon. Runs on PR.
  - Document the canonical rich-oro spec in `docs/RICH_ORO_FEATURE_SPEC.md` with both C# and Python file pointers.

### Path B — train from C# dumps, port only the live-feature pivot to Python

`train_4a.py` reads pre-dumped parquets produced by the existing C#
`dump-oro-features` command. `predict_4a.py` builds live features in Python.

- **Pros:** C# builder stays the single source of truth at train time; smaller Python footprint (~150 LOC instead of ~280).
- **Cons:** trains from a snapshot — fragile if anything between the dump and train_4a.py mutates state, opaque debugging (training fails because of dump issues, hard to trace). Predict-time still needs Python rich-oro builder anyway, so we end up doing the same Python work *plus* the dump-coupling. Worst of both worlds. **Reject.**

### Path C — move 4a training to C# (replace BART with LightGBM)

Use Microsoft.ML LightGBM, drop BART entirely.

- **Pros:** all-C#, no rpy2 / R install pain in CI.
- **Cons:** today's overnight test 2 found that **per-station LightGBM is *worse* than pooled LightGBM** for 3o. BART is what makes per-station win for 4a. Switching to LightGBM would throw away the ~5% gain we're shipping. **Reject.**

## Concrete change list (Path A)

### 1. `scripts/_shared.py` — new Python rich+oro feature builder

Add two functions, callable from both train_4a.py and predict_4a.py so the
spec lives in one place:

- **`build_rich_features_via_duckdb(station_friendly, lead, *, min_valid_time, run_time_source)`** — extends the existing `build_features_via_duckdb` pattern. Same DuckDB SQL skeleton as the lean version, with extra per-NWP aggregations:
  - Per-NWP: precip, temp, dewp, RH, cloud (low/mid/high), wind speed, wind direction (sin/cos), surface pressure, CAPE
  - Cross-NWP aggregates: mean, std, max, agreement_wet, dew_depression_mean, cloud_*_mean
  - Time encodings: hour_sin/cos, doy_sin/cos (already in `build_live_features`)
  - 7 NWPs × ~5 aggregated fields + 9 cross-model aggregates + 4 time encodings ≈ 48 columns; combined with precip per-NWP and cross-aggregates ≈ 59 features (matching the C# rich spec exactly).
  - `run_time_source` parameter: `'offset_day'` at train time, `'reported'` at predict time. Same SQL otherwise.

- **`compose_v1_terrain_block(station_slug, row_df)`** — mirrors `PrecipRichOroFeatureBuilder.ComposeTerrainBlock` in C#:
  - Reads `WeatherBlend/data/static/orographic/{slug}.json`
  - Takes a DataFrame with `oro_wind_sin`, `oro_wind_cos` columns (computed in the rich pivot above as NWP-mean wind components)
  - Per row: pick sector from wind direction, look up `upwind_gain_5km`, compute uplift from terrain gradient × wind vector, compute uplift × q (where q derived from dewpoint + surface pressure via Magnus equation)
  - Returns 9 columns: 3 static (elevation_vs_cell, relief_5km, ruggedness_5km), 5 dynamic (wind_sin, wind_cos, upwind_gain_per_wind_5km, uplift_m_per_s, uplift_x_q_g_per_kg), 1 station_id
  - Already wrote ~half this logic for last night's `compose_dynamic_lee` and `compose_v3_climatology` helpers in `run_phase6_dbarts_pooled_oro.py` — extract and reuse the wind-sector picker + JSON loader patterns.

Estimated: ~280 LOC new in `_shared.py`.

### 2. `scripts/train_4a.py`
- Replace the `build_features_via_duckdb` + `add_synoptic_features` calls with `build_rich_features_via_duckdb` + `compose_v1_terrain_block` (using the new builder).
- Add lead 96/120 support to the bake-off in Phase 4 before shipping (see below).
- BART hyperparams and per-cell loop stay identical.
- `training_metadata.json` `DataSource` stamp updates to mention "rich-oro + dbarts BART".
- `feature_schema.json` reflects the new 68-feature spec.
- `preprocess.json` per-lead block gets ~3× larger (68 features × scaler params).

Estimated: ~40 LOC changed in train_4a.py itself (mostly removing old imports + swapping function calls).

### 3. `scripts/predict_4a.py`
- `build_live_features` currently has its own ~80-LOC inline DuckDB SQL for lean live features. Replace with a call to `build_rich_features_via_duckdb(run_time_source='reported')` followed by `compose_v1_terrain_block`.
- Same function used at train + predict time = no source-of-truth divergence.

Estimated: ~60 LOC removed (the inline SQL), ~10 LOC of new function call wiring.

### 4. `tests/test_rich_oro_python_vs_csharp.py` — bit-equivalence smoke

- Run the C# `dump-oro-features --feature-sets rich-oro --leads 24 --stations Bellever Dartmoor` to produce a reference parquet (small, fast).
- Run the Python `build_rich_features_via_duckdb` + `compose_v1_terrain_block` on the same `(station, lead, min_valid_time)`.
- Inner-join on `ValidTimeUtc`, assert all 68 columns match within `1e-6` (or whatever's right for float aggregations across the two duckdb runtimes).
- If diverged: fail loudly, the message tells you which columns and how much.
- Runs on PR via the existing WP CI.

Estimated: ~100 LOC.

### 4. `phases.yaml`
- 4a entry stays the same except for an optional `featureSet: "rich-oro"` tag (currently absent; phases.yaml only stamps featureSet for tier-divergent phases). The bundle's `feature_schema.json` is the authoritative feature list — phases.yaml stamping is informational.
- `minValidTime: "2024-01-01"` stays unchanged.
- `locations: ["bonehill_rocks"]` stays (Membury 4a not in scope).

Estimated: ~3 lines.

### 5. RetrainGuard handling
- The guard compares the new run's `training_summary.json` against the previous run's. Switching from 25 to 68 features will fail the guard's `features-effective` check (default `0` — any change aborts).
- Two options:
  - **One-time override:** add `--retrain-allow-feature-set-change` flag or env var. Run the first new bundle with the override, subsequent runs see 68 → 68 and pass normally.
  - **Reset baseline:** delete the previous `training_summary.json` so the first new run has nothing to compare against (the guard treats first-ever as a free pass).
- Recommendation: reset baseline by deleting the prior `training_summary.json` from the most recent production bundle dirs (3 files). First new run accepts whatever it sees, then the steady-state guard tightens around the new feature count.

### 6. Bundle versioning
- Bundle dir name pattern `v{timestamp}_phase4a` stays — no need to introduce a new phase tag (`4a_rich` etc.). The feature-set lives inside the bundle's metadata, not the dir name.
- `find_latest_bundle` in `predict_4a.py` will pick up the new bundle as expected. Old lean bundles become harmless orphans.
- The 4b mint command in WB doesn't care which 4a feature-set generated the predictions — it just blends the per-row P(wet) values.

### 7. Princetown
- Bake-off had Princetown numbers (BSS +0.45 at 24h with rich-per-station, in line with the other stations) but no production baseline. Once we ship, Princetown's first 4a bundle will be rich+oro. That's fine — first-time stations don't have a "did it get worse" question.
- Memory `project_train_4a_no_manifest_promote` noted train_4a was historically buggy on first-ever promotion; that's since been fixed.

## Phased rollout

### Phase 1: Python rich+oro builder + bit-equivalence test (1.5 days)
- Add `build_rich_features_via_duckdb` and `compose_v1_terrain_block` to `_shared.py`.
- Write `tests/test_rich_oro_python_vs_csharp.py` — run C# dump for one cell, run Python builder for the same cell, diff with tolerance.
- Iterate until bit-equivalent (or close-enough — flag any columns that consistently differ and decide per-column whether the difference is principled).
- **Gate:** test passes before moving to Phase 2.

### Phase 2: train_4a.py swap + local validation (1 day)
- Swap `train_4a.py` from lean to the new rich+oro builder.
- Local run: `train_4a.py --stations Bellever Bovey Hexworthy Princetown`. Confirm:
  - All 5 leads train successfully
  - Bundle artefacts saved correctly
  - Test predictions parquet Brier matches the bake-off numbers within fit-noise
- This validates the train path before we touch CI.

### Phase 3: predict_4a.py swap (half day)
- Replace the inline lean SQL with the shared `build_rich_features_via_duckdb` + `compose_v1_terrain_block` calls.
- Smoke test: run `predict_4a.py --anchor 2026-05-30` against the local rich+oro bundle. Confirm:
  - Predictions parquet has same shape as before
  - Per-row P(wet) values are sensible (no NaN-heavy rows, no -inf/+inf)
  - The first 4b mint cycle in WB reads them without error
- The fact that train and predict share the SAME `_shared.py` function means the spec match is automatic.

### Phase 3.5: CI integration (half day)
- No C# dump step needed — train_4a.py is now self-contained in Python.
- Add `--retrain-allow-feature-set-change` flag to train_4a.py for the first promotion cycle.
- Confirm `tests/test_rich_oro_python_vs_csharp.py` runs on PR.

### Phase 4: Bake-off the long leads (half day, parallel to Phase 3)
- Run the bake-off for leads 96 + 120 with rich-per-station (extend `--leads` in the runner). 4 stations × 2 leads × ~3 min ≈ 24 min.
- Confirm the per-station-beats-pooled pattern holds at long lead. If 96/120 show a regression, we ship 24-72 only and keep lean for 96/120 (per-lead champion à la 2d).

### Phase 5: Deploy + verify (2 days, Sunday + Monday cycle)
- Reset the previous bundles' `training_summary.json` to clear RetrainGuard.
- Trigger Sunday auto-retrain manually (`gh workflow run retrain-python.yml -f force=true`).
- First cycle should produce 4 stations × 5 leads = 20 cell fits.
- Monday verify cycle should show 4a Brier improvement at the 3 prior stations.
- Princetown 4a starts producing predictions (first-time enable on the site).

### Phase 6: Retire (post-validation, 2-4 weeks)
- After 2 weeks of stable production with no verify-drift issues:
  - Update memory: `project_phase4a_shipped` superseded by a new entry.
  - Delete old lean-feature bundle dirs (they're harmless but disk-noise).
  - Consider promoting 4a from challenger to champion if its aligned Brier sustains beating 3a.

## Risks before ship

1. **96/120 untested.** Bake-off only covered 24/48/72. If long-lead per-station BART regresses (could happen — long-lead skill is dominated by upper-air structure which the existing rich+oro spec doesn't capture), we'd ship a regression at the leads where 4b feeds the long-range site widgets. **Mitigation: Phase 4 bake-off above. Half-day to derisk.**
2. **Python vs C# rich-oro spec drift over time.** Two implementations (Python in 4a, C# in 3o) means a future spec change to one may not propagate to the other. **Mitigation: bit-equivalence smoke test in Phase 1, runs on PR; docs/RICH_ORO_FEATURE_SPEC.md as the canonical reference, with both files pointing back to it; convention that 3o/4a feature-set changes touch both implementations + the doc in the same PR. Realistic exposure: low — the rich-oro spec hasn't moved in ~6 months.**
3. **RetrainGuard first-cycle pass.** Without resetting the baseline, the first new bundle fails the guard and never promotes. **Mitigation: documented above; reset the 3 prior `training_summary.json` files before triggering.**
4. **Multi-level NWP backfill (yesterday's work) creates an opportunity but no requirement.** The pressure-level columns are now in R2 but aren't consumed by any builder. The rich-per-station spec doesn't use them. They sit ready for a future v4 feature-builder experiment — *don't* try to add them in this shipping cycle, that's scope creep.

## Estimate roll-up

| phase | wall time |
|---|---|
| 1. Python rich+oro builder + bit-equivalence test | ~1.5 days |
| 2. train_4a.py swap + local validation | ~1 day |
| 3. predict_4a.py swap | ~half day |
| 3.5. CI integration | ~half day |
| 4. 96/120 bake-off (parallel to Phase 3/3.5) | ~half day |
| 5. Deploy + verify (Sun+Mon) | passive |

Total developer time: **3-4 days**. Total elapsed including Sunday verify cycle: **~1 week from start to confidently-deployed**.

## Open questions

- **Membury 4a?** Currently not trained at all (config has it but no bundles). Out of scope for this shipping cycle — Membury's per-station 4a is a separate decision (would need its own bake-off; the rich+oro design assumes orographic context which is much weaker for lowland Membury).
- **Phase tag rename?** Could call the new flavour `4a_rich` or `4a_oro` to differentiate from the lean predecessor. Recommend: don't. The bundle's metadata is enough; renaming would require also touching 4b mint, phases.yaml, site rendering, etc. Keep phase tag `4a`.
- **Promote from challenger to champion?** 3a is currently champion; 4a is challenger. With ~5% improvement vs lean 4a (and 4a was already beating 3a in 9/9 cells per the original 4a bake-off), the new 4a should beat 3a by an even larger margin. Worth re-running the 4a-vs-3a aligned comparison once the new bundles land, and considering a champion promotion. Not blocking for the first ship.
