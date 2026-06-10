# Overnight test results — 2026-05-29 → 30

Three experiments, all on Bonehill 4 stations × leads {24, 48, 72}.

---

## Test 1a — per-station BART + dynamic lee_obstruction (rich + v1 oro + 1 dynlee)

**Verdict: no help over baseline rich-per-station.** Adding the wind-direction-picked lee_obstruction feature did not move the needle.

Aligned vs production 4a — mean Δ% across 3 baseline-comparable stations:

| lead | rich-per-station (baseline) | **rich-per-station-dynlee** | diff |
|---|---:|---:|---:|
| 24 | −7.56% | −7.44% | +0.12 (slightly worse) |
| 48 | −4.12% | −3.60% | +0.52 (slightly worse) |
| 72 | −3.38% | −3.23% | +0.15 (slightly worse) |
| overall | **−5.02%** | **−4.75%** | +0.27 (tied within noise) |

The dynamic lee feature is correlated with v1's existing upwind_gain (both proxy "terrain in upwind direction") and BART didn't find independent signal in it.

---

## Test 1b — per-station BART + v3 climatology (rich + v1 oro + 6 dynamic climo features)

**Verdict: slight regression vs baseline.** Adding climatology features made per-station 4a marginally worse.

| lead | rich-per-station | **rich-per-station-v3** | diff |
|---|---:|---:|---:|
| 24 | −7.56% | −6.19% | +1.37 (worse) |
| 48 | −4.12% | −2.26% | +1.86 (worse) |
| 72 | −3.38% | −2.15% | +1.23 (worse) |
| overall | **−5.02%** | **−3.53%** | +1.49 (clear regression) |

Climatological lookups are static-per-(sector, month) — 96 unique values per station × 6 features. BART seems to be using these as noise rather than signal: the current NWP forecast already captures upper-air state implicitly via the rich precip features, so climatology averages add nothing and slightly dilute.

---

## Test 2 — per-station 3o LightGBM vs pooled 3o LightGBM (Python)

**Verdict: pooled marginally beats per-station for LightGBM.** Opposite architectural conclusion to BART.

Per-cell results (Python LightGBM, 4 stations × 3 leads):

| station | lead | pooled | per-station | Δ% |
|---|---:|---:|---:|---:|
| Bellever | 24 | 0.1140 | 0.1182 | **+3.68** (per-stn worse) |
| Bellever | 48 | 0.1336 | 0.1352 | +1.20 |
| Bellever | 72 | 0.1438 | 0.1509 | **+4.94** (per-stn worse) |
| Bovey | 24 | 0.0871 | 0.0897 | +2.99 |
| Bovey | 48 | 0.1040 | 0.1065 | +2.40 |
| Bovey | 72 | 0.1135 | 0.1145 | +0.88 |
| Hexworthy | 24 | 0.1372 | 0.1396 | +1.75 |
| Hexworthy | 48 | 0.1574 | 0.1543 | **−1.97** (per-stn better) |
| Hexworthy | 72 | 0.1667 | 0.1651 | −0.96 |
| Princetown | 24 | 0.1165 | 0.1185 | +1.72 |
| Princetown | 48 | 0.1383 | 0.1355 | **−2.02** (per-stn better) |
| Princetown | 72 | 0.1473 | 0.1473 | 0.00 |

Aggregate per lead (mean across 4 stations):

| lead | mean pooled | mean per-station | Δ% |
|---|---:|---:|---:|
| 24 | 0.1137 | 0.1165 | +2.53 (pooled wins) |
| 48 | 0.1333 | 0.1329 | −0.10 (tied) |
| 72 | 0.1428 | 0.1444 | +1.22 (pooled wins) |
| **overall** | **0.1300** | **0.1313** | **+1.22 (pooled marginally wins)** |

⚠️ _Python lightgbm 4.x with reasonable defaults, NOT ML.NET LightGBM. Absolute numbers won't match production 3o; what matters is the per-station-vs-pooled delta within this script._

---

## What this all means

1. **Today's per-station 4a finding stands and is the real deliverable.** Baseline rich-per-station (-5.02% overall vs production 4a) is the winner. Neither dynlee nor v3 climatology extensions added anything.

2. **The pooled-vs-per-station answer is model-dependent.** 
   - **BART (4a)** strongly prefers per-station (~5% Brier improvement).
   - **LightGBM (3o)** slightly prefers pooled (~1% Brier improvement, both directions at different cells).
   
   Plausible explanation: LightGBM can use the `oro_station_id` feature as a high-utility split, effectively learning per-station partitions inside a pooled fit. BART's gentler tree priors don't exploit station_id as aggressively, so pooling washes out per-station signal that per-station BART captures. The "must pool to extract static terrain" hypothesis that drove the 3o design is **correct for LightGBM and wrong for BART**.

3. **Production 3o (pooled LightGBM) is architecturally correct.** No change needed there.

4. **Production 4a should move to rich-per-station as today's finding suggested.** Lean per-cell BART → rich+oro per-cell BART. Plan-bar comfortably cleared.

---

## Side-channel — GH backfills completed too

All 10 multi-level NWP backfill workflows completed overnight:
- 5 × ERA5: 2-3 min each
- 5 × previous-runs: 26-32 min each
- Multi-level pressure-level data (T/Z/wind/RH at 500/700/850 hPa) now in R2 for 2024-01-01 → 2026-05-29.
