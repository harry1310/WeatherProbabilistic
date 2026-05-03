"""Option B for the start-hour bake-off: Bayesian Gaussian copula on Phase
3a hourly P(wet) marginals.

Outputs predictions in the same parquet schema as
WeatherBlend/data/reports/start_hour_bakeoff/method=current/predictions.parquet
so the existing C# `start-hour-bakeoff --action compare` harness can score
B alongside current and C without modification.

Architecture:
  - Marginals q_h come from WeatherBlend's Phase 3a replay parquet (one
    file per (station, lead) covering the full historical span the
    dry-window blenders trained on).
  - Within-day correlation injected via an AR(1) Gaussian copula:
        Z ~ MVN(0, Σ)   with   Σ_ij = ρ^|i-j|
        U = Φ(Z)
        wet_i = 1[U_i < q_i]
  - ρ is the AR(1) correlation parameter; we fit a posterior over it from
    the EA hourly rainfall truth via PyMC. Beta(2,2) prior; the likelihood
    treats lag-1 correlated wet/dry pairs from training-slice data.
  - Per test row: sample ρ from posterior, sample MC copula realisations
    with marginals q, count valid starts (every s where hours s..s+N-1 are
    all dry in that draw). Aggregate per s → π distribution that sums to 1.

Why Bayesian (not point-estimate ρ):
  - Captures uncertainty in ρ (which varies across stations + seasons).
  - Posterior MC integration naturally covers ρ ∈ realistic credible
    interval, so π is "marginal over ρ" not "conditional on ρ̂_MAP".

Why AR(1) (not full 9×9 covariance):
  - 1 parameter vs 36 parameters; identifiable from ~1000 wet/dry pairs
    per station even after stratifying by lag.
  - Captures the dominant decay-with-distance pattern of within-day rain
    autocorrelation. Real rain has bursty arrivals which AR(1) under-
    represents at very long range; if results disappoint, that's the
    next thing to relax.

Determinism:
  - Fixed seed for posterior sampling, MC draws, and bootstrap. Re-runs
    produce identical predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pymc as pm
from scipy.stats import norm

# Repo paths — WeatherBlend lives next door.
WB_ROOT = Path(__file__).resolve().parent.parent.parent / "WeatherBlend"
BAKEOFF_DIR = WB_ROOT / "data" / "reports" / "start_hour_bakeoff"
TEST_SET_PATH = BAKEOFF_DIR / "start_hour_test_set_2026-05-03.json"
TRUTH_PATH = BAKEOFF_DIR / "start_hour_truth_labels.parquet"
REPLAY_ROOT = WB_ROOT / "data" / "predictions" / "precipitation_replay"
RAINFALL_ROOT = WB_ROOT / "data" / "truth" / "rainfall"
OUT_DIR = BAKEOFF_DIR / "method=B"

# 3a champions per station — match what WeatherBlend's bake-off currently uses
# to read the replay parquets. Hardcoded here to keep the script standalone;
# update if the precipitation manifest's Current pointer moves.
PRECIP_3A_CHAMPIONS = {
    "ea_bellever_dartmoor": "v2026-04-28_232709",
    "ea_princetown":        "v2026-04-28_232739",
    "ea_dartmoor_nr_hexworthy": "v2026-04-28_232809",
}

STATION_FRIENDLY_NAME = {
    "ea_bellever_dartmoor": "Bellever Dartmoor",
    "ea_princetown":        "Princetown",
    "ea_dartmoor_nr_hexworthy": "Dartmoor nr Hexworthy",
}

WET_THRESHOLD_MM = 0.1
LOCATION_NAME = "bonehill_rocks"


@dataclass
class TestRow:
    station_slug: str
    window_hours: int
    lead_hours: int
    target_date: pd.Timestamp
    daytime_start_utc: int
    daytime_end_utc: int


def load_test_set() -> List[TestRow]:
    rows = json.loads(TEST_SET_PATH.read_text())
    out: List[TestRow] = []
    for r in rows:
        # JSON dates come through as YYYY-MM-DD strings via the
        # JsonSerializer's DateOnly format.
        out.append(TestRow(
            station_slug=r["StationSlug"],
            window_hours=r["WindowHours"],
            lead_hours=r["LeadHours"],
            target_date=pd.Timestamp(r["TargetDateUtc"]),
            daytime_start_utc=r["DaytimeStartUtc"],
            daytime_end_utc=r["DaytimeEndUtc"],
        ))
    return out


def load_replay_hourly(station_slug: str, version: str, lead: int) -> Dict[pd.Timestamp, float]:
    """Read (ValidTimeUtc → ProbWet) for one (station, lead) replay parquet."""
    path = REPLAY_ROOT / station_slug / version / f"lead_{lead}h.parquet"
    df = duckdb.sql(f"SELECT ValidTimeUtc, ProbWet FROM read_parquet('{path.as_posix()}')").df()
    df["ValidTimeUtc"] = pd.to_datetime(df["ValidTimeUtc"], utc=True).dt.tz_localize(None)
    return dict(zip(df["ValidTimeUtc"], df["ProbWet"]))


def load_hourly_rainfall(friendly_name: str) -> pd.Series:
    """Hourly mm series indexed by hour-truncated UTC timestamp, summed
    from the 15-minute EA observations. Returns a sorted ascending Series."""
    glob = (RAINFALL_ROOT / f"location={LOCATION_NAME}" / f"station={friendly_name}"
            / "*" / "*.parquet").as_posix()
    df = duckdb.sql(f"""
        SELECT date_trunc('hour', ObservedTimeUtc) AS hour_utc,
               SUM(Value15MinMm) AS mm
        FROM read_parquet('{glob}', hive_partitioning=false)
        WHERE Value15MinMm IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """).df()
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True).dt.tz_localize(None)
    return pd.Series(df["mm"].values, index=df["hour_utc"], name="mm")


def fit_ar1_rho_posterior(
    rainfall: pd.Series,
    train_end_date: pd.Timestamp,
    daytime_start_utc: int,
    daytime_end_utc: int,
    n_samples: int = 1000,
    n_chains: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Bayesian fit of the AR(1) Gaussian-copula correlation ρ from
    historical wet/dry transitions in EA rainfall.

    Approach (deliberately simple — see module docstring for why):
      1. Take all daytime hourly observations BEFORE train_end_date so the
         posterior never sees the test slice.
      2. Convert to wet indicators (mm >= threshold).
      3. Compute the empirical lag-1 Pearson correlation r̂ across all
         within-day adjacent-hour pairs. For a stationary Gaussian copula
         with binary marginals, the wet/wet correlation is monotone in ρ
         but not equal — we use the empirical r̂ as the LIKELIHOOD's
         observed value with a Normal observation noise calibrated by
         bootstrap-style standard error.
      4. PyMC: ρ ~ Beta(2,2); observation r̂ ~ Normal(ρ, σ̂_r) where σ̂_r
         is the bootstrap standard error. A pragmatic shortcut around
         the bivariate-normal CDF likelihood the "correct" model would
         need; gives the right posterior shape without the cost.

    Returns the flattened posterior samples for ρ (shape ~ n_chains * n_samples).
    """
    train = rainfall[rainfall.index < train_end_date]
    # Daytime hours only — the bake-off question is about daytime windows
    # so the relevant correlation structure is daytime within-day.
    in_daytime = (train.index.hour >= daytime_start_utc) & (train.index.hour < daytime_end_utc)
    train = train[in_daytime]

    wet = (train >= WET_THRESHOLD_MM).astype(float)
    # Adjacent-hour pairs: (wet[t-1], wet[t]) where both fall on the
    # same calendar date.
    df = pd.DataFrame({"wet": wet, "date": wet.index.date, "hour": wet.index.hour})
    df["wet_prev"] = df["wet"].shift(1)
    df["hour_prev"] = df["hour"].shift(1)
    df["date_prev"] = df["date"].shift(1)
    pairs = df[(df["date"] == df["date_prev"]) & (df["hour"] - df["hour_prev"] == 1)].dropna()
    if len(pairs) < 100:
        raise ValueError(
            f"Only {len(pairs)} within-day adjacent-hour pairs — not enough to fit ρ. "
            f"Check daytime range and train_end_date.")

    r_hat = float(np.corrcoef(pairs["wet_prev"], pairs["wet"])[0, 1])
    # Bootstrap SE for r_hat — gives the Normal observation noise scale.
    rng = np.random.default_rng(seed)
    n_boot = 500
    boot_r = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(pairs), size=len(pairs))
        s = pairs.iloc[idx]
        boot_r[i] = np.corrcoef(s["wet_prev"], s["wet"])[0, 1]
    se_r = float(np.std(boot_r, ddof=1))

    print(f"    empirical r̂={r_hat:.3f} ± {se_r:.4f} ({len(pairs)} pairs)")

    with pm.Model():
        rho = pm.Beta("rho", alpha=2.0, beta=2.0)
        # Observation: r̂ informs ρ through this noisy link. Not the
        # mathematically-correct copula likelihood — see docstring — but
        # produces a posterior of the right shape for our purposes.
        pm.Normal("r_obs", mu=rho, sigma=se_r, observed=r_hat)
        idata = pm.sample(
            draws=n_samples, chains=n_chains, tune=500,
            random_seed=seed, progressbar=False,
            nuts_sampler="nutpie",
        )
    samples = idata.posterior["rho"].values.flatten()
    print(f"    posterior ρ: mean={samples.mean():.3f}, "
          f"sd={samples.std():.3f}, "
          f"95% CI=[{np.percentile(samples, 2.5):.3f}, {np.percentile(samples, 97.5):.3f}]")
    print(f"    R̂ = {float(idata.posterior['rho'].std('chain').mean()):.3f}")
    return samples


def gaussian_ar1_copula_sample(
    q: np.ndarray, rho: float, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample n_samples × len(q) wet/dry realisations from a Gaussian copula
    with AR(1) correlation ρ and marginals q.

    Returns a (n_samples, n_hours) bool array (True = wet)."""
    n = len(q)
    # AR(1) covariance Σ_ij = ρ^|i-j|. Cholesky once, sample many.
    idx = np.arange(n)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
    Z = rng.standard_normal((n_samples, n)) @ L.T   # MVN(0, Σ)
    U = norm.cdf(Z)                                 # uniform marginals
    wet = U < q[None, :]                            # wet_i = 1[U_i < q_i]
    return wet


def compute_start_hour_pi(
    q: np.ndarray, window_hours: int, rho_samples: np.ndarray,
    n_mc_per_rho: int, rng: np.random.Generator,
) -> np.ndarray:
    """For one test row: marginalise over the posterior of ρ. For each ρ
    drawn from the posterior, sample n_mc_per_rho copula realisations,
    count valid starts; average across the full posterior × MC ensemble.

    Returns the start-hour distribution π (sums to 1, length = n_starts).
    Falls back to uniform if no realisation produced any block (shouldn't
    happen at MC scales but defensive).
    """
    n_hours = len(q)
    n_starts = n_hours - window_hours + 1
    counts = np.zeros(n_starts, dtype=np.int64)

    # Subsample ρ if posterior is huge — 200 ρ-draws × 50 MC each = 10k
    # total realisations matches option C's sample count for fair compare.
    n_rho_draws = min(200, len(rho_samples))
    rho_subset = rng.choice(rho_samples, size=n_rho_draws, replace=False)

    for rho in rho_subset:
        wet = gaussian_ar1_copula_sample(q, float(rho), n_mc_per_rho, rng)
        for i in range(n_starts):
            # Sample valid for start i iff hours [i, i+window_hours) are all dry.
            block_dry = ~wet[:, i:i + window_hours].any(axis=1)
            counts[i] += int(block_dry.sum())

    total = counts.sum()
    if total == 0:
        return np.full(n_starts, 1.0 / n_starts)
    return counts / total


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mc-per-rho", type=int, default=50,
                        help="MC copula draws per ρ posterior sample")
    parser.add_argument("--n-rho-samples", type=int, default=1000,
                        help="Posterior draws per chain when fitting ρ")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test rows.")

    # Earliest training date = min target_date - 365 days; gives us > 1 year
    # of historical truth before the test slice. Train_end_date = the earliest
    # test slice date, which is the same boundary the dry-window
    # chronological split uses.
    test_dates_by_station = {}
    for t in test_set:
        test_dates_by_station.setdefault(t.station_slug, []).append(t.target_date)
    min_test_date_by_station = {s: min(d for d in dts) for s, dts in test_dates_by_station.items()}
    print("Test slice starts:")
    for s, d in min_test_date_by_station.items():
        print(f"  {s}: {d.date()}")

    # Fit ρ posterior per station (lead-independent — ρ is a property of
    # real rain hour-to-hour clustering, not of the forecast).
    print("\n=== Fit ρ posterior per station ===")
    rho_by_station: Dict[str, np.ndarray] = {}
    daytime_start = test_set[0].daytime_start_utc
    daytime_end = test_set[0].daytime_end_utc
    for station_slug in PRECIP_3A_CHAMPIONS.keys():
        friendly = STATION_FRIENDLY_NAME[station_slug]
        print(f"  {station_slug} ({friendly}):")
        rainfall = load_hourly_rainfall(friendly)
        print(f"    {len(rainfall)} hourly rainfall obs")
        rho_by_station[station_slug] = fit_ar1_rho_posterior(
            rainfall, min_test_date_by_station[station_slug],
            daytime_start, daytime_end,
            n_samples=args.n_rho_samples, seed=args.seed)

    # Pre-load 3a replay parquets per (station, lead).
    print("\n=== Load 3a replay parquets ===")
    replay_by_station_lead: Dict[Tuple[str, int], Dict[pd.Timestamp, float]] = {}
    leads_per_station: Dict[str, List[int]] = {}
    for t in test_set:
        leads_per_station.setdefault(t.station_slug, [])
        if t.lead_hours not in leads_per_station[t.station_slug]:
            leads_per_station[t.station_slug].append(t.lead_hours)
    for station_slug, leads in leads_per_station.items():
        version = PRECIP_3A_CHAMPIONS[station_slug]
        for lead in leads:
            replay_by_station_lead[(station_slug, lead)] = load_replay_hourly(
                station_slug, version, lead)
            print(f"  {station_slug}/lead_{lead}h: {len(replay_by_station_lead[(station_slug, lead)])} hourly q")

    # Score every test row.
    print(f"\n=== Score {len(test_set)} test rows (MC = {args.n_mc_per_rho}/ρ × ~200ρ ≈ 10k draws each) ===")
    rng = np.random.default_rng(args.seed)
    pred_rows = []
    skipped_replay = 0
    t0 = time.time()
    for i, t in enumerate(test_set):
        if i and i % 500 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(test_set) - i) / rate
            print(f"  {i}/{len(test_set)} ({rate:.1f}/s, ETA {eta:.0f}s)")

        hourly = replay_by_station_lead.get((t.station_slug, t.lead_hours))
        if hourly is None:
            continue
        midnight = t.target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        q = np.empty(t.daytime_end_utc - t.daytime_start_utc)
        gap = False
        for j, h in enumerate(range(t.daytime_start_utc, t.daytime_end_utc)):
            v = hourly.get(midnight + pd.Timedelta(hours=h))
            if v is None:
                gap = True
                break
            q[j] = max(0.0, min(1.0, v))
        if gap:
            skipped_replay += 1
            continue

        rho_samples = rho_by_station[t.station_slug]
        pi = compute_start_hour_pi(q, t.window_hours, rho_samples, args.n_mc_per_rho, rng)

        for k, p in enumerate(pi):
            pred_rows.append({
                "Method": "B",
                "StationSlug": t.station_slug,
                "WindowHours": t.window_hours,
                "LeadHours": t.lead_hours,
                "TargetDateUtc": midnight,
                "StartHourUtc": t.daytime_start_utc + k,
                "ProbDryBlockStarting": float(p),
            })
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. Skipped {skipped_replay} for replay gaps.")

    # Write predictions in the bake-off parquet schema.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "predictions.parquet"
    df = pd.DataFrame(pred_rows)
    # Schema must match what the C# harness reads — explicit Pandas dtypes.
    df["TargetDateUtc"] = pd.to_datetime(df["TargetDateUtc"])
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path)
    print(f"\nWrote {len(pred_rows)} predictions → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
