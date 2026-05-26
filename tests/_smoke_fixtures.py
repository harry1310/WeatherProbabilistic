"""Synthetic parquet-fixture builders for the smoke-test harness.

The smoke harness's contract (see ``WeatherBlend/docs/SMOKE_TEST_PLAN.md``)
is to invoke production entry points (train_3f, predict_3f, train_4a, …)
against a temp-directory parquet tree that mimics the on-disk layout the
production loaders read. The fixtures here produce that tree.

Key constraints:

* **Schemas must match production exactly.** The DuckDB SQL in
  ``scripts/_shared.py`` and ``scripts/predict_3f.py`` is column-name +
  type sensitive. A missing or mistyped column silently drops rows from
  the inner join and the smoke ends up passing on zero training rows.
* **RunTimeSource='offset_day' rows are mandatory** for train_3f because
  the build_features_via_duckdb SQL filters on that exact literal. A
  fixture that only writes RunTimeSource='reported' would build_features
  to an empty frame.
* **Realistic distributions matter.** NGBoost-LogNormal will refuse to
  fit (or hit early-stop instantly) if precip features are uniform or
  if the wet rate is too extreme. We seed with a fixed RNG so any test
  failure is deterministic.

Tempdir scoping is the caller's job — pass it as ``root``. Every writer
returns the path it wrote so the caller can assert.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Mirrors _shared.MODELS_LEAN — the 7 NWPs train_3f trains on. Pinned
# here (not imported) so the fixture stays usable even when _shared has
# been monkey-patched in a test.
MODELS_LEAN_FULL: tuple[str, ...] = (
    "gfs_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "meteofrance_seamless",
    "gem_seamless",
    "ecmwf_aifs025_single",
    "jma_seamless",
)

DEFAULT_LEADS: tuple[int, ...] = (24, 48, 72, 96, 120)


# --------------------------------------------------------------------------
# Forecast tree
# --------------------------------------------------------------------------

def make_forecast_tree(
    root: Path,
    location: str,
    start_date: datetime,
    n_days: int,
    *,
    models: Iterable[str] = MODELS_LEAN_FULL,
    leads: Iterable[int] = DEFAULT_LEADS,
    run_time_source: str = "offset_day",
    rng: np.random.Generator | None = None,
) -> Path:
    """Write ``data/forecasts/location={loc}/model={m}/date={d}/run={HH}.parquet``
    files mirroring the production Open-Meteo previous-runs layout.

    One parquet per (model, valid-time-date) covering all 24 hours of that
    valid day for every lead in ``leads``. RunTimeUtc = ValidTimeUtc -
    LeadHours so the offset_day stamp is mathematically valid.

    Returns the root forecasts/ directory.
    """
    rng = rng or np.random.default_rng(seed=42)
    forecasts_root = root / "forecasts" / f"location={location}"
    forecasts_root.mkdir(parents=True, exist_ok=True)

    models_t = tuple(models)
    leads_t = tuple(leads)
    n_models = len(models_t)

    # Seed a per-valid-time "wet signal" + temperature signal that's
    # consistent across NWPs (with model-specific noise). NGBoost needs
    # to find a usable mapping precip_mean → wet probability — pure noise
    # would fail to converge.
    total_hours = n_days * 24
    valid_times = [start_date + timedelta(hours=h) for h in range(total_hours)]

    # Base "true precip" signal — diurnal + storm clusters. Used as the
    # shared mean across models; each NWP perturbs it. Tuned for ≈25%
    # wet rate (rng-dependent on n_days) so train_3f's MIN_WET_TRAIN_ROWS
    # gate is comfortably cleared.
    hours = np.arange(total_hours)
    diurnal = 0.5 + 0.4 * np.sin(2 * np.pi * hours / 24.0 - np.pi / 2)
    storms = np.zeros(total_hours)
    storm_starts = rng.choice(total_hours, size=max(2, 2 * n_days), replace=False)
    for s in storm_starts:
        width = rng.integers(2, 10)
        end = min(total_hours, s + width)
        storms[s:end] += rng.exponential(2.5, size=end - s)

    base_precip = np.maximum(0.0, storms * diurnal)
    base_temp = 12.0 + 6.0 * np.sin(2 * np.pi * hours / (24.0 * 365.0)) \
                + 4.0 * np.sin(2 * np.pi * hours / 24.0 - np.pi / 2)

    for model in models_t:
        model_dir = forecasts_root / f"model={model}"
        # Per-model noise scale: some NWPs are wetter/drier than the
        # ensemble mean. Captures the spread features 3a/3f rely on.
        precip_bias = rng.uniform(0.7, 1.3)
        temp_bias = rng.uniform(-1.5, 1.5)

        for d in range(n_days):
            day_start_hour = d * 24
            valid_day = (start_date + timedelta(days=d)).replace(hour=0)
            rows: list[dict] = []
            for lead in leads_t:
                for h in range(24):
                    idx = day_start_hour + h
                    v = valid_times[idx]
                    run_time = v - timedelta(hours=lead)
                    # NWP-noisy version of the shared signal.
                    p = max(0.0, base_precip[idx] * precip_bias
                            + rng.normal(0.0, 0.15))
                    t = base_temp[idx] + temp_bias + rng.normal(0.0, 0.6)
                    rh = float(np.clip(
                        80.0 - 5.0 * t + 20.0 * (p > 0.1) + rng.normal(0.0, 8.0),
                        20.0, 100.0,
                    ))
                    dp = t - (100.0 - rh) / 5.0
                    rows.append({
                        "LocationName": location,
                        "Model": model,
                        "RunTimeUtc": run_time,
                        "ValidTimeUtc": v,
                        "LeadHours": int(lead),
                        "RunTimeSource": run_time_source,
                        "Precipitation": float(p),
                        "RelativeHumidity2m": rh,
                        "Temperature2m": float(t),
                        "DewPoint2m": float(dp),
                        "CloudCoverLow":  float(np.clip(50 + 40 * (p > 0.1) + rng.normal(0, 10), 0, 100)),
                        "CloudCoverMid":  float(np.clip(40 + 30 * (p > 0.1) + rng.normal(0, 10), 0, 100)),
                        "CloudCoverHigh": float(np.clip(30 + rng.normal(0, 15), 0, 100)),
                        "Cape": float(max(0.0, rng.normal(50.0, 80.0))),
                        "WindSpeed10m": float(max(0.0, 8.0 + rng.normal(0, 3))),
                        "WindDirection10m": float(rng.uniform(0, 360)),
                        "SurfacePressure": float(1013.0 + rng.normal(0, 8)),
                    })
            day_dir = model_dir / f"date={valid_day.date().isoformat()}"
            day_dir.mkdir(parents=True, exist_ok=True)
            # File name encodes run_time_source so callers can write both
            # 'offset_day' (for train_3f) and 'reported' (for predict_3f)
            # to the same day_dir without one overwriting the other.
            df = pd.DataFrame(rows)
            df.to_parquet(day_dir / f"run=00_{run_time_source}.parquet", index=False)

    return forecasts_root


# --------------------------------------------------------------------------
# Rainfall truth tree (EA 15-min readings)
# --------------------------------------------------------------------------

def make_rainfall_truth(
    root: Path,
    location: str,
    station_friendly: str,
    start_date: datetime,
    n_days: int,
    *,
    rng: np.random.Generator | None = None,
    forecast_seed: int = 42,
) -> Path:
    """Write ``data/truth/rainfall/location={loc}/station={slug}/date={d}/rainfall.parquet``
    files. 15-min resolution with the strict-4-of-4 rule in mind: we emit
    all four quarter-hour readings per hour so the train_3f SQL's
    ``HAVING COUNT(*) = 4`` filter keeps every hour.

    The synthetic wet/dry pattern is correlated with the forecast tree's
    base signal (via ``forecast_seed``) so NGBoost has a signal to fit.
    """
    rng = rng or np.random.default_rng(seed=forecast_seed + 7)
    # Re-derive the same base_precip signal as make_forecast_tree so truth
    # is correlated with forecasts (but not identical — adds observation
    # noise so the wet/dry label isn't a perfect linear function of the
    # NWPs).
    forecast_rng = np.random.default_rng(seed=forecast_seed)
    total_hours = n_days * 24
    hours = np.arange(total_hours)
    diurnal = 0.5 + 0.4 * np.sin(2 * np.pi * hours / 24.0 - np.pi / 2)
    storms = np.zeros(total_hours)
    storm_starts = forecast_rng.choice(
        total_hours, size=max(2, 2 * n_days), replace=False
    )
    for s in storm_starts:
        width = forecast_rng.integers(2, 10)
        end = min(total_hours, s + width)
        storms[s:end] += forecast_rng.exponential(2.5, size=end - s)
    base_precip = np.maximum(0.0, storms * diurnal)
    # Observation noise — some wet hours read as dry and vice versa.
    obs_precip = np.maximum(0.0, base_precip + rng.normal(0.0, 0.05, size=total_hours))

    slug = _ea_slug(station_friendly)
    station_root = root / "truth" / "rainfall" / f"location={location}" / f"station={slug}"
    station_root.mkdir(parents=True, exist_ok=True)

    for d in range(n_days):
        valid_day = (start_date + timedelta(days=d)).replace(hour=0)
        rows: list[dict] = []
        for h in range(24):
            idx = d * 24 + h
            hourly_mm = float(obs_precip[idx])
            # Distribute hourly mm across four 15-min readings — uniformly
            # for simplicity. Production has clumpier per-quarter rates,
            # but the train_3f SQL only cares about the hourly SUM.
            per_quarter = hourly_mm / 4.0
            base = valid_day + timedelta(hours=h)
            for q in range(4):
                rows.append({
                    "ObservedTimeUtc": base + timedelta(minutes=15 * q),
                    "Value15MinMm": per_quarter,
                    "LocationName": location,
                    "StationName": station_friendly,
                })
        day_dir = station_root / f"date={valid_day.date().isoformat()}"
        day_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(day_dir / "rainfall.parquet", index=False)

    return station_root


# --------------------------------------------------------------------------
# 3a-style predictions parquet (used by predict_3f's bound-stage-1 read)
# --------------------------------------------------------------------------

def make_3a_predictions(
    root: Path,
    station_friendly: str,
    version: str,
    anchor: datetime,
    *,
    leads: Iterable[int] = DEFAULT_LEADS,
    rng: np.random.Generator | None = None,
    n_cycles: int = 4,
    target_hours_per_lead: int = 24,
) -> Path:
    """Write ``data/predictions/precipitation/{slug}/model_version={v}/date={anchor}/predictions.parquet``
    in the PrecipPredictionRow shape predict_3f.load_bound_3a_pi expects.

    Defaults to 4 cycles × 24 valid-time hours × 5 leads = 480 rows.
    Mirrors the on-disk shape predict-and-render produces — the
    load_bound_3a_pi DuckDB dedup keeps the freshest cycle per (V, L).

    Returns the predictions.parquet path.
    """
    rng = rng or np.random.default_rng(seed=99)
    slug = _ea_slug(station_friendly)
    leads_t = tuple(leads)

    rows: list[dict] = []
    for cycle in range(n_cycles):
        cycle_time = anchor.replace(hour=0, tzinfo=None) + timedelta(hours=6 * cycle)
        for lead in leads_t:
            for h in range(target_hours_per_lead):
                v = anchor.replace(hour=0, tzinfo=None) \
                    + timedelta(hours=lead) \
                    + timedelta(hours=h)
                rows.append({
                    "LocationName": "membury_devon",
                    "TruthStation": slug,
                    "ModelVersion": version,
                    "PredictionMadeAtUtc": cycle_time,
                    "ValidTimeUtc": v,
                    "LeadHours": int(lead),
                    "ProbWet": float(np.clip(rng.beta(2.0, 5.0) + 0.05 * (cycle + 1), 0.0, 1.0)),
                    "ClimatologyPWet": 0.25,
                    "FeatureVectorHash": "smoke",
                })
    target_dir = (
        root
        / "predictions"
        / "precipitation"
        / slug
        / f"model_version={version}"
        / f"date={anchor.date().isoformat()}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    parquet = target_dir / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    return parquet


# --------------------------------------------------------------------------
# MANIFEST.json — Stations-keyed, matches load_active manifest_promote shape
# --------------------------------------------------------------------------

def make_manifest(
    root: Path,
    target: str,
    station_slug: str,
    active_versions: Iterable[str],
    *,
    champion_phase: str | None = None,
) -> Path:
    """Write ``data/models/{target}/MANIFEST.json`` with the Stations
    layout the production loaders use (resolve_bound_3a_version reads
    ``Stations.{slug}.Active``).

    Falls through gracefully if the manifest already exists — merges the
    incoming station entry rather than clobbering siblings, so a smoke
    test that calls make_manifest twice (e.g. once for 3a and once for
    3f) keeps both stations resident.
    """
    manifest_path = root / "models" / target / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"Stations": {}}
    stations = manifest.setdefault("Stations", {})
    entry = stations.setdefault(station_slug, {})
    entry["Active"] = list(active_versions)
    if champion_phase:
        entry["ChampionPhase"] = champion_phase
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _ea_slug(friendly_name: str) -> str:
    """Mirror src.weatherblend_config._ea_slug so the fixture path matches
    what the production loaders compose. Duplicated here to keep this
    module import-light (no src.* side effects)."""
    bare = "_".join(friendly_name.lower().split())
    return f"ea_{bare}"
