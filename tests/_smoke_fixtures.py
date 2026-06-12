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
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Superset of every NWP id any production Python trainer reads. train_3f
# / train_4a filter to their own (overlapping) MODELS_LEAN lists and
# ignore extras; train_wind_mvn pulls 6 NWPs that include `ukmo_seamless`,
# which the 3f/4a MODELS_LEAN lists do NOT include — so we emit ukmo as
# well to keep all three smoke harnesses able to share one forecast tree.
MODELS_LEAN_FULL: tuple[str, ...] = (
    "gfs_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "meteofrance_seamless",
    "gem_seamless",
    "ecmwf_aifs025_single",
    "jma_seamless",
    "ukmo_seamless",
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
                    wsp = float(max(0.0, 8.0 + rng.normal(0, 3)))
                    # Gust ≈ wsp × 1.4 + noise (matches the realistic ratio
                    # range the wind_gust bake-off saw). The wind_mvn /
                    # wind_gust smoke tests pivot WindGusts10m so without
                    # this column the gust ratio features go all-NaN.
                    gust = float(max(wsp, wsp * 1.4 + rng.normal(0, 0.5)))
                    cc_low = float(np.clip(50 + 40 * (p > 0.1) + rng.normal(0, 10), 0, 100))
                    cc_mid = float(np.clip(40 + 30 * (p > 0.1) + rng.normal(0, 10), 0, 100))
                    cc_high = float(np.clip(30 + rng.normal(0, 15), 0, 100))
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
                        "CloudCoverLow":  cc_low,
                        "CloudCoverMid":  cc_mid,
                        "CloudCoverHigh": cc_high,
                        # Total CloudCover (single composite) used by the
                        # element blenders' SPREAD features. Approximation:
                        # max-overlap of the three layers.
                        "CloudCover":     float(max(cc_low, cc_mid, cc_high)),
                        "Cape": float(max(0.0, rng.normal(50.0, 80.0))),
                        "WindSpeed10m": wsp,
                        "WindDirection10m": float(rng.uniform(0, 360)),
                        "WindGusts10m": gust,
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
# MIDAS Open Dunkeswell SYNOP truth (BADC-CSV) — wind_mvn smoke
# --------------------------------------------------------------------------

# parse_badc in train_wind_mvn.py only needs the literal "data" + "end data"
# fences and the column header line between them. Everything before "data"
# is preamble. We emit a single-line preamble so the fixture stays compact
# but the file shape still matches the real MIDAS Open download pattern.
_MIDAS_PREAMBLE = (
    "Conventions,G,BADC-CSV,1\n"
    "title,G,uk-hourly-weather-obs\n"
    "observation_station,G,dunkeswell-aerodrome\n"
    "midas_station_id,G,01383\n"
)


def make_dunkeswell_midas_truth(
    root: Path,
    years: Iterable[int] = (2022, 2023, 2024),
    *,
    days_per_year: int = 30,
    rng: np.random.Generator | None = None,
) -> Path:
    """Write ``data/truth/midas/raw/midas-open*_01383_*_{year}.csv`` BADC-CSV
    files matching the shape ``train_wind_mvn.parse_badc`` expects. Only
    the four columns ``load_dunkeswell`` actually reads (ob_time,
    met_domain_name, wind_speed (kt), wind_direction (deg)) are populated;
    every other MIDAS column is omitted (parse_badc just splits on the
    column header).

    The signal is a low-frequency direction sweep + bounded speed walk so
    the wind_mvn MLP can find *something* to fit, but the smoke contract
    is "code runs end-to-end", not "model converges to skill".
    """
    rng = rng or np.random.default_rng(seed=2026_05_28)
    midas_root = root / "truth" / "midas" / "raw"
    midas_root.mkdir(parents=True, exist_ok=True)

    for y in years:
        # days_per_year × 24 hourly SYNOP obs starting at YYYY-01-01 00:00.
        start = datetime(y, 1, 1)
        n_hours = days_per_year * 24
        # Slow direction sweep (one full rotation across the window) +
        # noise. wind_speed_unit_id=4 in real MIDAS = "anemometer, knots",
        # so we emit knots; train_wind_mvn.load_dunkeswell multiplies by
        # KT_TO_MS.
        base_dir = (np.linspace(0, 360, n_hours, endpoint=False)
                    + rng.normal(0.0, 8.0, size=n_hours)) % 360.0
        base_spd_kt = np.clip(
            12.0 + 6.0 * np.sin(np.linspace(0, 4 * np.pi, n_hours))
            + rng.normal(0.0, 2.0, size=n_hours),
            0.5, 50.0,
        )
        lines = [_MIDAS_PREAMBLE.rstrip("\n"),
                 "data",
                 "ob_time,met_domain_name,wind_speed,wind_direction"]
        for h in range(n_hours):
            ob = start + timedelta(hours=h)
            lines.append(
                f"{ob.strftime('%Y-%m-%d %H:%M:%S')},SYNOP,"
                f"{base_spd_kt[h]:.1f},{base_dir[h]:.1f}"
            )
        lines.append("end data")
        # Filename pattern mirrors retrain-python.yml's rclone include
        # filter (``midas-open*_01383_*.csv``).
        fname = (
            f"midas-open_uk-hourly-weather-obs_dv-smoke_devon_01383"
            f"_dunkeswell-aerodrome_qcv-1_{y}.csv"
        )
        (midas_root / fname).write_text("\n".join(lines), encoding="utf-8")

    return midas_root


def make_era5_wind_truth(
    root: Path,
    location: str,
    start: datetime,
    n_days: int,
    *,
    rng: np.random.Generator | None = None,
) -> Path:
    """Write ``data/truth/era5/location={loc}/date=YYYY-MM-DD/data.parquet``
    files matching the WB Era5Client truth layout — the wind_mvn truth
    source for sennen_cove (2026-06-12). Only the columns
    ``train_wind_mvn.load_era5_wind`` reads (ValidTimeUtc, WindSpeed10m,
    WindDirection10m) plus LocationName are emitted; production files
    carry ~32 columns, all ignored by the loader's explicit SELECT.

    Signal recipe mirrors :func:`make_dunkeswell_midas_truth` (slow
    direction sweep + bounded speed walk) but values are already m/s —
    the ERA5 path has no knots conversion."""
    rng = rng or np.random.default_rng(seed=2026_06_12)
    base = root / "truth" / "era5" / f"location={location}"
    n_hours = n_days * 24
    base_dir = (np.linspace(0, 360, n_hours, endpoint=False)
                + rng.normal(0.0, 8.0, size=n_hours)) % 360.0
    base_spd_ms = np.clip(
        6.0 + 3.0 * np.sin(np.linspace(0, 4 * np.pi, n_hours))
        + rng.normal(0.0, 1.0, size=n_hours),
        0.3, 30.0,
    )
    for d in range(n_days):
        day = (start + timedelta(days=d)).replace(hour=0)
        idx = slice(d * 24, (d + 1) * 24)
        day_dir = base / f"date={day.date().isoformat()}"
        day_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "LocationName": location,
            "ValidTimeUtc": [day + timedelta(hours=h) for h in range(24)],
            "WindSpeed10m": base_spd_ms[idx],
            "WindDirection10m": base_dir[idx],
        }).to_parquet(day_dir / "data.parquet", index=False)
    return base


# --------------------------------------------------------------------------
# Static orographic JSON — wind_mvn (and 4a synoptic-feature lookups)
# --------------------------------------------------------------------------

# train_wind_mvn.load_oro_static only reads three keys. The production JSON
# is much bigger (climatology_by_sector_month etc.) but anything we don't
# emit here is simply ignored downstream.
_DEFAULT_UPWIND_GAIN_5KM = {
    "N": -26.6, "NE": -80.2, "E": -46.1, "SE": -42.0,
    "S": -56.0, "SW": -88.8, "W": -37.7, "NW": 50.4,
}


def make_orographic_static(
    root: Path,
    slug: str,
    *,
    terrain_gradient_dx: float = 0.10,
    terrain_gradient_dy: float = 0.05,
    upwind_gain_5km: dict | None = None,
    elevation_vs_cell_m: float = 50.0,
    relief_5km_m: float = 200.0,
    terrain_ruggedness_5km_m: float = 80.0,
) -> Path:
    """Write ``data/static/orographic/{slug}.json``. ``slug`` can be either
    a location (e.g. ``bonehill_rocks``) for train_wind_mvn's per-location
    lookup, or a station slug (e.g. ``ea_bellever_dartmoor``) for train_4a's
    per-station rich+oro builder.

    Includes both:
      * ``upwind_gain_5km`` + ``terrain_gradient_dx/dy`` — read by
        train_wind_mvn's oro_dynamic and 4a's v1 terrain block.
      * ``elevation_vs_cell_m`` / ``relief_5km_m`` / ``terrain_ruggedness_5km_m``
        — three additional static scalars read by 4a's v1 terrain block.
    Numerically modelled on Bonehill's real values so the resulting feature
    magnitudes match production within an order of magnitude.
    """
    static_root = root / "static" / "orographic"
    static_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "slug": slug,
        "terrain_gradient_dx": float(terrain_gradient_dx),
        "terrain_gradient_dy": float(terrain_gradient_dy),
        "upwind_gain_5km": dict(upwind_gain_5km or _DEFAULT_UPWIND_GAIN_5KM),
        "elevation_vs_cell_m":      float(elevation_vs_cell_m),
        "relief_5km_m":             float(relief_5km_m),
        "terrain_ruggedness_5km_m": float(terrain_ruggedness_5km_m),
    }
    out = static_root / f"{slug}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


# --------------------------------------------------------------------------
# sync_train_data.sh invocation
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_train_data.sh"


def _locate_bash() -> str:
    """Locate a bash executable.

    On ubuntu-latest a bare ``"bash"`` resolves via PATH; on Windows the
    Python interpreter's PATH typically exposes Git's ``cmd/`` directory
    (which has ``git.exe``) but NOT ``bin/`` or ``usr/bin/`` (where Git
    Bash itself lives), so ``subprocess.run(["bash", …])`` fails with
    FileNotFoundError. Prefer a known absolute path on Windows; fall
    back to PATH search on every platform. Honour ``WB_BASH`` as an
    explicit override for non-standard Git installs.
    """
    override = os.environ.get("WB_BASH")
    if override and os.path.isfile(override):
        return override
    if sys.platform == "win32":
        for candidate in (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ):
            if os.path.isfile(candidate):
                return candidate
    found = shutil.which("bash")
    if found:
        return found
    raise RuntimeError(
        "Could not locate a bash executable. Install Git for Windows "
        "(provides Git Bash) or set WB_BASH to an explicit bash.exe path."
    )


def add_decoy_bundles(station_dir: Path, phase_tag: str) -> None:
    """Sibling version dirs the predict-side bundle resolvers must SKIP —
    the production tree shapes behind real incidents (2026-05-11 stale
    lead-pooled 4a; the shared models/wind dir where .NET champion + python
    challenger bundles coexist). Drop these next to a freshly-trained
    bundle so the predict smokes exercise SELECTION, not just the happy
    single-dir tree:

      (a) a LEXICALLY NEWER foreign-phase version — a naive
          "sort | take last" resolver would pick it;
      (b) a lexically newer UNSUFFIXED version (the .NET champion shape);
      (c) an OLD same-phase dir with a legacy artefact layout (for 4a the
          lead-pooled state.rds without per-lead files — find_latest_bundle
          must filter it on artefact shape, not name alone).
    """
    foreign = station_dir / "v2099-01-01_000000_phasedecoy"
    foreign.mkdir(parents=True, exist_ok=True)
    (foreign / "model.zip").write_text("decoy")
    unsuffixed = station_dir / "v2099-01-02_000000"
    unsuffixed.mkdir(parents=True, exist_ok=True)
    (unsuffixed / "lead_24h.zip").write_text("decoy")
    legacy = station_dir / f"v2020-01-01_000000_{phase_tag}"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "state.rds").write_text("decoy")
    (legacy / "preprocess.json").write_text("{}")


def run_sync_train_data(
    *,
    location: str,
    phases: str,
    r2_source: Path,
    local_root: Path,
    mode: str = "train",
    predict_pull_anchor: str | None = None,
) -> None:
    """Invoke ``scripts/sync_train_data.sh`` against a local fake-R2 dir.

    Mirrors the production "Pull data trees from R2" step in
    ``retrain-python.yml`` (train) / ``predict-*.yml`` (predict) exactly —
    same script, same arg shape — so a missing pull declaration fails the
    smoke at PR time rather than "Loaded 0 rows"/FileNotFound in production.

    ``mode`` ("train" | "predict") sets the MODE env the script reads:
    predict pulls the feature trees + latest model bundle a phase reads at
    predict time, train pulls feature trees + truth labels + the manifest.

    Required layout in ``r2_source``: fixtures written to
    ``r2_source/data/forecasts/…``, ``r2_source/data/truth/…`` etc. (the
    sync script's source prefix includes ``data/``; the destination root
    does not, mirroring R2 vs the on-runner ``../WeatherBlend/data``
    layout).

    Raises ``RuntimeError`` if the script is missing or rclone/bash are
    not on PATH — surface as a clear test failure instead of a silent
    skip.
    """
    if not SYNC_SCRIPT.exists():
        raise RuntimeError(
            f"sync_train_data.sh not found at {SYNC_SCRIPT}; smoke harness "
            f"cannot validate pull declarations. Run from a WP checkout."
        )
    env = os.environ.copy()
    env["R2_SOURCE"] = str(r2_source)
    env["LOCAL_ROOT"] = str(local_root)
    env["MODE"] = mode
    # Predict-mode pulls a [anchor-14d, anchor+6d] date-partition window
    # (2026-06-11). Fixtures use ABSOLUTE dates, so predict-mode tests must
    # pin the window to their fixture anchor — wall-clock default would
    # exclude every fixture partition.
    if predict_pull_anchor is not None:
        env["PREDICT_PULL_ANCHOR"] = predict_pull_anchor
    bash_exe = _locate_bash()
    result = subprocess.run(
        [bash_exe, str(SYNC_SCRIPT), location, phases],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "sync_train_data.sh failed:\n"
            f"  cmd: bash {SYNC_SCRIPT} {location} {phases}\n"
            f"  R2_SOURCE={r2_source}\n"
            f"  LOCAL_ROOT={local_root}\n"
            f"  exit={result.returncode}\n"
            f"  stdout:\n{result.stdout}\n"
            f"  stderr:\n{result.stderr}"
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _ea_slug(friendly_name: str) -> str:
    """Mirror src.weatherblend_config._ea_slug so the fixture path matches
    what the production loaders compose. Duplicated here to keep this
    module import-light (no src.* side effects)."""
    bare = "_".join(friendly_name.lower().split())
    return f"ea_{bare}"


# --------------------------------------------------------------------------
# Sennen sea-state fixtures (wave_height_lgb smoke, 2026-06-12)
# --------------------------------------------------------------------------

WAVE_MODELS = ("meteofrance_wave", "ecmwf_wam025", "gwam", "ewam", "ncep_gfswave025")
WAVE_BEST = "best_match"


def make_marine_tree(
    data_root: Path,
    location: str,
    start: datetime,
    n_days: int,
    kind: str = "hist_forecast",
    rng: np.random.Generator | None = None,
) -> None:
    """Synthetic marine (wave) forecast tree matching the WB MarineClient
    writers. ``kind`` picks the file flavour the wave scripts read:
    ``hist_forecast`` (train features: hist_forecast.parquet per valid-date,
    RunTimeSource=hist_forecast, lead-unlabelled) or ``live`` (predict
    features: run=00.parquet per run-date, RunTimeSource=synthesised, each
    run covering 7 forecast days).

    Hs follows a shared seasonal+stormy signal with per-model noise so the
    blend has something real to regress; best_match additionally carries the
    site extras (tide / SST / secondary swell) like production.
    """
    rng = rng or np.random.default_rng(seed=2026_06_12)
    base = data_root / "marine" / f"location={location}"

    def day_rows(model: str, day: datetime, run_time: datetime, source: str,
                 hours: int) -> "pd.DataFrame":
        valid = [day + timedelta(hours=h) for h in range(hours)]
        t = np.array([(v - start).total_seconds() / 86400 for v in valid])
        signal = 1.6 + 0.9 * np.sin(2 * np.pi * t / 14) + 0.4 * np.sin(2 * np.pi * t / 3.1)
        hs = np.clip(signal + rng.normal(0, 0.25, len(valid)), 0.05, None)
        period = np.clip(7 + 2.5 * np.sin(2 * np.pi * t / 9) + rng.normal(0, 0.6, len(valid)), 2, None)
        direction = (270 + 40 * np.sin(2 * np.pi * t / 11) + rng.normal(0, 12, len(valid))) % 360
        is_best = model == WAVE_BEST
        return pd.DataFrame({
            "LocationName": location,
            "Model": model,
            "RunTimeUtc": run_time,
            "ValidTimeUtc": valid,
            "LeadHours": [int((v - run_time).total_seconds() // 3600) if source == "synthesised" else 0
                          for v in valid],
            "RunTimeSource": source,
            "WaveHeight": hs,
            "WavePeriod": period,
            "WaveDirection": direction,
            "WindWaveHeight": np.clip(hs * 0.3 + rng.normal(0, 0.1, len(valid)), 0, None),
            "WindWavePeriod": period * 0.5,
            "WindWaveDirection": direction,
            "SwellWaveHeight": np.clip(hs * 0.8 + rng.normal(0, 0.1, len(valid)), 0, None),
            "SwellWavePeriod": period * 1.3,
            "SwellWaveDirection": direction,
            "SecondarySwellWaveHeight": (hs * 0.15) if is_best else np.nan,
            "SecondarySwellWavePeriod": (period * 1.6) if is_best else np.nan,
            "SecondarySwellWaveDirection": direction if is_best else np.nan,
            "SeaLevelHeightMsl": (2.2 * np.sin(2 * np.pi * t * 24 / 12.42)) if is_best else np.nan,
            "SeaSurfaceTemperature": (12.5 + 2 * np.sin(2 * np.pi * t / 180)) if is_best else np.nan,
        })

    for model in (*WAVE_MODELS, WAVE_BEST):
        if kind == "hist_forecast":
            for d in range(n_days):
                day = start + timedelta(days=d)
                day_dir = base / f"model={model}" / f"date={day:%Y-%m-%d}"
                day_dir.mkdir(parents=True, exist_ok=True)
                day_rows(model, day, run_time=day, source="hist_forecast", hours=24) \
                    .to_parquet(day_dir / "hist_forecast.parquet", index=False)
        else:
            # One live run per day at 00Z, each covering 7 forecast days —
            # predict picks the lexicographically newest, like production.
            for d in range(n_days):
                run = start + timedelta(days=d)
                day_dir = base / f"model={model}" / f"date={run:%Y-%m-%d}"
                day_dir.mkdir(parents=True, exist_ok=True)
                day_rows(model, run, run_time=run, source="synthesised", hours=7 * 24) \
                    .to_parquet(day_dir / "run=00.parquet", index=False)


def make_wave_truth_tree(
    data_root: Path,
    location: str,
    start: datetime,
    n_days: int,
    rng: np.random.Generator | None = None,
) -> None:
    """era5_ocean wave truth matching the WB WriteWaveTruthAsync layout —
    the same shared signal as make_marine_tree (different noise draw) so the
    fixture blend genuinely has signal to find."""
    rng = rng or np.random.default_rng(seed=2026_06_13)
    base = data_root / "truth" / "waves" / f"location={location}" / "source=era5_ocean"
    for d in range(n_days):
        day = start + timedelta(days=d)
        valid = [day + timedelta(hours=h) for h in range(24)]
        t = np.array([(v - start).total_seconds() / 86400 for v in valid])
        signal = 1.6 + 0.9 * np.sin(2 * np.pi * t / 14) + 0.4 * np.sin(2 * np.pi * t / 3.1)
        day_dir = base / f"date={day:%Y-%m-%d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "LocationName": location,
            "Source": "era5_ocean",
            "ValidTimeUtc": valid,
            "WaveHeight": np.clip(signal + rng.normal(0, 0.12, 24), 0.05, None),
            "WavePeriod": 7 + 2.5 * np.sin(2 * np.pi * t / 9),
            "WaveDirection": (270 + 40 * np.sin(2 * np.pi * t / 11)) % 360,
            "PeakPeriod": np.nan,
            "DirectionalSpread": np.nan,
            "SeaSurfaceTemperature": np.nan,
        }).to_parquet(day_dir / "data.parquet", index=False)
