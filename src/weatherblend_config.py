"""Cross-repo configuration loader.

WeatherBlend's ``src/WeatherBlend/config.yaml`` is the single source of
truth for **what locations exist + which EA rainfall stations belong to
each + lat/lon/elevation + the NWP catalogue**. This module loads it and
exposes the slices WeatherProbabilistic actually consumes (station
slugs by location, friendly names by slug, location metadata).

Loaded once at import time and cached in module-level constants.
Re-importing or restarting the process is the supported refresh.

Load order (first hit wins):

  1. ``$WEATHERBLEND_CONFIG`` — explicit path override (used by tests,
     unusual setups).
  2. ``./config/config.yaml`` — the workflow-relative path used by every
     WeatherProbabilistic CI workflow, which rclone-copies
     ``r2:${R2_BUCKET}/config/config.yaml`` to this exact spot before
     any Python runs (matches the existing phases.yaml fetch pattern;
     see retrain-python.yml's "Fetch phases.yaml from R2" step).
  3. ``../WeatherBlend/src/WeatherBlend/config.yaml`` — sibling checkout
     on a dev machine. Lets a local run of any predict / bake-off script
     pick up edits to the WB config without copying or rclone'ing.

If none are found we raise — a missing config means the workflow's R2
fetch step is broken or the dev hasn't checked out WeatherBlend
alongside, and silently falling back to a hardcoded default is exactly
the drift that motivated this loader.

Locations and stations match the partition keys WeatherBlend writes to
the parquet trees, so the slugs returned here are the SAME strings used
in ``data/predictions/precipitation/{station_slug}/...`` paths and the
``LocationName`` column on every prediction row.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import yaml


class Station(NamedTuple):
    """One EA rainfall gauge. ``slug`` is the ``ea_`` form used in
    parquet station partitions; ``name`` is the friendly form stored
    in the ``StationName`` column."""
    slug: str
    name: str


class Location(NamedTuple):
    """One configured target location."""
    name: str            # config slug: bonehill_rocks / membury_devon
    display_name: str    # human-facing label
    latitude: float
    longitude: float
    elevation_meters: float
    stations: tuple[Station, ...]


def _ea_slug(friendly_name: str) -> str:
    """Mirror C# ``StationSlug.WithEaPrefix(name)``: ``ea_`` + the
    friendly name lowercased with spaces collapsed to underscores.
    ``"Bellever Dartmoor"`` → ``"ea_bellever_dartmoor"``."""
    bare = "_".join(friendly_name.lower().split())
    return f"ea_{bare}"


def _candidate_paths() -> list[Path]:
    env = os.environ.get("WEATHERBLEND_CONFIG")
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.append(Path("config/config.yaml"))
    paths.append(Path(__file__).resolve().parent.parent.parent
                 / "WeatherBlend" / "src" / "WeatherBlend" / "config.yaml")
    return paths


def _load_yaml() -> dict:
    for p in _candidate_paths():
        if p.is_file():
            with open(p) as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError(
        "No WeatherBlend config.yaml found. Looked in: "
        + ", ".join(str(p) for p in _candidate_paths())
        + ". CI workflows should rclone the file from r2:${R2_BUCKET}"
          "/config/config.yaml to ./config/config.yaml before invoking "
          "any Python script that imports src.weatherblend_config."
    )


def _parse(doc: dict) -> tuple[Location, ...]:
    locs_raw = doc.get("locations") or []
    locs: list[Location] = []
    for entry in locs_raw:
        stations_raw = (entry.get("rainfall") or {}).get("stations") or []
        stations = tuple(
            Station(slug=_ea_slug(s["name"]), name=s["name"])
            for s in stations_raw
            if "name" in s
        )
        locs.append(Location(
            name=entry["name"],
            display_name=entry.get("displayName") or entry["name"],
            latitude=float(entry["latitude"]),
            longitude=float(entry["longitude"]),
            elevation_meters=float(entry.get("elevationMeters", 0.0)),
            stations=stations,
        ))
    return tuple(locs)


# ---------------------------------------------------------------------
# Module-level derived constants — load once, used everywhere downstream.
# ---------------------------------------------------------------------

LOCATIONS: tuple[Location, ...] = _parse(_load_yaml())
"""All configured locations, in YAML order. First entry is the primary."""

LOCATIONS_BY_NAME: dict[str, Location] = {loc.name: loc for loc in LOCATIONS}

STATIONS_BY_LOCATION: dict[str, tuple[str, ...]] = {
    loc.name: tuple(s.slug for s in loc.stations) for loc in LOCATIONS
}
"""Slug-only tuple per location — replaces the hand-curated dict that
used to sit in ``src/data.py``."""

STATION_NAME_BY_SLUG: dict[str, str] = {
    s.slug: s.name for loc in LOCATIONS for s in loc.stations
}
"""Friendly-name lookup keyed by slug — replaces the hand-curated dict
that used to sit in ``scripts/_shared.py`` (which had stale Princetown
and was missing all of Membury until this rewrite, 2026-05-20)."""
