"""Python-side phase registry loader.

Mirrors WeatherBlend's ``Models/PhaseRegistry.cs`` for the Python phases
(currently 4a / 3f). Reads ``phases.yaml`` from the same R2-fetched
location every other WP script uses (``./config/phases.yaml``, with
sibling-checkout fallback for local dev) and exposes the fields each
script consumes.

Today only ``min_valid_time`` is consumed (2026-05-26 — see
``project_3a_3b_data_drift_2026-05-26`` memory): 4a carries
``minValidTime: "2024-01-01"`` in ``phases.yaml`` because the
JMA-extension surfaced that pre-2024 NWP archives are NULL-padded,
diluting the training signal. 3f does NOT (Membury-only, data
starts 2024 anyway). The lookup returns ``None`` for any phase
without the field, which the call site treats as "no cutoff".

Load order mirrors ``weatherblend_config._candidate_paths``:
  1. ``$WEATHERBLEND_PHASES`` — explicit override (tests).
  2. ``./config/phases.yaml`` — CI path; retrain-python.yml's
     "Fetch phases.yaml from R2" step rclone-copies the WB file here
     before any Python step runs.
  3. ``../WeatherBlend/src/WeatherBlend/Config/phases.yaml`` —
     sibling-checkout fallback for laptop runs.

Strict failure modes: invalid YAML, invalid date format, all raise.
Silent fallbacks (default min_valid_time to None on parse failure)
would re-introduce the regression mode this whole thing exists to
fix.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _candidate_paths() -> list[Path]:
    env = os.environ.get("WEATHERBLEND_PHASES")
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.append(Path("config/phases.yaml"))
    paths.append(Path(__file__).resolve().parent.parent.parent
                 / "WeatherBlend" / "src" / "WeatherBlend" / "Config" / "phases.yaml")
    return paths


def _resolve_phases_path() -> Path:
    for p in _candidate_paths():
        if p.is_file():
            return p
    raise FileNotFoundError(
        "phases.yaml not found in any of: "
        + ", ".join(str(p) for p in _candidate_paths())
        + ". CI workflows rclone-copy this from R2; check the "
        "'Fetch phases.yaml from R2' step is firing."
    )


def _parse_min_valid_time(raw: Any, target: str, phase_id: str) -> datetime | None:
    """Parse the optional ``minValidTime`` field. Mirrors C#
    ``PhaseRegistry.ParseMinValidTime`` — accepts an ISO date or
    timestamp, returns UTC midnight (or the parsed timestamp's UTC
    representation), raises with target+phase context on parse failure.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, datetime):
        # YAML auto-parsed it (uncommon for bare dates but covers the
        # case where someone writes a full ISO timestamp without quotes).
        return raw.astimezone(timezone.utc) if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if not isinstance(raw, str):
        raise ValueError(
            f"Target {target!r} phase {phase_id!r} has invalid minValidTime "
            f"{raw!r} (type {type(raw).__name__}). Expected an ISO-8601 "
            f"date or timestamp string."
        )
    try:
        # datetime.fromisoformat handles both 'YYYY-MM-DD' and full
        # 'YYYY-MM-DDTHH:MM:SS[+TZ]' — same surface as C#'s DateTime.TryParse
        # in invariant culture.
        dt = datetime.fromisoformat(raw)
    except ValueError as e:
        raise ValueError(
            f"Target {target!r} phase {phase_id!r} has invalid minValidTime "
            f"{raw!r}. Expected an ISO-8601 date (e.g. '2024-01-01') or "
            f"full timestamp. ({e})"
        ) from e
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class PhaseRegistry:
    """Loaded phases.yaml exposing the slices Python scripts need.

    Construct via :meth:`load` (preferred) for explicit-path tests, or
    use the module-level :data:`Default` singleton in production code
    paths."""

    def __init__(self, by_target: dict[str, list[dict[str, Any]]]) -> None:
        self._by_target = by_target

    @classmethod
    def load(cls, path: Path | str | None = None) -> "PhaseRegistry":
        target_path = Path(path) if path else _resolve_phases_path()
        with open(target_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        targets = doc.get("targets") or {}
        if not targets:
            raise ValueError(f"phases.yaml at {target_path} has no 'targets' block, or it's empty.")
        by_target: dict[str, list[dict[str, Any]]] = {}
        for target, body in targets.items():
            phases = (body or {}).get("phases") or []
            entries: list[dict[str, Any]] = []
            for raw in phases:
                phase_id = raw.get("id")
                if not phase_id:
                    raise ValueError(f"Target {target!r} has a phase with no id.")
                entries.append({
                    "id": str(phase_id),
                    "role": raw.get("role"),
                    "impl": raw.get("impl"),
                    "locations": raw.get("locations") or [],
                    "min_valid_time": _parse_min_valid_time(
                        raw.get("minValidTime"), target, str(phase_id)),
                })
            by_target[target] = entries
        return cls(by_target)

    def min_valid_time_for(self, target: str, phase_id: str) -> datetime | None:
        """Return the parsed ``minValidTime`` for one phase, or ``None``
        if absent (= no cutoff). Returns ``None`` for unknown
        (target, phase_id) pairs too — call sites should treat unknown
        as "no cutoff" because the registry is the source of truth for
        what phases exist."""
        for entry in self._by_target.get(target, []):
            if entry["id"] == phase_id:
                return entry["min_valid_time"]
        return None

    def all_phase_ids(self, target: str) -> list[str]:
        """Phase IDs (in YAML order) for a target."""
        return [e["id"] for e in self._by_target.get(target, [])]


_default: PhaseRegistry | None = None


def default() -> PhaseRegistry:
    """Lazy module-level cache. Re-import / restart the process to refresh."""
    global _default
    if _default is None:
        _default = PhaseRegistry.load()
    return _default


def min_valid_time_for(target: str, phase_id: str) -> datetime | None:
    """Convenience: ``default().min_valid_time_for(target, phase_id)``.
    Returns naive UTC datetime (tzinfo stripped) for ergonomic comparison
    against pandas datetime columns that are usually tz-naive too."""
    v = default().min_valid_time_for(target, phase_id)
    if v is None:
        return None
    # Strip tz for pandas-comparison ergonomics; same as the WB side which
    # returns DateTimeKind.Utc and is then formatted into a tz-naive
    # TIMESTAMP SQL literal anyway.
    return v.replace(tzinfo=None) if v.tzinfo else v
