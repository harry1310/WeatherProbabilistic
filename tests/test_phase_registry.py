"""Regression tests for ``src.phase_registry`` — the Python-side mirror
of WeatherBlend's ``Models/PhaseRegistry.cs``.

Pins:
  * ``minValidTime`` ISO-date strings parse to UTC datetimes (tz-naive
    for the convenience helper).
  * Absent / unknown phases return ``None`` (= no cutoff at the call site).
  * Invalid date strings raise with target+phase context.
  * The shipped ``phases.yaml`` carries ``minValidTime: "2024-01-01"``
    on 4a + 5a (the Python-side cutoff phases) — regression for the
    2026-05-26 JMA-extension data-drift decision.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.phase_registry import PhaseRegistry, min_valid_time_for  # noqa: E402


def _write(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "phases.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return p


def test_min_valid_time_iso_date_parses_to_utc(tmp_path):
    yaml_text = """
targets:
  precipitation:
    phases:
      - id: "4a"
        role: challenger
        impl: python
        minValidTime: "2024-01-01"
"""
    reg = PhaseRegistry.load(_write(tmp_path, yaml_text))
    got = reg.min_valid_time_for("precipitation", "4a")
    assert got == datetime(2024, 1, 1, tzinfo=__import__("datetime").timezone.utc)


def test_min_valid_time_absent_returns_none(tmp_path):
    yaml_text = """
targets:
  rainfall_amount:
    phases:
      - id: "3f"
        role: champion
        impl: python
"""
    reg = PhaseRegistry.load(_write(tmp_path, yaml_text))
    assert reg.min_valid_time_for("rainfall_amount", "3f") is None


def test_min_valid_time_unknown_phase_returns_none(tmp_path):
    yaml_text = """
targets:
  precipitation:
    phases:
      - id: "4a"
        role: challenger
        impl: python
        minValidTime: "2024-01-01"
"""
    reg = PhaseRegistry.load(_write(tmp_path, yaml_text))
    # Unknown phase is "no cutoff", same semantics as absent field.
    assert reg.min_valid_time_for("precipitation", "5a") is None
    assert reg.min_valid_time_for("nonexistent_target", "anything") is None


def test_min_valid_time_invalid_raises_with_context(tmp_path):
    yaml_text = """
targets:
  precipitation:
    phases:
      - id: "4a"
        role: challenger
        impl: python
        minValidTime: "not a date"
"""
    with pytest.raises(ValueError, match="precipitation.*4a.*not a date"):
        PhaseRegistry.load(_write(tmp_path, yaml_text))


def test_module_helper_strips_tzinfo_for_pandas_ergonomics(monkeypatch, tmp_path):
    yaml_text = """
targets:
  precipitation:
    phases:
      - id: "4a"
        role: challenger
        impl: python
        minValidTime: "2024-01-01"
"""
    p = _write(tmp_path, yaml_text)
    monkeypatch.setenv("WEATHERBLEND_PHASES", str(p))
    # Drop module-level cache so the env override is picked up.
    import src.phase_registry as pr
    pr._default = None  # type: ignore[attr-defined]

    got = min_valid_time_for("precipitation", "4a")
    assert got == datetime(2024, 1, 1)
    assert got.tzinfo is None, "module helper returns naive UTC for pandas comparison ergonomics"


def test_production_phases_yaml_pins_2024_cutoff_on_python_phases():
    """Sibling-checkout fallback: load the actual WB-side phases.yaml.
    Locks the 2026-05-26 decision in the shipped YAML for 4a + 5a.
    """
    # Sibling-checkout path the loader falls back to in dev.
    sibling = (Path(__file__).resolve().parent.parent.parent
               / "WeatherBlend" / "src" / "WeatherBlend" / "Config" / "phases.yaml")
    if not sibling.is_file():
        pytest.skip("WB sibling checkout not present; production-yaml test skipped.")
    reg = PhaseRegistry.load(sibling)
    expected = datetime(2024, 1, 1, tzinfo=__import__("datetime").timezone.utc)
    assert reg.min_valid_time_for("precipitation", "4a") == expected
    assert reg.min_valid_time_for("precipitation", "5a") == expected
    # 3f is Membury-only, data starts 2024 anyway → no cutoff.
    assert reg.min_valid_time_for("rainfall_amount", "3f") is None
