"""Shared pytest configuration.

Registers the ``parity`` marker and makes parity-marked tests OPT-IN:
they re-build the full multi-year feature set against the live data tree
(~20 min — the cost is whole-tree DuckDB scans, not the comparison), and
they are a SPEC-CHANGE contract, not a per-PR regression canary. Run them
explicitly after touching the rich-oro feature math on either side:

    # 1. regenerate the C# reference against the current tree state
    cd ../WeatherBlend && dotnet run --project src/WeatherBlend -- dump-oro-features
    # 2. compare (same tree state — don't let a noon refresh in between)
    pytest tests/test_rich_oro_python_vs_csharp.py --run-parity
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-parity", action="store_true", default=False,
        help="run the (slow, full-tree) C#↔Python feature parity tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parity: slow C#↔Python bit-parity contract test (opt-in via --run-parity)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-parity"):
        return
    skip = pytest.mark.skip(
        reason="parity contract test is opt-in (--run-parity) — run after "
               "any rich-oro feature-spec change, with a fresh dump-oro-features fixture")
    for item in items:
        if "parity" in item.keywords:
            item.add_marker(skip)
