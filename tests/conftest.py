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
    config.addinivalue_line(
        "markers",
        "slow: model-training integration smoke (every test_smoke_* test is "
        "auto-tagged). Default `pytest tests/` still runs them; for the fast "
        "inner loop run `pytest -m \"not slow\"`.",
    )


def pytest_collection_modifyitems(config, items):
    # Auto-tag every test in a test_smoke_*.py module as `slow` so the inner
    # loop can `-m "not slow"` without dropping coverage from a plain run.
    # These train real models (NGBoost / BART / PyTorch MVN / LightGBM CQR) end
    # to end — they catch integration wiring bugs, so they stay ON by default
    # and are the pre-push gate, mirroring WeatherBlend's [Category=Smoke] split.
    for item in items:
        # nodeid starts with the file path, e.g. "tests/test_smoke_3f.py::..."
        # — robust whether or not tests/ is an importable package.
        if "test_smoke_" in item.nodeid.split("::", 1)[0].rsplit("/", 1)[-1]:
            item.add_marker(pytest.mark.slow)

    if config.getoption("--run-parity"):
        return
    skip = pytest.mark.skip(
        reason="parity contract test is opt-in (--run-parity) — run after "
               "any rich-oro feature-spec change, with a fresh dump-oro-features fixture")
    for item in items:
        if "parity" in item.keywords:
            item.add_marker(skip)
