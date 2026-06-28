"""Station-slug + pool-only handling between WeatherProbabilistic and WeatherBlend.

Regression guards for the 2026-06-28 4a-retrain break. 4a is a PER-STATION, Bonehill-only
model that loads each gauge's orographic static. Two config facts it depends on:

  * Pool-only gauges (Princetown, Manaton) feed ONLY the WeatherBlend C# 3o pool as training
    augmentation — they are never a per-station product, so the WP config must EXCLUDE them.
    Including them made 4a try to build a model + load an oro static for Manaton, whose
    WeatherLink slug (wl_manaton) the parser mis-spelled as ea_manaton -> FileNotFoundError.
  * WeatherLink *product* gauges (Lands End at Sennen) need a wl_ slug, not ea_.
"""
from pathlib import Path

import pytest

from src.weatherblend_config import LOCATIONS, STATIONS_BY_LOCATION, _slug


def test_slug_is_source_aware():
    assert _slug({"name": "Manaton", "source": "weatherlink"}) == "wl_manaton"
    assert _slug({"name": "Lands End", "source": "WeatherLink"}) == "wl_lands_end"
    assert _slug({"name": "Bellever Dartmoor"}) == "ea_bellever_dartmoor"
    assert _slug({"name": "Princetown", "source": "ea"}) == "ea_princetown"


def test_pool_only_gauges_are_excluded_from_products():
    # Princetown + Manaton are pool-only -> they must NOT appear as per-station WP products
    # (any source-prefix), while the real product gauges remain.
    bonehill = STATIONS_BY_LOCATION.get("bonehill_rocks", ())
    for s in ("wl_manaton", "ea_manaton", "ea_princetown"):
        assert s not in bonehill, f"pool-only gauge {s} leaked into per-station products"
    for s in ("ea_bellever_dartmoor", "ea_bovey_tracey", "ea_dartmoor_nr_hexworthy"):
        assert s in bonehill, f"product gauge {s} missing from bonehill"


def test_weatherlink_product_gauge_keeps_wl_slug():
    # Lands End is a WeatherLink Sennen gauge that IS a product (not pool-only) -> wl_ slug.
    sennen = STATIONS_BY_LOCATION.get("sennen_cove", ())
    assert "wl_lands_end" in sennen and "ea_lands_end" not in sennen


def test_oro_location_gauges_each_have_a_static():
    # Per-location all-or-nothing: an oro location (Bonehill) must have a {slug}.json for every
    # gauge 4a loads; a non-oro location (flat-coastal Sennen) has none and is skipped.
    try:
        from src.data import WEATHERBLEND_DATA_ROOT
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"src.data not importable: {e}")
    oro = Path(WEATHERBLEND_DATA_ROOT) / "static" / "orographic"
    if not oro.is_dir():
        pytest.skip(f"orographic dir not present ({oro})")
    for loc in LOCATIONS:
        present = {st.slug: (oro / f"{st.slug}.json").is_file() for st in loc.stations}
        if not any(present.values()):
            continue  # non-oro location
        missing = [s for s, ok in present.items() if not ok]
        assert not missing, f"{loc.name}: oro gauges with no static (slug<->file mismatch): {missing}"
