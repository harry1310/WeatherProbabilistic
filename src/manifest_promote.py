"""Promote a freshly-trained per-station bundle into WeatherBlend's
``MANIFEST.json`` Active list (and optionally Current) — Python mirror of
the .NET ``ModelArtifact.PromoteStationVersionAsChampion / AsChallenger``
helpers in ``src/WeatherBlend/Train/ModelArtifact.cs``.

Why this exists: every WB-side trainer (2b/2c/2d/3a/3c/3d/3b/3p/element)
calls a Promote helper at the end of training to update MANIFEST.json so
the new version surfaces in the site's Active list. The Python-side
trainer (`scripts/train_4a.py` for per-cell BART) had no equivalent — it
wrote bundle files to R2 but never touched the manifest. Result: predict
workflows scored the new bundles, but the site only renders versions in
Active, so freshly retrained 4a fell off the site every Sunday and
stayed off until someone hand-merged the manifest. Symptom (verified
2026-05-12 ~00:30 UTC): "no 4a predictions on the site since midday
May 10" — the per-cell 4a refactor minted a new bundle name; nothing
promoted it.

Usage::

    from manifest_promote import promote_station_version
    promote_station_version(
        models_root,                                 # Path to data/models/precipitation parent
        target="precipitation",
        station_slug="ea_bellever_dartmoor",
        version="v2026-05-10_220841_phase4a",
        phase_tag="phase4a",                         # used to strip stale entries of SAME phase
        role="challenger",                           # "champion" sets Current too
    )

The function is idempotent — re-promoting the same version is a no-op
beyond a refreshed write timestamp.

The push to R2 happens in the workflow (retrain-python.yml's "Push
MANIFEST.json to R2" step). Local-only test runs can stop after this
helper has updated the on-disk MANIFEST.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


def promote_station_version(
    models_root: Path,
    target: str,
    station_slug: str,
    version: str,
    phase_tag: str,
    role: Literal["champion", "challenger"] = "challenger",
) -> dict:
    """Update ``models_root/{target}/MANIFEST.json`` to mark
    ``version`` as Active for ``station_slug``.

    Returns the resulting StationEntry dict so the caller can log + assert.
    Creates the manifest file (with empty top-level + a single station
    entry) if it doesn't exist yet — but in production the manifest is
    always present (created by the first .NET train of the same target).
    """
    manifest_path = Path(models_root) / target / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "Target": target, "Current": "", "Versions": [],
            "Active": [], "ChampionByLead": {}, "Stations": {},
        }

    stations = manifest.setdefault("Stations", {})
    s = stations.setdefault(station_slug, {
        "Current": "", "Versions": [], "Active": [], "ChampionByLead": {},
    })

    if version not in s["Versions"]:
        s["Versions"].append(version)

    # Strip any prior entries for the SAME phase from Active so we don't
    # accumulate stale versions (e.g. last week's 4a hanging next to this
    # week's 4a). Mirrors the .NET PromoteStationVersionAs* behaviour.
    s["Active"] = [v for v in s["Active"] if phase_tag not in v]
    s["Active"].append(version)

    if role == "champion":
        s["Current"] = version

    # Write back. Use a temp-file + rename so a Ctrl-C mid-write doesn't
    # leave a half-truncated manifest (the production sequence pulls the
    # manifest from R2, mutates it, pushes back; a corrupted local file
    # would clobber R2 if the push step doesn't notice).
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    tmp_path.replace(manifest_path)

    return s
