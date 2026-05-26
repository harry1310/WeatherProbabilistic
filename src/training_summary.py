"""Pre-train data sanity sidecar — captured at fit time and persisted next
to the model artefact as ``training_summary.json``. Read at the start of
the next retrain by :mod:`retrain_guard` to detect upstream data shifts
before a bad model gets written.

Mirrors the .NET ``WeatherBlend.Train.Common.TrainingSummary`` schema
exactly (PascalCase keys, same field shape) so a 4a / 3f summary is
interchangeable with a 2b / 3a one. The .NET serialiser uses
``System.Text.Json`` defaults which write PascalCase property names;
this module emits the same so a Python-trained phase's summary parses
cleanly under the .NET ``ModelArtifact.TryLoadTrainingSummary`` if
anything ever wants to cross-load.

Per-feature stats are computed on the **TRAIN slice only** (val/test
are evaluation slices; shifts there mean a different problem, not an
upstream data issue). NaN%/mean/std/p01/p99 use the non-NaN subset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA_VERSION = "1"


@dataclass
class FeatureStats:
    """Per-feature train-slice descriptive stats. NaN-aware: mean/std and
    quantiles are computed over the non-NaN subset; ``NanPct`` is the
    fraction that WAS NaN. Field names are PascalCase to match the .NET
    on-disk JSON shape."""

    NanPct: float = 0.0
    Mean: float = 0.0
    Std: float = 0.0
    P01: float = 0.0
    P99: float = 0.0


@dataclass
class TrainingSummary:
    """Mirrors WeatherBlend.Train.Common.TrainingSummary 1:1.

    Composite is the same key the site's ``ModelSummary`` uses so a
    Python-side guard's ``try_load_previous_summary`` can resolve the
    same lineage the .NET trainer would.
    """

    SchemaVersion: str = SCHEMA_VERSION
    Composite: str = ""
    Phase: str = ""
    Version: str = ""
    # Configured location whose NWP fed the training data (Phase A
    # multi-location safety, 2026-05-12). Mirrors the .NET TrainingSummary
    # field — required post-backfill. ``from_json`` raises on a JSON file
    # that lacks the field; the in-memory default of "" is for fixtures /
    # legacy callers that build summaries by hand without it, which the
    # guard's same-empty check tolerates without flagging a breach.
    LocationName: str = ""
    # ISO 8601 UTC string; matches the .NET DateTime serialisation
    # convention (which uses "yyyy-MM-ddTHH:mm:ss.fffffffZ"). We use
    # microsecond precision since Python's datetime is limited to that.
    ComputedAtUtc: str = ""
    RowsTrain: int = 0
    RowsVal: int = 0
    RowsTest: int = 0
    FeaturesEffective: int = 0
    PerFeature: dict[str, FeatureStats] = field(default_factory=dict)
    LabelRates: dict[str, float] = field(default_factory=dict)


def compute_feature_stats(
    X: np.ndarray,
    feature_names: list[str],
) -> dict[str, FeatureStats]:
    """Per-column NaN%, mean, std, P01, P99 over ``X[:, j]`` for each j.

    Uses NumPy's ``nanmean`` / ``nanstd`` / ``nanquantile`` family so
    NaN rows are dropped from each statistic without distorting the
    mean. Quantiles use linear interpolation (``method="linear"`` =
    R-7, matching the .NET ``Quantile`` implementation).

    A column that's 100% NaN gets ``{NanPct=1.0, Mean=0, Std=0, P01=0,
    P99=0}`` — same sentinel the .NET side emits, so the guard's
    "feature appeared/disappeared" check (FeaturesEffective delta)
    fires while per-feature checks no-op.

    Parameters
    ----------
    X : (n_rows, n_features) array — the TRAIN slice features as
        floats. NaN treated as missing.
    feature_names : column names; length must equal ``X.shape[1]``.
    """
    n_rows, n_cols = X.shape
    if len(feature_names) != n_cols:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X has {n_cols} columns"
        )

    out: dict[str, FeatureStats] = {}
    for j, name in enumerate(feature_names):
        col = X[:, j].astype(np.float64, copy=False)
        nan_mask = np.isnan(col)
        nan_pct = float(nan_mask.mean()) if n_rows > 0 else 0.0
        n_non_nan = int(np.count_nonzero(~nan_mask))
        if n_non_nan == 0:
            out[name] = FeatureStats(NanPct=nan_pct, Mean=0.0, Std=0.0, P01=0.0, P99=0.0)
            continue
        finite = col[~nan_mask]
        # ddof=1 mirrors the .NET Welford variance (which divides by n-1).
        mean = float(np.mean(finite))
        std = float(np.std(finite, ddof=1)) if n_non_nan > 1 else 0.0
        p01 = float(np.quantile(finite, 0.01, method="linear"))
        p99 = float(np.quantile(finite, 0.99, method="linear"))
        out[name] = FeatureStats(NanPct=nan_pct, Mean=mean, Std=std, P01=p01, P99=p99)
    return out


def build_summary(
    composite: str,
    phase: str,
    version: str,
    rows_train: int,
    rows_val: int,
    rows_test: int,
    train_features: np.ndarray,
    feature_names: list[str],
    label_rates: Mapping[str, float] | None = None,
    computed_at_utc: datetime | None = None,
    # Configured location whose NWP fed the training data; pinned at train
    # time so the guard can refuse a retrain that swapped the location
    # underneath the manifest slot (Phase A safety, 2026-05-12). Optional
    # during the backfill window — None means "legacy summary, location
    # unknown" and the guard skips the LocationName check with a note.
    location_name: str | None = None,
) -> TrainingSummary:
    """Compose a complete TrainingSummary. Convenience wrapper that ties
    the per-feature stats to the metadata fields. Caller is responsible
    for the train/val/test row counts.
    """
    when = computed_at_utc or datetime.now(timezone.utc)
    return TrainingSummary(
        SchemaVersion=SCHEMA_VERSION,
        Composite=composite,
        Phase=phase,
        Version=version,
        LocationName=location_name,
        ComputedAtUtc=when.isoformat(),
        RowsTrain=rows_train,
        RowsVal=rows_val,
        RowsTest=rows_test,
        FeaturesEffective=len(feature_names),
        PerFeature=compute_feature_stats(train_features, feature_names),
        LabelRates=dict(label_rates) if label_rates else {},
    )


def to_json(summary: TrainingSummary, path: Path | str) -> None:
    """Serialise to disk. PascalCase keys matching the .NET writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_dict(summary)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def from_json(path: Path | str) -> TrainingSummary | None:
    """Load from disk; returns None if the file doesn't exist. Tolerates
    extra fields (forwards-compat with future schema bumps).

    Legacy handling (added 2026-05-25): if the file exists but its
    top-level dict is missing the LocationName *key entirely*, treat as
    a pre-2026-05-12 write — warn loudly and return None so the caller
    skips it as if absent. First successful retrain after this overwrites
    with a current-schema summary, restoring normal baseline-checking.

    The strict ValueError in `_from_dict` still fires when LocationName
    *is* present but blank/whitespace — that's a real corrupt write, not
    a legacy file, and merits the loud failure the original tightening
    intended.
    """
    import logging
    path = Path(path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if "LocationName" not in raw:
        logging.warning(
            "Ignoring legacy training_summary at %s — pre-2026-05-12 schema, "
            "no 'LocationName' key. Treating as no-previous-baseline; the "
            "next successful retrain writes a current-schema summary. Delete "
            "this file from R2 at your leisure to silence the warning.",
            path,
        )
        return None
    return _from_dict(raw)


def _to_dict(summary: TrainingSummary) -> dict:
    """asdict() preserves PascalCase since the dataclass field names ARE
    PascalCase — Python dataclasses round-trip the field names verbatim."""
    return asdict(summary)


def _from_dict(raw: dict) -> TrainingSummary:
    # Phase A tightening (Task #21): LocationName is required at load time.
    # A summary on disk without it is a corrupt/incomplete write — refusing
    # to load is loud and obvious, vs the warn-then-fallback path which
    # silently let pre-backfill bundles through.
    if "LocationName" not in raw or not str(raw.get("LocationName") or "").strip():
        raise ValueError(
            "training_summary.json missing required field 'LocationName' — "
            "every summary written post-2026-05-12 backfill carries it. "
            "Drop the file and let the next retrain produce a fresh baseline.",
        )
    per_feature = {
        k: FeatureStats(
            NanPct=v.get("NanPct", 0.0),
            Mean=v.get("Mean", 0.0),
            Std=v.get("Std", 0.0),
            P01=v.get("P01", 0.0),
            P99=v.get("P99", 0.0),
        )
        for k, v in (raw.get("PerFeature") or {}).items()
    }
    return TrainingSummary(
        SchemaVersion=raw.get("SchemaVersion", SCHEMA_VERSION),
        Composite=raw.get("Composite", ""),
        Phase=raw.get("Phase", ""),
        Version=raw.get("Version", ""),
        LocationName=raw["LocationName"],
        ComputedAtUtc=raw.get("ComputedAtUtc", ""),
        RowsTrain=raw.get("RowsTrain", 0),
        RowsVal=raw.get("RowsVal", 0),
        RowsTest=raw.get("RowsTest", 0),
        FeaturesEffective=raw.get("FeaturesEffective", 0),
        PerFeature=per_feature,
        LabelRates=dict(raw.get("LabelRates") or {}),
    )
