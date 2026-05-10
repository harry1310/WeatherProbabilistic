"""Pre-train sanity gate for the WeatherProbabilistic Python phases (4a
BART, 5a Bayesian logreg). Mirror of WeatherBlend's
``WeatherBlend.Train.Common.RetrainGuard`` so a Python-trained phase's
guard behaviour matches a .NET-trained phase's exactly: same tolerance
bands, same breach reasons, same first-ever-train passing semantics.

Usage from a trainer script::

    from src.retrain_guard import build_check_and_save_versioned, DEFAULTS

    result = build_check_and_save_versioned(
        log,
        version_dir=Path("data/models/precipitation/ea_bellever/v..._phase4a"),
        composite="precipitation/ea_bellever",
        phase="4a",
        version="v..._phase4a",
        rows_train=14000, rows_val=3000, rows_test=3000,
        train_features=X_train,           # (n, k) float ndarray
        feature_names=feature_names,
        label_rates={"ea_bellever": 0.31},
    )
    if not result.passed:
        sys.exit(4)   # workflow webhook fires [ci-fail] retrain-* issue

For 5a (single hierarchical model, overwrite-style bundle), use
``build_check_and_save_singleton`` which compares against whatever the
existing ``training_summary.json`` says was last written and overwrites
on pass.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.training_summary import (
    TrainingSummary,
    build_summary,
    from_json as load_summary,
    to_json as save_summary,
)


# ----- tolerance bands + result shape -----------------------------------


@dataclass
class GuardTolerances:
    """Per-metric tolerance band thresholds. Defaults match the .NET side.

    rows_delta_pct: max relative row-count change (0.30 = ±30%)
    nan_pct_absolute: max absolute change in NaN fraction per feature (0..1)
    label_rate_delta_absolute: max absolute change in label rate per station
    features_effective_delta: max absolute change in featuresEffective
        (0 = any change aborts; bumping above 0 effectively disables the check)
    """

    rows_delta_pct: float = 0.30
    nan_pct_absolute: float = 0.20
    label_rate_delta_absolute: float = 0.10
    features_effective_delta: int = 0


DEFAULTS = GuardTolerances()


@dataclass
class GuardBreach:
    """One field's tolerance breach for structured logging."""

    field: str
    previous: Any
    current: Any
    threshold: Any
    reason: str


@dataclass
class GuardResult:
    """Outcome of a guard check. Caller aborts the bundle write when
    ``passed`` is False.
    """

    passed: bool
    breaches: list[GuardBreach] = field(default_factory=list)
    note: str = ""


# ----- core check -------------------------------------------------------


def check(
    current: TrainingSummary,
    previous: TrainingSummary | None,
    tolerances: GuardTolerances = DEFAULTS,
) -> GuardResult:
    """Compare current against previous; return a GuardResult.

    First-ever-train (``previous is None``) returns ``passed=True`` with
    a "no baseline" note — caller writes the new summary as the baseline
    and proceeds with the bundle write.
    """
    if previous is None:
        return GuardResult(
            passed=True,
            breaches=[],
            note="First-ever train (no previous summary on disk) — guard skipped, current summary becomes the baseline.",
        )

    breaches: list[GuardBreach] = []

    # Row counts: relative ±rows_delta_pct. Skip on zero baseline.
    _check_relative("rowsTrain", previous.RowsTrain, current.RowsTrain,
                    tolerances.rows_delta_pct, breaches)
    _check_relative("rowsVal", previous.RowsVal, current.RowsVal,
                    tolerances.rows_delta_pct, breaches)
    _check_relative("rowsTest", previous.RowsTest, current.RowsTest,
                    tolerances.rows_delta_pct, breaches)

    # FeaturesEffective: absolute (default 0 = any change aborts).
    feat_delta = current.FeaturesEffective - previous.FeaturesEffective
    if abs(feat_delta) > tolerances.features_effective_delta:
        breaches.append(GuardBreach(
            field="featuresEffective",
            previous=previous.FeaturesEffective,
            current=current.FeaturesEffective,
            threshold=tolerances.features_effective_delta,
            reason=(
                f"feature count changed by {feat_delta:+d} (tolerance "
                f"±{tolerances.features_effective_delta}) — a column "
                "appearing or disappearing always signals an upstream "
                "schema shift; abort and inspect"
            ),
        ))

    # Per-feature NaN%: only on intersection (FeaturesEffective check
    # already handles set differences).
    shared_features = set(current.PerFeature.keys()) & set(previous.PerFeature.keys())
    for key in shared_features:
        prev_nan = previous.PerFeature[key].NanPct
        curr_nan = current.PerFeature[key].NanPct
        delta = abs(curr_nan - prev_nan)
        if delta > tolerances.nan_pct_absolute:
            breaches.append(GuardBreach(
                field=f"perFeature.{key}.nanPct",
                previous=prev_nan,
                current=curr_nan,
                threshold=tolerances.nan_pct_absolute,
                reason=(
                    f"NaN fraction changed by {delta:.2f} (tolerance "
                    f"{tolerances.nan_pct_absolute:.2f}) — feature column "
                    "reliability shifted, often signals an upstream model "
                    "dropping out or a feature builder change"
                ),
            ))

    # Per-station label rate: intersection only.
    shared_stations = set(current.LabelRates.keys()) & set(previous.LabelRates.keys())
    for key in shared_stations:
        prev_rate = previous.LabelRates[key]
        curr_rate = current.LabelRates[key]
        delta = abs(curr_rate - prev_rate)
        if delta > tolerances.label_rate_delta_absolute:
            breaches.append(GuardBreach(
                field=f"labelRates.{key}",
                previous=prev_rate,
                current=curr_rate,
                threshold=tolerances.label_rate_delta_absolute,
                reason=(
                    f"label rate changed by {delta:.2f} (tolerance "
                    f"{tolerances.label_rate_delta_absolute:.2f}) — base "
                    "rate shift this large at a single retrain interval is "
                    "unusual; suspect truth-source change or upstream gauge "
                    "issue"
                ),
            ))

    return GuardResult(
        passed=len(breaches) == 0,
        breaches=breaches,
        note=(
            "All bands within tolerance."
            if not breaches
            else f"{len(breaches)} breach(es); abort the bundle write and inspect."
        ),
    )


def _check_relative(
    field_name: str, previous: int, current: int,
    pct_tolerance: float, breaches: list[GuardBreach],
) -> None:
    if previous <= 0:
        return  # undefined relative on zero baseline
    delta = abs(current - previous) / previous
    if delta > pct_tolerance:
        breaches.append(GuardBreach(
            field=field_name,
            previous=previous,
            current=current,
            threshold=pct_tolerance,
            reason=(
                f"row count changed by {delta:.1%} (tolerance "
                f"{pct_tolerance:.1%}) — large drift this size at one "
                "retrain interval signals an upstream collector / refresh "
                "issue"
            ),
        ))


# ----- previous-summary loaders ----------------------------------------


def try_load_previous_summary_versioned(
    parent_dir: Path,
    phase: str,
    exclude_version: str,
) -> TrainingSummary | None:
    """For per-version-subdir bundles (4a layout — one dir per train run).

    Walks ``parent_dir`` for sibling version subdirs (sorted lex-desc =
    newest first since timestamp prefix is fixed-width), skips the
    in-progress one, filters by phase via training_metadata.json, and
    returns the first ``training_summary.json`` it finds.
    """
    parent_dir = Path(parent_dir)
    if not parent_dir.is_dir():
        return None

    candidates = sorted(
        (p for p in parent_dir.iterdir() if p.is_dir() and p.name != exclude_version),
        key=lambda p: p.name,
        reverse=True,
    )

    for ver_dir in candidates:
        # Phase-match via metadata. Skip dirs lacking it (partial writes).
        meta_path = ver_dir / "training_metadata.json"
        if not meta_path.exists():
            continue
        try:
            import json as _json
            meta = _json.loads(meta_path.read_text())
            if meta.get("Phase") != phase:
                continue
        except Exception:
            continue
        summary = load_summary(ver_dir / "training_summary.json")
        if summary is not None:
            return summary
        # Phase matches but no summary on this version (predates Phase 1a) —
        # keep walking back.
    return None


def try_load_previous_summary_singleton(path: Path) -> TrainingSummary | None:
    """For overwrite-style bundles (5a layout — one summary file, gets
    overwritten each run). The "previous" is whatever's currently on
    disk. Returns None on first-ever train when the file doesn't exist.
    """
    return load_summary(Path(path))


# ----- trainer-facing entry points -------------------------------------


def build_check_and_save_versioned(
    log: logging.Logger,
    version_dir: Path,
    composite: str,
    phase: str,
    version: str,
    rows_train: int,
    rows_val: int,
    rows_test: int,
    train_features,  # np.ndarray
    feature_names: list[str],
    label_rates: dict[str, float] | None = None,
    tolerances: GuardTolerances = DEFAULTS,
) -> GuardResult:
    """Versioned-bundle entry point (4a). Builds the summary, finds the
    previous in the parent dir, runs the guard, and on pass writes the
    new summary alongside training_metadata.json.

    Returns the GuardResult so the caller can ``sys.exit(4)`` on fail
    (the orphan version dir stays on disk but never gets promoted —
    R2 sync skips dirs without a training_summary.json).
    """
    version_dir = Path(version_dir)
    parent_dir = version_dir.parent
    return _run_guard(
        log,
        save_path=version_dir / "training_summary.json",
        previous=try_load_previous_summary_versioned(parent_dir, phase, version_dir.name),
        composite=composite, phase=phase, version=version,
        rows_train=rows_train, rows_val=rows_val, rows_test=rows_test,
        train_features=train_features, feature_names=feature_names,
        label_rates=label_rates, tolerances=tolerances,
    )


def build_check_and_save_singleton(
    log: logging.Logger,
    summary_path: Path,
    composite: str,
    phase: str,
    version: str,
    rows_train: int,
    rows_val: int,
    rows_test: int,
    train_features,  # np.ndarray
    feature_names: list[str],
    label_rates: dict[str, float] | None = None,
    tolerances: GuardTolerances = DEFAULTS,
) -> GuardResult:
    """Singleton-bundle entry point (5a). Loads the existing summary at
    ``summary_path`` as the "previous" baseline, runs the guard, and on
    pass overwrites the file with the new summary. On fail, leaves the
    existing file untouched so the next retrain attempt sees the same
    baseline.
    """
    summary_path = Path(summary_path)
    return _run_guard(
        log,
        save_path=summary_path,
        previous=try_load_previous_summary_singleton(summary_path),
        composite=composite, phase=phase, version=version,
        rows_train=rows_train, rows_val=rows_val, rows_test=rows_test,
        train_features=train_features, feature_names=feature_names,
        label_rates=label_rates, tolerances=tolerances,
    )


def _run_guard(
    log: logging.Logger,
    save_path: Path,
    previous: TrainingSummary | None,
    composite: str,
    phase: str,
    version: str,
    rows_train: int,
    rows_val: int,
    rows_test: int,
    train_features,
    feature_names: list[str],
    label_rates: dict[str, float] | None,
    tolerances: GuardTolerances,
) -> GuardResult:
    """Shared body. Builds → checks → saves on pass / preserves on fail."""
    if train_features is None or len(train_features) == 0 or len(feature_names) == 0:
        log.warning(
            "Retrain guard skipped for %s/%s %s — empty train slice (no summary written, no guard check).",
            composite, phase, version,
        )
        return GuardResult(
            passed=True,
            breaches=[],
            note="Empty train slice — guard skipped, no summary written.",
        )

    summary = build_summary(
        composite=composite, phase=phase, version=version,
        rows_train=rows_train, rows_val=rows_val, rows_test=rows_test,
        train_features=train_features, feature_names=feature_names,
        label_rates=label_rates,
    )

    result = check(summary, previous, tolerances)

    if not result.passed:
        log.error(
            "Retrain guard FAIL for %s/%s %s: %s",
            composite, phase, version, result.note,
        )
        for breach in result.breaches:
            log.error(
                "  %s: previous=%s, current=%s, threshold=%s — %s",
                breach.field, breach.previous, breach.current,
                breach.threshold, breach.reason,
            )
        return result

    save_summary(summary, save_path)
    log.info(
        "Retrain guard PASS for %s/%s %s — train=%d val=%d test=%d features=%d%s",
        composite, phase, version,
        summary.RowsTrain, summary.RowsVal, summary.RowsTest, summary.FeaturesEffective,
        " (no previous baseline; this becomes the new baseline)" if previous is None else "",
    )
    return result
