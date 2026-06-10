#!/usr/bin/env bash
#
# WeatherProbabilistic/scripts/sync_train_data.sh
#
# Sibling of WeatherBlend's scripts/sync_train_data.sh — same contract,
# scoped to the python (impl=python) train phases this repo owns:
#   - 4a             per-cell BART precipitation
#   - 3f             NGBoost-LogNormal rainfall amount
#   - wind_mvn       PyTorch bivariate-normal wind direction
#   - wind_speed_lgb quantile-LGB + cross-conformal CQR wind speed
#
# Single source of truth for "which data trees does each python phase
# consume?". Called by:
#   - .github/workflows/retrain-python.yml (Sunday + ad-hoc retrains)
#   - tests/test_smoke_*.py (via subprocess against a local fake-R2 dir)
#
# Adding a new python phase = update phases.yaml AND the case table
# below. Both the workflow + smoke pick up the new pull declaration
# automatically. Missing a declaration here = smoke fails at PR time
# rather than "Loaded 0 rows" in a Sunday retrain.
#
# Usage:
#   sync_train_data.sh <location> <phases-csv>
#
# Required env:
#   R2_SOURCE   "r2:<bucket>" in production, or a local fake-R2 path in smoke.
# Optional env:
#   LOCAL_ROOT  Destination root (default: ../WeatherBlend/data — the
#               production layout where this repo runs alongside the WB
#               checkout on the runner).
#   MODE        train (default) | predict. train = feature trees + truth
#               labels + promote-target MANIFEST. predict = feature trees +
#               the latest saved model bundle per cell, no labels/MANIFEST.
#               Predict workflows set MODE=predict so they pull via the SAME
#               declaration as train (no more hand-rolled rclone that drifts).
#
# Exit codes:
#   0  success
#   2  bad args
#   3  unknown phase id
#   *  rclone exit code
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <location> <phases-csv>"
  echo "  e.g. $0 bonehill_rocks 4a,3f,wind_mvn"
  exit 2
fi

location="$1"
phases_csv="$2"
: "${R2_SOURCE:?R2_SOURCE env var required (e.g. r2:weatherblend or /tmp/fake-r2)}"
LOCAL_ROOT="${LOCAL_ROOT:-../WeatherBlend/data}"

# MODE selects the pull set (2026-05-31). Default `train` keeps every existing
# caller (retrain-python.yml + train smoke) unchanged. `predict` pulls the
# feature trees a phase reads at PREDICT time PLUS its saved model bundle, and
# skips truth-as-label trees + the promote-target MANIFEST. Same single source
# of truth drives both, so predict workflows can stop hand-rolling rclone (the
# divergence that caused the 2026-05-31 predict-4a failures).
MODE="${MODE:-train}"
case "$MODE" in
  train|predict) ;;
  *) echo "::error::MODE must be 'train' or 'predict' (got '$MODE')"; exit 2 ;;
esac

mkdir -p "$LOCAL_ROOT"
cd "$LOCAL_ROOT"

# ----------------------------------------------------------------------------
# PER-PHASE NEEDS TABLE
# ----------------------------------------------------------------------------
# Each python phase declares which trees it consumes. Same shape as the
# WB script — keeps the two repos' pull contracts in lockstep when a
# phase grows a new dependency.
#
# WP's forecast pull is FILTERED to the seven Open-Meteo seamless models
# (the python trainers only consume those; the raw S3 archive models
# are blender inputs only). The filter list is hard-coded inside the
# pull below — see PYTHON_NWP_MODELS.

need_forecasts=0
need_rainfall=0
need_midas=0
need_orographic=0
need_precip_manifest=0
need_rainfall_amount_manifest=0
need_wind_direction_manifest=0
need_wind_manifest=0
# Predict-mode: name the model-bundle tree (target) + phase-version glob to
# pull the latest bundle per cell for. Empty = no bundle pull.
bundle_precipitation=""
bundle_rainfall_amount=""
bundle_wind_direction=""
bundle_wind=""

IFS=',' read -ra phases <<< "$phases_csv"
for p in "${phases[@]}"; do
  case "$p" in
    # Per-cell BART precipitation. Needs forecast tree + rainfall truth.
    # Precipitation MANIFEST is critical — train_4a promotes new versions
    # into the EXISTING Active list (preserving WB-side .NET phase
    # entries). Without the MANIFEST pull, promote starts blank and
    # clobbers Bonehill on the next push.
    # static/orographic pull added 2026-05-30 alongside the rich+oro
    # feature swap — compose_v1_terrain_block reads
    # data/static/orographic/{station_slug}.json for the 9-feature
    # terrain block. Without this pull the Sunday retrain would
    # FileNotFoundError on the first cell.
    # Per-cell BART precipitation. Rich+oro builder reads forecasts +
    # orographic + antecedent EA rainfall — at BOTH train and predict
    # (rainfall is the label AND the ea_rain_prev_* features). Train also
    # needs the precip MANIFEST (train_4a promotes into the existing Active
    # list); predict instead needs the latest phase4a bundle per cell.
    4a)
      need_forecasts=1; need_orographic=1; need_rainfall=1
      if [ "$MODE" = "train" ]; then need_precip_manifest=1
      else bundle_precipitation="phase4a"; fi ;;

    # NGBoost-LogNormal rainfall amount. Train: same data as 4a + the
    # rainfall_amount MANIFEST. Predict: forecasts + latest phase3f bundle.
    # NOTE: 3f predict ALSO needs the bound 3a champion PREDICTIONS, but
    # the version is pinned in the 3f bundle's training_metadata at runtime,
    # so that pull stays inline in predict-3f.yml (can't be declared here).
    3f)
      need_forecasts=1
      if [ "$MODE" = "train" ]; then
        need_rainfall=1; need_precip_manifest=1; need_rainfall_amount_manifest=1
      else
        bundle_rainfall_amount="phase3f"
      fi ;;

    # Phase 2 wind_mvn (PyTorch MLP MVN). Forecasts + static orographic at
    # BOTH modes. Train: Dunkeswell MIDAS truth (label) + wind_direction
    # MANIFEST. Predict: the latest wind_mvn bundle.
    wind_mvn)
      need_forecasts=1; need_orographic=1
      if [ "$MODE" = "train" ]; then need_midas=1; need_wind_direction_manifest=1
      else bundle_wind_direction="wind_mvn"; fi ;;

    # wind_speed_lgb (quantile-LGB + cross-conformal CQR, the Python port of
    # the .NET WindSpeedLgbBlender — Option B cutover 2026-06-10). Same
    # feature recipe as wind_mvn (train_wind_mvn.build_features): forecasts +
    # static orographic at BOTH modes. Train: Dunkeswell MIDAS truth (label)
    # + the wind MANIFEST (train promotes as challenger into the EXISTING
    # Active list, alongside the .NET-trained `wind` champion entries).
    # Predict: the latest *_wind_speed_lgb bundle under models/wind/{loc}.
    wind_speed_lgb)
      need_forecasts=1; need_orographic=1
      if [ "$MODE" = "train" ]; then need_midas=1; need_wind_manifest=1
      else bundle_wind="wind_speed_lgb"; fi ;;

    *)
      echo "::error::sync_train_data: unknown python phase '$p' — add it to scripts/sync_train_data.sh."
      exit 3 ;;
  esac
done

# ----------------------------------------------------------------------------
# Pulls — WP-specific shapes
# ----------------------------------------------------------------------------
# Union of every Open-Meteo seamless model any python trainer consumes.
# Per-trainer NWPS lists (each trainer filters down to its own subset):
#   - train_4a.py  / train_3f.py:    7 — incl mf + aifs, NO ukmo
#   - train_wind_mvn.py:             6 — incl ukmo, NO mf, NO aifs
# Sync pulls the UNION (8) so every trainer finds its inputs. Each
# trainer's own NWPS filter narrows further at feature-build time, so
# over-pulling is cheap and load-bearing — under-pulling silently drops
# columns from the pivot. The legacy inline list in retrain-python.yml
# (replaced by this script 2026-05-28) omitted ukmo_seamless, which
# would have silently shrunk wind_mvn from 6→5 NWPs in production.
PYTHON_NWP_MODELS=(
  gfs_seamless ecmwf_ifs025 icon_seamless meteofrance_seamless
  gem_seamless ecmwf_aifs025_single jma_seamless ukmo_seamless
)

# Forecast tree: per-NWP filter (matches the legacy inline loop). Each
# rclone copy is its own tree pull so the include/exclude semantics
# stay simple.
if [ "$need_forecasts" -gt 0 ]; then
  for m in "${PYTHON_NWP_MODELS[@]}"; do
    echo "::group::sync_train_data: pull forecasts/location=$location/model=$m"
    mkdir -p "forecasts/location=$location/model=$m"
    rclone copy "${R2_SOURCE%/}/data/forecasts/location=$location/model=$m" \
      "forecasts/location=$location/model=$m" \
      --fast-list --transfers 16 --checkers 32 --s3-no-check-bucket \
      || echo "::warning::no rows for model=$m at location=$location (may be expected on first runs)"
    echo "::endgroup::"
  done
fi

if [ "$need_rainfall" -gt 0 ]; then
  echo "::group::sync_train_data: pull truth/rainfall"
  mkdir -p truth/rainfall
  rclone copy "${R2_SOURCE%/}/data/truth/rainfall" truth/rainfall \
    --fast-list --transfers 16 --checkers 32 --s3-no-check-bucket
  echo "::endgroup::"
fi

if [ "$need_midas" -gt 0 ]; then
  echo "::group::sync_train_data: pull truth/midas/raw"
  mkdir -p truth/midas/raw
  rclone copy "${R2_SOURCE%/}/data/truth/midas/raw" truth/midas/raw \
    --include 'midas-open*_01383_*.csv' \
    --fast-list --transfers 4 --checkers 8 --s3-no-check-bucket
  echo "::endgroup::"
fi

if [ "$need_orographic" -gt 0 ]; then
  echo "::group::sync_train_data: pull static/orographic"
  mkdir -p static/orographic
  rclone copy "${R2_SOURCE%/}/data/static/orographic" static/orographic \
    --include '*.json' --checkers 4 --s3-no-check-bucket
  echo "::endgroup::"
fi

# Manifests are single-file copytos. Soft on missing source — first
# Sunday for a fresh target the manifest may not exist yet; promote
# helpers create it.
copyto_manifest() {
  local target="$1"
  echo "::group::sync_train_data: pull models/$target/MANIFEST.json"
  mkdir -p "models/$target"
  rclone copyto "${R2_SOURCE%/}/data/models/$target/MANIFEST.json" \
    "models/$target/MANIFEST.json" --s3-no-check-bucket \
    || echo "::notice::No $target MANIFEST.json on R2 yet — train_$target will create one."
  echo "::endgroup::"
}

if [ "$need_precip_manifest" -gt 0 ]; then
  copyto_manifest "precipitation"
fi
if [ "$need_rainfall_amount_manifest" -gt 0 ]; then
  copyto_manifest "rainfall_amount"
fi
if [ "$need_wind_direction_manifest" -gt 0 ]; then
  copyto_manifest "wind_direction"
fi
if [ "$need_wind_manifest" -gt 0 ]; then
  copyto_manifest "wind"
fi

# Predict-mode model bundles. Pull the LATEST bundle matching the phase-version
# glob per cell (station or location) under models/<target>/. Mirrors the
# hand-rolled loops the predict workflows used (lexicographic max of the
# phase-tagged version dirs) — now declared once so predict + smoke agree.
pull_latest_bundle() {
  local target="$1" phase_glob="$2"
  echo "::group::sync_train_data: pull latest *${phase_glob}* bundles under models/${target}"
  mkdir -p "models/${target}"
  local cell latest
  for cell in $(rclone lsf "${R2_SOURCE%/}/data/models/${target}/" --dirs-only 2>/dev/null); do
    cell="${cell%/}"
    # `|| true`: a cell with NO bundle of this phase is normal — e.g.
    # models/precipitation holds ALL locations' gauges but only Bonehill has
    # phase4a. Without it, grep's no-match exit 1 + `set -euo pipefail` kills
    # the whole sync (the 2026-06-10 predict-4a failure, run 27264913136 —
    # died on ea_chards_snowdon_hill, the first 4a-less Membury gauge; the
    # pre-MODE=predict hand-rolled loops survived because workflow shells
    # have -e but NOT pipefail). The empty-`latest` continue below is the
    # intended skip path.
    latest=$(rclone lsf "${R2_SOURCE%/}/data/models/${target}/${cell}/" --dirs-only 2>/dev/null \
      | grep "$phase_glob" | sort | tail -1 || true)
    latest="${latest%/}"
    [ -z "$latest" ] && continue
    echo "  ${target}/${cell}: ${latest}"
    rclone copy "${R2_SOURCE%/}/data/models/${target}/${cell}/${latest}" \
      "models/${target}/${cell}/${latest}" \
      --fast-list --transfers 16 --checkers 32 --s3-no-check-bucket
  done
  echo "::endgroup::"
}

[ -n "$bundle_precipitation" ]    && pull_latest_bundle "precipitation"    "$bundle_precipitation"
[ -n "$bundle_rainfall_amount" ]  && pull_latest_bundle "rainfall_amount"  "$bundle_rainfall_amount"
[ -n "$bundle_wind_direction" ]   && pull_latest_bundle "wind_direction"   "$bundle_wind_direction"
[ -n "$bundle_wind" ]             && pull_latest_bundle "wind"             "$bundle_wind"

echo "sync_train_data: done (mode=$MODE)."
echo "  forecasts=$need_forecasts rainfall=$need_rainfall midas=$need_midas orographic=$need_orographic"
echo "  manifests: precip=$need_precip_manifest rainfall_amount=$need_rainfall_amount_manifest wind_direction=$need_wind_direction_manifest wind=$need_wind_manifest"
echo "  bundles: precip='$bundle_precipitation' rainfall_amount='$bundle_rainfall_amount' wind_direction='$bundle_wind_direction' wind='$bundle_wind'"
