#!/usr/bin/env bash
#
# WeatherProbabilistic/scripts/pull_3a_predictions.sh
#
# Pull the bound 3a (stage-1) prediction partitions predict_3f.py reads.
# predict_3f resolves precip_3a_version from each 3f bundle's
# training_metadata, so the pull is deliberately broad: for every station
# that has a rainfall_amount bundle, the last 3 UNSUFFIXED (= 3a) version
# dirs' anchor-day partition. Smallest cheap pull that always contains the
# stamped version; predict_3f errors loudly if its version is missing.
#
# Extracted from predict-3f.yml 2026-06-11: this loop was the last
# grep-in-pipeline living in workflow shell. Workflow shells are `bash -e`
# WITHOUT pipefail, so `... | grep -v ... | tail` surviving a no-match was
# a property of WHERE the code ran, not of the code — anyone "hardening"
# the step with `set -euo pipefail` would have recreated the 2026-06-10
# predict-4a incident (run 27264913136). Here the no-match cases are
# explicit (`|| true`) under the same strict mode the sync script uses.
#
# Usage: pull_3a_predictions.sh [anchor-yyyy-mm-dd]
#   anchor defaults to today (UTC). R2_SOURCE + LOCAL_ROOT as in
#   sync_train_data.sh (LOCAL_ROOT default ../WeatherBlend/data).
set -euo pipefail

: "${R2_SOURCE:?R2_SOURCE env var required (e.g. r2:weatherblend)}"
LOCAL_ROOT="${LOCAL_ROOT:-../WeatherBlend/data}"
anchor="${1:-$(date -u +%F)}"

mkdir -p "$LOCAL_ROOT/predictions/precipitation"
for st in $(rclone lsf "${R2_SOURCE%/}/data/models/rainfall_amount/" --dirs-only 2>/dev/null || true); do
  st="${st%/}"
  [ "$st" = "MANIFEST.json" ] && continue
  versions=$(rclone lsf "${R2_SOURCE%/}/data/predictions/precipitation/${st}/" --dirs-only 2>/dev/null \
    | grep -v 'phase' | sort | tail -3 || true)
  for v in $versions; do
    v="${v%/}"
    src="${R2_SOURCE%/}/data/predictions/precipitation/${st}/${v}/date=${anchor}"
    dest="$LOCAL_ROOT/predictions/precipitation/${st}/${v}/date=${anchor}"
    if rclone lsf "${src}/" 2>/dev/null | grep -q predictions.parquet; then
      rclone copy "${src}" "${dest}" --checkers 4
      echo "3a pi for ${st}: ${v}/date=${anchor}"
    fi
  done
done
echo "pull_3a_predictions: done (anchor=$anchor)."
