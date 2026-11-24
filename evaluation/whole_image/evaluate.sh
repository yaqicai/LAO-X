#!/usr/bin/env bash
# COCO evaluation of the per-image JSON files written by predict.sh.
# Usage: GT_FILE=<coco_gt.json> bash evaluate.sh [extra evaluate.py args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
LAOX_RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${REPO_ROOT}/results}"
PREDICT_DIR="${PREDICT_DIR:-${LAOX_RESULTS_ROOT}/whole_image_eval}"
if [[ -z "${GT_FILE:-}" ]]; then
  echo "Set GT_FILE to the COCO ground-truth json." >&2
  exit 1
fi

"${PYTHON}" "${SCRIPT_DIR}/evaluate.py" \
    --predict-directory "${PREDICT_DIR}" \
    --gt-file "${GT_FILE}" \
    "$@"
