#!/usr/bin/env bash
# Whole-image automatic mask generation at several granularities; one COCO json per image.
# Usage: CHECKPOINT=... IMAGE_DIR=... bash predict.sh [extra predict.py args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
LAOX_DATA_ROOT="${LAOX_DATA_ROOT:-${REPO_ROOT}/datasets}"
LAOX_RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${REPO_ROOT}/results}"
IMAGE_DIR="${IMAGE_DIR:-${LAOX_DATA_ROOT}/pixray/PIXray}"
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${LAOX_RESULTS_ROOT}/whole_image_eval}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-100}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Set CHECKPOINT to a trained UnSAMv2 checkpoint (.pt)." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON}" "${SCRIPT_DIR}/predict.py" \
    --image-dir "${IMAGE_DIR}" \
    --model-config configs/unsamv2_small.yaml \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --granularities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --start-index "${START_INDEX}" \
    --end-index "${END_INDEX}" \
    "$@"
