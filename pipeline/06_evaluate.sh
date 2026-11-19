#!/usr/bin/env bash
# Step 6: class-agnostic box evaluation of the extreme-stage checkpoint (epoch 48)
# at granularities 0.1-0.8, mask threshold 0.5.
#   bash pipeline/06_evaluate.sh                 # held-out DET-COMPASS images
#   DATASET=CLCXray bash pipeline/06_evaluate.sh # cross-dataset (CLCXray, DvXray, HiXray, pidray, PIXray)
# Output: <repo>/output_eval/<run>/ (predictions + COCO bbox metrics)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DATASET="${DATASET:-det_compass_xp}"
if [[ "${DATASET}" == det_compass_xp ]]; then
  require_file "${HELDOUT_COCO}" "${SOURCE_DIR}/xray"
  export VAL_JSON="${VAL_JSON:-${HELDOUT_COCO}}"
  export IMAGE_ROOT="${IMAGE_ROOT:-${SOURCE_DIR}/xray}"
fi

DATASET="${DATASET}" \
STAGES="${EVAL_STAGES:-extreme}" \
GRAS="${GRAS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}" \
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}" \
CKPT_SUFFIX="${CKPT_SUFFIX:-48}" \
  bash "${LAOX_REPO}/scripts/task_test_augmentation_restart.sh" "${EXPERIMENT}" "${LAOX_RESULTS_ROOT}/train"
