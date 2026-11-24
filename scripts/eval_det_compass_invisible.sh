#!/usr/bin/env bash
# Evaluate a curriculum-trained checkpoint on the DET-COMPASS `invisible` split
# (class-agnostic ground truth built by prepare_det_compass_invisible.py).
#
# Usage: bash scripts/eval_det_compass_invisible.sh [EXPERIMENT_NAME] [TRAIN_ROOT]
# Environment: GROUND_TRUTH, IMAGE_ROOT, GPUS, GRAS [1.0], STAGES [extreme],
# CKPT_SUFFIX [48], MASK_THRESHOLD [0.5], EVAL_SEED [1], PYTHON, LAOX_DATA_ROOT,
# LAOX_RESULTS_ROOT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-python}"
LAOX_DATA_ROOT="${LAOX_DATA_ROOT:-${REPO_ROOT}/datasets}"
LAOX_RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${REPO_ROOT}/results}"

COMMENT="${1:-crop_musc_aug_fusion_bs16_xp}"
TRAIN_ROOT="${2:-${LAOX_RESULTS_ROOT}/train}"
GROUND_TRUTH="${GROUND_TRUTH:-${LAOX_RESULTS_ROOT}/annotations/det_compass_xp_coco_invisible_agnostic.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${LAOX_DATA_ROOT}/det_compass_xp/xray}"

"${PYTHON}" scripts/prepare_det_compass_invisible.py \
  --output "${GROUND_TRUTH}" \
  --image-root "${IMAGE_ROOT}"

MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}" \
DATASET=det_compass_xp \
VAL_JSON="${GROUND_TRUTH}" \
IMAGE_ROOT="${IMAGE_ROOT}" \
START_ID=0 \
END_ID=355 \
GPUS="${GPUS:-0}" \
GRAS="${GRAS:-1.0}" \
STAGES="${STAGES:-extreme}" \
CKPT_SUFFIX="${CKPT_SUFFIX:-48}" \
EVAL_SEED="${EVAL_SEED:-1}" \
bash scripts/eval_curriculum.sh "${COMMENT}" "${TRAIN_ROOT}"

PREDICTIONS="output_eval/det_compass_xp_img0_355_unsamv2_${GRAS:-1.0}_augmentation_restart_${COMMENT}_extreme_m${MASK_THRESHOLD:-0.5}_${CKPT_SUFFIX:-48}/coco_instances_results.json"
if [[ ! -f "${PREDICTIONS}" ]]; then
  echo "Inference did not produce predictions: ${PREDICTIONS}" >&2
  exit 1
fi
"${PYTHON}" scripts/score_det_compass_invisible.py \
  --ground-truth "${GROUND_TRUTH}" \
  --predictions "${PREDICTIONS}" \
  --output-prefix "output_eval/det_compass_invisible_${COMMENT}_extreme_m${MASK_THRESHOLD:-0.5}_${CKPT_SUFFIX:-48}_s3_1p5"
