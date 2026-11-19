#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

COMMENT="${1:-crop_musc_aug_fusion_bs16_xp}"
TRAIN_ROOT="${2:-/path/to/results/train}"
GROUND_TRUTH="${GROUND_TRUTH:-${REPO_ROOT}/annotations/det_compass_xp_coco_invisible_agnostic.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/path/to/datasets/det_compass_xp/xray}"

python scripts/prepare_det_compass_invisible_eval.py \
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
bash scripts/task_test_augmentation_restart.sh "${COMMENT}" "${TRAIN_ROOT}"

PREDICTIONS="output_eval/det_compass_xp_img0_355_unsamv2_${GRAS:-1.0}_augmentation_restart_${COMMENT}_extreme_m${MASK_THRESHOLD:-0.5}_${CKPT_SUFFIX:-48}/coco_instances_results.json"
if [[ ! -f "${PREDICTIONS}" ]]; then
  echo "Inference did not produce predictions: ${PREDICTIONS}" >&2
  exit 1
fi
python scripts/evaluate_det_compass_invisible.py \
  --ground-truth "${GROUND_TRUTH}" \
  --predictions "${PREDICTIONS}" \
  --output-prefix "output_eval/det_compass_invisible_${COMMENT}_extreme_m${MASK_THRESHOLD:-0.5}_${CKPT_SUFFIX:-48}_s3_1p5"
