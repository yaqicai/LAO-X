#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="/path/to/datasets/unsam_aug"
LOG_ROOT="/path/to/LAO-X/results/train/augmentation_resume"
CONFIG="configs/unsamv2_small_training_augmentation.yaml"

STAGES=("easy" "hard" "extreme")
RESUME_FROM=""
STAGE_EPOCHS=20

for stage_idx in "${!STAGES[@]}"; do
  stage="${STAGES[$stage_idx]}"
  IMG_FOLDER="${DATA_ROOT}/${stage}/xray"
  GT_FOLDER="${DATA_ROOT}/${stage}"
  EXPERIMENT_LOG_DIR="${LOG_ROOT}/${stage}/log"

  if [[ -f "${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint.pt" && -f "${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint_20.pt" ]]; then
    echo "==> Stage: ${stage} already finished, skipping."
    RESUME_FROM="${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint.pt"
    continue
  fi

  echo "==> Stage: ${stage}"
  echo "    IMG_FOLDER=${IMG_FOLDER}"
  echo "    GT_FOLDER=${GT_FOLDER}"
  echo "    EXPERIMENT_LOG_DIR=${EXPERIMENT_LOG_DIR}"
  if [[ -n "${RESUME_FROM}" ]]; then
    echo "    RESUME_FROM=${RESUME_FROM}"
  fi

  TOTAL_EPOCHS=$((STAGE_EPOCHS * (stage_idx + 1)))
  echo "    TOTAL_EPOCHS=${TOTAL_EPOCHS}"

  if [[ -n "${RESUME_FROM}" ]]; then
    RESUME_ENV=("RESUME_FROM=${RESUME_FROM}")
  else
    RESUME_ENV=()
  fi

  TOTAL_ENV=("TOTAL_EPOCHS=${TOTAL_EPOCHS}")

  env \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    IMG_FOLDER="${IMG_FOLDER}" \
    GT_FOLDER="${GT_FOLDER}" \
    EXPERIMENT_LOG_DIR="${EXPERIMENT_LOG_DIR}" \
    "${TOTAL_ENV[@]}" \
    "${RESUME_ENV[@]}" \
    python "${SCRIPT_DIR}/train.py" \
      -c "${CONFIG}" \
      --use-cluster 0 \
      --num-gpus 1

  RESUME_FROM=${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint.pt
  # RESUME_FROM="${LOG_ROOT}/${stage}/checkpoints/checkpoint.pt"
done
