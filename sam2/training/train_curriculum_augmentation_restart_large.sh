#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="/path/to/datasets/unsam_aug"
LOG_ROOT="/path/to/LAO-X/results/train/augmentation_restart_large"
CONFIG="configs/unsamv2_large_training_augmentation.yaml"

STAGES=("easy" "hard" "extreme")
INIT_CKPT=""

for stage in "${STAGES[@]}"; do
  IMG_FOLDER="${DATA_ROOT}/${stage}/xray"
  GT_FOLDER="${DATA_ROOT}/${stage}"
  EXPERIMENT_LOG_DIR="${LOG_ROOT}/${stage}/log"

  if [[ -f "${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint.pt" && -f "${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint_20.pt" ]]; then
    echo "==> Stage: ${stage} already finished, skipping."
    INIT_CKPT="${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint.pt"
    continue
  fi

  echo "==> Stage: ${stage}"
  echo "    IMG_FOLDER=${IMG_FOLDER}"
  echo "    GT_FOLDER=${GT_FOLDER}"
  echo "    EXPERIMENT_LOG_DIR=${EXPERIMENT_LOG_DIR}"
  if [[ -n "${INIT_CKPT}" ]]; then
    echo "    INIT_CKPT=${INIT_CKPT}"
  fi

  if [[ -n "${INIT_CKPT}" ]]; then
    INIT_ENV=("INIT_CKPT=${INIT_CKPT}")
  else
    INIT_ENV=()
  fi

  env \
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    IMG_FOLDER="${IMG_FOLDER}" \
    GT_FOLDER="${GT_FOLDER}" \
    EXPERIMENT_LOG_DIR="${EXPERIMENT_LOG_DIR}" \
    "${INIT_ENV[@]}" \
    python "${SCRIPT_DIR}/train.py" \
      -c "${CONFIG}" \
      --use-cluster 0 \
      --num-gpus 1

  INIT_CKPT="${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint.pt"
done
