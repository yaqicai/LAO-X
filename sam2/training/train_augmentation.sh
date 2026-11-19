#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
IMG_FOLDER="${IMG_FOLDER:-/path/to/datasets/unsam_aug/easy/xray}" \
GT_FOLDER="${GT_FOLDER:-/path/to/datasets/unsam_aug/easy/anno}" \
EXPERIMENT_LOG_DIR="${EXPERIMENT_LOG_DIR:-/path/to/LAO-X/results/train/augmentation/log}" \
python "${SCRIPT_DIR}/train.py" \
  -c configs/unsamv2_small_training_augmentation.yaml \
  --use-cluster 0 \
  --num-gpus 1
