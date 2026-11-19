#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Example alternative config:
#   -c configs/sam2.1_training/0827_unsup_dinov3.yaml

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python "${SCRIPT_DIR}/train.py" \
  -c configs/unsamv2_small_training_det_compass_xp.yaml \
  --use-cluster 0 \
  --num-gpus 1
