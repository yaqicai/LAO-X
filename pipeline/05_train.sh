#!/usr/bin/env bash
# Step 5: easy -> hard -> extreme curriculum training of UnSAMv2 (SAM 2.1 Hiera-S),
# 8 GPUs x batch size 2. Each stage starts from the previous stage's checkpoint.
# Output: ${LAOX_RESULTS_ROOT}/train/augmentation_restart_${EXPERIMENT}/<stage>/log/
# Needs: ${LAOX_CKPT_ROOT}/sam2.1_hiera_small.pt (bash sam2/checkpoints/download_ckpts.sh)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

NUM_GPUS="$(awk -F, '{print NF}' <<< "${GPUS}")"
CUDA_VISIBLE_DEVICES="${GPUS}" bash "${LAOX_REPO}/sam2/training/train_curriculum.sh" "${EXPERIMENT}" \
  --data-root-name "${SYN_DATASET}" \
  --anno-comment fusion \
  --stages easy,hard,extreme \
  --stage-ranges easy:0-4000,hard:0-6000,extreme:0-8000 \
  --stage-base-lrs easy:1.6e-4,hard:1.2e-4,extreme:1e-4 \
  --stage-vision-lrs easy:4.8e-6,hard:3.2e-6,extreme:3e-6 \
  --stage-epochs easy:80,hard:64,extreme:48 \
  --batch-size 2 \
  --num-gpus "${NUM_GPUS}" \
  "$@"
