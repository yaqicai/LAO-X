#!/usr/bin/env bash
# Step 3: oversegmentation pseudo-labels (granularity divide-and-conquer).
# Output: ${LAOX_DATA_ROOT}/${SYN_DATASET}/<stage>/anno_cutler/*.json
# Needs: ${LAOX_CKPT_ROOT}/{cutler_cascade_final.pth,dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth},
#        the DINOv3 clone (see INSTALL.md) and the GPUs listed in ${GPUS}.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

for stage in ${STAGES}; do
  require_file "${LAOX_DATA_ROOT}/${SYN_DATASET}/${stage}/coco.json"
  bash "${LAOX_REPO}/granularity_divide_and_conquer/run.sh" "${SYN_DATASET}" "${stage}"
done
