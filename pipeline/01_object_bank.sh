#!/usr/bin/env bash
# Step 1: object bank. MuSc anomaly heatmaps -> CutLER x MuSc fused masks.
# Output: ${OBJECT_BANK_JSON} (default <data>/det_compass_xp/annotations/coco_cutler_musc.json)
# Needs: ${SOURCE_DIR}/xray/*.png, ${LAOX_CKPT_ROOT}/cutler_cascade_final.pth, one GPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

IMAGE_DIR="${IMAGE_DIR:-${SOURCE_DIR}/xray}"
HEATMAP_DIR="${HEATMAP_DIR:-${LAOX_RESULTS_ROOT}/object_bank/musc_heatmaps}"
require_file "${IMAGE_DIR}" "${LAOX_CKPT_ROOT}/cutler_cascade_final.pth"
if [[ -e "${OBJECT_BANK_JSON}" ]]; then
  echo "Object bank already exists: ${OBJECT_BANK_JSON} (delete it to regenerate)" >&2
  exit 1
fi

bash "${LAOX_REPO}/third_party/MuSc/scripts/laox_heatmaps.sh" \
  "${IMAGE_DIR}" "${LAOX_RESULTS_ROOT}/object_bank/musc_output" "${HEATMAP_DIR}"
bash "${LAOX_REPO}/third_party/CutLER/cutler/demo/laox_object_bank.sh" \
  "${IMAGE_DIR}" "${HEATMAP_DIR}" "${OBJECT_BANK_JSON}"
