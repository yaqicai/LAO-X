#!/usr/bin/env bash
# Step 2: synthesize the easy / hard / extreme curriculum (8,000 images each).
# One container per image; colour rendering with the DET-COMPASS high/low LUTs.
# CPU only. Output: ${LAOX_DATA_ROOT}/${SYN_DATASET}/<stage>/{xray/,coco.json}
#
# One-time preparation (needs the DET-COMPASS dataset through HuggingFace):
#   ${PYTHON} synthesis/build_luts.py all            # writes ${LAOX_LUT_DIR}/*.pkl
#   ${PYTHON} synthesis/generate.py match --xray-dir ${SOURCE_DIR}/xray --output ${MATCHES_JSON} --online
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

WORKERS="${WORKERS:-16}"
NUM_GENERATED="${NUM_GENERATED:-8000}"
SEED="${SEED:-40}"
require_file "${OBJECT_BANK_JSON}" "${HELDOUT_COCO}" "${MATCHES_JSON}" \
  "${LAOX_LUT_DIR}/xray_highlow_to_colour_dict_RAW.pkl" \
  "${LAOX_LUT_DIR}/xray_highlow_to_grey_dict_RAW.pkl" \
  "${LAOX_LUT_DIR}/xray_highlow_to_density_dict_RAW.pkl"

for stage in ${STAGES}; do
  bash "${LAOX_REPO}/synthesis/run_stage.sh" \
    --object-bank-json "${OBJECT_BANK_JSON}" \
    --test-coco "${HELDOUT_COCO}" \
    --matches-json "${MATCHES_JSON}" \
    --dataset-root "${LAOX_DATA_ROOT}" \
    --dataset-gen "${SYN_DATASET}" \
    --stage "${stage}" \
    --num-objects "$(stage_objects "${stage}")" \
    --gamma "$(stage_gamma "${stage}")" \
    --num-generated "${NUM_GENERATED}" \
    --workers "${WORKERS}" \
    --io-workers 2 \
    --seed "${SEED}" \
    "$@"
done
