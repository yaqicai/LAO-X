#!/usr/bin/env bash
# Step 4: class-agnostic synthetic labels + fusion with oversegmentation labels.
# Output per stage: coco_agnostic.json, anno/, anno_cutler_clean/, anno_fusion/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "${LAOX_REPO}"
for stage in ${STAGES}; do
  require_file "${LAOX_DATA_ROOT}/${SYN_DATASET}/${stage}/anno_cutler"
  "${PYTHON}" pseudo_label_generate.py --to-agnostic --dataset "${SYN_DATASET}" --stage "${stage}"
  "${PYTHON}" pseudo_label_generate.py --dataset "${SYN_DATASET}" --coco-name coco_agnostic.json --stage "${stage}"
  "${PYTHON}" pseudo_label_generate.py --fusion --fusion-name anno_fusion --dataset "${SYN_DATASET}" \
    --stage "${stage}" --fusion-overlap-thresh 0.95
done
