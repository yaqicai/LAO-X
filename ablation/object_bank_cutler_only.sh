#!/usr/bin/env bash
# Object-bank ablation: build the curriculum from CutLER-only masks instead of
# CutLER x MuSc, then train with the paper schedule.
# N1 / N3 = number of CutLER predictions kept per image:
#   VARIANT=n1 (default): 1 prediction per image  -> coco_cutler_ori.json
#   VARIANT=n3          : 3 predictions per image -> coco_cutler_ori_N3.json
# Create the object-bank JSON first:
#   n1: MODE=ori bash third_party/CutLER/cutler/demo/laox_object_bank.sh
#   n3: MODE=ori EXTRA_OPTS="TEST.DETECTIONS_PER_IMAGE 3" \
#         COCO_OUTPUT=$LAOX_DATA_ROOT/det_compass_xp/annotations/coco_cutler_ori_N3.json \
#         bash third_party/CutLER/cutler/demo/laox_object_bank.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

case "${VARIANT:-n1}" in
  n1) bank="${ANNO_DIR}/coco_cutler_ori.json" ;;
  n3) bank="${ANNO_DIR}/coco_cutler_ori_N3.json" ;;
  *) echo "VARIANT must be n1 or n3" >&2; exit 2 ;;
esac
dataset="crop_aug_cutler_${VARIANT:-n1}_xp"
require_file "${bank}"

build_curriculum "${dataset}" --object-bank-json "${bank}"
train_curriculum "crop_aug_cutler_${VARIANT:-n1}_fusion_bs16" "${dataset}" fusion
