#!/usr/bin/env bash
# Object-bank ablation: build the curriculum from CutLER-only masks instead of
# CutLER x MuSc, then train with the paper schedule.
#   VARIANT=n1 (default): 1 detection per image  -> coco_cutler_ori.json
#   VARIANT=n3          : 3 detections per image -> coco_cutler_ori_N3.json
# Create the object-bank JSON first, e.g.
#   MODE=ori bash third_party/CutLER/cutler/demo/laox_object_bank.sh
# (see third_party/CutLER/LAOX_CHANGES.md for the N3 variant).
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
