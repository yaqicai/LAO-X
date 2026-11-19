#!/usr/bin/env bash
# LAO-X: MuSc zero-shot heatmaps on bbox/mask crops of COCO-format X-ray datasets
# (dataset_name in models/musc.py:COCO_BBOX_DATASETS, loader datasets/coco_bbox.py).
# The original MVTec AD / VisA / BTAD commands are in the upstream repository.
# The object-bank heatmaps (mvtec_xp, full images) are produced by scripts/laox_heatmaps.sh.
#
# Usage (from anywhere):
#   bash scripts/musc.sh                     # pidray HiXray (default)
#   bash scripts/musc.sh pixray CLCXray DvXray
# Environment:
#   LAOX_DATA_ROOT     dataset root (default: <LAO-X repo>/datasets)
#   LAOX_RESULTS_ROOT  results root (default: <LAO-X repo>/results)
#   OUTPUT_DIR         MuSc output dir (default: $LAOX_RESULTS_ROOT/musc/output_xray_v2)
#   PYTHON, DEVICE
set -euo pipefail

MUSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAOX_ROOT="$(cd "${MUSC_ROOT}/../.." && pwd)"
DATA_ROOT="${LAOX_DATA_ROOT:-${LAOX_ROOT}/datasets}"
RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${LAOX_ROOT}/results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/musc/output_xray_v2}"
PYTHON="${PYTHON:-python}"

cd "${MUSC_ROOT}"
export PYTHONPATH="${MUSC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS=(
  --device "${DEVICE:-0}"
  --class_name ALL
  --backbone_name ViT-L-14-336
  --pretrained openai
  --feature_layers 5 11 17 23
  --img_resize 560
  --divide_num 1
  --r_list 1 3 5
  --batch_size 4
  --max_images 100
  --crop_mode ratio_expand
  --output_dir "${OUTPUT_DIR}"
  --vis True
  --save_excel True
)

run_xray_dataset() {
  local dataset_name="$1"
  local data_path="$2"
  local annotation_path="$3"
  local image_root="$4"
  local annotation_mode="$5"

  "${PYTHON}" examples/musc_main.py \
    --data_path "${data_path}" \
    --dataset_name "${dataset_name}" \
    --annotation_path "${annotation_path}" \
    --image_root "${image_root}" \
    --annotation_mode "${annotation_mode}" \
    "${COMMON_ARGS[@]}"
}

# dataset name -> data_path, COCO annotation json, image root, annotation mode
run_named() {
  local d="${DATA_ROOT}/$1"
  case "$1" in
    pixray)  run_xray_dataset pixray  "${d}" "${d}/coco.json"                             "${d}/PIXray"                  mask ;;
    CLCXray) run_xray_dataset CLCXray "${d}" "${d}/coco_agnostic_val.json"                "${d}/val2017"                 bbox ;;
    DvXray)  run_xray_dataset DvXray  "${d}" "${d}/coco_agnostic.json"                    "${d}/DvXray_Positive_Samples" bbox ;;
    pidray)  run_xray_dataset pidray  "${d}" "${d}/pidray/annotations/coco_agnostic.json" "${d}/pidray/train"            mask ;;
    HiXray)  run_xray_dataset HiXray  "${d}" "${d}/coco_agnostic.json"                    "${d}/test/test_image"         bbox ;;
    *) echo "unknown dataset: $1 (expected pixray|CLCXray|DvXray|pidray|HiXray)" >&2; return 2 ;;
  esac
}

if [ "$#" -eq 0 ]; then
  set -- pidray HiXray
fi
for name in "$@"; do
  run_named "${name}"
done
