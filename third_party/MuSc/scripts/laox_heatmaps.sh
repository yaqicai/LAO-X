#!/usr/bin/env bash
# LAO-X object bank, step 1: MuSc zero-shot anomaly heatmaps for a flat folder of
# X-ray images, then collect the forward ("vanilla") heatmaps into one flat folder
# whose file names match the source images (input for CutLER --musc_dir, step 2).
# Inverse maps (*_musc_inverse_vanilla.png) are never used.
#
# Usage:
#   bash third_party/MuSc/scripts/laox_heatmaps.sh [IMAGE_DIR [MUSC_OUTPUT_DIR [HEATMAP_DIR]]]
# Defaults:
#   IMAGE_DIR        $LAOX_DATA_ROOT/det_compass_xp/xray
#   MUSC_OUTPUT_DIR  $LAOX_RESULTS_ROOT/object_bank/musc_output   (raw MuSc output)
#   HEATMAP_DIR      $LAOX_RESULTS_ROOT/object_bank/musc_heatmaps (collected heatmaps)
# Environment: LAOX_DATA_ROOT (default <LAO-X repo>/datasets),
#   LAOX_RESULTS_ROOT (default <LAO-X repo>/results), PYTHON (default python),
#   DEVICE (default 0), COLLECT_MODE=symlink|copy (default symlink),
#   SKIP_MUSC=1 to only (re)collect from an existing MUSC_OUTPUT_DIR.
# Use a fresh MUSC_OUTPUT_DIR: collection fails if an image has 0 or >1 candidate maps.
set -euo pipefail

MUSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAOX_ROOT="$(cd "${MUSC_ROOT}/../.." && pwd)"
DATA_ROOT="${LAOX_DATA_ROOT:-${LAOX_ROOT}/datasets}"
RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${LAOX_ROOT}/results}"
PYTHON="${PYTHON:-python}"
COLLECT_MODE="${COLLECT_MODE:-symlink}"

IMAGE_DIR="${1:-${IMAGE_DIR:-${DATA_ROOT}/det_compass_xp/xray}}"
MUSC_OUTPUT_DIR="${2:-${MUSC_OUTPUT_DIR:-${RESULTS_ROOT}/object_bank/musc_output}}"
HEATMAP_DIR="${3:-${HEATMAP_DIR:-${RESULTS_ROOT}/object_bank/musc_heatmaps}}"

if [ ! -d "${IMAGE_DIR}" ]; then
  echo "image dir not found: ${IMAGE_DIR}" >&2
  exit 1
fi
mkdir -p "${MUSC_OUTPUT_DIR}" "${HEATMAP_DIR}"
# Resolve to absolute paths before changing directory.
IMAGE_DIR="$(cd "${IMAGE_DIR}" && pwd)"
MUSC_OUTPUT_DIR="$(cd "${MUSC_OUTPUT_DIR}" && pwd)"
HEATMAP_DIR="$(cd "${HEATMAP_DIR}" && pwd)"

if [ "${SKIP_MUSC:-0}" != "1" ]; then
  cd "${MUSC_ROOT}"
  export PYTHONPATH="${MUSC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  # Note: examples/musc_main.py maps any --img_resize to (H, W) = (560, 840).
  "${PYTHON}" examples/musc_main.py --device "${DEVICE:-0}" \
    --data_path "${IMAGE_DIR}" --dataset_name mvtec_xp --class_name ALL \
    --backbone_name ViT-L-14-336 --pretrained openai --feature_layers 5 11 17 23 \
    --img_resize 560 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
    --output_dir "${MUSC_OUTPUT_DIR}" --vis True --save_excel True
fi

# Index forward heatmaps <stem>_musc_vanilla.png anywhere below the MuSc output dir.
declare -A n_cand=()
declare -A cand=()
while IFS= read -r -d '' f; do
  b="$(basename "${f}")"
  stem="${b%_musc_vanilla.png}"
  n_cand["${stem}"]=$(( ${n_cand["${stem}"]:-0} + 1 ))
  cand["${stem}"]="${f}"
done < <(find "${MUSC_OUTPUT_DIR}" -type f -name '*_musc_vanilla.png' -print0)

shopt -s nullglob
images=("${IMAGE_DIR}"/*.png)
shopt -u nullglob
if [ "${#images[@]}" -eq 0 ]; then
  echo "no *.png images in ${IMAGE_DIR}" >&2
  exit 1
fi

errors=0
for img in "${images[@]}"; do
  name="$(basename "${img}")"
  stem="${name%.*}"
  n="${n_cand["${stem}"]:-0}"
  if [ "${n}" -ne 1 ]; then
    echo "ERROR: ${name}: ${n} candidate *_musc_vanilla.png heatmaps (expected exactly 1)" >&2
    errors=$((errors + 1))
  fi
done
if [ "${errors}" -ne 0 ]; then
  echo "heatmap collection failed for ${errors}/${#images[@]} images; nothing collected" >&2
  exit 1
fi

for img in "${images[@]}"; do
  name="$(basename "${img}")"
  src="${cand["${name%.*}"]}"
  if [ "${COLLECT_MODE}" = "copy" ]; then
    cp -f "${src}" "${HEATMAP_DIR}/${name}"
  else
    ln -sfn "${src}" "${HEATMAP_DIR}/${name}"
  fi
done
echo "collected ${#images[@]} forward heatmaps into ${HEATMAP_DIR}"
