#!/usr/bin/env bash
# LAO-X: CutLER-only ablation banks for DET-COMPASS X-ray images.
#   n1   MaskCut (DINO ViT-B/8, tau 0.15), N=1 mask/image, 960 px
#        -> $OUT_DIR/imagenet_train_fixsize960_tau0.15_N1_0_1.json
#   n3   MaskCut (DINO ViT-B/8, tau 0.15), N=3 masks/image, 480 px
#        -> $OUT_DIR/imagenet_train_fixsize480_tau0.15_N3_0_1.json
#   vis  overlay the N=1 and N=3 MaskCut jsons on the images (visualize_maskcut.py)
#   ori  CutLER detector without MuSc fusion -> coco_cutler_ori.json
#        (same as MODE=ori cutler/demo/laox_object_bank.sh)
#
# Usage: bash third_party/CutLER/maskcut.sh [n1] [n3] [vis] [ori]   (default: n1 n3)
# Environment: LAOX_DATA_ROOT (default <LAO-X repo>/datasets), LAOX_RESULTS_ROOT
#   (default <LAO-X repo>/results), PYTHON (default python),
#   IMAGE_DIR (default $LAOX_DATA_ROOT/det_compass_xp/xray),
#   OUT_DIR (default $LAOX_RESULTS_ROOT/object_bank/maskcut/det_compass_xp),
#   NUM_IMAGES (default -1 = all; >0 processes images [12, 12+NUM_IMAGES) of the
#   sorted list, see maskcut/maskcut.py).
# With a flat image folder MaskCut writes image file_name as "<first entry>/<file>"
# (upstream folder-per-class convention); use the basename when matching images.
set -euo pipefail

CUTLER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAOX_ROOT="$(cd "${CUTLER_ROOT}/../.." && pwd)"
DATA_ROOT="${LAOX_DATA_ROOT:-${LAOX_ROOT}/datasets}"
RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${LAOX_ROOT}/results}"
PYTHON="${PYTHON:-python}"
IMAGE_DIR="${IMAGE_DIR:-${DATA_ROOT}/det_compass_xp/xray}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/object_bank/maskcut/det_compass_xp}"
NUM_IMAGES="${NUM_IMAGES:--1}"

mkdir -p "${OUT_DIR}"
IMAGE_DIR="$(cd "${IMAGE_DIR}" && pwd)"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

run_maskcut() {
  local n="$1" size="$2"
  (cd "${CUTLER_ROOT}/maskcut" && "${PYTHON}" maskcut.py \
    --dataset-path "${IMAGE_DIR}" --out-dir "${OUT_DIR}" \
    --vit-arch base --patch-size 8 --pretrain-type dino --tau 0.15 \
    --N "${n}" --fixed_size "${size}" --num_images "${NUM_IMAGES}")
}

run_vis() {
  local ann
  for ann in imagenet_train_fixsize960_tau0.15_N1_0_1.json imagenet_train_fixsize480_tau0.15_N3_0_1.json; do
    if [ -f "${OUT_DIR}/${ann}" ]; then
      "${PYTHON}" "${CUTLER_ROOT}/visualize_maskcut.py" --flat \
        --dataset-path "${IMAGE_DIR}" --out-dir "${OUT_DIR}" \
        --ann-path "${OUT_DIR}/${ann}" --vis-dir "${OUT_DIR}/vis_${ann%.json}"
    fi
  done
}

if [ "$#" -eq 0 ]; then
  set -- n1 n3
fi
for step in "$@"; do
  case "${step}" in
    n1)  run_maskcut 1 960 ;;
    n3)  run_maskcut 3 480 ;;
    vis) run_vis ;;
    ori) MODE=ori IMAGE_DIR="${IMAGE_DIR}" bash "${CUTLER_ROOT}/cutler/demo/laox_object_bank.sh" ;;
    *) echo "unknown step: ${step} (expected n1|n3|vis|ori)" >&2; exit 2 ;;
  esac
done
