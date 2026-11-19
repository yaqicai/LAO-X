#!/usr/bin/env bash
# LAO-X object bank, step 2: CutLER (cascade Mask R-CNN, DETECTIONS_PER_IMAGE=1 from
# the demo config) on X-ray images fused with MuSc heatmaps (mul_blend_mul_v2), written
# as a COCO json. Step 1 is third_party/MuSc/scripts/laox_heatmaps.sh.
#
# Usage:
#   bash third_party/CutLER/cutler/demo/laox_object_bank.sh [IMAGE_DIR [HEATMAP_DIR [COCO_OUTPUT]]]
# Defaults:
#   IMAGE_DIR    $LAOX_DATA_ROOT/det_compass_xp/xray
#   HEATMAP_DIR  $LAOX_RESULTS_ROOT/object_bank/musc_heatmaps
#   COCO_OUTPUT  $LAOX_DATA_ROOT/det_compass_xp/annotations/coco_cutler_musc.json
#                (MODE=ori: .../coco_cutler_ori.json)
#   VIS_DIR      $LAOX_RESULTS_ROOT/object_bank/cutler_<MODE>_vis
#   CUTLER_WEIGHTS  $LAOX_CKPT_ROOT/cutler_cascade_final.pth
# Environment: LAOX_DATA_ROOT (default <LAO-X repo>/datasets), LAOX_RESULTS_ROOT
#   (default <LAO-X repo>/results), LAOX_CKPT_ROOT (default <LAO-X repo>/checkpoints),
#   PYTHON (default python),
#   MODE=musc|ori  musc (default): MuSc-guided bank; ori: CutLER-only ablation bank
#                  (no heatmap fusion),
#   EXTRA_OPTS     extra Detectron2 KEY VALUE pairs appended after MODEL.WEIGHTS,
#                  e.g. EXTRA_OPTS="TEST.DETECTIONS_PER_IMAGE 3".
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLER_PKG="$(cd "${DEMO_DIR}/.." && pwd)"     # third_party/CutLER/cutler
CUTLER_ROOT="$(cd "${CUTLER_PKG}/.." && pwd)"  # third_party/CutLER
LAOX_ROOT="$(cd "${CUTLER_ROOT}/../.." && pwd)"
DATA_ROOT="${LAOX_DATA_ROOT:-${LAOX_ROOT}/datasets}"
RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${LAOX_ROOT}/results}"
CKPT_ROOT="${LAOX_CKPT_ROOT:-${LAOX_ROOT}/checkpoints}"
PYTHON="${PYTHON:-python}"
MODE="${MODE:-musc}"

case "${MODE}" in
  musc) default_coco="coco_cutler_musc.json" ;;
  ori)  default_coco="coco_cutler_ori.json" ;;
  *) echo "MODE must be musc or ori, got: ${MODE}" >&2; exit 2 ;;
esac

IMAGE_DIR="${1:-${IMAGE_DIR:-${DATA_ROOT}/det_compass_xp/xray}}"
HEATMAP_DIR="${2:-${HEATMAP_DIR:-${RESULTS_ROOT}/object_bank/musc_heatmaps}}"
COCO_OUTPUT="${3:-${COCO_OUTPUT:-${DATA_ROOT}/det_compass_xp/annotations/${default_coco}}}"
VIS_DIR="${VIS_DIR:-${RESULTS_ROOT}/object_bank/cutler_${MODE}_vis}"
CUTLER_WEIGHTS="${CUTLER_WEIGHTS:-${CKPT_ROOT}/cutler_cascade_final.pth}"
read -r -a extra_opts <<< "${EXTRA_OPTS:-}"

if [ ! -d "${IMAGE_DIR}" ]; then
  echo "image dir not found: ${IMAGE_DIR}" >&2; exit 1
fi
if [ ! -f "${CUTLER_WEIGHTS}" ]; then
  echo "CutLER checkpoint not found: ${CUTLER_WEIGHTS}" >&2
  echo "download: http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth" >&2
  exit 1
fi
mkdir -p "${VIS_DIR}" "$(dirname "${COCO_OUTPUT}")"
# Resolve to absolute paths before changing directory.
IMAGE_DIR="$(cd "${IMAGE_DIR}" && pwd)"
VIS_DIR="$(cd "${VIS_DIR}" && pwd)"
COCO_OUTPUT="$(cd "$(dirname "${COCO_OUTPUT}")" && pwd)/$(basename "${COCO_OUTPUT}")"
CUTLER_WEIGHTS="$(cd "$(dirname "${CUTLER_WEIGHTS}")" && pwd)/$(basename "${CUTLER_WEIGHTS}")"

fusion_args=()
if [ "${MODE}" = "musc" ]; then
  if [ ! -d "${HEATMAP_DIR}" ]; then
    echo "heatmap dir not found: ${HEATMAP_DIR} (run third_party/MuSc/scripts/laox_heatmaps.sh)" >&2
    exit 1
  fi
  HEATMAP_DIR="$(cd "${HEATMAP_DIR}" && pwd)"
  fusion_args=(--musc_dir "${HEATMAP_DIR}" --fusion_mode mul_blend_mul_v2)
fi

cd "${CUTLER_PKG}"
export PYTHONPATH="${CUTLER_PKG}:${CUTLER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# The input glob is quoted on purpose: demo.py expands it itself.
"${PYTHON}" demo/demo.py \
  --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml \
  --input "${IMAGE_DIR}/*.png" \
  --output "${VIS_DIR}" \
  ${fusion_args[@]+"${fusion_args[@]}"} \
  --coco_output "${COCO_OUTPUT}" \
  --vis_demo \
  --opts MODEL.WEIGHTS "${CUTLER_WEIGHTS}" ${extra_opts[@]+"${extra_opts[@]}"}
echo "wrote ${COCO_OUTPUT}"
