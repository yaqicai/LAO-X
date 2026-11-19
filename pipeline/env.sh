# Shared settings for the LAO-X pipeline scripts. Source it; do not run it.
# Every value can be overridden from the environment before calling a script.

LAOX_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LAOX_REPO
export LAOX_DATA_ROOT="${LAOX_DATA_ROOT:-${LAOX_REPO}/datasets}"
export LAOX_RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${LAOX_REPO}/results}"
export LAOX_CKPT_ROOT="${LAOX_CKPT_ROOT:-${LAOX_REPO}/checkpoints}"
export LAOX_LUT_DIR="${LAOX_LUT_DIR:-${LAOX_REPO}/synthesis/luts}"
export PYTHON="${PYTHON:-python}"
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# Source dataset (DET-COMPASS, local copy) and its annotation files.
SOURCE_DIR="${SOURCE_DIR:-${LAOX_DATA_ROOT}/det_compass_xp}"
ANNO_DIR="${ANNO_DIR:-${SOURCE_DIR}/annotations}"
OBJECT_BANK_JSON="${OBJECT_BANK_JSON:-${ANNO_DIR}/coco_cutler_musc.json}"
HELDOUT_COCO="${HELDOUT_COCO:-${ANNO_DIR}/coco_agnostic.json}"
MATCHES_JSON="${MATCHES_JSON:-${ANNO_DIR}/huggingface_matches.json}"

# Synthetic curriculum dataset and the main experiment name.
SYN_DATASET="${SYN_DATASET:-crop_aug_xp}"
EXPERIMENT="${EXPERIMENT:-crop_musc_aug_fusion_bs16_xp}"
STAGES="${STAGES:-easy hard extreme}"

# Stage -> "objects-per-image gamma-range" used for synthesis.
stage_objects() {
  case "$1" in
    easy) echo "2,4" ;; hard) echo "4,7" ;; extreme) echo "6,9" ;;
    *) echo "unknown stage: $1" >&2; return 1 ;;
  esac
}
stage_gamma() {
  case "$1" in
    easy) echo "0.0,0.05" ;; hard) echo "0.0,0.3" ;; extreme) echo "0.0,0.5" ;;
    *) echo "unknown stage: $1" >&2; return 1 ;;
  esac
}

require_file() {
  local f
  for f in "$@"; do
    [[ -e "${f}" ]] || { echo "Missing: ${f}" >&2; return 1; }
  done
}
