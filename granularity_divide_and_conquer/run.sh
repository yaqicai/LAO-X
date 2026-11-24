#!/usr/bin/env bash
# Granularity divide-and-conquer oversegmentation (CutLER + DINOv3 + CascadePSP).
#
# Usage:
#   bash run.sh <dataset> [stage]
#       input : $LAOX_DATA_ROOT/<dataset>[/<stage>]/xray
#       output: $LAOX_DATA_ROOT/<dataset>[/<stage>]/anno_cutler   (one JSON per image)
#       e.g.  GPUS=0,1,2,3 bash run.sh crop_aug_xp easy
#
#   bash run.sh --input-dir DIR --output-dir DIR
#       any flat folder of images, e.g.
#       bash run.sh --input-dir "$LAOX_DATA_ROOT/pixray/PIXray" \
#           --output-dir "$LAOX_RESULTS_ROOT/cutler/pixray/granularity_divide_and_conquer/vis"
#
# Environment:
#   GPUS               comma-separated GPU ids; the sorted image list is split into one
#                      contiguous shard per GPU, run in parallel (default: 0)
#   PYTHON             Python interpreter (default: python)
#   LAOX_DATA_ROOT     dataset root (default: <repo>/datasets)
#   LAOX_CKPT_ROOT     checkpoint dir holding cutler_cascade_final.pth and
#                      dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth (default: <repo>/checkpoints)
#   LAOX_RESULTS_ROOT  results root, only used in the example above (default: <repo>/results)
#   DINOV3_REPO        local DINOv3 clone (default: <this dir>/dinov3, see INSTALL.md)
#
# Resuming / duplicate runs: images whose JSON already exists are skipped, so a
# re-run resumes. A stage whose output was updated within the last
# OVERSEG_ACTIVE_WINDOW_SECONDS (default 300) is treated as owned by another run
# and skipped; set OVERSEG_FORCE_RESUME=1 to take over after stopping the old run.
# A lock file prevents two runs on the same output; set OVERSEG_ALLOW_CONCURRENT=1
# only when duplicate workers are intentional.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${SCRIPT_DIR}"

usage() {
  sed -n '2,13p' "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")" | sed 's/^# \{0,1\}//' >&2
}

die() {
  echo "[gra_dico] ERROR: $*" >&2
  exit 1
}

INPUT_DIR=""
OUTPUT_DIR=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir) INPUT_DIR="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) usage; die "unknown option: $1" ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

DATA_ROOT="${LAOX_DATA_ROOT:-${REPO_ROOT}/datasets}"
CKPT_ROOT="${LAOX_CKPT_ROOT:-${REPO_ROOT}/checkpoints}"
CUTLER_WEIGHTS="${CKPT_ROOT}/cutler_cascade_final.pth"
DINOV3_WEIGHTS="${CKPT_ROOT}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DINOV3_REPO="${DINOV3_REPO:-${SCRIPT_DIR}/dinov3}"
PY="${PYTHON:-python}"

if [[ -n "${INPUT_DIR}" || -n "${OUTPUT_DIR}" ]]; then
  [[ -n "${INPUT_DIR}" && -n "${OUTPUT_DIR}" ]] || die "--input-dir and --output-dir must be given together."
  [[ "${#POSITIONAL[@]}" -eq 0 ]] || die "do not combine <dataset> [stage] with --input-dir/--output-dir."
  LABEL="${INPUT_DIR}"
else
  [[ "${#POSITIONAL[@]}" -ge 1 && "${#POSITIONAL[@]}" -le 2 ]] || { usage; exit 1; }
  DATASET="${POSITIONAL[0]}"
  STAGE="${POSITIONAL[1]:-}"
  BASE_DIR="${DATA_ROOT}/${DATASET}${STAGE:+/${STAGE}}"
  INPUT_DIR="${BASE_DIR}/xray"
  OUTPUT_DIR="${BASE_DIR}/anno_cutler"
  LABEL="${DATASET}${STAGE:+/${STAGE}}"
fi

GPUS="${GPUS:-0}"
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NUM_GPUS="${#GPU_LIST[@]}"
[[ "${NUM_GPUS}" -gt 0 ]] || die "no GPUs specified in GPUS."

command -v "${PY}" >/dev/null 2>&1 || die "Python executable not found: ${PY} (set PYTHON)."
[[ -d "${INPUT_DIR}" ]] || die "input directory not found: ${INPUT_DIR}"
[[ -f "${CUTLER_WEIGHTS}" ]] || die "CutLER checkpoint not found: ${CUTLER_WEIGHTS} (set LAOX_CKPT_ROOT)."
[[ -f "${DINOV3_WEIGHTS}" ]] || die "DINOv3 weights not found: ${DINOV3_WEIGHTS} (set LAOX_CKPT_ROOT)."
[[ -f "${DINOV3_REPO}/hubconf.py" ]] || die "DINOv3 repository not found at ${DINOV3_REPO} (clone it as described in INSTALL.md, or set DINOV3_REPO)."

TOTAL_IMAGES="$(find "${INPUT_DIR}" -maxdepth 1 -type f ! -name "*.json" ! -name ".DS_Store" | wc -l)"
[[ "${TOTAL_IMAGES}" -gt 0 ]] || die "no images found in ${INPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

EXISTING_JSON="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)"
if [[ "${EXISTING_JSON}" -ge "${TOTAL_IMAGES}" ]]; then
  echo "[gra_dico] ${LABEL} already complete (${EXISTING_JSON}/${TOTAL_IMAGES}); skipping."
  exit 0
fi

ACTIVE_WINDOW="${OVERSEG_ACTIVE_WINDOW_SECONDS:-300}"
if [[ "${EXISTING_JSON}" -gt 0 \
    && "${OVERSEG_FORCE_RESUME:-0}" != "1" \
    && "${OVERSEG_ALLOW_CONCURRENT:-0}" != "1" ]]; then
  LATEST_MTIME="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%T@\n' | sort -nr | sed -n '1p')"
  LATEST_EPOCH="${LATEST_MTIME%.*}"
  NOW_EPOCH="$(date +%s)"
  OUTPUT_AGE=$(( NOW_EPOCH - LATEST_EPOCH ))
  if [[ "${OUTPUT_AGE}" -le "${ACTIVE_WINDOW}" ]]; then
    echo "[gra_dico] ${LABEL} has an active writer (${EXISTING_JSON}/${TOTAL_IMAGES}, latest ${OUTPUT_AGE}s ago); skipping this duplicate run."
    exit 0
  fi
fi

if [[ "${OVERSEG_ALLOW_CONCURRENT:-0}" != "1" ]]; then
  LOCK_FILE="${OUTPUT_DIR}/.oversegmentation.lock"
  exec {STAGE_LOCK_FD}>"${LOCK_FILE}"
  if ! flock -n "${STAGE_LOCK_FD}"; then
    echo "[gra_dico] ${LABEL} is locked by another run; skipping this duplicate run."
    exit 0
  fi
fi

if [[ "${EXISTING_JSON}" -gt 0 ]]; then
  echo "[gra_dico] resuming ${LABEL} from ${EXISTING_JSON}/${TOTAL_IMAGES} existing JSON files."
fi

echo "[gra_dico] Python: $(command -v "${PY}")"
"${PY}" -c 'import sys, torch; print(f"[gra_dico] Python version: {sys.version.split()[0]}"); print(f"[gra_dico] PyTorch: {torch.__version__}")'

if ! "${PY}" -c 'import torch; import MultiScaleDeformableAttention' >/dev/null 2>&1; then
  die "the MultiScaleDeformableAttention op is missing or incompatible with this PyTorch.
  Build it from source (see INSTALL.md):
    (cd ${SCRIPT_DIR}/mask2former/modeling/pixel_decoder/ops && bash make.sh)"
fi
"${PY}" -c 'import torch; import MultiScaleDeformableAttention as op; print(f"[gra_dico] MSDA: {op.__file__}")'

CHUNK_SIZE=$(( (TOTAL_IMAGES + NUM_GPUS - 1) / NUM_GPUS ))

PIDS=()
run_shard() {
  local gpu="$1"
  local start_id="$2"
  local end_id="$3"
  echo "[gra_dico] ${LABEL} gpu=${gpu} start=${start_id} end=${end_id} total=${TOTAL_IMAGES}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" gra_divide_conquer.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --cutler-weights "${CUTLER_WEIGHTS}" \
    --dinov3-repo "${DINOV3_REPO}" \
    --dinov3-weights "${DINOV3_WEIGHTS}" \
    --confidence-threshold 0.3 \
    --thetas 0.9 0.8 0.7 0.6 0.5 \
    --NMS-iou 0.8 \
    --patch-size 16 \
    --local-size 768 \
    --postprocess True \
    --preprocess True \
    --granularity True \
    --start-id "${start_id}" \
    --end-id "${end_id}" \
    --detection-per-image 1000 &
  PIDS+=("$!")
}

for i in "${!GPU_LIST[@]}"; do
  start_id=$(( i * CHUNK_SIZE ))
  end_id=$(( (i + 1) * CHUNK_SIZE ))
  if [[ "${end_id}" -gt "${TOTAL_IMAGES}" ]]; then
    end_id="${TOTAL_IMAGES}"
  fi
  if [[ "${start_id}" -ge "${end_id}" ]]; then
    continue
  fi
  run_shard "${GPU_LIST[$i]}" "${start_id}" "${end_id}"
done

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAILED=$(( FAILED + 1 ))
done
[[ "${FAILED}" -eq 0 ]] || die "${FAILED} shard(s) failed for ${LABEL}."
echo "[gra_dico] ${LABEL} done: $(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)/${TOTAL_IMAGES} JSON files in ${OUTPUT_DIR}"
