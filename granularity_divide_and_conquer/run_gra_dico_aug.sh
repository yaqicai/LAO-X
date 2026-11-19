#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_PYTHON="/path/to/conda/envs/UnSAMv2_5090/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

SUFFIX="${1:-}"
STAGE="${2:-}"

if [[ -z "${STAGE}" ]]; then
  echo "Usage: $0 [SUFFIX] <STAGE>" >&2
  exit 1
fi

if [[ -n "${SUFFIX}" ]]; then
  DATA_ROOT="/path/to/datasets/${SUFFIX}"
else
  DATA_ROOT="/path/to/datasets/unsam_aug_v2"
fi

GPUS="${GPUS:-0}"
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPUs specified in GPUS." >&2
  exit 1
fi

INPUT_DIR="${DATA_ROOT}/${STAGE}/xray"
OUTPUT_DIR="${DATA_ROOT}/${STAGE}/anno_cutler"

TOTAL_IMAGES="$(find "${INPUT_DIR}" -maxdepth 1 -type f ! -name "*.json" ! -name ".DS_Store" | wc -l)"
if [[ "${TOTAL_IMAGES}" -le 0 ]]; then
  echo "No images found in ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Avoid duplicate work when separate containers target the same dataset stage.
# A recently updated output directory is treated as owned by another run. Set
# OVERSEG_FORCE_RESUME=1 to take over immediately after stopping the old run.
# Set OVERSEG_ALLOW_CONCURRENT=1 only when duplicate workers are intentional.
EXISTING_JSON="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)"
if [[ "${EXISTING_JSON}" -ge "${TOTAL_IMAGES}" ]]; then
  echo "[gra_dico] stage=${STAGE} already complete (${EXISTING_JSON}/${TOTAL_IMAGES}); skipping."
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
    echo "[gra_dico] stage=${STAGE} has an active writer (${EXISTING_JSON}/${TOTAL_IMAGES}, latest ${OUTPUT_AGE}s ago); skipping this duplicate run."
    exit 0
  fi
fi

if [[ "${OVERSEG_ALLOW_CONCURRENT:-0}" != "1" ]]; then
  LOCK_FILE="${OUTPUT_DIR}/.oversegmentation.lock"
  exec {STAGE_LOCK_FD}>"${LOCK_FILE}"
  if ! flock -n "${STAGE_LOCK_FD}"; then
    echo "[gra_dico] stage=${STAGE} is locked by another container; skipping this duplicate run."
    exit 0
  fi
fi

if [[ "${EXISTING_JSON}" -gt 0 ]]; then
  echo "[gra_dico] resuming stage=${STAGE} from ${EXISTING_JSON}/${TOTAL_IMAGES} existing JSON files."
fi

echo "[gra_dico] Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys, torch; print(f"[gra_dico] Python version: {sys.version.split()[0]}"); print(f"[gra_dico] PyTorch: {torch.__version__}")'

if ! "${PYTHON_BIN}" -c 'import torch; import MultiScaleDeformableAttention' >/dev/null 2>&1; then
  MSDA_OPS_DIR="${SCRIPT_DIR}/mask2former/modeling/pixel_decoder/ops"
  MSDA_WHEEL="${MSDA_OPS_DIR}/prebuilt/torch2.7.1_cu128_py310_sm120/multiscaledeformableattention-1.0-cp310-cp310-linux_x86_64.whl"
  echo "[gra_dico] MultiScaleDeformableAttention is missing or incompatible."

  if [[ -f "${MSDA_WHEEL}" ]] && "${PYTHON_BIN}" -m pip install \
      --no-deps --force-reinstall "${MSDA_WHEEL}"; then
    echo "[gra_dico] Installed the prebuilt CUDA 12.8/sm_120 extension."
  else
    echo "[gra_dico] Prebuilt installation unavailable; rebuilding the extension."
    PYTHON_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
    env \
      PATH="$(dirname "${PYTHON_BIN}"):${PATH}" \
      CONDA_PREFIX="${PYTHON_PREFIX}" \
      bash "${MSDA_OPS_DIR}/make.sh"
  fi
fi

"${PYTHON_BIN}" -c 'import torch; import MultiScaleDeformableAttention as op; print(f"[gra_dico] MSDA: {op.__file__}")'

CHUNK_SIZE=$(( (TOTAL_IMAGES + NUM_GPUS - 1) / NUM_GPUS ))

run_shard() {
  local gpu="$1"
  local start_id="$2"
  local end_id="$3"
  echo "[gra_dico] stage=${STAGE} gpu=${gpu} start=${start_id} end=${end_id} total=${TOTAL_IMAGES}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" gra_divide_conquer.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
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
    --detection-per-image 1000 \
    --opts MODEL.WEIGHTS /path/to/LAO-X/granularity_divide_and_conquer/ckpts/cutler_cascade_final.pth &
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

wait
