#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

COMMENT="${1:-}"
TRAIN_ROOT="${2:-${TRAIN_ROOT:-results/train}}"
CKPT_PATH_OVERRIDE="${3:-${CKPT_PATH:-}}"

if [[ -n "${COMMENT}" && -d "${TRAIN_ROOT}/${COMMENT}" ]]; then
  CKPT_ROOT="${TRAIN_ROOT}/${COMMENT}"
  COMMENT_SUFFIX="_${COMMENT}"
elif [[ -n "${COMMENT}" ]]; then
  CKPT_ROOT="${TRAIN_ROOT}/augmentation_restart_${COMMENT}"
  COMMENT_SUFFIX="_augmentation_restart_${COMMENT}"
else
  CKPT_ROOT="${TRAIN_ROOT}/augmentation_restart"
  COMMENT_SUFFIX="_augmentation_restart"
fi

DATASET="${DATASET:-det_compass_xp}"
STAGE="${STAGE:-easy}"
GRA="${GRA:-1.0}"
GRAS="${GRAS:-${GRA}}"
GPUS="${GPUS:-0}"
START_ID="${START_ID:-0}"
SAM_TYPE="${SAM_TYPE:-unsamv2}"
if [[ -z "${SAM2_CFG:-}" ]]; then
  if [[ "${SAM_TYPE}" == "sam2" ]]; then
    SAM2_CFG="sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
  else
    SAM2_CFG="sam2/configs/unsamv2_small.yaml"
  fi
fi
DEVICE="${DEVICE:-cuda}"
N_CLICKS="${N_CLICKS:-20}"
MIN_N_CLICKS="${MIN_N_CLICKS:-1}"
MAX_IMAGES="${MAX_IMAGES:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CKPT_SUFFIX="${CKPT_SUFFIX:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluation/interactive/outputs/xray_eval}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

if [[ -z "${END_ID:-}" ]]; then
  case "${DATASET}" in
    det_compass_xp) END_ID=307 ;;
    CLCXray) END_ID=956 ;;
    DvXray) END_ID=9155 ;;
    PIXray) END_ID=5047 ;;
    HiXray) END_ID=9069 ;;
    pidray) END_ID=18220 ;;
    *) END_ID=1928 ;;
  esac
fi

ckpt_name="checkpoint.pt"
if [[ -n "${CKPT_SUFFIX}" ]]; then
  if [[ "${CKPT_SUFFIX}" == checkpoint*.pt || "${CKPT_SUFFIX}" == *.pt ]]; then
    ckpt_name="${CKPT_SUFFIX}"
  else
    ckpt_name="checkpoint_${CKPT_SUFFIX}.pt"
  fi
fi

if [[ -n "${CKPT_PATH_OVERRIDE}" ]]; then
  CKPT_PATH="${CKPT_PATH_OVERRIDE}"
elif [[ -d "${CKPT_ROOT}/${STAGE}/log/checkpoints" ]]; then
  CKPT_PATH="${CKPT_ROOT}/${STAGE}/log/checkpoints/${ckpt_name}"
else
  CKPT_PATH="${CKPT_ROOT}/log/checkpoints/${ckpt_name}"
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
IFS=',' read -ra GRA_LIST <<< "${GRAS}"

NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPUs specified in GPUS." >&2
  exit 1
fi

run_eval() {
  local gra="$1"
  local gpu="$2"
  local file_tag="${DATASET}${COMMENT_SUFFIX}_${STAGE}_gra${gra}"
  if [[ -n "${CKPT_PATH_OVERRIDE}" ]]; then
    local ckpt_tag
    ckpt_tag="$(basename "${CKPT_PATH}")"
    ckpt_tag="${ckpt_tag%.pt}"
    file_tag="${file_tag}_${ckpt_tag}"
  fi
  local out_path="${OUTPUT_DIR}/${file_tag}.json"
  local log_path="${LOG_DIR}/${file_tag}.log"

  echo "[interactive_eval] dataset=${DATASET} stage=${STAGE} gra=${gra} sam_type=${SAM_TYPE} gpu=${gpu} ckpt=${CKPT_PATH}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python evaluation/interactive/xray_interactive_eval.py \
    --dataset "${DATASET}" \
    --sam2-ckpt "${CKPT_PATH}" \
    --sam-type "${SAM_TYPE}" \
    --sam2-cfg "${SAM2_CFG}" \
    --device "${DEVICE}" \
    --start-id "${START_ID}" \
    --end-id "${END_ID}" \
    --max-images "${MAX_IMAGES}" \
    --batch-size "${BATCH_SIZE}" \
    --n-clicks "${N_CLICKS}" \
    --min-n-clicks "${MIN_N_CLICKS}" \
    --gra "${gra}" \
    --output "${out_path}" | tee "${log_path}" &
}

job_count=0
idx=0
for gra in "${GRA_LIST[@]}"; do
  gpu="${GPU_LIST[$((idx % NUM_GPUS))]}"
  run_eval "${gra}" "${gpu}"
  idx=$((idx + 1))
  job_count=$((job_count + 1))
  if [[ "${job_count}" -ge "${NUM_GPUS}" ]]; then
    wait -n
    job_count=$((job_count - 1))
  fi
done
wait
