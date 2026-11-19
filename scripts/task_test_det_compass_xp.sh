#!/usr/bin/env bash
# Evaluate a single-run UnSAMv2 model trained directly on DET-COMPASS pseudo-labels
# (sam2/training/train.sh configs/unsamv2_small_training_det_compass_xp*.yaml).
#
# Usage: [VAR=value ...] bash scripts/task_test_det_compass_xp.sh [RUN_NAME]
# Checkpoint: ${TRAIN_ROOT}/det_compass_xp_<RUN_NAME>/log/checkpoints/checkpoint.pt
# (TRAIN_ROOT default ${LAOX_RESULTS_ROOT}/train), e.g. RUN_NAME=divide_and_conquer_1000_clean
# for configs/unsamv2_small_training_det_compass_xp.yaml.
# Environment: DATASET [det_compass_xp], GRAS [0.1,...,1.0], GPUS [0],
# MASK_THRESHOLD [0], START_ID [0], END_ID [full test split], PYTHON [python].
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-python}"
LAOX_RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${REPO_ROOT}/results}"
TRAIN_ROOT="${TRAIN_ROOT:-${LAOX_RESULTS_ROOT}/train}"
COMMENT="${1:-}"

DATASET="${DATASET:-det_compass_xp}"
START_ID="${START_ID:-0}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
if [[ -z "${END_ID}" ]]; then
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
SAM2_PROFILE="${SAM2_PROFILE:-0}"
GPUS="${GPUS:-0}"
GRAS="${GRAS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
IFS=',' read -ra GRA_LIST <<< "${GRAS}"

NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPUs specified in GPUS." >&2
  exit 1
fi

if [[ -n "${COMMENT}" ]]; then
  CKPT_ROOT="${TRAIN_ROOT}/det_compass_xp_${COMMENT}"
  COMMENT_SUFFIX="_det_compass_xp_${COMMENT}_m${MASK_THRESHOLD}"
else
  CKPT_ROOT="${TRAIN_ROOT}/det_compass_xp"
  COMMENT_SUFFIX="_det_compass_xp"
fi

run_eval() {
  local gra="$1"
  local gpu="$2"
  local ckpt_path="${CKPT_ROOT}/log/checkpoints/checkpoint.pt"
  echo "[eval] dataset=${DATASET} gra=${gra} gpu=${gpu} ckpt=${ckpt_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    SAM2_PROFILE="${SAM2_PROFILE}" \
    "${PYTHON}" test_det_v2.py \
      --method unsamv2 \
      --dataset "${DATASET}" \
      --sam2-ckpt "${ckpt_path}" \
      --gra "${gra}" \
      --eval-mode bbox \
      --start_id "${START_ID}" \
      --end_id "${END_ID}" \
      --comment "${COMMENT_SUFFIX}" \
      --mask-threshold "${MASK_THRESHOLD}" \
      --device-id 0 &
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
