#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/sam2/training/train_curriculum_augmentation_restart.sh"

COMMENTS="${COMMENTS:-${1:-}}"
if [[ -z "${COMMENTS}" ]]; then
  echo "COMMENTS is required (comma-separated), e.g. COMMENTS=fusion_095_v2,fusion_090_v1" >&2
  exit 1
fi

GPUS="${GPUS:-0}"
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
IFS=',' read -ra COMMENT_LIST <<< "${COMMENTS}"

NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPUs specified in GPUS." >&2
  exit 1
fi

run_train() {
  local comment="$1"
  local gpu="$2"
  echo "[train] comment=${comment} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    bash "${TRAIN_SCRIPT}" "${comment}" &
}

job_count=0
idx=0
for comment in "${COMMENT_LIST[@]}"; do
  gpu="${GPU_LIST[$((idx % NUM_GPUS))]}"
  run_train "${comment}" "${gpu}"
  idx=$((idx + 1))
  job_count=$((job_count + 1))
  if [[ "${job_count}" -ge "${NUM_GPUS}" ]]; then
    wait -n
    job_count=$((job_count - 1))
  fi
done
wait
