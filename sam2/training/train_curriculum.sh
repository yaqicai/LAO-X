#!/usr/bin/env bash
# Easy -> hard -> extreme curriculum training of UnSAMv2 (SAM 2.1 Hiera-S) on
# the LAO-X synthetic X-ray stages. Every stage is a separate run that starts
# from the final checkpoint of the previous stage (the first stage starts from
# INIT_CKPT); optimizer state and epoch counters are reset between stages.
#
# Usage:
#   bash sam2/training/train_curriculum.sh EXPERIMENT_NAME --data-root-name NAME [options]
#
# Paper recipe (8 GPUs x batch size 2 = global batch 16):
#   bash sam2/training/train_curriculum.sh crop_musc_aug_xp_fusion_bs16 \
#     --data-root-name crop_aug_xp --anno-comment fusion \
#     --stages easy,hard,extreme \
#     --stage-ranges easy:0-4000,hard:0-6000,extreme:0-8000 \
#     --stage-base-lrs easy:1.6e-4,hard:1.2e-4,extreme:1e-4 \
#     --stage-vision-lrs easy:4.8e-6,hard:3.2e-6,extreme:3e-6 \
#     --stage-epochs easy:80,hard:64,extreme:48 \
#     --batch-size 2 --num-gpus 8
# The same global batch on 4 GPUs: --num-gpus 4 --batch-size 4.
#
# Data layout: ${LAOX_DATA_ROOT}/<data-root-name>/<stage>/xray/*.png with
# pseudo-labels in <stage>/anno_<anno-comment>/ (or <stage>/anno/ when
# --anno-comment is empty, "none" or "base"). A stage name that is not a
# directory is mapped to easy/hard/extreme/merge by substring.
# Outputs: ${LOG_ROOT}_<EXPERIMENT_NAME>/<stage>/log/checkpoints/checkpoint[_<epoch>].pt
# Finished stages (checkpoint.pt and checkpoint_<stage-epochs>.pt present) are
# skipped, so an interrupted run can simply be relaunched.
#
# Options:
#   --data-root-name NAME     dataset directory under LAOX_DATA_ROOT (or set DATA_ROOT)
#   --anno-comment NAME       pseudo-label folder suffix (anno_NAME)
#   --stages a,b,c            stages to train in order (default: easy,hard,extreme)
#   --stage-ranges s:a-b,...  per-stage image id range [a, b)
#   --stage-base-lrs s:v,...  per-stage base LR        (default: config value)
#   --stage-vision-lrs s:v,.. per-stage vision LR      (default: config value)
#   --stage-epochs s:v,...    per-stage epochs         (default: config value)
#   --base-lr/--vision-lr/--epochs/--start-id/--end-id V   values for all stages
#   --batch-size N            per-GPU batch size       (default: config value)
#   --num-workers N           dataloader workers per GPU
#   --phases-per-epoch N
#   --num-gpus N              GPUs per run (default: number of GPUs in
#                             CUDA_VISIBLE_DEVICES/GPUS, otherwise 8)
#   --seed N                  training seed, TRAIN_SEED/PYTHONHASHSEED (default: 1)
#   --config NAME             Hydra config (default: configs/train_laox_small.yaml)
#   --log-root DIR            default: ${LAOX_RESULTS_ROOT}/train/augmentation_restart
#
# Environment: LAOX_DATA_ROOT, LAOX_RESULTS_ROOT, LAOX_CKPT_ROOT (default
# <repo>/datasets, <repo>/results, <repo>/checkpoints), INIT_CKPT (default
# ${LAOX_CKPT_ROOT}/sam2.1_hiera_small.pt; use the Hiera-L checkpoint with
# configs/train_laox_large.yaml), PYTHON (default python),
# CUDA_VISIBLE_DEVICES or GPUS.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

export LAOX_DATA_ROOT="${LAOX_DATA_ROOT:-${REPO_ROOT}/datasets}"
export LAOX_RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${REPO_ROOT}/results}"
export LAOX_CKPT_ROOT="${LAOX_CKPT_ROOT:-${REPO_ROOT}/checkpoints}"

DATA_ROOT_NAME=""
LOG_ROOT="${LOG_ROOT:-${LAOX_RESULTS_ROOT}/train/augmentation_restart}"
CONFIG="configs/train_laox_small.yaml"
COMMENT=""
ANNO_COMMENT=""
STAGES_CSV=""
BASE_LR=""
VISION_LR=""
TRAIN_BATCH_SIZE=""
TOTAL_EPOCHS=""
NUM_TRAIN_WORKERS=""
PHASES_PER_EPOCH=""
START_ID=""
END_ID=""
STAGE_RANGES=""
STAGE_BASE_LRS=""
STAGE_VISION_LRS=""
STAGE_EPOCHS=""
NUM_GPUS=""
SEED="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages) STAGES_CSV="$2"; shift 2 ;;
    --anno-comment) ANNO_COMMENT="$2"; shift 2 ;;
    --data-root-name) DATA_ROOT_NAME="$2"; shift 2 ;;
    --base-lr) BASE_LR="$2"; shift 2 ;;
    --vision-lr) VISION_LR="$2"; shift 2 ;;
    --batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --num-workers) NUM_TRAIN_WORKERS="$2"; shift 2 ;;
    --phases-per-epoch) PHASES_PER_EPOCH="$2"; shift 2 ;;
    --start-id) START_ID="$2"; shift 2 ;;
    --end-id) END_ID="$2"; shift 2 ;;
    --stage-ranges) STAGE_RANGES="$2"; shift 2 ;;
    --stage-base-lrs) STAGE_BASE_LRS="$2"; shift 2 ;;
    --stage-vision-lrs) STAGE_VISION_LRS="$2"; shift 2 ;;
    --stage-epochs) STAGE_EPOCHS="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^set -e$/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
      exit 0
      ;;
    *)
      if [[ -z "${COMMENT}" ]]; then
        COMMENT="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "${DATA_ROOT_NAME}" && -z "${DATA_ROOT:-}" ]]; then
  echo "Missing --data-root-name (or DATA_ROOT)." >&2
  exit 1
fi
DATA_ROOT="${DATA_ROOT:-${LAOX_DATA_ROOT}/${DATA_ROOT_NAME}}"

# GPUs: CUDA_VISIBLE_DEVICES, else GPUS, else 0..NUM_GPUS-1.
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-${GPUS:-}}"
if [[ -z "${NUM_GPUS}" ]]; then
  if [[ -n "${VISIBLE_GPUS}" ]]; then
    IFS=',' read -ra _GPU_LIST <<< "${VISIBLE_GPUS}"
    NUM_GPUS="${#_GPU_LIST[@]}"
  else
    NUM_GPUS=8
  fi
fi
if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--num-gpus must be a positive integer." >&2
  exit 1
fi
if [[ -z "${VISIBLE_GPUS}" ]]; then
  VISIBLE_GPUS="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]] || (( SEED > 4294967295 )); then
  echo "--seed must be an integer in [0, 4294967295]." >&2
  exit 1
fi

STAGES_CSV="${STAGES_CSV:-${STAGES:-easy,hard,extreme}}"
IFS=',' read -ra STAGES <<< "${STAGES_CSV}"
if [[ -n "${COMMENT}" ]]; then
  EXPERIMENT_ROOT="${LOG_ROOT}_${COMMENT}"
else
  EXPERIMENT_ROOT="${LOG_ROOT}"
fi
# Initial weights of the first stage; later stages start from the previous stage.
INIT_CKPT="${INIT_CKPT:-${LAOX_CKPT_ROOT}/sam2.1_hiera_small.pt}"

# lookup_stage_value "s1:v1,s2:v2" STAGE DATA_STAGE DEFAULT
lookup_stage_value() {
  local spec="$1" stage="$2" data_stage="$3" value="$4" item name
  if [[ -n "${spec}" ]]; then
    IFS=',' read -ra _ITEMS <<< "${spec}"
    for item in "${_ITEMS[@]}"; do
      name="${item%%:*}"
      if [[ "${name}" == "${stage}" || "${name}" == "${data_stage}" ]]; then
        value="${item#*:}"
        break
      fi
    done
  fi
  printf '%s' "${value}"
}

echo "==> Experiment root: ${EXPERIMENT_ROOT}"
echo "    DATA_ROOT=${DATA_ROOT} CONFIG=${CONFIG} NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS}"

for stage in "${STAGES[@]}"; do
  data_stage="${stage}"
  if [[ -d "${DATA_ROOT}/${stage}" ]]; then
    data_stage="${stage}"
  elif [[ "${stage}" == *"easy"* ]]; then
    data_stage="easy"
  elif [[ "${stage}" == *"hard"* ]]; then
    data_stage="hard"
  elif [[ "${stage}" == *"extreme"* ]]; then
    data_stage="extreme"
  elif [[ "${stage}" == *"merge"* ]]; then
    data_stage="merge"
  fi

  stage_start_id="${START_ID}"
  stage_end_id="${END_ID}"
  stage_range="$(lookup_stage_value "${STAGE_RANGES}" "${stage}" "${data_stage}" "")"
  if [[ -n "${stage_range}" ]]; then
    stage_start_id="${stage_range%-*}"
    stage_end_id="${stage_range#*-}"
  fi
  stage_base_lr="$(lookup_stage_value "${STAGE_BASE_LRS}" "${stage}" "${data_stage}" "${BASE_LR}")"
  stage_vision_lr="$(lookup_stage_value "${STAGE_VISION_LRS}" "${stage}" "${data_stage}" "${VISION_LR}")"
  stage_epochs="$(lookup_stage_value "${STAGE_EPOCHS}" "${stage}" "${data_stage}" "${TOTAL_EPOCHS}")"

  IMG_FOLDER="${DATA_ROOT}/${data_stage}/xray"
  if [[ -z "${ANNO_COMMENT}" || "${ANNO_COMMENT}" == "none" || "${ANNO_COMMENT}" == "base" ]]; then
    GT_FOLDER="${DATA_ROOT}/${data_stage}/anno"
  else
    GT_FOLDER="${DATA_ROOT}/${data_stage}/anno_${ANNO_COMMENT}"
  fi
  EXPERIMENT_LOG_DIR="${EXPERIMENT_ROOT}/${stage}/log"

  stage_ckpt="${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint.pt"
  stage_epoch_ckpt=""
  if [[ -n "${stage_epochs}" ]]; then
    stage_epoch_ckpt="${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint_${stage_epochs}.pt"
  fi
  if [[ -f "${stage_ckpt}" && ( -z "${stage_epoch_ckpt}" || -f "${stage_epoch_ckpt}" ) ]]; then
    echo "==> Stage: ${stage} already finished, skipping."
    INIT_CKPT="${stage_ckpt}"
    continue
  fi

  if [[ ! -f "${INIT_CKPT}" ]]; then
    echo "Initial checkpoint not found: ${INIT_CKPT}" >&2
    echo "Download it with sam2/checkpoints/download_ckpts.sh or set INIT_CKPT." >&2
    exit 1
  fi

  echo "==> Stage: ${stage}"
  echo "    IMG_FOLDER=${IMG_FOLDER}"
  echo "    GT_FOLDER=${GT_FOLDER}"
  echo "    EXPERIMENT_LOG_DIR=${EXPERIMENT_LOG_DIR}"
  echo "    INIT_CKPT=${INIT_CKPT}"
  if [[ -n "${stage_start_id}" || -n "${stage_end_id}" ]]; then
    echo "    START_ID=${stage_start_id} END_ID=${stage_end_id}"
  fi
  [[ -n "${stage_base_lr}" ]] && echo "    BASE_LR=${stage_base_lr}"
  [[ -n "${stage_vision_lr}" ]] && echo "    VISION_LR=${stage_vision_lr}"
  [[ -n "${stage_epochs}" ]] && echo "    TOTAL_EPOCHS=${stage_epochs}"
  [[ -n "${TRAIN_BATCH_SIZE}" ]] && echo "    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} (global batch=$((NUM_GPUS * TRAIN_BATCH_SIZE)))"
  echo "    SEED=${SEED}"

  # Only set what was given; everything else falls back to the config defaults.
  HYPER_ENV=()
  [[ -n "${stage_base_lr}" ]] && HYPER_ENV+=("BASE_LR=${stage_base_lr}")
  [[ -n "${stage_vision_lr}" ]] && HYPER_ENV+=("VISION_LR=${stage_vision_lr}")
  [[ -n "${TRAIN_BATCH_SIZE}" ]] && HYPER_ENV+=("TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}")
  [[ -n "${stage_epochs}" ]] && HYPER_ENV+=("TOTAL_EPOCHS=${stage_epochs}")
  [[ -n "${NUM_TRAIN_WORKERS}" ]] && HYPER_ENV+=("NUM_TRAIN_WORKERS=${NUM_TRAIN_WORKERS}")
  [[ -n "${PHASES_PER_EPOCH}" ]] && HYPER_ENV+=("PHASES_PER_EPOCH=${PHASES_PER_EPOCH}")
  [[ -n "${stage_start_id}" ]] && HYPER_ENV+=("START_ID=${stage_start_id}")
  [[ -n "${stage_end_id}" ]] && HYPER_ENV+=("END_ID=${stage_end_id}")

  # The training config enables torch deterministic algorithms, which needs a
  # fixed cuBLAS workspace configuration.
  env \
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" \
    CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
    PYTHONHASHSEED="${SEED}" \
    TRAIN_SEED="${SEED}" \
    IMG_FOLDER="${IMG_FOLDER}" \
    GT_FOLDER="${GT_FOLDER}" \
    EXPERIMENT_LOG_DIR="${EXPERIMENT_LOG_DIR}" \
    INIT_CKPT="${INIT_CKPT}" \
    "${HYPER_ENV[@]}" \
    "${PYTHON}" "${SCRIPT_DIR}/train.py" \
      -c "${CONFIG}" \
      --use-cluster 0 \
      --num-gpus "${NUM_GPUS}"

  INIT_CKPT="${stage_ckpt}"
done
