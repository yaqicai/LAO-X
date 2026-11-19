#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS=8
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

DATA_ROOT_BASE="/path/to/datasets"
DATA_ROOT_NAME=""
DATA_PROFILE=""
WRITE_ROOT=""
LOG_ROOT="${LOG_ROOT:-/path/to/LAO-X/results/train/augmentation_restart}"
CONFIG="configs/unsamv2_small_training_augmentation.yaml"
COMMENT=""
# The paper's original curriculum experiments use the fused annotations.
# Pass --anno-comment none explicitly only for an ablation using the base annotations.
ANNO_COMMENT="fusion"
STAGES_CSV=""
BASE_LR=""
VISION_LR=""
TRAIN_BATCH_SIZE=""
SEED="1"
TOTAL_EPOCHS=""
NUM_TRAIN_WORKERS=""
PHASES_PER_EPOCH=""
START_ID=""
END_ID=""
STAGE_RANGES=""
STAGE_BASE_LRS=""
STAGE_VISION_LRS=""
STAGE_EPOCHS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages)
      STAGES_CSV="$2"
      shift 2
      ;;
    --anno-comment)
      ANNO_COMMENT="$2"
      shift 2
      ;;
    --data-root-name)
      DATA_ROOT_NAME="$2"
      shift 2
      ;;
    --data-profile)
      DATA_PROFILE="$2"
      shift 2
      ;;
    --write-root)
      WRITE_ROOT="$2"
      shift 2
      ;;
    --base-lr)
      BASE_LR="$2"
      shift 2
      ;;
    --vision-lr)
      VISION_LR="$2"
      shift 2
      ;;
    --batch-size)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --epochs)
      TOTAL_EPOCHS="$2"
      shift 2
      ;;
    --num-workers)
      NUM_TRAIN_WORKERS="$2"
      shift 2
      ;;
    --phases-per-epoch)
      PHASES_PER_EPOCH="$2"
      shift 2
      ;;
    --start-id)
      START_ID="$2"
      shift 2
      ;;
    --end-id)
      END_ID="$2"
      shift 2
      ;;
    --stage-ranges)
      STAGE_RANGES="$2"
      shift 2
      ;;
    --stage-base-lrs)
      STAGE_BASE_LRS="$2"
      shift 2
      ;;
    --stage-vision-lrs)
      STAGE_VISION_LRS="$2"
      shift 2
      ;;
    --stage-epochs)
      STAGE_EPOCHS="$2"
      shift 2
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

if [[ -z "${DATA_ROOT_NAME}" ]]; then
  echo "Missing --data-root-name." >&2
  exit 1
fi

if [[ ! "${SEED}" =~ ^[0-9]+$ ]] || (( SEED > 4294967295 )); then
  echo "--seed must be an integer in [0, 4294967295]." >&2
  exit 1
fi
if [[ -n "${DATA_PROFILE}" && ! "${DATA_PROFILE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--data-profile may contain only letters, numbers, dot, underscore, and dash." >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_BASE}/${DATA_ROOT_NAME}}"
RUNTIME_ENV=()
if [[ -n "${WRITE_ROOT}" ]]; then
  if [[ "${WRITE_ROOT}" != /* || "${WRITE_ROOT}" == "/" ]]; then
    echo "--write-root must be an absolute directory other than /." >&2
    exit 1
  fi
  WRITE_ROOT="${WRITE_ROOT%/}"
  RUNTIME_ROOT="${WRITE_ROOT}/runtime"
  LOG_ROOT="${WRITE_ROOT}/results/train/augmentation_restart"
  RUNTIME_DIRS=(
    "${RUNTIME_ROOT}/home"
    "${RUNTIME_ROOT}/tmp"
    "${RUNTIME_ROOT}/cache"
    "${RUNTIME_ROOT}/cache/cuda"
    "${RUNTIME_ROOT}/cache/huggingface"
    "${RUNTIME_ROOT}/cache/matplotlib"
    "${RUNTIME_ROOT}/cache/numba"
    "${RUNTIME_ROOT}/cache/pycache"
    "${RUNTIME_ROOT}/cache/torch"
    "${RUNTIME_ROOT}/cache/torch_extensions"
    "${RUNTIME_ROOT}/cache/triton"
    "${RUNTIME_ROOT}/config"
    "${RUNTIME_ROOT}/data"
    "${RUNTIME_ROOT}/distributed"
  )
  for runtime_dir in "${RUNTIME_DIRS[@]}"; do
    if ! mkdir -p "${runtime_dir}" || [[ ! -w "${runtime_dir}" ]]; then
      echo "Cannot create or write runtime directory: ${runtime_dir}" >&2
      exit 1
    fi
  done
  RUNTIME_ENV=(
    "HOME=${RUNTIME_ROOT}/home"
    "TMPDIR=${RUNTIME_ROOT}/tmp"
    "TMP=${RUNTIME_ROOT}/tmp"
    "TEMP=${RUNTIME_ROOT}/tmp"
    "XDG_CACHE_HOME=${RUNTIME_ROOT}/cache"
    "XDG_CONFIG_HOME=${RUNTIME_ROOT}/config"
    "XDG_DATA_HOME=${RUNTIME_ROOT}/data"
    "PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/cache/pycache"
    "TORCH_HOME=${RUNTIME_ROOT}/cache/torch"
    "TORCH_EXTENSIONS_DIR=${RUNTIME_ROOT}/cache/torch_extensions"
    "TRITON_CACHE_DIR=${RUNTIME_ROOT}/cache/triton"
    "CUDA_CACHE_PATH=${RUNTIME_ROOT}/cache/cuda"
    "HF_HOME=${RUNTIME_ROOT}/cache/huggingface"
    "MPLCONFIGDIR=${RUNTIME_ROOT}/cache/matplotlib"
    "NUMBA_CACHE_DIR=${RUNTIME_ROOT}/cache/numba"
    "EXP_DIR=${RUNTIME_ROOT}/distributed"
  )
fi
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
STAGES_CSV="${STAGES_CSV:-${STAGES:-easy,hard,extreme}}"
IFS=',' read -ra STAGES <<< "${STAGES_CSV}"
if [[ -n "${COMMENT}" ]]; then
  EXPERIMENT_ROOT="${LOG_ROOT}_${COMMENT}"
else
  EXPERIMENT_ROOT="${LOG_ROOT}"
fi
if [[ -n "${DATA_PROFILE}" ]]; then
  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}_data${DATA_PROFILE}"
fi
INIT_CKPT=""

echo "==> Experiment root: ${EXPERIMENT_ROOT}"
if [[ -n "${WRITE_ROOT}" ]]; then
  echo "    WRITE_ROOT=${WRITE_ROOT}"
fi
if [[ -n "${DATA_PROFILE}" ]]; then
  echo "    DATA_PROFILE=${DATA_PROFILE}"
fi

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
  if [[ -n "${STAGE_RANGES}" ]]; then
    IFS=',' read -ra RANGE_ITEMS <<< "${STAGE_RANGES}"
    for item in "${RANGE_ITEMS[@]}"; do
      name="${item%%:*}"
      range="${item#*:}"
      if [[ "${name}" == "${stage}" || "${name}" == "${data_stage}" ]]; then
        stage_start_id="${range%-*}"
        stage_end_id="${range#*-}"
        break
      fi
    done
  fi

  stage_base_lr="${BASE_LR}"
  stage_vision_lr="${VISION_LR}"
  stage_epochs="${TOTAL_EPOCHS}"
  if [[ -n "${STAGE_BASE_LRS}" ]]; then
    IFS=',' read -ra LR_ITEMS <<< "${STAGE_BASE_LRS}"
    for item in "${LR_ITEMS[@]}"; do
      name="${item%%:*}"
      val="${item#*:}"
      if [[ "${name}" == "${stage}" || "${name}" == "${data_stage}" ]]; then
        stage_base_lr="${val}"
        break
      fi
    done
  fi
  if [[ -n "${STAGE_VISION_LRS}" ]]; then
    IFS=',' read -ra VLR_ITEMS <<< "${STAGE_VISION_LRS}"
    for item in "${VLR_ITEMS[@]}"; do
      name="${item%%:*}"
      val="${item#*:}"
      if [[ "${name}" == "${stage}" || "${name}" == "${data_stage}" ]]; then
        stage_vision_lr="${val}"
        break
      fi
    done
  fi
  if [[ -n "${STAGE_EPOCHS}" ]]; then
    IFS=',' read -ra EPOCH_ITEMS <<< "${STAGE_EPOCHS}"
    for item in "${EPOCH_ITEMS[@]}"; do
      name="${item%%:*}"
      val="${item#*:}"
      if [[ "${name}" == "${stage}" || "${name}" == "${data_stage}" ]]; then
        stage_epochs="${val}"
        break
      fi
    done
  fi

  IMG_FOLDER="${DATA_ROOT}/${data_stage}/xray"
  if [[ "${ANNO_COMMENT}" == "none" || "${ANNO_COMMENT}" == "base" ]]; then
    GT_FOLDER="${DATA_ROOT}/${data_stage}/anno"
  elif [[ -n "${ANNO_COMMENT}" ]]; then
    GT_FOLDER="${DATA_ROOT}/${data_stage}/anno_${ANNO_COMMENT}"
  else
    GT_FOLDER="${DATA_ROOT}/${data_stage}/anno"
  fi
  if [[ ! -d "${GT_FOLDER}" ]]; then
    echo "Annotation folder does not exist: ${GT_FOLDER}" >&2
    exit 1
  fi
  EXPERIMENT_LOG_DIR="${EXPERIMENT_ROOT}/${stage}/log"
  if ! mkdir -p "${EXPERIMENT_LOG_DIR}"; then
    echo "Cannot create experiment log directory: ${EXPERIMENT_LOG_DIR}" >&2
    echo "Grant this account write access to $(dirname "${EXPERIMENT_ROOT}") or set LOG_ROOT to a writable path." >&2
    exit 1
  fi
  if [[ ! -w "${EXPERIMENT_LOG_DIR}" ]]; then
    echo "Experiment log directory is not writable: ${EXPERIMENT_LOG_DIR}" >&2
    echo "Grant this account write access or set LOG_ROOT to a writable path." >&2
    exit 1
  fi

  stage_ckpt="${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint.pt"
  stage_epoch_ckpt=""
  if [[ -n "${stage_epochs}" ]]; then
    stage_epoch_ckpt="${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint_${stage_epochs}.pt"
  fi
  if [[ -f "${stage_ckpt}" && ( -z "${stage_epoch_ckpt}" || -f "${stage_epoch_ckpt}" ) ]]; then
    resolved_config="${EXPERIMENT_LOG_DIR}/config_resolved.yaml"
    if [[ -f "${resolved_config}" ]] && ! grep -Fq "gt_folder: ${GT_FOLDER}" "${resolved_config}"; then
      echo "Existing stage uses a different annotation folder: ${stage}" >&2
      echo "Expected: ${GT_FOLDER}" >&2
      grep -m1 "gt_folder:" "${resolved_config}" >&2 || true
      echo "Use a new experiment comment; the existing checkpoint must not be reused." >&2
      exit 1
    fi
    echo "==> Stage: ${stage} already finished, skipping."
    INIT_CKPT="${stage_ckpt}"
    continue
  fi

  echo "==> Stage: ${stage}"
  echo "    IMG_FOLDER=${IMG_FOLDER}"
  echo "    GT_FOLDER=${GT_FOLDER}"
  echo "    EXPERIMENT_LOG_DIR=${EXPERIMENT_LOG_DIR}"
  echo "    SEED=${SEED}"
  echo "    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} (global batch=$((NUM_GPUS * TRAIN_BATCH_SIZE)))"
  if [[ -n "${INIT_CKPT}" ]]; then
    echo "    INIT_CKPT=${INIT_CKPT}"
  fi
  if [[ -n "${stage_start_id}" || -n "${stage_end_id}" ]]; then
    echo "    START_ID=${stage_start_id} END_ID=${stage_end_id}"
  fi
  if [[ -n "${stage_base_lr}" ]]; then
    echo "    BASE_LR=${stage_base_lr}"
  fi
  if [[ -n "${stage_vision_lr}" ]]; then
    echo "    VISION_LR=${stage_vision_lr}"
  fi
  if [[ -n "${stage_epochs}" ]]; then
    echo "    TOTAL_EPOCHS=${stage_epochs}"
  fi

  if [[ -n "${INIT_CKPT}" ]]; then
    INIT_ENV=("INIT_CKPT=${INIT_CKPT}")
  else
    INIT_ENV=()
  fi
  HYPER_ENV=()
  if [[ -n "${stage_base_lr}" ]]; then
    HYPER_ENV+=("BASE_LR=${stage_base_lr}")
  fi
  if [[ -n "${stage_vision_lr}" ]]; then
    HYPER_ENV+=("VISION_LR=${stage_vision_lr}")
  fi
  if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
    HYPER_ENV+=("TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}")
  fi
  if [[ -n "${stage_epochs}" ]]; then
    HYPER_ENV+=("TOTAL_EPOCHS=${stage_epochs}")
  fi
  if [[ -n "${NUM_TRAIN_WORKERS}" ]]; then
    HYPER_ENV+=("NUM_TRAIN_WORKERS=${NUM_TRAIN_WORKERS}")
  fi
  if [[ -n "${PHASES_PER_EPOCH}" ]]; then
    HYPER_ENV+=("PHASES_PER_EPOCH=${PHASES_PER_EPOCH}")
  fi
  if [[ -n "${stage_start_id}" ]]; then
    HYPER_ENV+=("START_ID=${stage_start_id}")
  fi
  if [[ -n "${stage_end_id}" ]]; then
    HYPER_ENV+=("END_ID=${stage_end_id}")
  fi

  env \
    "${RUNTIME_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" \
    CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
    PYTHONHASHSEED="${SEED}" \
    TRAIN_SEED="${SEED}" \
    IMG_FOLDER="${IMG_FOLDER}" \
    GT_FOLDER="${GT_FOLDER}" \
    EXPERIMENT_LOG_DIR="${EXPERIMENT_LOG_DIR}" \
    "${INIT_ENV[@]}" \
    "${HYPER_ENV[@]}" \
    python "${SCRIPT_DIR}/train.py" \
      -c "${CONFIG}" \
      --use-cluster 0 \
      --num-gpus "${NUM_GPUS}"

  INIT_CKPT="${EXPERIMENT_LOG_DIR}/checkpoints/checkpoint.pt"
done
