#!/usr/bin/env bash
# Class-agnostic detection evaluation of curriculum-trained UnSAMv2 checkpoints
# (sam2/training/train_curriculum.sh) with evaluate.py. One job per
# (stage, granularity) pair, distributed over the GPUs in GPUS.
#
# Usage:
#   [VAR=value ...] bash scripts/eval_curriculum.sh [EXPERIMENT_NAME] [TRAIN_ROOT]
#
# Checkpoint: ${TRAIN_ROOT}/augmentation_restart_<EXPERIMENT_NAME>/<stage>/log/checkpoints/checkpoint_<CKPT_SUFFIX>.pt
# (TRAIN_ROOT default ${LAOX_RESULTS_ROOT}/train), or an explicit SAM2_CKPT.
# Predictions and metrics: output_eval/<dataset, lower-case>_img<START_ID>_<END_ID>_unsamv2_<gra>_augmentation_restart_<EXPERIMENT_NAME>_<stage>_m<MASK_THRESHOLD>_<CKPT_SUFFIX>/
#
# Paper evaluation on DET-COMPASS:
#   DATASET=det_compass_xp STAGES=extreme GRAS=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
#   MASK_THRESHOLD=0.5 CKPT_SUFFIX=48 GPUS=0,1,2,3,4,5,6,7 \
#   bash scripts/eval_curriculum.sh crop_musc_aug_xp_fusion_bs16
# Cross-dataset: the same with DATASET=CLCXray, DvXray, HiXray, pidray or PIXray.
#
# Environment (defaults in brackets):
#   DATASET [det_compass_xp]   STAGES [easy]   GRAS [0.1,...,1.0]   GPUS [0]
#   MASK_THRESHOLD [0.5]   CKPT_SUFFIX [] (e.g. 48 -> checkpoint_48.pt; empty -> checkpoint.pt)
#   START_ID [0]   END_ID [full test split of DATASET]   SAVE_EVERY [100]
#   VAL_JSON, IMAGE_ROOT   override the annotation file / image directory
#   SAM2_CKPT              evaluate this checkpoint instead of the experiment's
#   MODEL_SCALE            passed to --model-scale (e.g. large); default: evaluate.py default
#   EVAL_SEED [1]   EVAL_WANDB [0]   MODEL_SEED, TRAIN_DATA_PROFILE (provenance only)
#   PYTHON [python]   LAOX_DATA_ROOT, LAOX_RESULTS_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-python}"
LAOX_RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${REPO_ROOT}/results}"

COMMENT="${1:-}"
TRAIN_ROOT="${2:-${TRAIN_ROOT:-${LAOX_RESULTS_ROOT}/train}}"
if [[ -n "${COMMENT}" ]]; then
  CKPT_ROOT="${TRAIN_ROOT}/augmentation_restart_${COMMENT}"
  COMMENT_SUFFIX="_augmentation_restart_${COMMENT}"
else
  CKPT_ROOT="${TRAIN_ROOT}/augmentation_restart"
  COMMENT_SUFFIX="_augmentation_restart"
fi

GPUS="${GPUS:-0}"
STAGES="${STAGES:-easy}"
GRAS="${GRAS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
DATASET="${DATASET:-det_compass_xp}"
VAL_JSON="${VAL_JSON:-}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
START_ID="${START_ID:-0}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
SAVE_EVERY="${SAVE_EVERY:-100}"
EVAL_SEED="${EVAL_SEED:-${SEED:-1}}"
EVAL_WANDB="${EVAL_WANDB:-0}"
MODEL_SEED="${MODEL_SEED:-}"
MODEL_SCALE="${MODEL_SCALE:-}"
TRAIN_DATA_PROFILE="${TRAIN_DATA_PROFILE:-}"
if [[ -z "${END_ID}" ]]; then
  case "${DATASET}" in
    det_compass_xp) END_ID=307 ;;
    CLCXray) END_ID=956 ;;
    DvXray) END_ID=9155 ;;
    PIXray) END_ID=5046 ;;
    HiXray) END_ID=9069 ;;
    pidray) END_ID=18220 ;;
    *) END_ID=1928 ;;
  esac
fi
CKPT_SUFFIX="${CKPT_SUFFIX:-}"
SAM2_PROFILE="${SAM2_PROFILE:-0}"
SAM2_CKPT="${SAM2_CKPT:-}"

DATASET_ARGS=()
if [[ -n "${VAL_JSON}" ]]; then
  if [[ ! -f "${VAL_JSON}" ]]; then
    echo "VAL_JSON does not exist: ${VAL_JSON}" >&2
    exit 1
  fi
  DATASET_ARGS+=(--val_json "${VAL_JSON}")
fi
if [[ -n "${IMAGE_ROOT}" ]]; then
  if [[ ! -d "${IMAGE_ROOT}" ]]; then
    echo "IMAGE_ROOT does not exist: ${IMAGE_ROOT}" >&2
    exit 1
  fi
  DATASET_ARGS+=(--image-root "${IMAGE_ROOT}")
fi
if [[ -n "${MODEL_SCALE}" ]]; then
  DATASET_ARGS+=(--model-scale "${MODEL_SCALE}")
fi

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
IFS=',' read -ra STAGE_LIST <<< "${STAGES}"
IFS=',' read -ra GRA_LIST <<< "${GRAS}"

NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPUs specified in GPUS." >&2
  exit 1
fi

run_eval() {
  local stage="$1"
  local gra="$2"
  local gpu="$3"
  local ckpt_name="checkpoint.pt"
  if [[ -n "${CKPT_SUFFIX}" ]]; then
    if [[ "${CKPT_SUFFIX}" == checkpoint*.pt || "${CKPT_SUFFIX}" == *.pt ]]; then
      ckpt_name="${CKPT_SUFFIX}"
    else
      ckpt_name="checkpoint_${CKPT_SUFFIX}.pt"
    fi
  fi

  local ckpt_path=""
  if [[ -n "${SAM2_CKPT}" ]]; then
    ckpt_path="${SAM2_CKPT}"
  else
    ckpt_path="${CKPT_ROOT}/${stage}/log/checkpoints/${ckpt_name}"
  fi
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "Checkpoint not found: ${ckpt_path}" >&2
    echo "Set SAM2_CKPT to evaluate a specific checkpoint." >&2
    exit 1
  fi

  local ckpt_suffix="${CKPT_SUFFIX}"
  if [[ -z "${ckpt_suffix}" ]]; then
    local ckpt_file
    ckpt_file="$(basename "${ckpt_path}")"
    ckpt_file="${ckpt_file%.pt}"
    if [[ "${ckpt_file}" == checkpoint_* ]]; then
      ckpt_suffix="${ckpt_file#checkpoint_}"
    elif [[ "${ckpt_file}" != "checkpoint" ]]; then
      ckpt_suffix="${ckpt_file}"
    fi
  fi

  local comment="${COMMENT_SUFFIX}_${stage}_m${MASK_THRESHOLD}"
  local provenance_args=(--train-data-profile "${TRAIN_DATA_PROFILE}")
  local wandb_args=(--no-wandb)
  if [[ "${EVAL_WANDB}" == "1" ]]; then
    wandb_args=(--wandb)
  fi
  if [[ -n "${MODEL_SEED}" ]]; then
    provenance_args+=(--model-seed "${MODEL_SEED}")
  fi
  echo "[eval] stage=${stage} gra=${gra} gpu=${gpu} ckpt=${ckpt_path} ckpt_suffix=${ckpt_suffix} model_seed=${MODEL_SEED:-unknown} data_profile=${TRAIN_DATA_PROFILE:-unknown}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
    PYTHONHASHSEED="${EVAL_SEED}" \
    SAM2_PROFILE="${SAM2_PROFILE}" \
    "${PYTHON}" evaluate.py \
      --method unsamv2 \
      --seed "${EVAL_SEED}" \
      "${wandb_args[@]}" \
      "${provenance_args[@]}" \
      --deterministic \
      --dataset "${DATASET}" \
      "${DATASET_ARGS[@]}" \
      --sam2-ckpt "${ckpt_path}" \
      --ckpt-suffix "${ckpt_suffix}" \
      --gra "${gra}" \
      --eval-mode bbox \
      --start_id "${START_ID}" \
      --end_id "${END_ID}" \
      --comment "${comment}" \
      --mask-threshold "${MASK_THRESHOLD}" \
      --save-every "${SAVE_EVERY}" \
      --resume-from-results \
      --device-id 0 &
}

job_count=0
idx=0
for stage in "${STAGE_LIST[@]}"; do
  for gra in "${GRA_LIST[@]}"; do
    gpu="${GPU_LIST[$((idx % NUM_GPUS))]}"
    run_eval "${stage}" "${gra}" "${gpu}"
    idx=$((idx + 1))
    job_count=$((job_count + 1))
    if [[ "${job_count}" -ge "${NUM_GPUS}" ]]; then
      wait -n
      job_count=$((job_count - 1))
    fi
  done
done
wait
