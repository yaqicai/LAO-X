SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
COMMENT="${1:-}"
TRAIN_ROOT="${2:-${TRAIN_ROOT:-results/train}}"
# TRAIN_ROOT="results/train"
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
TRAIN_DATA_PROFILE="${TRAIN_DATA_PROFILE:-}"
ALLOW_CHECKPOINT_FALLBACK="${ALLOW_CHECKPOINT_FALLBACK:-0}"
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
FALLBACK_TRAIN_ROOT="${FALLBACK_TRAIN_ROOT:-/path/to/results/train}"
UNSAMV2_OFFICIAL_CKPT="${UNSAMV2_OFFICIAL_CKPT:-/path/to/results/official_ckpt/unsamv2_plus_ckpt.pt}"

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
    if [[ ! -f "${SAM2_CKPT}" ]]; then
      echo "SAM2_CKPT does not exist: ${SAM2_CKPT}" >&2
      exit 1
    fi
    ckpt_path="${SAM2_CKPT}"
  else
    local primary_ckpt_path="${CKPT_ROOT}/${stage}/log/checkpoints/${ckpt_name}"
    if [[ -f "${primary_ckpt_path}" ]]; then
      ckpt_path="${primary_ckpt_path}"
    elif [[ "${ALLOW_CHECKPOINT_FALLBACK}" == "1" ]]; then
      local fallback_ckpt_root=""
      if [[ -n "${COMMENT}" ]]; then
        fallback_ckpt_root="${FALLBACK_TRAIN_ROOT}/augmentation_restart_${COMMENT}"
      else
        fallback_ckpt_root="${FALLBACK_TRAIN_ROOT}/augmentation_restart"
      fi
      local fallback_ckpt_path="${fallback_ckpt_root}/${stage}/log/checkpoints/${ckpt_name}"
      if [[ -f "${fallback_ckpt_path}" ]]; then
        ckpt_path="${fallback_ckpt_path}"
      elif [[ -f "${UNSAMV2_OFFICIAL_CKPT}" ]]; then
        ckpt_path="${UNSAMV2_OFFICIAL_CKPT}"
      else
        echo "Checkpoint not found for stage=${stage}. Tried:" >&2
        echo "  - ${primary_ckpt_path}" >&2
        echo "  - ${fallback_ckpt_path}" >&2
        echo "  - ${UNSAMV2_OFFICIAL_CKPT}" >&2
        exit 1
      fi
    else
      echo "Checkpoint not found: ${primary_ckpt_path}" >&2
      echo "Fallback is disabled to prevent evaluating the wrong model." >&2
      echo "Set ALLOW_CHECKPOINT_FALLBACK=1 only when fallback is intentional." >&2
      exit 1
    fi
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
    python test_det_v2.py \
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
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_easy
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_easy 
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_easy 
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_easy 

# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_hard
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_hard
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_hard 
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_hard
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_hard

# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_xp --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 1928 --comment ${COMMENT_SUFFIX}_extreme

# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_easy
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_easy 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_easy 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_easy 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_easy 

# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_hard
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_hard
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_hard 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_hard
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_hard

# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_extreme
