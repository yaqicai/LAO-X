#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

COMMENT="${1:-}"
if [[ -n "${COMMENT}" ]]; then
  COMMENT_SUFFIX="_${COMMENT}"
else
  COMMENT_SUFFIX=""
fi

# DATASETS="${DATASETS:-det_compass_xp,DvXray,CLCXray,PIXray,HiXray,pidray}"
DATASETS="${DATASETS:-det_compass_xp,DvXray,CLCXray,PIXray,HiXray,pidray}"
GPUS="${GPUS:-0}"

CONFIG="${CONFIG:-${REPO_ROOT}/config.py}"
MAX_IMAGES="${MAX_IMAGES:-100}"

PRED_IOU_MIN="${PRED_IOU_MIN:-0.0}"
PRED_IOU_MAX="${PRED_IOU_MAX:-0.3}"
PRED_IOU_STEP="${PRED_IOU_STEP:-0.05}"

SAM2_CKPT="${SAM2_CKPT:-${REPO_ROOT}/sam2/checkpoints/sam2.1_hiera_small.pt}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_s.yaml}"
POINTS_PER_SIDE="${POINTS_PER_SIDE:-64}"
POINTS_PER_BATCH="${POINTS_PER_BATCH:-512}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.0}"
STABILITY_SCORE_THRESH="${STABILITY_SCORE_THRESH:-0.95}"
STABILITY_SCORE_OFFSET="${STABILITY_SCORE_OFFSET:-1.0}"
BOX_NMS_THRESH="${BOX_NMS_THRESH:-0.7}"
CROP_N_LAYERS="${CROP_N_LAYERS:-0}"
CROP_NMS_THRESH="${CROP_NMS_THRESH:-0.7}"
CROP_N_POINTS_DOWNSCALE_FACTOR="${CROP_N_POINTS_DOWNSCALE_FACTOR:-1}"
MIN_MASK_REGION_AREA="${MIN_MASK_REGION_AREA:-0}"
GRA="${GRA:-1.0}"
USE_M2M="${USE_M2M:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results_pred_iou_thresh_sam2}"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPUs specified in GPUS." >&2
  exit 1
fi

mapfile -t THRESHOLDS < <(python - "${PRED_IOU_MIN}" "${PRED_IOU_MAX}" "${PRED_IOU_STEP}" <<'PY'
import sys
import numpy as np

vmin = float(sys.argv[1])
vmax = float(sys.argv[2])
step = float(sys.argv[3])
vals = np.arange(vmin, vmax + 1e-9, step)
for v in vals:
    print(f"{v:.4f}".rstrip("0").rstrip("."))
PY
)

NUM_THRESH="${#THRESHOLDS[@]}"
if [[ "${NUM_THRESH}" -le 0 ]]; then
  echo "No thresholds computed." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${USE_M2M}" == "1" || "${USE_M2M}" == "true" || "${USE_M2M}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--use-m2m)
fi
if [[ -n "${DEVICE:-}" ]]; then
  EXTRA_ARGS+=(--device "${DEVICE}")
fi

IFS=',' read -ra DATASET_LIST <<< "${DATASETS}"
for DATASET in "${DATASET_LIST[@]}"; do
  OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}${COMMENT_SUFFIX}"
  LOG_DIR="${OUTPUT_DIR}/logs"
  mkdir -p "${LOG_DIR}"

  CHUNK_SIZE=$(((NUM_THRESH + NUM_GPUS - 1) / NUM_GPUS))
  job_count=0

  for idx in "${!GPU_LIST[@]}"; do
    chunk_start=$((idx * CHUNK_SIZE))
    if [[ "${chunk_start}" -ge "${NUM_THRESH}" ]]; then
      continue
    fi
    chunk_end=$((chunk_start + CHUNK_SIZE))
    if [[ "${chunk_end}" -gt "${NUM_THRESH}" ]]; then
      chunk_end="${NUM_THRESH}"
    fi

    chunk_vals=("${THRESHOLDS[@]:chunk_start:$((chunk_end - chunk_start))}")
    PRED_IOU_VALUES="$(IFS=,; echo "${chunk_vals[*]}")"
    if [[ -z "${PRED_IOU_VALUES}" ]]; then
      continue
    fi

    gpu="${GPU_LIST[$idx]}"
    LOG_FILE="${LOG_DIR}/pred_iou_${chunk_start}_${chunk_end}_gpu${gpu}.log"
    CHUNK_OUT="${OUTPUT_DIR}/chunk_${chunk_start}_${chunk_end}"
    echo "[eval] dataset=${DATASET} pred_iou=${PRED_IOU_VALUES} gpu=${gpu} log=${LOG_FILE}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python scripts/sweep_pred_iou_thresh_sam2.py \
        --config "${CONFIG}" \
        --datasets "${DATASET}" \
        --max-images "${MAX_IMAGES}" \
        --pred-iou-values="${PRED_IOU_VALUES}" \
        --checkpoint "${SAM2_CKPT}" \
        --model-config "${SAM2_CFG}" \
        --points-per-side "${POINTS_PER_SIDE}" \
        --points-per-batch "${POINTS_PER_BATCH}" \
        --mask-threshold "${MASK_THRESHOLD}" \
        --stability-score-thresh "${STABILITY_SCORE_THRESH}" \
        --stability-score-offset "${STABILITY_SCORE_OFFSET}" \
        --box-nms-thresh "${BOX_NMS_THRESH}" \
        --crop-n-layers "${CROP_N_LAYERS}" \
        --crop-nms-thresh "${CROP_NMS_THRESH}" \
        --crop-n-points-downscale-factor "${CROP_N_POINTS_DOWNSCALE_FACTOR}" \
        --min-mask-region-area "${MIN_MASK_REGION_AREA}" \
        --gra "${GRA}" \
        --output-dir "${CHUNK_OUT}" \
        "${EXTRA_ARGS[@]}" \
      > "${LOG_FILE}" 2>&1 &

    job_count=$((job_count + 1))
    if [[ "${job_count}" -ge "${NUM_GPUS}" ]]; then
      wait -n
      job_count=$((job_count - 1))
    fi
  done
  wait

  echo "Logs: ${LOG_DIR}"
done
