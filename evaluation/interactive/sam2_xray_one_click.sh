#!/usr/bin/env bash
# One-click interactive evaluation on an X-ray COCO dataset (paths from config.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
LAOX_CKPT_ROOT="${LAOX_CKPT_ROOT:-${REPO_ROOT}/checkpoints}"
DATASET="${DATASET:-det_compass_xp}"
SAM2_CKPT="${SAM2_CKPT:-${LAOX_CKPT_ROOT}/sam2.1_hiera_small.pt}"
SAM2_CFG="${SAM2_CFG:-configs/unsamv2_small.yaml}"
DEVICE="${DEVICE:-cuda}"
START_ID="${START_ID:-0}"
END_ID="${END_ID:-}"
MAX_IMAGES="${MAX_IMAGES:-0}"

# config.py lives in the repo root; isegm is found next to the script.
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON}" evaluation/interactive/sam2_xray_one_click.py \
  --dataset "${DATASET}" \
  --sam2-ckpt "${SAM2_CKPT}" \
  --sam2-cfg "${SAM2_CFG}" \
  --device "${DEVICE}" \
  --start-id "${START_ID}" \
  ${END_ID:+--end-id "${END_ID}"} \
  ${MAX_IMAGES:+--max-images "${MAX_IMAGES}"}
