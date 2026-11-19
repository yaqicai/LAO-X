#!/usr/bin/env bash
set -euo pipefail

cd /path/to/LAO-X

DATASET="${DATASET:-det_compass_xp}"
SAM2_CKPT="${SAM2_CKPT:-/path/to/LAO-X/sam2/checkpoints/sam2.1_hiera_small.pt}"
SAM2_CFG="${SAM2_CFG:-configs/unsamv2_small.yaml}"
DEVICE="${DEVICE:-cuda}"
START_ID="${START_ID:-0}"
END_ID="${END_ID:-}"
MAX_IMAGES="${MAX_IMAGES:-0}"

python evaluation/interactive/sam2_xray_one_click.py \
  --dataset "${DATASET}" \
  --sam2-ckpt "${SAM2_CKPT}" \
  --sam2-cfg "${SAM2_CFG}" \
  --device "${DEVICE}" \
  --start-id "${START_ID}" \
  ${END_ID:+--end-id "${END_ID}"} \
  ${MAX_IMAGES:+--max-images "${MAX_IMAGES}"}
