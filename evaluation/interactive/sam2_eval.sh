#!/usr/bin/env bash
# Interactive (click-based) evaluation on the natural-image GraCo benchmarks.
# Dataset locations come from config.yml (default: ${LAOX_DATA_ROOT}/interactive/...).
# The SAM 2 model config path is relative to the sam2 package.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"
LAOX_CKPT_ROOT="${LAOX_CKPT_ROOT:-${REPO_ROOT}/checkpoints}"
CHECKPOINT="${CHECKPOINT:-${LAOX_CKPT_ROOT}/unsamv2_plus_ckpt.pt}"
LOGS_PATH="${LOGS_PATH:-${SCRIPT_DIR}/outputs/evaluation_logs}"
DATASETS="${DATASETS:-GrabCut,Berkeley,DAVIS,PartImageNet,PascalPart,SBD,SA1B}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-45341}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    sam2_evaluate.py NoBRS \
    --datasets "${DATASETS}" \
    --logs-path "${LOGS_PATH}" \
    --checkpoint "${CHECKPOINT}" \
    --sam-cfg-path "configs/unsamv2_small.yaml" \
    --sam-type SAM2 \
    --graco \
    --distributed
