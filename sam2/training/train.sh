#!/usr/bin/env bash
# Single (non-curriculum) UnSAMv2 training run with a fixed Hydra config.
# For the LAO-X curriculum use train_curriculum.sh instead.
#
# Usage:
#   bash sam2/training/train.sh [CONFIG] [extra sam2/training/train.py args]
#
# CONFIG (default configs/unsamv2_small_training.yaml) is a config name under
# sam2/sam2/configs, e.g.
#   configs/unsamv2_small_training_det_compass_xp.yaml
#       UnSAMv2 trained directly on DET-COMPASS oversegmentation pseudo-labels
#       (results/cutler/det_compass_xp/granularity_divide_and_conquer_1000_clean,
#       test images removed with sam2/training/scripts/copy_non_test_annotations.py);
#       evaluate with scripts/task_test_det_compass_xp.sh
#   the same with CutLER(+MuSc) pseudo-label variants:
#       configs/unsamv2_small_training_det_compass_xp_cutler_musc.yaml
#       configs/unsamv2_small_training_det_compass_xp_cutler_musc_gra_bins10.yaml
#       configs/unsamv2_small_training_det_compass_xp_cutler_musc_gra_minmax.yaml
#       configs/unsamv2_small_training_det_compass_xp_cutler_ori.yaml
#   configs/unsamv2_small_training_pixray.yaml
#       UnSAMv2 trained on PIXray oversegmentation pseudo-labels
# Data, pseudo-label, checkpoint and log paths are set in the config relative
# to LAOX_DATA_ROOT / LAOX_RESULTS_ROOT / LAOX_CKPT_ROOT (default
# <repo>/datasets, <repo>/results, <repo>/checkpoints).
#
# Environment: NUM_GPUS (default 1), CUDA_VISIBLE_DEVICES or GPUS (default 0),
# PYTHON (default python).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

export LAOX_DATA_ROOT="${LAOX_DATA_ROOT:-${REPO_ROOT}/datasets}"
export LAOX_RESULTS_ROOT="${LAOX_RESULTS_ROOT:-${REPO_ROOT}/results}"
export LAOX_CKPT_ROOT="${LAOX_CKPT_ROOT:-${REPO_ROOT}/checkpoints}"

CONFIG="configs/unsamv2_small_training.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi
NUM_GPUS="${NUM_GPUS:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS:-0}}" \
"${PYTHON}" "${SCRIPT_DIR}/train.py" \
  -c "${CONFIG}" \
  --use-cluster 0 \
  --num-gpus "${NUM_GPUS}" \
  "$@"
