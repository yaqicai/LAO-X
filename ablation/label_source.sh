#!/usr/bin/env bash
# Label-source ablation on the main curriculum data (run pipeline steps 1-4 first).
#   bash ablation/label_source.sh holistic   # synthetic object masks only      (anno/)
#   bash ablation/label_source.sh part       # cleaned oversegmentation only    (anno_cutler_clean/)
# The full method fuses both (anno_fusion/, pipeline/05_train.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

case "${1:?usage: $0 {holistic|part}}" in
  holistic) train_curriculum crop_musc_aug_holistic_only_bs16 "${SYN_DATASET}" base ;;
  part)     train_curriculum crop_musc_aug_part_only_clean_bs16 "${SYN_DATASET}" cutler_clean ;;
  *) echo "unknown label source: $1" >&2; exit 2 ;;
esac
