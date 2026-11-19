#!/usr/bin/env bash
# Difficulty ablation: replace the extreme stage by a harder variant and continue
# the main run's curriculum from its finished easy/hard stages.
#   bash ablation/extreme_difficulty.sh {00_07|00_09|9_12|12_15}
#     00_07 : 6-9 objects,   gamma 0.0-0.7
#     00_09 : 6-9 objects,   gamma 0.0-0.9
#     9_12  : 9-12 objects,  gamma 0.0-0.5
#     12_15 : 12-15 objects, gamma 0.0-0.5
# These runs used 4 GPUs x batch size 4 (same global batch of 16).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

variant="${1:?usage: $0 {00_07|00_09|9_12|12_15}}"
case "${variant}" in
  00_07) objects="6,9";   gamma="0.0,0.7" ;;
  00_09) objects="6,9";   gamma="0.0,0.9" ;;
  9_12)  objects="9,12";  gamma="0.0,0.5" ;;
  12_15) objects="12,15"; gamma="0.0,0.5" ;;
  *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
esac
stage="extreme_${variant}"
GPUS="${GPUS_ABLATION:-0,1,2,3}"

require_file "${OBJECT_BANK_JSON}"
build_stage "${SYN_DATASET}" "${stage}" "${objects}" "${gamma}" --object-bank-json "${OBJECT_BANK_JSON}"
train_curriculum "${EXPERIMENT}" "${SYN_DATASET}" fusion "${stage}" 4
