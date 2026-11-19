#!/usr/bin/env bash
# Composition ablation: paste objects in RGB space (copy-paste overwrite) instead
# of Beer-Lambert composition in the high/low-energy domain.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_file "${OBJECT_BANK_JSON}"
build_curriculum crop_aug_rgb_xp --object-bank-json "${OBJECT_BANK_JSON}" --composition-mode rgb
train_curriculum crop_aug_rgb_fusion_bs16 crop_aug_rgb_xp fusion
