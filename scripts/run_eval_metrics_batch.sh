#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
 
OUTPUT_EVAL_DIR="/path/to/LAO-X/output_eval"
# OUTPUT_EVAL_DIR="/path/to/results/output_eval"
# OUTPUT_EVAL_DIR="/path/to/sam3/output_eval"
START_ID="${START_ID:-0}"
SKIP_FUSION="${SKIP_FUSION:-1}"
EVAL_MODE="${EVAL_MODE:-bbox}"
PROTOCOL_PRESET="${PROTOCOL_PRESET:-xray_partiall_addi}" # xray_partial_all # xray_partiall_addi
PROTOCOL_INCLUDE_BASELINE="${PROTOCOL_INCLUDE_BASELINE:-1}"
PROTOCOL_REGION_MODES="${PROTOCOL_REGION_MODES:-}"
PROTOCOL_FILTER_MODES="${PROTOCOL_FILTER_MODES:-}"
PROTOCOL_EXPAND_RATIOS="${PROTOCOL_EXPAND_RATIOS:-}"
PROTOCOL_OVERLAP_METRICS="${PROTOCOL_OVERLAP_METRICS:-}"
PROTOCOL_OVERLAP_THRESHOLDS="${PROTOCOL_OVERLAP_THRESHOLDS:-}"
PROTOCOL_DILATE_PXS="${PROTOCOL_DILATE_PXS:-}"
PROTOCOL_DILATE_RATIOS="${PROTOCOL_DILATE_RATIOS:-}"
WRITE_ABO="${WRITE_ABO:-1}"
UNSAMV2_CKPT_TAGS="${UNSAMV2_CKPT_TAGS:-unsamv2_plus_ckpt,unsamv2_ckpt}"

SUFFIXES=(

  # "_augmentation_restart_fusion_095_v2_easy"
  # "_augmentation_restart_musc_aug_easy"
  # "_augmentation_restart_musc_aug_hard"
  # "_augmentation_restart_musc_aug_extreme"
  # "_augmentation_restart_musc_aug_hard_only"
  # "_augmentation_restart_musc_aug_extreme_only"
  # "_augmentation_restart_musc_aug_fusion_easy"
  # "_augmentation_restart_musc_aug_fusion_hard"
  # "_augmentation_restart_musc_aug_fusion_extreme"
  # "_augmentation_restart_musc_aug_fusion_hard_only"
  # "_augmentation_restart_musc_aug_fusion_extreme_only"
  # "_augmentation_restart_musc_aug_fusion_8gpus_easy"
  # "_augmentation_restart_crop_musc_aug_bs16_easy"
  # "_augmentation_restart_crop_musc_aug_bs16_hard"
  # "_augmentation_restart_crop_musc_aug_bs16_extreme"
  # "_augmentation_restart_crop_musc_aug_fusion_easy_40"
  # "_augmentation_restart_crop_musc_aug_fusion_hard_32"
  # "_augmentation_restart_crop_musc_aug_fusion_extreme_24"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_easy_80"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_hard_64"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_extreme_48"

  # "_augmentation_restart_crop_musc_aug_bs16_xp_extreme_m0.5_48"


  # Main Table
  # "_det_compass_xp_divide_and_conquer_1000_m0.5"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_extreme_m0.5_48"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_xp_extreme_m0.5_48"
  # "_sam2"
  # "_sam3"

  # Ablations
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_easy_m0.5_80"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_hard_m0.5_64"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_merge_m0.5"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_xp_extreme_00_07_m0.5_48"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_xp_extreme_00_09_m0.5_48"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_xp_extreme_9_12_m0.5_48"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_xp_extreme_12_15_m0.5_48"
  # "_augmentation_restart_musc_aug_fusion_extreme_only_m0.5_{UNSAMV2_CKPT}"
  "_augmentation_restart_crop_musc_aug_holistic_only_bs16_extreme_m0.5_48"
  "_augmentation_restart_crop_musc_aug_part_only_unsamv2_bs16_extreme_m0.5_48"
  "_augmentation_restart_crop_aug_rgb_fusion_bs16_extreme_m0.5_48"
  "_augmentation_restart_musc_aug_fusion_extreme_only_m0.5_unsamv2_plus_ckpt"
  "_augmentation_restart_musc_aug_fusion_extreme_only_m0.5_unsamv2_ckpt"
)

DATASETS=(
  # det_compass_xp
  # CLCXray
  # DvXray
  PIXray
  # HiXray
  pidray
)
# get_end_id() {
#   local dataset="$1"
#   case "${dataset}" in
#     det_compass_xp) echo 100 ;;
#     CLCXray) echo 100 ;;
#     DvXray) echo 100 ;;
#     PIXray) echo 100 ;;
#     HiXray) echo 100 ;;
#     pidray) echo 100 ;;
#     *) echo 100 ;;
#   esac
# }
get_end_id() {
  local dataset="$1"
  case "${dataset}" in
    det_compass_xp) echo 307 ;;
    CLCXray) echo 956 ;;
    DvXray) echo 9155 ;;
    PIXray) echo 5047 ;;
    HiXray) echo 9069 ;;
    pidray) echo 18220 ;;
    *) echo 1928 ;;
  esac
}

IFS=',' read -ra _UNSAMV2_CKPT_TAG_LIST <<< "${UNSAMV2_CKPT_TAGS}"
EXPANDED_SUFFIXES=()
for _raw_suffix in "${SUFFIXES[@]}"; do
  if [[ "${_raw_suffix}" == *"{UNSAMV2_CKPT}"* ]]; then
    for _ckpt_tag in "${_UNSAMV2_CKPT_TAG_LIST[@]}"; do
      _ckpt_tag="$(echo "${_ckpt_tag}" | xargs)"
      if [[ -z "${_ckpt_tag}" ]]; then
        continue
      fi
      EXPANDED_SUFFIXES+=("${_raw_suffix//\{UNSAMV2_CKPT\}/${_ckpt_tag}}")
    done
  else
    EXPANDED_SUFFIXES+=("${_raw_suffix}")
  fi
done
if [[ "${#EXPANDED_SUFFIXES[@]}" -le 0 ]]; then
  echo "No valid suffix found after expansion." >&2
  exit 1
fi

for SUFFIX in "${EXPANDED_SUFFIXES[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    if [[ -n "${END_ID}" ]]; then
      DATASET_END_ID="${END_ID}"
    else
      DATASET_END_ID="$(get_end_id "${DATASET}")"
    fi
    python - <<PY
from util_xray import evaluate_output_eval_metrics

def _parse_csv(raw, cast=None):
    raw = str(raw).strip()
    if not raw:
        return None
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if cast is None:
        return vals
    out = []
    for v in vals:
        out.append(cast(v))
    return out

evaluate_output_eval_metrics(
    suffix="${SUFFIX}",
    start_id=${START_ID},
    end_id=${DATASET_END_ID},
    datasets=["${DATASET}"],
    output_eval_dir="${OUTPUT_EVAL_DIR}",
    skip_fusion=bool(int("${SKIP_FUSION}")),
    eval_mode="${EVAL_MODE}",
    protocol_preset="${PROTOCOL_PRESET}".strip() or None,
    protocol_region_modes=_parse_csv("${PROTOCOL_REGION_MODES}"),
    protocol_filter_modes=_parse_csv("${PROTOCOL_FILTER_MODES}"),
    protocol_expand_ratios=_parse_csv("${PROTOCOL_EXPAND_RATIOS}", float),
    protocol_overlap_metrics=_parse_csv("${PROTOCOL_OVERLAP_METRICS}"),
    protocol_overlap_thresholds=_parse_csv("${PROTOCOL_OVERLAP_THRESHOLDS}", float),
    protocol_dilate_pxs=_parse_csv("${PROTOCOL_DILATE_PXS}", int),
    protocol_dilate_ratios=_parse_csv("${PROTOCOL_DILATE_RATIOS}", float),
    protocol_include_baseline=bool(int("${PROTOCOL_INCLUDE_BASELINE}")),
    protocol_write_detail_sheet=True,
    write_abo=bool(int("${WRITE_ABO}")),
)
PY
  done
done
