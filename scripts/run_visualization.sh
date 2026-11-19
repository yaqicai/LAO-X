#!/usr/bin/env bash
# Per-image AP comparison and qualitative figures for several methods
# (util_xray.merge_image_ap_by_suffix / visualization_select) on the
# cross-dataset benchmarks. Methods, granularities and datasets are configured
# in the aligned arrays below; VIS_IMAGE_IDS_MATRIX holds the images shown in
# the paper (set it to an empty string to select images by AP instead).
# Prediction roots: OUTPUT_EVAL_DIR [${LAOX_OUTPUT_EVAL:-<repo>/output_eval}],
# OUTPUT_EVAL_DIR_OURS [OUTPUT_EVAL_DIR], OUTPUT_EVAL_DIR_SAM3 [${SAM3_OUTPUT_EVAL:-OUTPUT_EVAL_DIR}]
# (SAM 3 predictions are produced outside this repository).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_EVAL_DIR="${OUTPUT_EVAL_DIR:-${LAOX_OUTPUT_EVAL:-${REPO_ROOT}/output_eval}}"
OUTPUT_EVAL_DIR_OURS="${OUTPUT_EVAL_DIR_OURS:-${OUTPUT_EVAL_DIR}}"
OUTPUT_EVAL_DIR_SAM3="${OUTPUT_EVAL_DIR_SAM3:-${SAM3_OUTPUT_EVAL:-${OUTPUT_EVAL_DIR}}}"

START_ID="${START_ID:-0}"
VIS_AP1="${VIS_AP1:-0}"
OUTPUT_VIS_DIR="${OUTPUT_VIS_DIR:-${REPO_ROOT}/output_visualization_reorganized}"
VIS_SCORE_THRESH="${VIS_SCORE_THRESH:-0.0}"
AP_PROTOCOL_PRESET="${AP_PROTOCOL_PRESET:-s3_r1p5_center}"
VIS_GROUP_BY_GT_REGION="${VIS_GROUP_BY_GT_REGION:-1}"
VIS_TOPN="${VIS_TOPN:-100}"
VIS_SCORE_SORTED="${VIS_SCORE_SORTED:-1}"
VIS_SCORE_SORTED_TOPK="${VIS_SCORE_SORTED_TOPK:-5}"
VIS_SCORE_SORTED_LOWK="${VIS_SCORE_SORTED_LOWK:-10}"
VIS_SCORE_SORTED_ALPHA="${VIS_SCORE_SORTED_ALPHA:-0.45}"
VIS_COMPACT_TOPK_PANEL="${VIS_COMPACT_TOPK_PANEL:-0}"
VIS_COMPACT_TOPK_K="${VIS_COMPACT_TOPK_K:-5}"
VIS_COMPACT_TOPK_COLS="${VIS_COMPACT_TOPK_COLS:-5}"
VIS_COMPACT_TOPK_ALPHA="${VIS_COMPACT_TOPK_ALPHA:-0.45}"
VIS_COMPACT_TOPK_SHOW_SCORE="${VIS_COMPACT_TOPK_SHOW_SCORE:-0}"
# 0 means keep all matched masks in each GT-expand region; >0 keeps top-k by score.
VIS_GT_REGION_MASK_TOPK="${VIS_GT_REGION_MASK_TOPK:-0}"
# Save GT-expand region reference crops.
VIS_SAVE_GT_REGION_ORIGIN="${VIS_SAVE_GT_REGION_ORIGIN:-1}"
VIS_SAVE_GT_REGION_GRAY="${VIS_SAVE_GT_REGION_GRAY:-1}"
SINGLE_MASK_VIS_MODE="${SINGLE_MASK_VIS_MODE:-edge_fill}"
# Independent color control (edge/fill/edge_fill). Falls back to SINGLE_MASK_VIS_MODE.
VIS_MASK_COLOR_MODE="${VIS_MASK_COLOR_MODE:-${SINGLE_MASK_VIS_MODE}}"
VIS_GRAYSCALE_IMAGE="${VIS_GRAYSCALE_IMAGE:-1}"
VIS_SAVE_ORIGINAL_IMAGE="${VIS_SAVE_ORIGINAL_IMAGE:-1}"
# Legacy switch kept for compatibility; use VIS_SAMPLE_FILTER_ONLY for sample-selection flow.
VIS_OUTPUT_MODE="${VIS_OUTPUT_MODE:-sample_filter}"
# Independent sample-selection control (1: selection-oriented outputs, 0: full outputs).
VIS_SAMPLE_FILTER_ONLY="${VIS_SAMPLE_FILTER_ONLY:-}"
VIS_DRAW_GT="${VIS_DRAW_GT:-}"
VIS_SAVE_DETAILED_OUTPUTS="${VIS_SAVE_DETAILED_OUTPUTS:-}"
VIS_SKIP_EXISTING="${VIS_SKIP_EXISTING:-0}"
# When manual image ids are provided for a dataset row, skip per-image AP merge/eval.
VIS_SKIP_EVAL_WHEN_MANUAL_IDS="${VIS_SKIP_EVAL_WHEN_MANUAL_IDS:-1}"

LEGACY_SAMPLE_FILTER_ONLY=0
case "${VIS_OUTPUT_MODE}" in
  sample_filter|filter_only|sample)
    LEGACY_SAMPLE_FILTER_ONLY=1
    ;;
esac

if [[ -z "${VIS_SAMPLE_FILTER_ONLY}" ]]; then
  VIS_SAMPLE_FILTER_ONLY="${LEGACY_SAMPLE_FILTER_ONLY}"
fi
if [[ -z "${VIS_DRAW_GT}" ]]; then
  if [[ "${LEGACY_SAMPLE_FILTER_ONLY}" == "1" ]]; then
    VIS_DRAW_GT=0
  else
    VIS_DRAW_GT=1
  fi
fi
if [[ -z "${VIS_SAVE_DETAILED_OUTPUTS}" ]]; then
  if [[ "${LEGACY_SAMPLE_FILTER_ONLY}" == "1" ]]; then
    VIS_SAVE_DETAILED_OUTPUTS=0
  else
    VIS_SAVE_DETAILED_OUTPUTS=1
  fi
fi

# Optional manual image-id matrix aligned to DATASETS.
# Format: row1;row2;...;rowN, where each row is comma-separated ids.
# Example (5 datasets): "12,35,91;7,8;101;55,66,77;9001"
VIS_IMAGE_IDS_MATRIX="${VIS_IMAGE_IDS_MATRIX-78,204,225,271,312,336,408,428;259,439,593,1011,1090,1159,1630,2162,2204,2224,2228,2242,2342,2405,2410,2763,2941,3135,3406,3577,3615,3652,3738,3740,3786,3790,3802,3959,3992,4078,4186,4264,4604,5244,5278,5296,5364,5645,5661;419,459,471,532,544,732,963,1007,1019,1134;7331,7420,7536,7573,8128,8848;170,919,1232,1500,1506,1570,2310,2775,2866,3089,3171,3559,3713,4015,5853,6310,6565,8105,8115,13518,14472,15294,15295,25747,18079}"

# Arrays must be aligned one-to-one (suffix <-> model_name <-> gra row).
SUFFIXES=(
  "_augmentation_restart_crop_musc_aug_fusion_bs16_xp_extreme_m0.5_48"
  "_det_compass_xp_divide_and_conquer_1000_m0.5"
  "_sam2"
  "_sam3"
)

MODEL_NAMES=(
  "unsamv2"
  "unsamv2"
  "sam2"
  "sam3"
)

METHOD_ALIASES=(
  "ours"
  "unsamv2"
  "sam2"
  "sam3"
)

# One GRA value per dataset (aligned to DATASETS) for each method; these are the
# best granularities under the s3_r1p5_center protocol.
GRAS_BY_METHOD=(
  "1.0,1.0,1.0,0.3,1.0"
  "0.5,0.3,0.3,0.3,0.5"
  "0.0,0.0,0.0,0.0,0.0"
  "0.0,0.0,0.0,0.0,0.0"
)

DATASETS=(
  CLCXray
  DvXray
  PIXray
  HiXray
  pidray
)

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

if [[ "${#SUFFIXES[@]}" -ne "${#MODEL_NAMES[@]}" ]]; then
  echo "SUFFIXES and MODEL_NAMES length mismatch." >&2
  exit 1
fi

if [[ "${#SUFFIXES[@]}" -ne "${#GRAS_BY_METHOD[@]}" ]]; then
  echo "SUFFIXES and GRAS_BY_METHOD length mismatch." >&2
  exit 1
fi

if [[ "${#SUFFIXES[@]}" -ne "${#METHOD_ALIASES[@]}" ]]; then
  echo "SUFFIXES and METHOD_ALIASES length mismatch." >&2
  exit 1
fi

VIS_IMAGE_ID_ROWS=()
if [[ -n "${VIS_IMAGE_IDS_MATRIX}" ]]; then
  IFS=';' read -ra VIS_IMAGE_ID_ROWS <<< "${VIS_IMAGE_IDS_MATRIX}"
  if [[ "${#VIS_IMAGE_ID_ROWS[@]}" -ne "${#DATASETS[@]}" ]]; then
    echo "VIS_IMAGE_IDS_MATRIX row count must match DATASETS length." >&2
    exit 1
  fi
fi

for di in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$di]}"
  END_ID="$(get_end_id "${DATASET}")"
  VIS_IMAGE_IDS_ROW="${VIS_IMAGE_ID_ROWS[$di]:-}"

  GRA_LIST=()
  for mi in "${!GRAS_BY_METHOD[@]}"; do
    IFS=',' read -ra GRA_ROW <<< "${GRAS_BY_METHOD[$mi]}"
    if [[ "${#GRA_ROW[@]}" -ne "${#DATASETS[@]}" ]]; then
      echo "GRAS_BY_METHOD row ${mi} length mismatch with DATASETS." >&2
      exit 1
    fi
    GRA_LIST+=("${GRA_ROW[$di]}")
  done

  SUFFIXES_PY=$(printf "'%s', " "${SUFFIXES[@]}")
  MODEL_NAMES_PY=$(printf "'%s', " "${MODEL_NAMES[@]}")
  GRA_LIST_PY=$(printf "'%s', " "${GRA_LIST[@]}")
  METHOD_ALIASES_PY=$(printf "'%s', " "${METHOD_ALIASES[@]}")
  OUTPUT_DIRS=()
  for mi in "${!METHOD_ALIASES[@]}"; do
    method_alias="${METHOD_ALIASES[$mi]}"
    case "${method_alias}" in
      ours)
        OUTPUT_DIRS+=("${OUTPUT_EVAL_DIR_OURS}")
        ;;
      sam3)
        OUTPUT_DIRS+=("${OUTPUT_EVAL_DIR_SAM3}")
        ;;
      *)
        OUTPUT_DIRS+=("${OUTPUT_EVAL_DIR}")
        ;;
    esac
  done
  OUTPUT_DIRS_PY=$(printf "'%s', " "${OUTPUT_DIRS[@]}")

  "${PYTHON:-python}" - <<PY
import os
from util_xray import merge_image_ap_by_suffix, select_image_ids_by_ap_gap, visualization_select

ap_protocol_preset = "${AP_PROTOCOL_PRESET}".strip() or None
ap_tag = ap_protocol_preset or "default"
dataset = "${DATASET}"
vis_output_mode = "${VIS_OUTPUT_MODE}".strip().lower() or "full"
draw_gt_flag = bool(int("${VIS_DRAW_GT}"))
sample_filter_only = bool(int("${VIS_SAMPLE_FILTER_ONLY}"))
save_detailed_outputs = bool(int("${VIS_SAVE_DETAILED_OUTPUTS}"))
mask_color_mode = "${VIS_MASK_COLOR_MODE}".strip() or "${SINGLE_MASK_VIS_MODE}".strip() or "edge"
manual_ids_raw = "${VIS_IMAGE_IDS_ROW}".strip()
skip_eval_when_manual_ids = bool(int("${VIS_SKIP_EVAL_WHEN_MANUAL_IDS}"))
image_ids = []
if manual_ids_raw:
    normalized = manual_ids_raw.replace("，", ",")
    parsed = []
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid image id token in VIS_IMAGE_IDS_MATRIX: {token}") from exc
    # Preserve order while removing duplicates.
    image_ids = list(dict.fromkeys(parsed))

merged_df = None
if not (skip_eval_when_manual_ids and image_ids):
    merged_df = merge_image_ap_by_suffix(
        dataset="${DATASET}",
        gra=[${GRA_LIST_PY}],
        suffixes=[${SUFFIXES_PY}],
        model_names=[${MODEL_NAMES_PY}],
        start_id=${START_ID},
        end_id=${END_ID},
        output_eval_dir="${OUTPUT_EVAL_DIR}",
        output_eval_dirs=[${OUTPUT_DIRS_PY}],
        output_path="${OUTPUT_EVAL_DIR}/per_image_ap_${DATASET}_${ap_tag}.xlsx",
        sort_suffix="${SUFFIXES[0]}",
        region_protocol_preset=ap_protocol_preset,
    )

if (not image_ids) and int("${VIS_AP1}") == 1:
    if merged_df is None:
        raise RuntimeError(
            "VIS_AP1=1 requires merged AP results, but evaluation was skipped. "
            "Set VIS_SKIP_EVAL_WHEN_MANUAL_IDS=0 or provide manual ids."
        )
    first_suffix = "${SUFFIXES[0]}"
    first_model = "${MODEL_NAMES[0]}"
    first_col = f"{first_model}{first_suffix}"
    all_cols = [f"{m}{s}" for s, m in zip([${SUFFIXES_PY}], [${MODEL_NAMES_PY}])]
    other_cols = all_cols[1:]
    image_ids = select_image_ids_by_ap_gap(
        merged_df=merged_df,
        first_method_col=first_col,
        other_method_cols=other_cols,
        ap_value=1.0,
        tol=1e-6,
        top_n=int("${VIS_TOPN}"),
    )
if image_ids:
    output_dir = "${OUTPUT_VIS_DIR}"
    method_aliases = [${METHOD_ALIASES_PY}]
    if (
        save_detailed_outputs
        and (not bool(int("${VIS_SCORE_SORTED}")))
        and (not bool(int("${VIS_COMPACT_TOPK_PANEL}")))
    ):
        filtered_ids = []
        for img_id in image_ids:
            expected = [
                os.path.join(output_dir, f"{dataset}_{img_id}_{alias}_mask_b3.png")
                for alias in method_aliases
            ]
            if all(os.path.exists(path) for path in expected):
                continue
            filtered_ids.append(img_id)
        image_ids = filtered_ids
    if image_ids:
        visualization_select(
            dataset="${DATASET}",
            image_ids=image_ids,
            suffixes=[${SUFFIXES_PY}],
            gra=[${GRA_LIST_PY}],
            model_names=[${MODEL_NAMES_PY}],
            method_aliases=[${METHOD_ALIASES_PY}],
            start_id=${START_ID},
            end_id=${END_ID},
            output_eval_dir="${OUTPUT_EVAL_DIR}",
            output_eval_dirs=[${OUTPUT_DIRS_PY}],
            output_dir="${OUTPUT_VIS_DIR}",
            score_thresh=float("${VIS_SCORE_THRESH}"),
            max_det=100,
            draw_gt=draw_gt_flag,
            region_protocol_preset=ap_protocol_preset,
            group_by_gt_region=bool(int("${VIS_GROUP_BY_GT_REGION}")),
            score_sorted_vis=bool(int("${VIS_SCORE_SORTED}")),
            score_sorted_topk=int("${VIS_SCORE_SORTED_TOPK}"),
            score_sorted_lowk=int("${VIS_SCORE_SORTED_LOWK}"),
            score_sorted_alpha=float("${VIS_SCORE_SORTED_ALPHA}"),
            compact_topk_panel=bool(int("${VIS_COMPACT_TOPK_PANEL}")),
            compact_topk_k=int("${VIS_COMPACT_TOPK_K}"),
            compact_topk_cols=int("${VIS_COMPACT_TOPK_COLS}"),
            compact_topk_alpha=float("${VIS_COMPACT_TOPK_ALPHA}"),
            compact_topk_show_score=bool(int("${VIS_COMPACT_TOPK_SHOW_SCORE}")),
            gt_region_mask_topk=int("${VIS_GT_REGION_MASK_TOPK}"),
            save_gt_region_origin=bool(int("${VIS_SAVE_GT_REGION_ORIGIN}")),
            save_gt_region_gray=bool(int("${VIS_SAVE_GT_REGION_GRAY}")),
            single_mask_vis_mode="${SINGLE_MASK_VIS_MODE}",
            mask_color_mode=mask_color_mode,
            grayscale_image=bool(int("${VIS_GRAYSCALE_IMAGE}")),
            save_original_image=bool(int("${VIS_SAVE_ORIGINAL_IMAGE}")),
            sample_filter_only=sample_filter_only,
            save_detailed_outputs=save_detailed_outputs,
            output_mode=vis_output_mode,
            skip_existing=bool(int("${VIS_SKIP_EXISTING}")),
        )
PY
done
