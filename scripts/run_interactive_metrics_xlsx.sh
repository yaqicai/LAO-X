#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_GRAS="${GRAS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"

DATASETS=(
  det_compass_xp
  CLCXray
  DvXray
  PIXray
  HiXray
  pidray
)

# 按方法逐项配置，索引一一对应。
COMMENTS=(
  "crop_musc_aug_fusion_bs16_xp"
  "det_compass_xp_divide_and_conquer_1000_clean"
  ""
  ""
)

TRAIN_ROOTS=(
  "results/train"
  "/path/to/results/train"
  "results/train"
  "results/train"
)

STAGES=(
  "extreme"
  "extreme"
  "easy"
  "easy"
)

SAM_TYPES=(
  "unsamv2"
  "unsamv2"
  "sam2"
  "unsamv2"
)

CKPT_PATHS=(
  ""
  ""
  "/path/to/checkpoints/sam2.1_s.pt"
  "/path/to/results/official_ckpt/unsamv2_plus_ckpt.pt"
)

GRAS_LISTS=(
  "${DEFAULT_GRAS}"
  "${DEFAULT_GRAS}"
  "1.0"
  "${DEFAULT_GRAS}"
)


OUTPUT_DIRS=(
  "${REPO_ROOT}/evaluation/interactive/outputs/xray_eval"
  "${REPO_ROOT}/evaluation/interactive/outputs/xray_eval"
  "${REPO_ROOT}/evaluation/interactive/outputs/xray_eval"
  "${REPO_ROOT}/evaluation/interactive/outputs/xray_eval"
)

if [[ "${#COMMENTS[@]}" -eq 0 ]]; then
  echo "COMMENTS is empty. Add at least one method configuration in scripts/run_interactive_metrics_xlsx.sh" >&2
  exit 1
fi

num_methods="${#COMMENTS[@]}"
for arr_name in TRAIN_ROOTS STAGES SAM_TYPES CKPT_PATHS GRAS_LISTS OUTPUT_DIRS; do
  eval "arr_len=\${#${arr_name}[@]}"
  if [[ "${arr_len}" -ne "${num_methods}" ]]; then
    echo "${arr_name} length (${arr_len}) must match COMMENTS length (${num_methods})." >&2
    exit 1
  fi
done

resolve_comment_suffix() {
  local comment="$1"
  local train_root="$2"
  if [[ -n "${comment}" && -d "${train_root}/${comment}" ]]; then
    printf '_%s' "${comment}"
  elif [[ -n "${comment}" ]]; then
    printf '_augmentation_restart_%s' "${comment}"
  else
    printf '_augmentation_restart'
  fi
}

build_output_path() {
  local output_dir="$1"
  local comment_suffix="$2"
  local stage="$3"
  local sam_type="$4"
  local ckpt_tag="$5"

  local parts=("interactive_metrics")
  if [[ -n "${comment_suffix}" ]]; then
    parts+=("${comment_suffix#_}")
  fi
  if [[ -n "${stage}" ]]; then
    parts+=("${stage}")
  fi
  if [[ -n "${sam_type}" ]]; then
    parts+=("${sam_type}")
  fi
  if [[ -n "${ckpt_tag}" ]]; then
    parts+=("${ckpt_tag}")
  fi

  local joined
  joined="$(IFS=_; echo "${parts[*]}")"
  printf '%s/%s.xlsx' "${output_dir}" "${joined}"
}

for ((i=0; i<num_methods; i++)); do
  comment="${COMMENTS[$i]}"
  train_root="${TRAIN_ROOTS[$i]}"
  stage="${STAGES[$i]}"
  sam_type="${SAM_TYPES[$i]}"
  ckpt_path="${CKPT_PATHS[$i]}"
  gras="${GRAS_LISTS[$i]}"
  output_dir="${OUTPUT_DIRS[$i]}"

  comment_suffix="$(resolve_comment_suffix "${comment}" "${train_root}")"
  ckpt_tag=""
  if [[ -n "${ckpt_path}" ]]; then
    ckpt_tag="$(basename "${ckpt_path}")"
    ckpt_tag="${ckpt_tag%.pt}"
  fi
  output_path="$(build_output_path "${output_dir}" "${comment_suffix}" "${stage}" "${sam_type}" "${ckpt_tag}")"

  echo "[interactive_xlsx] output=${output_path}"
  for dataset in "${DATASETS[@]}"; do
    python - <<PY
from util_xray import export_interactive_eval_metrics

export_interactive_eval_metrics(
    dataset="${dataset}",
    stage="${stage}",
    gras="${gras}",
    comment_suffix="${comment_suffix}",
    output_dir="${output_dir}",
    output_path="${output_path}",
    sam_type="${sam_type}",
    ckpt_tag="${ckpt_tag}",
)
PY
  done
done
