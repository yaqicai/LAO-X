#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-0}"
MAX_IMAGES="${MAX_IMAGES:-20}"
VIS="${VIS:-True}"
SAVE_EXCEL="${SAVE_EXCEL:-True}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output_clcxray_strict}"
LOG_DIR="${LOG_DIR:-./logs/musc_strict_clcxray}"

DATA_PATH="/path/to/datasets/CLCXray"
ANNOTATION_PATH="/path/to/datasets/CLCXray/coco_agnostic_val.json"
IMAGE_ROOT="/path/to/datasets/CLCXray/val2017"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
SUMMARY_PATH="${LOG_DIR}/summary.tsv"
printf "name\tmsm_topmin_min\tmsm_topmin_max\tr_list\tfeature_layers\tlog\n" > "${SUMMARY_PATH}"

run_combo() {
  local name="$1"
  local msm_min="$2"
  local msm_max="$3"
  local r_list="$4"
  local feature_layers="$5"
  local log_path="${LOG_DIR}/${name}.log"
  local output_dir="${OUTPUT_ROOT}/${name}"

  echo "===== ${name} ====="
  echo "msm_topmin_min=${msm_min} msm_topmin_max=${msm_max} r_list=${r_list} feature_layers=${feature_layers}"

  python examples/musc_main.py \
    --device "${DEVICE}" \
    --data_path "${DATA_PATH}" \
    --dataset_name CLCXray \
    --class_name ALL \
    --annotation_path "${ANNOTATION_PATH}" \
    --image_root "${IMAGE_ROOT}" \
    --annotation_mode bbox \
    --crop_mode ratio_expand \
    --max_images "${MAX_IMAGES}" \
    --backbone_name ViT-L-14-336 \
    --pretrained openai \
    --feature_layers ${feature_layers} \
    --img_resize 560 \
    --divide_num 1 \
    --r_list ${r_list} \
    --batch_size 4 \
    --msm_topmin_min "${msm_min}" \
    --msm_topmin_max "${msm_max}" \
    --output_dir "${output_dir}" \
    --vis "${VIS}" \
    --save_excel "${SAVE_EXCEL}" \
    2>&1 | tee "${log_path}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${name}" "${msm_min}" "${msm_max}" "${r_list}" "${feature_layers}" "${log_path}" >> "${SUMMARY_PATH}"
}

run_combo "baseline_msm030_r135_l5-11-17-23" "0" "0.30" "1 3 5" "5 11 17 23"
run_combo "strict_msm010_r135_l5-11-17-23" "0" "0.10" "1 3 5" "5 11 17 23"
run_combo "strict_skip005_msm020_r135_l5-11-17-23" "0.05" "0.20" "1 3 5" "5 11 17 23"
run_combo "deep_msm020_r135_l11-17-23" "0" "0.20" "1 3 5" "11 17 23"
run_combo "smooth_msm020_r35_l11-17-23" "0" "0.20" "3 5" "11 17 23"

python - <<'PY' "${SUMMARY_PATH}" "${LOG_DIR}/metrics_summary.tsv"
import re
import sys

summary_path, out_path = sys.argv[1], sys.argv[2]
image_re = re.compile(r"image-level, auroc:([^,]+), f1:([^,]+), ap:([^,]+)")
pixel_re = re.compile(r"pixel-level, auroc:([^,]+), f1:([^,]+), ap:([^,]+), aupro:([^,]+)")

with open(summary_path, "r") as f:
    rows = [line.rstrip("\n").split("\t") for line in f]

header = rows[0]
out_rows = [header[:-1] + ["auroc_px", "f1_px", "ap_px", "aupro", "auroc_sp", "f1_sp", "ap_sp", "log"]]

for row in rows[1:]:
    log_path = row[-1]
    with open(log_path, "r", errors="ignore") as f:
        text = f.read()
    image_matches = image_re.findall(text)
    pixel_matches = pixel_re.findall(text)
    image = image_matches[-1] if image_matches else ("NA", "NA", "NA")
    pixel = pixel_matches[-1] if pixel_matches else ("NA", "NA", "NA", "NA")
    out_rows.append(row[:-1] + list(pixel) + list(image) + [log_path])

with open(out_path, "w") as f:
    for row in out_rows:
        f.write("\t".join(row) + "\n")

print(f"Metrics summary written to {out_path}")
PY

echo "Summary written to ${SUMMARY_PATH}"
