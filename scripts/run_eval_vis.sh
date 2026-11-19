#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ROOT="${ROOT:-/path/to/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/LAO-X/output_eval}"

START_ID="${START_ID:-0}"
END_ID="${END_ID:-}"
GRAS="${GRAS:-0.1}"
MAX_VIS="${MAX_VIS:-0}"
SCORE_THRESH="${SCORE_THRESH:-0.0}"
VIS_GT="${VIS_GT:-1}"

SUFFIXES=(
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_easy_80"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_hard_64"
  # "_augmentation_restart_crop_musc_aug_fusion_bs16_extreme_48"
  "_augmentation_restart_crop_musc_aug_fusion_bs16_extreme_m0_48"
)

DATASETS=(
  # det_compass_xp
  # CLCXray
  # DvXray
  PIXray
  # HiXray
  # pidray
)

# Optional per-dataset gra list (must align with DATASETS order).
# Example: GRAS_BY_DATASET=(0.1 0.2 0.3 0.1 0.1 0.1)
GRAS_BY_DATASET=(
  # 1.0
  1.0
  # 1.0
  # 0.6
  # 0.3
  # 1.0
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

IFS=',' read -ra GRA_LIST <<< "${GRAS}"

for SUFFIX in "${SUFFIXES[@]}"; do
  for idx in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$idx]}"
    if [[ -n "${END_ID}" ]]; then
      DATASET_END_ID="${END_ID}"
    else
      DATASET_END_ID="$(get_end_id "${DATASET}")"
    fi

    DATA_ROOT=""
    DEFAULT_IMAGE_ROOT=""
    case "${DATASET}" in
      det_compass_xp)
        DATA_ROOT="${ROOT}/det_compass_xp"
        DEFAULT_IMAGE_ROOT="${ROOT}/det_compass_xp/xray"
        ;;
      det_compass_syn_100)
        DATA_ROOT="${ROOT}/det_compass_syn_100"
        DEFAULT_IMAGE_ROOT="${ROOT}/det_compass_syn_100/xray"
        ;;
      det_compass_syn_312)
        DATA_ROOT="${ROOT}/det_compass_syn_312"
        DEFAULT_IMAGE_ROOT="${ROOT}/det_compass_syn_312/xray"
        ;;
      PIXray)
        DATA_ROOT="${ROOT}/pixray"
        DEFAULT_IMAGE_ROOT="${ROOT}/pixray/PIXray"
        ;;
      CLCXray)
        DATA_ROOT="${ROOT}/CLCXray"
        DEFAULT_IMAGE_ROOT="${ROOT}/CLCXray/test2017"
        ;;
      DvXray)
        DATA_ROOT="${ROOT}/DvXray"
        DEFAULT_IMAGE_ROOT="${ROOT}/DvXray/DvXray_Positive_Samples"
        ;;
      pidray)
        DATA_ROOT="${ROOT}/pidray"
        DEFAULT_IMAGE_ROOT="${ROOT}/pidray/xray"
        ;;
      HiXray)
        DATA_ROOT="${ROOT}/HiXray"
        DEFAULT_IMAGE_ROOT="${ROOT}/HiXray/test/test_image"
        ;;
    esac

    COCO_JSON=""
    if [[ -n "${DATA_ROOT}" ]]; then
      if [[ -f "${DATA_ROOT}/val.json" ]]; then
        COCO_JSON="${DATA_ROOT}/val.json"
      elif [[ -f "${DATA_ROOT}/default_coco_xray_v2.json" ]]; then
        COCO_JSON="${DATA_ROOT}/default_coco_xray_v2.json"
      elif [[ -f "${DATA_ROOT}/coco_agnostic.json" ]]; then
        COCO_JSON="${DATA_ROOT}/coco_agnostic.json"
      elif [[ -f "${DATA_ROOT}/coco.json" ]]; then
        COCO_JSON="${DATA_ROOT}/coco.json"
      fi
    fi

    if [[ -z "${COCO_JSON}" || -z "${DEFAULT_IMAGE_ROOT}" ]]; then
      echo "Missing COCO json or image root for ${DATASET}, skipping." >&2
      continue
    fi

    if [[ "${#GRAS_BY_DATASET[@]}" -eq "${#DATASETS[@]}" && -n "${GRAS_BY_DATASET[$idx]:-}" ]]; then
      DATASET_GRAS=("${GRAS_BY_DATASET[$idx]}")
    else
      DATASET_GRAS=("${GRA_LIST[@]}")
    fi

    for GRA in "${DATASET_GRAS[@]}"; do
      DATASET_LOWER="${DATASET,,}"
      OUTPUT_SUBDIR="${DATASET_LOWER}_img${START_ID}_${DATASET_END_ID}_unsamv2_${GRA}${SUFFIX}"
      OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"
      PRED_JSON="${OUTPUT_DIR}/coco_instances_results.json"

      if [[ ! -f "${PRED_JSON}" ]]; then
        echo "Missing prediction file: ${PRED_JSON}" >&2
        continue
      fi

      VIS_DIR="${OUTPUT_DIR}/vis"
      mkdir -p "${VIS_DIR}"

      TEMP_JSON="${DATA_ROOT}/temp_${DATASET}_val_${START_ID}_${DATASET_END_ID}.json"
      python - "${COCO_JSON}" "${START_ID}" "${DATASET_END_ID}" "${TEMP_JSON}" <<'PY'
import json
import sys

coco_json, start_id, end_id, out_json = sys.argv[1:5]
start_id = int(start_id)
end_id = int(end_id)

with open(coco_json, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])
data["images"] = images[start_id:end_id]
valid_ids = {img["id"] for img in data["images"]}
data["annotations"] = [
    ann for ann in data.get("annotations", []) if ann.get("image_id") in valid_ids
]

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(data, f)
print(out_json)
PY

      python - "${TEMP_JSON}" "${PRED_JSON}" "${DEFAULT_IMAGE_ROOT}" "${VIS_DIR}" "${MAX_VIS}" "${SCORE_THRESH}" "${VIS_GT}" <<'PY'
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

coco_json, pred_json, image_root, vis_dir, max_vis, score_thresh, vis_gt = sys.argv[1:8]
max_vis = int(max_vis)
score_thresh = float(score_thresh)
vis_gt = int(vis_gt) != 0

def _decode_coco_mask(seg, h, w):
    if seg is None:
        return None
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict):
        rle = seg
    else:
        return None
    return mask_utils.decode(rle)

def _safe_name(file_name: str) -> str:
    return file_name.replace("/", "_").replace("\\", "_")

coco_gt = COCO(coco_json)
with open(pred_json, "r") as f:
    predictions = json.load(f)

pred_by_image = {}
for pred in predictions:
    pred_by_image.setdefault(int(pred.get("image_id", -1)), []).append(pred)

images = coco_gt.dataset.get("images", [])
if max_vis > 0:
    images = images[:max_vis]

for img_info in images:
    img_id = int(img_info["id"])
    file_name = img_info.get("file_name", "")
    if not file_name:
        continue
    img_path = file_name if os.path.isabs(file_name) else os.path.join(image_root, file_name)
    if not os.path.exists(img_path):
        print(f"Missing image: {img_path}")
        continue

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    base_name = _safe_name(file_name)

    if vis_gt:
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_list = coco_gt.loadAnns(ann_ids)

        gt_comp = np.array(img).astype(np.float32)
        gt_draw = []
        for ann in gt_list:
            mask_np = _decode_coco_mask(ann.get("segmentation"), h, w)
            color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
            if mask_np is not None:
                mask_bool = mask_np.astype(bool)
                gt_comp[mask_bool] = 0.6 * gt_comp[mask_bool] + 0.4 * color
            x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
            gt_draw.append(((int(x), int(y), int(x + bw), int(y + bh)), color))

        gt_img = Image.fromarray(np.clip(gt_comp, 0, 255).astype(np.uint8))
        if gt_draw:
            draw = ImageDraw.Draw(gt_img)
            for bbox, color in gt_draw:
                outline = tuple(int(c) for c in color.tolist())
                draw.rectangle(bbox, outline=outline, width=2)
        gt_out = os.path.join(vis_dir, f"{base_name[:-4]}_unsamv2_gt.png")
        gt_img.save(gt_out)

    pred_list = pred_by_image.get(img_id, [])
    pred_comp = np.array(img).astype(np.float32)
    pred_draw = []
    for pred in pred_list:
        score_val = float(pred.get("score", 0.0))
        if score_val < score_thresh:
            continue
        mask_np = _decode_coco_mask(pred.get("segmentation"), h, w)
        color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        if mask_np is not None:
            mask_bool = mask_np.astype(bool)
            pred_comp[mask_bool] = 0.6 * pred_comp[mask_bool] + 0.4 * color
        bbox = pred.get("bbox")
        if bbox is None and mask_np is not None:
            ys, xs = np.where(mask_bool)
            if ys.size:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]
        if bbox is not None:
            x, y, bw, bh = bbox
            pred_draw.append(((int(x), int(y), int(x + bw), int(y + bh)), color, score_val))

    pred_img = Image.fromarray(np.clip(pred_comp, 0, 255).astype(np.uint8))
    if pred_draw:
        draw = ImageDraw.Draw(pred_img)
        for bbox, color, score_val in pred_draw:
            outline = tuple(int(c) for c in color.tolist())
            draw.rectangle(bbox, outline=outline, width=2)
            text_pos = (bbox[0], max(0, bbox[1] - 10))
            draw.text(text_pos, f"{score_val:.3f}", fill=outline)
    pred_out = os.path.join(vis_dir, f"{base_name[:-4]}_unsamv2_pred.png")
    pred_img.save(pred_out)

    if vis_gt:
        print(f"Saved {gt_out} and {pred_out}")
    else:
        print(f"Saved {pred_out}")
PY
    done
  done
done
