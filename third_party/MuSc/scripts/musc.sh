#!/usr/bin/env bash
set -e

COMMON_ARGS=(
  --device 0
  --class_name ALL
  --backbone_name ViT-L-14-336
  --pretrained openai
  --feature_layers 5 11 17 23
  --img_resize 560
  --divide_num 1
  --r_list 1 3 5
  --batch_size 4
  --max_images 100
  --crop_mode ratio_expand
  --output_dir ./output_xray_v2
  --vis True
  --save_excel True
)

run_xray_dataset() {
  local dataset_name="$1"
  local data_path="$2"
  local annotation_path="$3"
  local image_root="$4"
  local annotation_mode="$5"

  python examples/musc_main.py \
    --data_path "${data_path}" \
    --dataset_name "${dataset_name}" \
    --annotation_path "${annotation_path}" \
    --image_root "${image_root}" \
    --annotation_mode "${annotation_mode}" \
    "${COMMON_ARGS[@]}"
}

# run_xray_dataset \
#   pixray \
#   /path/to/datasets/pixray \
#   /path/to/datasets/pixray/coco.json \
#   /path/to/datasets/pixray/PIXray \
#   mask

# run_xray_dataset \
#   CLCXray \
#   /path/to/datasets/CLCXray \
#   /path/to/datasets/CLCXray/coco_agnostic_val.json \
#   /path/to/datasets/CLCXray/val2017 \
#   bbox

# run_xray_dataset \
#   DvXray \
#   /path/to/datasets/DvXray \
#   /path/to/datasets/DvXray/coco_agnostic.json \
#   /path/to/datasets/DvXray/DvXray_Positive_Samples \
#   bbox

run_xray_dataset \
  pidray \
  /path/to/datasets/pidray \
  /path/to/datasets/pidray/pidray/annotations/coco_agnostic.json \
  /path/to/datasets/pidray/pidray/train \
  mask

run_xray_dataset \
  HiXray \
  /path/to/datasets/HiXray \
  /path/to/datasets/HiXray/coco_agnostic.json \
  /path/to/datasets/HiXray/test/test_image \
  bbox
