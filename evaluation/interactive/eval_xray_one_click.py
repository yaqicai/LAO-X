#!/usr/bin/env python3
import argparse
import json
import os

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.inference.evaluation import get_sam_input
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import XrayImgRoot, XrayRoot


def _resolve_coco_json(dataset_root):
    candidates = [
        os.path.join(dataset_root, "val.json"),
        os.path.join(dataset_root, "default_coco_xray_v2.json"),
        os.path.join(dataset_root, "coco_agnostic.json"),
        os.path.join(dataset_root, "coco.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _decode_coco_mask(seg, height, width):
    if seg is None:
        return None
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):
            rle = mask_utils.frPyObjects(seg, height, width)
        else:
            rle = seg
    else:
        return None
    mask = mask_utils.decode(rle)
    if mask.shape[0] != height or mask.shape[1] != width:
        mask = mask[:height, :width]
    return mask.astype(bool)


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 one-click evaluation on Xray COCO datasets.")
    parser.add_argument("--dataset", default="det_compass_xp")
    parser.add_argument("--coco-json", default="", help="Optional COCO json path override.")
    parser.add_argument("--image-root", default="", help="Optional image root override.")
    parser.add_argument("--sam2-ckpt", required=True, help="SAM2 checkpoint path.")
    parser.add_argument("--sam2-cfg", required=True, help="SAM2 config path (relative to sam2).")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all).")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = XrayRoot.get(args.dataset)
    if not dataset_root and not args.coco_json:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    coco_json = args.coco_json or _resolve_coco_json(dataset_root)
    if not coco_json:
        raise FileNotFoundError("Missing COCO json. Use --coco-json to specify.")

    image_root = args.image_root or XrayImgRoot.get(args.dataset) or dataset_root
    if not image_root:
        raise ValueError("Missing image root. Use --image-root to specify.")

    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann.get("image_id"), []).append(ann)

    start_id = max(args.start_id, 0)
    end_id = len(images) if args.end_id is None else min(args.end_id, len(images))

    sam2_model = build_sam2(args.sam2_cfg, args.sam2_ckpt, device=args.device, strict=False)
    predictor = SAM2ImagePredictor(
        sam2_model, mask_threshold=-1, max_hole_area=0, max_sprinkle_area=0
    )

    total_ious = []
    total_objects = 0

    for idx in range(start_id, end_id):
        if args.max_images and (idx - start_id) >= args.max_images:
            break
        img_info = images[idx]
        img_id = img_info.get("id")
        file_name = img_info.get("file_name")
        if file_name is None:
            continue
        img_path = file_name if os.path.isabs(file_name) else os.path.join(image_root, file_name)
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        predictor.set_image(image)
        anns = ann_by_image.get(img_id, [])
        for ann in anns:
            gt_mask = _decode_coco_mask(ann.get("segmentation"), height, width)
            if gt_mask is None or gt_mask.sum() == 0:
                continue

            clicker = Clicker(gt_mask=gt_mask)
            pred_mask = np.zeros_like(gt_mask, dtype=bool)
            clicker.make_next_click(pred_mask)
            point_coords, point_labels = get_sam_input(clicker)

            with torch.inference_mode():
                pred_probs, _, _ = predictor.predict(
                    point_coords,
                    point_labels,
                    mask_input=None,
                    multimask_output=False,
                    return_logits=True,
                    gra=None,
                    granularity=None,
                )
            pred_mask = pred_probs[0] > -1
            iou = utils.get_iou(gt_mask, pred_mask)
            total_ious.append(iou)
            total_objects += 1

    if total_objects == 0:
        print("No valid objects found for evaluation.")
        return

    mean_iou = float(np.mean(total_ious))
    print(f"One-click mIoU: {mean_iou:.4f} over {total_objects} objects")


if __name__ == "__main__":
    main()
