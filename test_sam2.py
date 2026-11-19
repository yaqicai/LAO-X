#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_util
from ultralytics import SAM

from config import XrayImgRoot, XrayRoot

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_ROOT = os.environ.get("LAOX_CKPT_ROOT", os.path.join(REPO_ROOT, "checkpoints"))


def _resolve_coco_json(dataset_root: str) -> str | None:
    dataset_name = os.path.basename(os.path.normpath(dataset_root)).lower()
    candidates = [
        *(
            [
                os.path.join(
                    dataset_root,
                    "annotations",
                    "instances_test2017_agnostic.json",
                )
            ]
            if dataset_name == "pidray"
            else []
        ),
        os.path.join(dataset_root, "val.json"),
        os.path.join(dataset_root, "default_coco_xray_v2.json"),
        os.path.join(dataset_root, "default_coco_xray_v1.json"),
        os.path.join(dataset_root, "coco_agnostic.json"),
        os.path.join(dataset_root, "annotations", "coco_agnostic.json"),
        os.path.join(dataset_root, "default_coco_xray.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_coco_images(coco_json: str) -> Tuple[List[dict], List[dict]]:
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = sorted(data.get("images", []), key=lambda x: x.get("id", 0))
    categories = data.get("categories", [])
    return images, categories


def _resolve_image_path(file_name: str, image_root: str, dataset_root: str) -> str:
    if os.path.isabs(file_name):
        return file_name
    cand = os.path.join(image_root, file_name)
    if os.path.exists(cand):
        return cand
    return os.path.join(dataset_root, file_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 inference via ultralytics")
    parser.add_argument("--dataset", default="det_compass_xp", help="Dataset name")
    parser.add_argument(
        "--sam2-ckpt",
        default=os.path.join(CKPT_ROOT, "sam2.1_s.pt"),
        help="Path to SAM2 checkpoint (ultralytics format; default: $LAOX_CKPT_ROOT/sam2.1_s.pt)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "output_eval"),
        help="Root output directory",
    )
    parser.add_argument("--start-id", type=int, default=0, help="Start image index")
    parser.add_argument("--end-id", type=int, default=None, help="End image index (exclusive)")
    parser.add_argument("--comment", default="", help="Optional comment suffix")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--coco-json", default=None, help="Override COCO json path")
    parser.add_argument("--image-root", default=None, help="Override image root directory")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold passed to SAM predict.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = XrayRoot.get(args.dataset)
    if not dataset_root:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    coco_json = args.coco_json or _resolve_coco_json(dataset_root)
    if not coco_json:
        raise FileNotFoundError(f"No COCO json found under {dataset_root}")

    images, categories = _load_coco_images(coco_json)
    if args.end_id is None:
        end_id = len(images)
    else:
        end_id = min(args.end_id, len(images))
    start_id = max(args.start_id, 0)

    image_root = args.image_root or XrayImgRoot.get(
        args.dataset, os.path.join(dataset_root, "xray")
    )

    comment_suffix = f"_{args.comment}" if args.comment else ""
    output_subdir = (
        f"{args.dataset.lower()}_img{start_id}_{end_id}_sam2{comment_suffix}_conf{args.conf}"
    )
    output_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    category_id = categories[0]["id"] if categories else 1

    model = SAM(args.sam2_ckpt).to(args.device).eval()

    results_list = []
    total = 0

    for idx in range(start_id, end_id):
        info = images[idx]
        image_id = info.get("id", idx)
        file_name = info.get("file_name")
        if not file_name:
            continue

        image_path = _resolve_image_path(file_name, image_root, dataset_root)
        if not os.path.exists(image_path):
            continue

        with torch.inference_mode():
            preds = model(image_path, conf=args.conf)

        if not preds:
            continue
        pred = preds[0]
        if pred.masks is None or pred.masks.data is None:
            continue

        masks = pred.masks.data
        scores = None
        if pred.boxes is not None and getattr(pred.boxes, "conf", None) is not None:
            scores = pred.boxes.conf.detach().cpu().tolist()

        for i, mask_t in enumerate(masks):
            mask = mask_t.detach().cpu().numpy().astype(np.uint8)
            if mask.sum() == 0:
                continue
            rle = mask_util.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("ascii")
            bbox = mask_util.toBbox(rle).tolist()
            score = float(scores[i]) if scores and i < len(scores) else 1.0
            results_list.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "bbox": bbox,
                    "score": score,
                }
            )
        total += 1

    out_path = os.path.join(output_dir, "coco_instances_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f)

    print(
        f"Saved {len(results_list)} instances from {total} images to {out_path}"
    )


if __name__ == "__main__":
    main()
