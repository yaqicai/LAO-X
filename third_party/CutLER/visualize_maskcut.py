#!/usr/bin/env python3
import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_util


def _list_jsons(out_dir):
    return sorted(
        [
            os.path.join(out_dir, name)
            for name in os.listdir(out_dir)
            if name.endswith(".json")
        ]
    )


def _load_annotations(ann_path):
    with open(ann_path, "r") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return images, anns_by_image


def _color_for_id(ann_id):
    rng = random.Random(ann_id)
    return (rng.randint(30, 225), rng.randint(30, 225), rng.randint(30, 225))


def _overlay_mask(base, mask, color, alpha=0.4):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    overlay = Image.new("RGB", base.size, color)
    mask_img = Image.fromarray(mask * int(255 * alpha), mode="L")
    base.paste(overlay, mask=mask_img)


def _resolve_image_path(dataset_path, rel_path, flat=False):
    if not flat:
        candidate = os.path.join(dataset_path, rel_path)
        if os.path.exists(candidate):
            return candidate
    base_name = os.path.basename(rel_path)
    candidate = os.path.join(dataset_path, base_name)
    if os.path.exists(candidate):
        return candidate
    return None


def visualize(dataset_path, ann_path, vis_dir, flat=False):
    images, anns_by_image = _load_annotations(ann_path)
    os.makedirs(vis_dir, exist_ok=True)

    for image_id, img_info in images.items():
        img_rel = img_info["file_name"]
        img_path = _resolve_image_path(dataset_path, img_rel, flat=flat)
        if img_path is None:
            print("Missing image for: {}".format(img_rel))
            continue
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for ann in anns_by_image.get(image_id, []):
            color = _color_for_id(ann["id"])
            seg = ann.get("segmentation", None)
            if seg:
                mask = mask_util.decode(seg)
                _overlay_mask(image, mask, color)
            bbox = ann.get("bbox", None)
            if bbox:
                x, y, w, h = bbox
                draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        out_path = os.path.join(vis_dir, os.path.basename(img_rel))
        image.save(out_path)


def main():
    parser = argparse.ArgumentParser("Visualize MaskCut outputs with bboxes")
    parser.add_argument("--dataset-path", required=True, help="Image root path")
    parser.add_argument("--out-dir", required=True, help="Output directory containing json")
    parser.add_argument("--ann-path", default=None, help="Explicit annotation json path")
    parser.add_argument("--vis-dir", default=None, help="Where to save visualizations")
    parser.add_argument("--flat", action="store_true", help="Dataset path contains images directly")
    args = parser.parse_args()

    ann_path = args.ann_path
    if ann_path is None:
        jsons = _list_jsons(args.out_dir)
        if len(jsons) != 1:
            raise ValueError(
                "Found {} jsons in {}. Provide --ann-path.".format(len(jsons), args.out_dir)
            )
        ann_path = jsons[0]

    vis_dir = args.vis_dir or os.path.join(args.out_dir, "vis")
    visualize(args.dataset_path, ann_path, vis_dir, flat=args.flat)
    print("Saved visualizations to {}".format(vis_dir))


if __name__ == "__main__":
    main()
