#!/usr/bin/env python3
"""Per-image pseudo labels and GT / oversegmentation fusion.

Paths are resolved under ``config.Root`` (``$LAOX_DATA_ROOT``, default
``<repo>/datasets``) as ``<Root>/<dataset>/<stage>``. Typical use per stage::

    # 0) class-agnostic COCO: <stage>/coco.json -> <stage>/coco_agnostic.json
    python make_pseudo_labels.py --to-agnostic --dataset crop_aug_xp --stage easy
    # 1) per-image GT labels: <stage>/anno/*.json
    python make_pseudo_labels.py --dataset crop_aug_xp --coco-name coco_agnostic.json --stage easy
    # 2) fuse GT with oversegmentation (<stage>/anno_cutler from
    #    granularity_divide_and_conquer/run.sh):
    #    writes <stage>/anno_cutler_clean and <stage>/anno_fusion
    python make_pseudo_labels.py --fusion --fusion-name anno_fusion --dataset crop_aug_xp \
        --stage easy --fusion-overlap-thresh 0.95
"""
import os
import cv2
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from xray_utils import (
    DEFAULT_CATEGORIES,
    DEFAULT_INFO,
    DEFAULT_LICENSES,
    convert_to_class_agnostic,
    generate_anno_per_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a COCO json into per-image annotation json files."
    )
    parser.add_argument(
        "--to-agnostic",
        nargs="?",
        const="",
        default=None,
        metavar="COCO_JSON",
        help=(
            "Only write a class-agnostic copy (coco_agnostic.json) next to COCO_JSON and exit. "
            "Without a value, uses <Root>/<dataset>/<stage>/coco.json."
        ),
    )
    parser.add_argument(
        "--dataset", default="", help="Dataset name for input/output paths."
    )
    parser.add_argument(
        "--coco-name",
        default="coco_cutler_musc.json",
        help="COCO JSON filename under dataset root.",
    )
    parser.add_argument(
        "--coco-json", default="", help="COCO JSON path or filename under dataset root."
    )
    parser.add_argument(
        "--output-dir", default="", help="Output directory for per-image JSONs."
    )
    parser.add_argument("--stage", default="", help="Stage name under dataset path.")
    parser.add_argument(
        "--name-mode",
        default="file_name",
        choices=["file_name", "id"],
        help="Output filename mode: use image file_name stem or image id.",
    )
    parser.add_argument(
        "--granularity-mode",
        default="fix1.0",
        choices=["fix1.0", "minmax", "bins10"],
        help="Granularity assignment: fixed 1.0, min-max normalized, or 10-bin mapping.",
    )
    parser.add_argument(
        "--fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fusion of GT masks and over-segmentation annotations.",
    )
    parser.add_argument(
        "--fusion-name",
        default="anno_fusion",
        help="Output folder name under {Root}/{dataset}/{stage}.",
    )
    parser.add_argument(
        "--fusion-overlap-thresh",
        type=float,
        default=0.85,
        help="Containment threshold for keeping over-segmentation masks.",
    )
    parser.add_argument(
        "--fusion-iou-dup-thresh",
        type=float,
        default=0.85,
        help="IoU threshold to drop cutler masks that duplicate GT.",
    )
    parser.add_argument(
        "--vis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable visualization outputs (vis_anno/vis_cutler/vis_fusion).",
    )
    return parser.parse_args()


def _decode_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return np.zeros((height, width), dtype=bool)
        polys = []
        for poly in segmentation:
            if not poly or len(poly) < 6 or (len(poly) % 2) != 0:
                continue
            polys.append(poly)
        if not polys:
            return np.zeros((height, width), dtype=bool)
        try:
            rles = mask_utils.frPyObjects(polys, height, width)
            mask = mask_utils.decode(mask_utils.merge(rles)).astype(bool)
        except Exception:
            return np.zeros((height, width), dtype=bool)
        return _ensure_mask_size(mask, height, width)
    if isinstance(segmentation, dict):
        seg = dict(segmentation)
        if "counts" in seg and isinstance(seg.get("counts"), list):
            if not seg.get("counts"):
                return np.zeros((height, width), dtype=bool)
            try:
                rle = mask_utils.frPyObjects(seg, height, width)
                mask = mask_utils.decode(rle).astype(bool)
            except Exception:
                return np.zeros((height, width), dtype=bool)
            return _ensure_mask_size(mask, height, width)
        # compressed RLE: force size to image size to fix off-by-one
        seg["size"] = [int(height), int(width)]
        try:
            mask = mask_utils.decode(seg).astype(bool)
        except Exception:
            return np.zeros((height, width), dtype=bool)
        return _ensure_mask_size(mask, height, width)
    raise ValueError("Unsupported segmentation format")


def _ensure_mask_size(mask, height, width):
    if mask is None:
        return None
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    h = min(mask.shape[0], height)
    w = min(mask.shape[1], width)
    out = np.zeros((height, width), dtype=bool)
    out[:h, :w] = mask[:h, :w]
    return out


def _load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from config import Root, XrayRoot

    return Root, XrayRoot


def _resolve_base_dir(root, dataset, stage):
    if not dataset:
        return ""
    return os.path.join(root, dataset, stage) if stage else os.path.join(root, dataset)


def _resolve_coco_json(dataset_root, coco_json, coco_name):
    name = coco_json or coco_name
    if not name:
        return ""
    if os.path.isabs(name) or not dataset_root:
        return name
    return os.path.join(dataset_root, name)


def _resolve_input_paths(args, root, xray_root):
    dataset_root = ""
    if args.dataset:
        dataset_root = xray_root.get(args.dataset) or _resolve_base_dir(
            root, args.dataset, args.stage
        )
    return _resolve_coco_json(dataset_root, args.coco_json, args.coco_name)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _mask_from_annotation(ann, height, width):
    seg = ann.get("segmentation")
    return _decode_mask(seg, height, width)


def _contains_ratio(pred_mask, gt_mask):
    pred_area = int(pred_mask.sum())
    if pred_area == 0:
        return 0.0
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    return inter / float(pred_area)


def _mask_iou(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return 0.0
    if mask_a.shape != mask_b.shape:
        h = min(mask_a.shape[0], mask_b.shape[0])
        w = min(mask_a.shape[1], mask_b.shape[1])
        mask_a = mask_a[:h, :w]
        mask_b = mask_b[:h, :w]
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return inter / float(union) if union > 0 else 0.0


def fusion_annotations(
    gt_dir,
    cutler_dir,
    output_dir,
    overlap_thresh,
    iou_dup_thresh,
    clean_dir=None,
    enable_vis=False,
):
    if not gt_dir or not cutler_dir or not output_dir:
        raise ValueError("Missing fusion paths.")
    os.makedirs(output_dir, exist_ok=True)
    if clean_dir is None:
        clean_dir = os.path.join(os.path.dirname(output_dir), "anno_cutler_clean")
    os.makedirs(clean_dir, exist_ok=True)
    vis_cutler_dir = None
    vis_cutler_clean_dir = None
    vis_anno_dir = None
    vis_fusion_dir = None
    if enable_vis:
        vis_root = os.path.dirname(output_dir)
        vis_cutler_dir = os.path.join(vis_root, "vis_cutler")
        vis_cutler_clean_dir = os.path.join(vis_root, "vis_cutler_clean")
        vis_anno_dir = os.path.join(vis_root, "vis_anno")
        vis_fusion_dir = os.path.join(vis_root, "vis_fusion")
        os.makedirs(vis_cutler_dir, exist_ok=True)
        os.makedirs(vis_cutler_clean_dir, exist_ok=True)
        os.makedirs(vis_anno_dir, exist_ok=True)
        os.makedirs(vis_fusion_dir, exist_ok=True)

    gt_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".json")) 
    for fname in tqdm(gt_files, desc="fusion"):
        gt_path = os.path.join(gt_dir, fname)
        cutler_path = os.path.join(cutler_dir, fname)
        if not os.path.isfile(cutler_path):
            continue

        gt_data = _load_json(gt_path)
        cutler_data = _load_json(cutler_path)

        image_info = gt_data.get("image", {})
        width = image_info.get("width")
        height = image_info.get("height")
        if width is None or height is None:
            cutler_image = cutler_data.get("image", {})
            width = cutler_image.get("width")
            height = cutler_image.get("height")
        if width is None or height is None:
            continue

        gt_anns = gt_data.get("annotations", [])
        cutler_anns = cutler_data.get("annotations", [])
        print(
            f"[fusion] {fname} gt={len(gt_anns)} cutler={len(cutler_anns)}", flush=True
        )

        gt_masks = []
        for ann in gt_anns:
            mask = _mask_from_annotation(ann, height, width)
            if mask is not None:
                gt_masks.append(mask.astype(bool))

        fused = list(gt_anns)
        clean_cutler = []
        next_id = max([ann.get("id", 0) for ann in fused], default=0) + 1

        for ann in cutler_anns:
            if ann.get("granularity") == 1.0:
                continue
            pred_mask = _mask_from_annotation(ann, height, width)
            if pred_mask is None:
                continue
            pred_mask = pred_mask.astype(bool)
            keep = False
            duplicate = False
            for gt_mask in gt_masks:
                if _mask_iou(pred_mask, gt_mask) >= iou_dup_thresh:
                    duplicate = True
                    break
                if _contains_ratio(pred_mask, gt_mask) >= overlap_thresh:
                    keep = True
            if duplicate:
                continue
            if not keep:
                continue
            clean_cutler.append(dict(ann))
            ann_copy = dict(ann)
            ann_copy["id"] = next_id
            next_id += 1
            fused.append(ann_copy)

        out_data = {
            "info": gt_data.get("info", DEFAULT_INFO),
            "licenses": gt_data.get("licenses", DEFAULT_LICENSES),
            "categories": gt_data.get("categories", DEFAULT_CATEGORIES),
            "image": image_info,
            "annotations": fused,
        }
        print(f"[fusion] {fname} fused={len(fused)}", flush=True)
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=True)

        clean_data = {
            "info": cutler_data.get("info", gt_data.get("info", DEFAULT_INFO)),
            "licenses": cutler_data.get("licenses", gt_data.get("licenses", DEFAULT_LICENSES)),
            "categories": cutler_data.get("categories", gt_data.get("categories", DEFAULT_CATEGORIES)),
            "image": cutler_data.get("image", image_info),
            "annotations": clean_cutler,
        }
        clean_path = os.path.join(clean_dir, fname)
        with open(clean_path, "w") as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=True)
        if not clean_cutler:
            print(f"[fusion] {fname} No cutler mask is clean", flush=True)

        file_name = image_info.get("file_name") or cutler_data.get("image", {}).get("file_name")
        if enable_vis and file_name:
            base_dir = os.path.dirname(gt_dir)
            if os.path.isabs(file_name):
                image_path = file_name
            else:
                image_path = os.path.join(base_dir, file_name)
            if not os.path.exists(image_path):
                image_path = os.path.join(base_dir, "xray", os.path.basename(file_name))

            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                seed = image_info.get("id", 0)

                def _color_palette(n):
                    if n <= 0:
                        return []
                    hsv = np.zeros((n, 3), dtype=np.float32)
                    hsv[:, 0] = np.linspace(0, 179, n, endpoint=False)
                    hsv[:, 1] = 220
                    hsv[:, 2] = 255
                    bgr = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(n, 3)
                    return bgr.astype(np.uint8)

                def _vis_anns(anns, out_dir):
                    overlay = img.astype(np.float32)
                    colors = _color_palette(max(len(anns), 1))
                    for idx, ann in enumerate(anns):
                        mask = _mask_from_annotation(ann, h, w)
                        if mask is None:
                            continue
                        mask = mask.astype(bool)
                        color = colors[idx % len(colors)].astype(np.float32)
                        overlay[mask] = 0.55 * overlay[mask] + 0.45 * color
                        bbox = ann.get("bbox")
                        if bbox is not None:
                            x, y, bw, bh = bbox
                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + bw), int(y + bh)
                            cv2.rectangle(
                                overlay,
                                (x1, y1),
                                (x2, y2),
                                color.tolist(),
                                2,
                            )
                    out_img = np.clip(overlay, 0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(out_dir, fname.replace(".json", ".png")), out_img)

                _vis_anns(gt_anns, vis_anno_dir)
                _vis_anns(cutler_anns, vis_cutler_dir)
                _vis_anns(clean_cutler, vis_cutler_clean_dir)
                _vis_anns(fused, vis_fusion_dir)


def main():
    args = parse_args()
    root_dir, xray_root = _load_config()

    if args.to_agnostic is not None:
        coco_path = args.to_agnostic or os.path.join(
            _resolve_base_dir(root_dir, args.dataset, args.stage), "coco.json"
        )
        convert_to_class_agnostic(coco_path)
        return

    if args.fusion:
        base_dir = _resolve_base_dir(root_dir, args.dataset, args.stage)
        convert_to_class_agnostic(os.path.join(base_dir, "coco.json"))
        fusion_annotations(
            os.path.join(base_dir, "anno"),
            os.path.join(base_dir, "anno_cutler"),
            os.path.join(base_dir, args.fusion_name),
            args.fusion_overlap_thresh,
            args.fusion_iou_dup_thresh,
            clean_dir=os.path.join(base_dir, "anno_cutler_clean"),
            enable_vis=args.vis,
        )
        print("Saved fused annotations to", os.path.join(base_dir, args.fusion_name))
        return

    coco_json = _resolve_input_paths(args, root_dir, xray_root)
    if not coco_json:
        raise SystemExit("Missing --coco-json.")
    with open(coco_json, "r") as f:
        coco = json.load(f)

    if not coco.get("images", []):
        raise SystemExit("No images found in COCO JSON.")

    base_dir = _resolve_base_dir(
        root_dir, args.dataset, args.stage
    ) or os.path.dirname(coco_json)
    generate_anno_per_image(
        coco,
        args.output_dir or os.path.join(base_dir or os.getcwd(), "anno"),
        name_mode=args.name_mode,
        granularity_mode=args.granularity_mode,
        default_info=DEFAULT_INFO,
        default_licenses=DEFAULT_LICENSES,
        default_categories=DEFAULT_CATEGORIES,
    )


if __name__ == "__main__":
    main()
