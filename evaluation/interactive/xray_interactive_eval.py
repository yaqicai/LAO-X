#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import sam2
import torch
from pycocotools import mask as mask_utils

# NumPy 2.0 compatibility for older imgaug/albumentations import chains.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_, np.object_],
    }

REPO_ROOT = Path(__file__).resolve().parents[2]
SAM2_REPO_ROOT = REPO_ROOT / "sam2"
SAM2_PKG_ROOT = SAM2_REPO_ROOT / "sam2"
SAM2_HYDRA_ROOT = Path(list(sam2.__path__)[0]).resolve()
for path in (REPO_ROOT, SAM2_REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import XrayImgRoot, XrayRoot
from isegm.inference.clicker import Clicker
from isegm.inference.evaluation import get_sam_input
from isegm.inference.utils import compute_noc_metric, get_iou
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Xray evaluation for NoC@80, NoC@90 and IoU@1."
    )
    parser.add_argument("--dataset", default="det_compass_xp")
    parser.add_argument("--coco-json", default="", help="Optional COCO json override.")
    parser.add_argument("--image-root", default="", help="Optional image root override.")
    parser.add_argument("--sam2-ckpt", required=True, help="Checkpoint path.")
    parser.add_argument(
        "--sam-type",
        default="unsamv2",
        choices=["unsamv2", "sam2"],
        help="Use unsamv2 granularity-aware logic or official sam2 logic.",
    )
    parser.add_argument(
        "--sam2-cfg",
        default="configs/unsamv2_small.yaml",
        help="SAM2 config path relative under sam2/ or absolute path.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-n-clicks", type=int, default=1)
    parser.add_argument("--n-clicks", type=int, default=20)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--gra", type=float, default=None)
    parser.add_argument("--output", default="", help="Optional txt/json output path.")
    return parser.parse_args()


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


def _resolve_image_path(file_name, image_root):
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(image_root, file_name)


def _decode_coco_mask(seg, height, width):
    if seg is None:
        return None
    try:
        if isinstance(seg, list):
            if len(seg) == 0:
                return None
            valid_polys = [poly for poly in seg if poly and len(poly) >= 6 and len(poly) % 2 == 0]
            if not valid_polys:
                return None
            rles = mask_utils.frPyObjects(valid_polys, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(seg, dict):
            if isinstance(seg.get("counts"), list):
                if not seg.get("counts"):
                    return None
                rle = mask_utils.frPyObjects(seg, height, width)
            else:
                rle = dict(seg)
                rle["size"] = [int(height), int(width)]
        else:
            return None
        mask = mask_utils.decode(rle)
    except Exception:
        return None
    if mask.shape[0] != height or mask.shape[1] != width:
        fixed = np.zeros((height, width), dtype=bool)
        h = min(height, mask.shape[0])
        w = min(width, mask.shape[1])
        fixed[:h, :w] = mask[:h, :w]
        mask = fixed
    return mask.astype(bool)


def _bbox_to_mask(bbox, height, width):
    if not bbox or len(bbox) != 4:
        return None
    x, y, w, h = bbox
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(width, int(x + w))
    y2 = min(height, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _build_predictor(cfg_path, ckpt_path, device):
    cfg_path = str(cfg_path)
    sam2_model = build_sam2(cfg_path, ckpt_path, device=device, strict=False)
    return SAM2ImagePredictor(
        sam2_model, mask_threshold=-1, max_hole_area=0, max_sprinkle_area=0
    )


def _resolve_sam2_cfg_path(cfg_arg):
    cfg_path = Path(cfg_arg)
    if cfg_path.is_absolute():
        try:
            return cfg_path.relative_to(SAM2_HYDRA_ROOT).as_posix()
        except ValueError:
            try:
                return cfg_path.relative_to(SAM2_PKG_ROOT).as_posix()
            except ValueError:
                return cfg_path.as_posix()

    # Support repo-relative paths like sam2/sam2/sam2_hiera_s.yaml.
    repo_root_candidate = REPO_ROOT / cfg_path
    if repo_root_candidate.exists():
        try:
            return repo_root_candidate.relative_to(SAM2_HYDRA_ROOT).as_posix()
        except ValueError:
            try:
                return repo_root_candidate.relative_to(SAM2_PKG_ROOT).as_posix()
            except ValueError:
                return repo_root_candidate.as_posix()

    hydra_candidate = SAM2_HYDRA_ROOT / cfg_path
    if hydra_candidate.exists():
        return hydra_candidate.relative_to(SAM2_HYDRA_ROOT).as_posix()

    package_candidate = SAM2_PKG_ROOT / cfg_path
    if package_candidate.exists():
        try:
            return package_candidate.relative_to(SAM2_HYDRA_ROOT).as_posix()
        except ValueError:
            return package_candidate.relative_to(SAM2_PKG_ROOT).as_posix()

    repo_candidate = SAM2_REPO_ROOT / cfg_path
    if repo_candidate.exists():
        try:
            return repo_candidate.relative_to(SAM2_HYDRA_ROOT).as_posix()
        except ValueError:
            try:
                return repo_candidate.relative_to(SAM2_PKG_ROOT).as_posix()
            except ValueError:
                return repo_candidate.as_posix()

    # Fallback: if the requested path does not exist but a config with the same
    # basename exists under the sam2 package root, use that Hydra config name.
    basename_hydra_candidate = SAM2_HYDRA_ROOT / cfg_path.name
    if basename_hydra_candidate.exists():
        return basename_hydra_candidate.relative_to(SAM2_HYDRA_ROOT).as_posix()
    basename_candidate = SAM2_PKG_ROOT / cfg_path.name
    if basename_candidate.exists():
        try:
            return basename_candidate.relative_to(SAM2_HYDRA_ROOT).as_posix()
        except ValueError:
            return basename_candidate.relative_to(SAM2_PKG_ROOT).as_posix()

    return cfg_arg


def _evaluate_object(image, gt_mask, predictor, gra, n_clicks, min_n_clicks, sam_type):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask, dtype=bool)
    prev_low_res_masks_np = None
    ious = []

    for click_idx in range(n_clicks):
        clicker.make_next_click(pred_mask)
        point_coords, point_labels = get_sam_input(clicker)
        predict_kwargs = dict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=prev_low_res_masks_np,
            multimask_output=False,
            return_logits=True,
        )
        if sam_type == "unsamv2":
            predict_kwargs["gra"] = gra
            predict_kwargs["granularity"] = (
                None if gra is None else torch.tensor([gra]).reshape(1, 1, 1)
            )
        with torch.inference_mode():
            pred_probs, _, low_res_masks_np = predictor.predict(**predict_kwargs)
        prev_low_res_masks_np = low_res_masks_np
        pred_mask = pred_probs[0] > -1
        iou = get_iou(gt_mask, pred_mask)
        ious.append(iou)
        if click_idx + 1 >= min_n_clicks and iou >= 0.90:
            break

    return np.array(ious, dtype=np.float32)


def _evaluate_object_batch_sam2(images, gt_masks, predictor, n_clicks):
    predictor.set_image_batch(images)
    pred_masks = [np.zeros_like(gt_mask, dtype=bool) for gt_mask in gt_masks]
    prev_low_res_masks = [None] * len(gt_masks)
    clickers = [Clicker(gt_mask=gt_mask) for gt_mask in gt_masks]
    ious_lists = [[] for _ in gt_masks]

    for _ in range(n_clicks):
        point_coords_batch = []
        point_labels_batch = []
        for task_idx, clicker in enumerate(clickers):
            clicker.make_next_click(pred_masks[task_idx])
            point_coords, point_labels = get_sam_input(clicker)
            point_coords_batch.append(point_coords)
            point_labels_batch.append(point_labels)

        pred_probs_batch, _, low_res_masks_batch = predictor.predict_batch(
            point_coords_batch=point_coords_batch,
            point_labels_batch=point_labels_batch,
            mask_input_batch=prev_low_res_masks,
            multimask_output=False,
            return_logits=True,
        )

        for task_idx, gt_mask in enumerate(gt_masks):
            prev_low_res_masks[task_idx] = low_res_masks_batch[task_idx]
            pred_mask = pred_probs_batch[task_idx][0] > -1
            pred_masks[task_idx] = pred_mask
            iou = get_iou(gt_mask, pred_mask)
            ious_lists[task_idx].append(iou)

    return [np.array(ious, dtype=np.float32) for ious in ious_lists]


def _iter_object_tasks(images, ann_by_image, image_root, start_id, end_id, max_images=0):
    processed_images = 0
    for idx in range(start_id, end_id):
        if max_images and processed_images >= max_images:
            break
        img_info = images[idx]
        file_name = img_info.get("file_name")
        img_id = img_info.get("id")
        if not file_name:
            continue
        img_path = _resolve_image_path(file_name, image_root)
        if not os.path.exists(img_path):
            continue
        anns = ann_by_image.get(img_id, [])
        if not anns:
            continue
        yielded = False
        for ann in anns:
            yield {
                "img_id": img_id,
                "file_name": file_name,
                "img_path": img_path,
                "ann": ann,
            }
            yielded = True
        if yielded:
            processed_images += 1


def main():
    args = parse_args()
    sam2_cfg_path = _resolve_sam2_cfg_path(args.sam2_cfg)

    dataset_root = XrayRoot.get(args.dataset)
    if not dataset_root and not args.coco_json:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    coco_json = args.coco_json or _resolve_coco_json(dataset_root)
    if not coco_json:
        raise FileNotFoundError("Missing COCO json.")

    image_root = args.image_root or XrayImgRoot.get(args.dataset) or dataset_root
    if not image_root:
        raise ValueError("Missing image root.")

    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann.get("image_id"), []).append(ann)

    start_id = max(args.start_id, 0)
    end_id = len(images) if args.end_id is None else min(args.end_id, len(images))

    predictor = _build_predictor(sam2_cfg_path, args.sam2_ckpt, args.device)

    all_ious = []
    total_objects = 0
    object_tasks = list(
        _iter_object_tasks(
            images,
            ann_by_image,
            image_root,
            start_id,
            end_id,
            max_images=args.max_images,
        )
    )
    processed_images = len({task["img_id"] for task in object_tasks})

    if args.sam_type == "sam2" and args.batch_size > 1:
        batch_size = max(1, int(args.batch_size))
        for batch_start in range(0, len(object_tasks), batch_size):
            task_batch = object_tasks[batch_start : batch_start + batch_size]
            image_batch = []
            gt_masks = []
            valid_tasks = []
            for task in task_batch:
                image = cv2.imread(task["img_path"])
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                ann = task["ann"]
                gt_mask = _decode_coco_mask(ann.get("segmentation"), height, width)
                if gt_mask is None or gt_mask.sum() == 0:
                    gt_mask = _bbox_to_mask(ann.get("bbox"), height, width)
                if gt_mask is None or gt_mask.sum() == 0:
                    continue
                image_batch.append(image)
                gt_masks.append(gt_mask)
                valid_tasks.append(task)
            if not valid_tasks:
                continue
            batch_ious = _evaluate_object_batch_sam2(
                image_batch,
                gt_masks,
                predictor,
                args.n_clicks,
            )
            for obj_ious in batch_ious:
                if obj_ious.size == 0:
                    continue
                all_ious.append(obj_ious)
                total_objects += 1
    else:
        if args.sam_type == "unsamv2" and args.batch_size > 1:
            print("batch_size>1 is currently only accelerated for sam2 mode; falling back to sequential unsamv2 evaluation.")
        for task in object_tasks:
            image = cv2.imread(task["img_path"])
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            gt_mask = _decode_coco_mask(task["ann"].get("segmentation"), height, width)
            if gt_mask is None or gt_mask.sum() == 0:
                gt_mask = _bbox_to_mask(task["ann"].get("bbox"), height, width)
            if gt_mask is None or gt_mask.sum() == 0:
                continue
            predictor.set_image(image)
            obj_ious = _evaluate_object(
                image,
                gt_mask,
                predictor,
                args.gra,
                args.n_clicks,
                args.min_n_clicks,
                args.sam_type,
            )
            if obj_ious.size == 0:
                continue
            all_ious.append(obj_ious)
            total_objects += 1

    if total_objects == 0:
        raise SystemExit("No valid objects found for evaluation.")

    noc_list, _, over_max_list = compute_noc_metric(
        all_ious, iou_thrs=[0.80, 0.90], max_clicks=args.n_clicks
    )
    iou_first = float(np.mean([ious[0] for ious in all_ious]))

    metrics = {
        "dataset": args.dataset,
        "num_images": processed_images,
        "num_objects": total_objects,
        "NoC80": float(noc_list[0]),
        "NoC90": float(noc_list[1]),
        "over_max_80": int(over_max_list[0]),
        "over_max_90": int(over_max_list[1]),
        "IoU@1": iou_first,
        "gra": args.gra,
        "sam_type": args.sam_type,
        "batch_size": args.batch_size,
        "checkpoint": args.sam2_ckpt,
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        else:
            with out_path.open("w", encoding="utf-8") as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
