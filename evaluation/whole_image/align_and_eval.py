#!/usr/bin/env python3
"""Align prediction image_ids to GT image_ids and run COCO evaluation."""

import argparse
import json
import os
from typing import Dict, Tuple

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _build_image_maps(gt_json: dict) -> Tuple[Dict[str, int], Dict[str, int]]:
    name_map: Dict[str, int] = {}
    stem_map: Dict[str, int] = {}
    for img in gt_json.get("images", []):
        file_name = img.get("file_name", "")
        image_id = img.get("id")
        if not file_name:
            continue
        base = os.path.basename(file_name)
        stem, _ = os.path.splitext(base)
        name_map[file_name] = image_id
        name_map[base] = image_id
        if stem:
            stem_map[stem] = image_id
    return name_map, stem_map


def _resolve_image_id(
    raw_id,
    name_map: Dict[str, int],
    stem_map: Dict[str, int],
    pred_id_to_name: Dict[int, str],
    stem_len_candidates: Tuple[int, ...],
):
    if raw_id is None:
        return None
    # 1) If prediction image_id points to a pred image list entry, map to filename.
    if isinstance(raw_id, int) and raw_id in pred_id_to_name:
        raw_id = pred_id_to_name[raw_id]
    elif isinstance(raw_id, str) and raw_id.isdigit():
        as_int = int(raw_id)
        if as_int in pred_id_to_name:
            raw_id = pred_id_to_name[as_int]

    raw_str = str(raw_id)
    # 2) Match by exact filename / basename / stem to GT image_id.
    if raw_str in name_map:
        return name_map[raw_str]
    base = os.path.basename(raw_str)
    if base in name_map:
        return name_map[base]
    stem, _ = os.path.splitext(base)
    if stem in stem_map:
        return stem_map[stem]
    if stem.isdigit() and stem_len_candidates:
        for ln in stem_len_candidates:
            if len(stem) >= ln:
                continue
            padded = stem.zfill(ln)
            if padded in stem_map:
                return stem_map[padded]
    return None


def align_predictions(gt_path: str, pred_path: str, out_path: str) -> list:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        pred = json.load(f)

    name_map, stem_map = _build_image_maps(gt)
    pred_id_to_name: Dict[int, str] = {}
    for img in pred.get("images", []):
        img_id = img.get("id")
        file_name = img.get("file_name")
        if isinstance(img_id, int) and file_name:
            pred_id_to_name[img_id] = file_name
    stem_len_candidates = sorted(
        {len(stem) for stem in stem_map.keys() if stem.isdigit()}
    )
    stem_len_candidates = tuple(stem_len_candidates)

    missing = []
    aligned = []
    for ann in pred.get("annotations", []):
        raw_id = ann.get("image_id")
        resolved = _resolve_image_id(
            raw_id,
            name_map,
            stem_map,
            pred_id_to_name,
            stem_len_candidates,
        )
        if resolved is None:
            missing.append(raw_id)
            continue
        ann["image_id"] = resolved
        ann.setdefault("category_id", 1)
        if "score" not in ann:
            ann["score"] = float(ann.get("predicted_iou", 1.0))
        aligned.append(ann)

    if missing:
        unique_missing = sorted({str(x) for x in missing})
        print(f"Warning: {len(unique_missing)} image_id values not matched. Sample: {unique_missing[:10]}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aligned, f)
    return aligned


def run_eval(gt_path: str, pred_path: str, iou_type: str):
    coco_gt = COCO(gt_path)
    with open(pred_path, "r", encoding="utf-8") as f:
        pred = json.load(f)
    if isinstance(pred, dict):
        anns = pred.get("annotations", [])
    else:
        anns = pred
    coco_dt = coco_gt.loadRes(anns)

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.useCats = 0
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metrics = {
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "APs": stats[3],
        "APm": stats[4],
        "APl": stats[5],
        "AR@1": stats[6],
        "AR@10": stats[7],
        "AR@100": stats[8],
        "ARs": stats[9],
        "ARm": stats[10],
        "ARl": stats[11],
    }
    print("\nCOCO metrics ({}):".format(iou_type))
    for k, v in metrics.items():
        print(f"{k}: {v * 100:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Align prediction image_ids and run COCO eval.")
    parser.add_argument("--gt-file", required=True, help="COCO GT json path")
    parser.add_argument("--pred-file", required=True, help="Prediction json path (merged_dt.json)")
    parser.add_argument("--out-file", default="merged_dt_aligned.json", help="Aligned prediction output")
    parser.add_argument("--iou-type", default="bbox", choices=["bbox", "segm"], help="IoU type")
    args = parser.parse_args()

    align_predictions(args.gt_file, args.pred_file, args.out_file)
    run_eval(args.gt_file, args.out_file, args.iou_type)


if __name__ == "__main__":
    main()
