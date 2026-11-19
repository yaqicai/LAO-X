#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple


def _polygon_to_segmentation(poly: List[List[float]]) -> List[List[float]] | None:
    if not poly or len(poly) < 3:
        return None
    flat = [float(v) for pt in poly for v in pt]
    if len(flat) < 6:
        return None
    return [flat]


def _polygon_bbox(poly: List[List[float]]) -> List[float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _polygon_area(poly: List[List[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Pixray polygons into coco.json as segmentation.")
    parser.add_argument(
        "--pixray-json-dir",
        default="/path/to/datasets/pixray/pixray_json",
        help="Directory containing per-image Pixray json annotations.",
    )
    parser.add_argument(
        "--coco-json",
        default="/path/to/datasets/pixray/coco.json",
        help="COCO json to update.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output path (default: overwrite coco-json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    cat_name_to_id = {c["name"]: c["id"] for c in categories if "name" in c and "id" in c}

    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    image_id_by_name = {}
    for img in images:
        file_name = img.get("file_name")
        if not file_name:
            continue
        base = os.path.splitext(os.path.basename(file_name))[0]
        image_id_by_name[base] = img.get("id")

    ann_by_image_cat: Dict[Tuple[int, int], List[int]] = {}
    for idx, ann in enumerate(anns):
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        if img_id is None or cat_id is None:
            continue
        ann_by_image_cat.setdefault((img_id, cat_id), []).append(idx)

    matched = 0
    skipped = 0
    total_objs = 0

    for fname in sorted(os.listdir(args.pixray_json_dir)):
        if not fname.endswith(".json"):
            continue
        base = os.path.splitext(fname)[0]
        img_id = image_id_by_name.get(base)
        if img_id is None:
            continue

        json_path = os.path.join(args.pixray_json_dir, fname)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        objects = data.get("objects", [])
        for obj in objects:
            label = obj.get("label")
            poly = obj.get("polygon")
            if label is None or not poly:
                skipped += 1
                continue
            cat_id = cat_name_to_id.get(label)
            if cat_id is None:
                skipped += 1
                continue

            seg = _polygon_to_segmentation(poly)
            if seg is None:
                skipped += 1
                continue
            bbox = _polygon_bbox(poly)

            total_objs += 1
            candidates = ann_by_image_cat.get((img_id, cat_id), [])
            if not candidates:
                skipped += 1
                continue

            best_idx = None
            best_iou = -1.0
            for ann_idx in candidates:
                ann_bbox = anns[ann_idx].get("bbox")
                if not ann_bbox:
                    continue
                iou = _bbox_iou(bbox, ann_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = ann_idx

            if best_idx is None:
                skipped += 1
                continue

            anns[best_idx]["segmentation"] = seg
            anns[best_idx]["area"] = _polygon_area(poly)
            matched += 1
            candidates.remove(best_idx)

    out_path = args.output or args.coco_json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(
        f"Done. matched={matched} skipped={skipped} total_objects={total_objs} "
        f"output={out_path}"
    )


if __name__ == "__main__":
    main()
