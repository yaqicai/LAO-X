import os
import shutil
import cv2
import json
import base64
import datetime
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict
from pycocotools import mask as mask_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Shared COCO defaults
DEFAULT_INFO = {
    "version": "1.0",
    "year": 2023,
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

DEFAULT_LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
    }
]

DEFAULT_CATEGORIES = [
    {
        "id": 1,
        "name": "fg",
        "supercategory": "fg",
    }
]

# from datasets import load_dataset


def split_train_val(
        json_path_full, 
        json_path_agnostic, 
        val_ratio=0.5, 
        random_split=False, 
        seed=0, 
        mode="w/o_invisible", 
        localizer=False, 
        old_classes=None, 
        new_classes=None, 
        dataset=None):
    """
    将 COCO 格式的标注文件划分为 train.json 和 val.json

    Args:
        json_path (str): 原始 COCO json 文件路径
        val_ratio (float): 验证集比例 (0-1)
        random_split (bool): 是否随机划分，默认 False（顺序划分）
        seed (int): 随机种子（仅在 random_split=True 时生效）

    Returns:
        tuple: (train_json_path, val_json_path)
    """
    json_path_v1 = '/path/to/datasets/det_compass_syn/default_coco_xray_v1.json'
    json_path_v1_PIXray = '/path/to/datasets/PIXray/default_coco_xray_v1.json'
    # 'Bat':球棒, 10 baseball bat
    # 'Battery':电池,  
    # 'Dart':镖, 不存在
    # 'Fireworks':烟花, 不存在
    # 'Gun':枪, 不存在
    # 'Hammer':锤, 110 hammer; 150 mallet
    # 'Knife':刀, 28 box cutter, 31 bread knife, 39 can opener, 44 carving knife, 71 craft knife, 76 dagger, 132 knife, 181 penknife
    # 'Lighter':打火机, 142 lighter
    # 'Pliers':钳子, 190 pliers
    # 'Pressure_vessel'压力容器:, 101 gas canister
    # 'Razor_blade':剃须刀片, 204 razor blades
    # 'Saw_blade':锯片, 105 hacksaw, 224 saw
    # 'Scissors':剪刀, 227 scissors
    # 'Screwdriver':螺丝刀, 229 screwdriver
    # 'Wrench':扳手 310 wrench
    # [10,28,31,39,44,71,76,101,105,110,132,142,150,181,190,204,224,227,229,310] 一共20个同义物品，刚好前110个类包含10个同义物品
    with open(json_path_full, "r", encoding="utf-8") as f:
        coco = json.load(f)
    with open(json_path_agnostic, "r", encoding="utf-8") as f:
        coco_agnostic = json.load(f)
    categories_pixry_in_det_compass_syn = [10,28,31,39,44,71,76,101,105,110]
    old_classes = set(old_classes) - set(categories_pixry_in_det_compass_syn)
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]
    annotations_agnostic = coco_agnostic["annotations"]
    categories_agnostic = coco_agnostic["categories"]

    # 确保排序一致（避免不同运行结果不一样）
    images = sorted(images, key=lambda x: x["id"])

    n_total = len(images)
    n_val = int(n_total * val_ratio)
    if dataset == 'PIXray':
        val_images = images
        train_images = images
    else:
        if random_split:
            random.seed(seed)
            val_images = random.sample(images, n_val)
        

        else:
            val_images = images[-n_val:]  # 取最后 val_ratio 部分
        train_images = [img for img in images if img not in val_images]

    # 获取 image_id 列表
    train_ids = set(img["id"] for img in train_images)
    val_ids = set(img["id"] for img in val_images)

    # 按 image_id 筛选 annotations
    train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids
                         and ann['category_id'] in old_classes]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_ids
                       and ann['category_id'] not in old_classes]
    val_annotations_agnostic = [ann for ann in annotations_agnostic if ann["image_id"] in val_ids]
    train_images_id = set([ann['image_id'] for ann in train_annotations])
    train_images = [img for img in train_images if img['id'] in train_images_id]
    for anno in train_annotations:
        anno['category_id'] = 1
    for anno in val_annotations:
        anno['category_id'] = 1
    # for anno in val_annotations_agnostic:
    #     anno['category_id'] = 1
        # anno['id'] += len(val_annotations)
    # val_annotations_agnostic = val_annotations + val_annotations_agnostic
    
    # val_categories = categories + [{'id': 313, 'name': 'object'}]
    val_categories = categories_agnostic
    # 组装 COCO 格式
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories_agnostic
    }

    val_coco = {
        "images": val_images,
        "annotations": val_annotations_agnostic if dataset == 'PIXray' else val_annotations,
        "categories": val_categories
    }
    for key in coco.keys():
        if key not in ["images", "annotations", "categories"]:
            train_coco[key] = coco[key]
            val_coco[key] = coco[key]
    

    # 保存到文件
    base_dir = os.path.dirname(json_path_full)
    train_json_path = os.path.join(base_dir, "train.json")
    val_json_path = os.path.join(base_dir, "val.json")

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=2)

    print(f"训练集: {len(train_images)} 张, 验证集: {len(val_images)} 张")
    print(f"保存到 {train_json_path} 和 {val_json_path}")

    return train_json_path, val_json_path

def convert_to_class_agnostic(save_dir=None):
    if not save_dir.endswith("json"):
        json_path = os.path.join(save_dir, f"coco.json")
    else:
        json_path = save_dir
        save_dir = os.path.dirname(save_dir)
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    categories = coco["categories"]
    coco["categories"] = [{'id': 1, 'name': 'object'}]
    for anno in coco["annotations"]:
        anno['category_id'] = 1

    
    json_path = os.path.join(save_dir, f"coco_agnostic.json")
    print("save class agnostic json file to {}".format(json_path))
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=4)


def _granularity_for_area(area, mode, min_area, max_area):
    if mode == "fix1.0":
        return 1.0
    if max_area <= min_area:
        return 1.0
    ratio = (area - min_area) / (max_area - min_area)
    ratio = max(0.0, min(1.0, ratio))
    if mode == "minmax":
        return ratio
    if mode == "bins10":
        bin_idx = min(int(ratio * 10), 9)
        return 0.1 * (bin_idx + 1)
    return 1.0


def _output_name(image, name_mode):
    if name_mode == "file_name":
        file_name = image.get("file_name")
        if file_name:
            return os.path.splitext(os.path.basename(file_name))[0] + ".json"
    return f"{image.get('image_id')}.json"


def generate_anno_per_image(
    coco,
    output_dir,
    name_mode="file_name",
    granularity_mode="fix1.0",
    default_info=None,
    default_licenses=None,
    default_categories=None,
):
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if not images:
        raise SystemExit("No images found in COCO JSON.")

    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann.get("image_id"), []).append(ann)

    areas = [ann.get("area", 0.0) for ann in annotations if ann.get("area") is not None]
    min_area = min(areas) if areas else 0.0
    max_area = max(areas) if areas else 0.0

    info = coco.get("info") or default_info or {}
    licenses = coco.get("licenses") or default_licenses or []
    categories = coco.get("categories") or default_categories or []

    os.makedirs(output_dir, exist_ok=True)
    base_output = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "image": {},
        "annotations": [],
    }
    for image in images:
        image_id = image.get("id")
        per_image = dict(base_output)
        per_image["image"] = image
        per_image_annotations = []
        for idx, ann in enumerate(ann_by_image.get(image_id, []), start=1):
            ann_copy = dict(ann)
            ann_copy["id"] = idx
            ann_copy["is_divide"] = True
            ann_area = ann_copy.get("area", 0.0)
            ann_copy["granularity"] = _granularity_for_area(
                ann_area, granularity_mode, min_area, max_area
            )
            ann_copy["instance_id"] = 0
            ann_copy["is_part"] = False
            ann_copy["is_conquer"] = False
            per_image_annotations.append(ann_copy)
        per_image["annotations"] = per_image_annotations

        file_name = _output_name(image, name_mode)
        out_path = os.path.join(output_dir, file_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(per_image, f, indent=2, ensure_ascii=True)

    print(f"Saved {len(images)} JSON files to {output_dir}")


def _ann_to_rle(ann, height, width):
    seg = ann.get("segmentation")
    if seg is None:
        return None
    if isinstance(seg, list):
        if len(seg) == 0:
            return None
        try:
            rles = mask_utils.frPyObjects(seg, height, width)
            return mask_utils.merge(rles)
        except Exception:
            return None
    if isinstance(seg, dict):
        try:
            if isinstance(seg.get("counts"), list):
                return mask_utils.frPyObjects(seg, height, width)
            return seg
        except Exception:
            return None
    return None


def _bbox_iou(a, b):
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
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
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


def _ap_from_pr(precision, recall):
    if precision.size == 0 or recall.size == 0:
        return 0.0
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    prec_at_rec = []
    for t in recall_thresholds:
        p = precision[recall >= t]
        prec_at_rec.append(p.max() if p.size else 0.0)
    return float(np.mean(prec_at_rec))


def _compute_image_ap(gt_boxes, dt_boxes, dt_scores, iou_thresholds):
    if len(gt_boxes) == 0:
        return np.nan, np.nan, np.nan
    if len(dt_boxes) == 0:
        return 0.0, 0.0, 0.0

    order = np.argsort(-np.array(dt_scores, dtype=np.float32))
    dt_boxes = [dt_boxes[i] for i in order]
    dt_scores = [dt_scores[i] for i in order]

    ap_list = []
    ap50 = None
    ap75 = None
    for thr in iou_thresholds:
        matched = [False] * len(gt_boxes)
        tp = []
        fp = []
        for det_box in dt_boxes:
            best_iou = 0.0
            best_j = -1
            for j, gt_box in enumerate(gt_boxes):
                if matched[j]:
                    continue
                iou = _bbox_iou_xywh(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= thr and best_j >= 0:
                matched[best_j] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        if not tp:
            ap = 0.0
        else:
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / max(1, len(gt_boxes))
            precision = tp_cum / np.maximum(1, tp_cum + fp_cum)
            ap = _ap_from_pr(precision, recall)
        ap_list.append(ap)
        if abs(thr - 0.5) < 1e-9:
            ap50 = ap
        if abs(thr - 0.75) < 1e-9:
            ap75 = ap
    ap_mean = float(np.mean(ap_list)) if ap_list else 0.0
    return ap_mean, float(ap50 or 0.0), float(ap75 or 0.0)


def _resolve_val_json(dataset_root):
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
        os.path.join(dataset_root, "coco_agnostic.json"),
        os.path.join(dataset_root, "annotations", "coco_agnostic.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _slice_coco_json(coco_json, start_id, end_id, out_path):
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "info" not in data:
        data["info"] = DEFAULT_INFO
    if "licenses" not in data:
        data["licenses"] = DEFAULT_LICENSES
    if "categories" not in data:
        data["categories"] = DEFAULT_CATEGORIES
    images = data.get("images", [])
    if end_id is None:
        end_id = len(images)
    images = images[start_id:end_id]
    valid_image_ids = {img["id"] for img in images}
    data["images"] = images
    data["annotations"] = [
        ann for ann in data.get("annotations", []) if ann.get("image_id") in valid_image_ids
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return out_path


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return int(default)


def _is_valid_bbox_xywh(box):
    if box is None or len(box) != 4:
        return False
    x, y, w, h = [_safe_float(v, 0.0) for v in box]
    return w > 0 and h > 0


def _xywh_to_xyxy(box):
    x, y, w, h = [_safe_float(v, 0.0) for v in box]
    return x, y, x + w, y + h


def _xyxy_to_xywh(x1, y1, x2, y2):
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def _clip_xyxy_to_image(x1, y1, x2, y2, width, height):
    x1 = min(max(0.0, x1), float(width))
    y1 = min(max(0.0, y1), float(height))
    x2 = min(max(0.0, x2), float(width))
    y2 = min(max(0.0, y2), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _clip_box_to_image_xywh(box, width, height):
    if not _is_valid_bbox_xywh(box):
        return None
    x1, y1, x2, y2 = _xywh_to_xyxy(box)
    clipped = _clip_xyxy_to_image(x1, y1, x2, y2, width, height)
    if clipped is None:
        return None
    return _xyxy_to_xywh(*clipped)


def _intersection_box_xywh(a, b):
    if not _is_valid_bbox_xywh(a) or not _is_valid_bbox_xywh(b):
        return None
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return _xyxy_to_xywh(x1, y1, x2, y2)


def _bbox_area_xywh(box):
    if not _is_valid_bbox_xywh(box):
        return 0.0
    _, _, w, h = [_safe_float(v, 0.0) for v in box]
    return max(0.0, w) * max(0.0, h)


def _bbox_center_xywh(box):
    x, y, w, h = [_safe_float(v, 0.0) for v in box]
    return x + 0.5 * w, y + 0.5 * h


def _is_inside_xywh(inner_box, outer_box, eps=1e-6):
    if not _is_valid_bbox_xywh(inner_box) or not _is_valid_bbox_xywh(outer_box):
        return False
    ix1, iy1, ix2, iy2 = _xywh_to_xyxy(inner_box)
    ox1, oy1, ox2, oy2 = _xywh_to_xyxy(outer_box)
    return ix1 >= ox1 - eps and iy1 >= oy1 - eps and ix2 <= ox2 + eps and iy2 <= oy2 + eps


def _point_inside_xywh(x, y, box, eps=1e-6):
    if not _is_valid_bbox_xywh(box):
        return False
    x1, y1, x2, y2 = _xywh_to_xyxy(box)
    return (x >= x1 - eps) and (x <= x2 + eps) and (y >= y1 - eps) and (y <= y2 + eps)


def _bbox_overlap_score(box, region, metric="iof"):
    inter = _intersection_box_xywh(box, region)
    if inter is None:
        return 0.0
    inter_area = _bbox_area_xywh(inter)
    box_area = max(_bbox_area_xywh(box), 1e-12)
    region_area = max(_bbox_area_xywh(region), 1e-12)
    metric = str(metric).lower().strip()
    if metric == "iou":
        union = box_area + region_area - inter_area
        return inter_area / union if union > 0 else 0.0
    if metric in {"ior", "region"}:
        return inter_area / region_area
    # default: intersection over prediction area
    return inter_area / box_area


def _expand_box_xywh(box, ratio, width, height):
    if not _is_valid_bbox_xywh(box):
        return None
    ratio = max(_safe_float(ratio, 1.0), 0.0)
    x, y, w, h = [_safe_float(v, 0.0) for v in box]
    cx, cy = x + 0.5 * w, y + 0.5 * h
    new_w = w * ratio
    new_h = h * ratio
    x1 = cx - 0.5 * new_w
    y1 = cy - 0.5 * new_h
    x2 = cx + 0.5 * new_w
    y2 = cy + 0.5 * new_h
    clipped = _clip_xyxy_to_image(x1, y1, x2, y2, width, height)
    if clipped is None:
        return None
    return _xyxy_to_xywh(*clipped)


def _expand_square_box_xywh(box, ratio, width, height):
    if not _is_valid_bbox_xywh(box):
        return None
    ratio = max(_safe_float(ratio, 1.0), 0.0)
    x, y, w, h = [_safe_float(v, 0.0) for v in box]
    side = max(w, h) * ratio
    cx, cy = x + 0.5 * w, y + 0.5 * h
    x1 = cx - 0.5 * side
    y1 = cy - 0.5 * side
    x2 = cx + 0.5 * side
    y2 = cy + 0.5 * side
    clipped = _clip_xyxy_to_image(x1, y1, x2, y2, width, height)
    if clipped is None:
        return None
    return _xyxy_to_xywh(*clipped)


def _decode_ann_mask_bool(ann, height, width):
    rle = _ann_to_rle(ann, height, width)
    if rle is None:
        return None
    try:
        mask = mask_utils.decode(rle)
    except Exception:
        return None
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def _mask_to_bbox_xywh(mask, width, height):
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask)
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    clipped = _clip_xyxy_to_image(x1, y1, x2, y2, width, height)
    if clipped is None:
        return None
    return _xyxy_to_xywh(*clipped)


def _build_region_from_gt_ann(ann, image_h, image_w, protocol):
    base_bbox = ann.get("bbox")
    if not _is_valid_bbox_xywh(base_bbox):
        return None
    region_mode = str(protocol.get("region_mode", "gt_bbox")).lower().strip()
    expand_ratio = _safe_float(protocol.get("expand_ratio", 1.0), 1.0)
    dilate_px = _safe_int(protocol.get("dilate_px", 0), 0)
    dilate_ratio = _safe_float(protocol.get("dilate_ratio", 0.0), 0.0)

    if region_mode == "gt_bbox":
        return _clip_box_to_image_xywh(base_bbox, image_w, image_h)
    if region_mode == "gt_bbox_expand":
        return _expand_box_xywh(base_bbox, expand_ratio, image_w, image_h)
    if region_mode == "gt_bbox_square_expand":
        return _expand_square_box_xywh(base_bbox, expand_ratio, image_w, image_h)
    if region_mode in {"gt_mask_bbox", "gt_mask_dilate_bbox"}:
        mask = _decode_ann_mask_bool(ann, image_h, image_w)
        if mask is None:
            if region_mode == "gt_mask_bbox":
                return _clip_box_to_image_xywh(base_bbox, image_w, image_h)
            return _expand_box_xywh(base_bbox, max(expand_ratio, 1.0), image_w, image_h)
        if region_mode == "gt_mask_dilate_bbox":
            if dilate_px <= 0 and dilate_ratio > 0:
                area = max(float(np.count_nonzero(mask)), 1.0)
                dilate_px = int(round(np.sqrt(area) * dilate_ratio))
            if dilate_px > 0:
                kernel_size = int(max(1, 2 * dilate_px + 1))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
        return _mask_to_bbox_xywh(mask, image_w, image_h)
    raise ValueError(f"Unsupported region_mode: {region_mode}")


def _build_regions_for_image(gt_anns, image_h, image_w, protocol):
    regions = []
    for ann in gt_anns:
        region = _build_region_from_gt_ann(ann, image_h, image_w, protocol)
        if _is_valid_bbox_xywh(region):
            regions.append(region)
    return regions


def _select_region_for_prediction(dt_box, regions, protocol):
    filter_mode = str(protocol.get("filter_mode", "inside")).lower().strip()
    overlap_metric = str(protocol.get("overlap_metric", "iof")).lower().strip()
    overlap_thr = _safe_float(protocol.get("overlap_threshold", 0.0), 0.0)

    if not regions:
        return False, None, 0.0

    if filter_mode == "inside":
        for region in regions:
            if _is_inside_xywh(dt_box, region):
                return True, region, 1.0
        return False, None, 0.0

    if filter_mode == "center":
        cx, cy = _bbox_center_xywh(dt_box)
        for region in regions:
            if _point_inside_xywh(cx, cy, region):
                return True, region, 1.0
        return False, None, 0.0

    best_score = 0.0
    best_region = None
    for region in regions:
        score = _bbox_overlap_score(dt_box, region, overlap_metric)
        if score > best_score:
            best_score = score
            best_region = region
    keep = best_score >= overlap_thr
    if filter_mode in {"overlap", "clip"}:
        return keep, best_region, best_score
    raise ValueError(f"Unsupported filter_mode: {filter_mode}")


def _clip_annotation_to_region(ann, region, image_h, image_w, clip_segmentation=False):
    inter = _intersection_box_xywh(ann.get("bbox"), region)
    if inter is None or _bbox_area_xywh(inter) <= 0:
        return None

    out = dict(ann)
    out["bbox"] = [float(v) for v in inter]
    out["area"] = float(_bbox_area_xywh(inter))

    if not clip_segmentation:
        return out

    mask = _decode_ann_mask_bool(ann, image_h, image_w)
    if mask is None:
        return out

    x, y, w, h = inter
    x1 = int(max(0, np.floor(x)))
    y1 = int(max(0, np.floor(y)))
    x2 = int(min(image_w, np.ceil(x + w)))
    y2 = int(min(image_h, np.ceil(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None

    clipped_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    clipped_region = mask[y1:y2, x1:x2]
    if clipped_region.size == 0 or not np.any(clipped_region):
        return None
    clipped_mask[y1:y2, x1:x2] = clipped_region.astype(np.uint8)

    rle = mask_utils.encode(np.asfortranarray(clipped_mask))
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    out["segmentation"] = rle
    try:
        new_bbox = mask_utils.toBbox(rle).tolist()
        out["bbox"] = [float(v) for v in new_bbox]
        out["area"] = float(mask_utils.area(rle))
    except Exception:
        pass
    return out


def _normalize_protocol(protocol, idx=1):
    p = dict(protocol or {})
    p["region_mode"] = str(p.get("region_mode", "gt_bbox")).lower().strip()
    p["filter_mode"] = str(p.get("filter_mode", "inside")).lower().strip()
    p["expand_ratio"] = _safe_float(p.get("expand_ratio", 1.0), 1.0)
    p["dilate_px"] = _safe_int(p.get("dilate_px", 0), 0)
    p["dilate_ratio"] = _safe_float(p.get("dilate_ratio", 0.0), 0.0)
    p["overlap_metric"] = str(p.get("overlap_metric", "iof")).lower().strip()
    p["overlap_threshold"] = _safe_float(p.get("overlap_threshold", 0.0), 0.0)
    p["clip_segmentation"] = bool(p.get("clip_segmentation", True))

    name = str(p.get("name", "")).strip()
    if not name:
        parts = [f"P{idx:02d}", p["region_mode"], p["filter_mode"]]
        if p["region_mode"] in {"gt_bbox_expand", "gt_bbox_square_expand"}:
            parts.append(f"r{p['expand_ratio']:.2f}")
        if p["region_mode"] == "gt_mask_dilate_bbox":
            if p["dilate_px"] > 0:
                parts.append(f"d{p['dilate_px']}")
            elif p["dilate_ratio"] > 0:
                parts.append(f"dr{p['dilate_ratio']:.3f}")
        if p["filter_mode"] in {"overlap", "clip"}:
            parts.append(f"{p['overlap_metric']}{p['overlap_threshold']:.2f}")
        name = "_".join(parts).replace(".", "p")
    p["name"] = name
    return p


def _protocol_to_description(protocol):
    region_map = {
        "gt_bbox": "Region=GT bbox",
        "gt_bbox_expand": f"Region=GT bbox center-expand (ratio={protocol['expand_ratio']:.2f})",
        "gt_bbox_square_expand": f"Region=GT bbox square-expand (ratio={protocol['expand_ratio']:.2f})",
        "gt_mask_bbox": "Region=GT mask bbox (fallback GT bbox)",
        "gt_mask_dilate_bbox": (
            f"Region=GT mask dilate bbox (dilate_px={protocol['dilate_px']}, "
            f"dilate_ratio={protocol['dilate_ratio']:.4f}, fallback expand ratio={max(protocol['expand_ratio'], 1.0):.2f})"
        ),
    }
    filter_map = {
        "inside": "Keep DT only if bbox fully inside any region",
        "center": "Keep DT only if bbox center is inside any region",
        "overlap": f"Keep DT if overlap_score({protocol['overlap_metric']}) >= {protocol['overlap_threshold']:.3f}",
        "clip": (
            f"Keep DT if overlap_score({protocol['overlap_metric']}) >= {protocol['overlap_threshold']:.3f}, "
            "then clip DT bbox to intersection with best region"
        ),
    }
    extra = ""
    if protocol["filter_mode"] == "clip":
        extra = f"; clip_segmentation={int(protocol['clip_segmentation'])}"
    return f"{region_map.get(protocol['region_mode'], protocol['region_mode'])}; {filter_map.get(protocol['filter_mode'], protocol['filter_mode'])}{extra}"


def build_region_eval_protocols(
    region_modes=None,
    filter_modes=None,
    expand_ratios=None,
    overlap_metrics=None,
    overlap_thresholds=None,
    dilate_pxs=None,
    dilate_ratios=None,
    clip_segmentation=True,
):
    """Build protocol combinations for region-constrained evaluation."""
    if region_modes is None:
        region_modes = ["gt_bbox"]
    if filter_modes is None:
        filter_modes = ["inside"]
    if expand_ratios is None:
        expand_ratios = [1.0]
    if overlap_metrics is None:
        overlap_metrics = ["iof"]
    if overlap_thresholds is None:
        overlap_thresholds = [0.0]
    if dilate_pxs is None:
        dilate_pxs = [0]
    if dilate_ratios is None:
        dilate_ratios = [0.0]

    protocols = []
    for region_mode in region_modes:
        region_mode = str(region_mode).lower().strip()
        region_expand_values = expand_ratios if region_mode in {"gt_bbox_expand", "gt_bbox_square_expand", "gt_mask_dilate_bbox"} else [1.0]
        region_dilate_px_values = dilate_pxs if region_mode == "gt_mask_dilate_bbox" else [0]
        region_dilate_ratio_values = dilate_ratios if region_mode == "gt_mask_dilate_bbox" else [0.0]
        for filter_mode in filter_modes:
            filter_mode = str(filter_mode).lower().strip()
            metric_values = overlap_metrics if filter_mode in {"overlap", "clip"} else ["iof"]
            thr_values = overlap_thresholds if filter_mode in {"overlap", "clip"} else [0.0]
            for expand_ratio in region_expand_values:
                for dilate_px in region_dilate_px_values:
                    for dilate_ratio in region_dilate_ratio_values:
                        for overlap_metric in metric_values:
                            for overlap_thr in thr_values:
                                proto = {
                                    "region_mode": region_mode,
                                    "filter_mode": filter_mode,
                                    "expand_ratio": _safe_float(expand_ratio, 1.0),
                                    "dilate_px": _safe_int(dilate_px, 0),
                                    "dilate_ratio": _safe_float(dilate_ratio, 0.0),
                                    "overlap_metric": str(overlap_metric).lower().strip(),
                                    "overlap_threshold": _safe_float(overlap_thr, 0.0),
                                    "clip_segmentation": bool(clip_segmentation),
                                }
                                protocols.append(proto)

    normalized = []
    seen = set()
    for idx, p in enumerate(protocols, start=1):
        p = _normalize_protocol(p, idx=idx)
        key = (
            p["region_mode"],
            p["filter_mode"],
            p["expand_ratio"],
            p["dilate_px"],
            p["dilate_ratio"],
            p["overlap_metric"],
            p["overlap_threshold"],
            p["clip_segmentation"],
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(p)
    return normalized


def _get_protocol_preset(protocol_preset):
    key = str(protocol_preset or "").lower().strip()
    if not key or key in {"none", "off", "baseline"}:
        return []

    if key in {"xray_partial_v1", "xray_incomplete_v1"}:
        return [
            _normalize_protocol(
                {
                    "name": "S1_gt_bbox_inside",
                    "region_mode": "gt_bbox",
                    "filter_mode": "inside",
                },
                idx=1,
            ),
            _normalize_protocol(
                {
                    "name": "S2_gt_bbox_expand1p25_inside",
                    "region_mode": "gt_bbox_expand",
                    "expand_ratio": 1.25,
                    "filter_mode": "inside",
                },
                idx=2,
            ),
            _normalize_protocol(
                {
                    "name": "S3_gt_bbox_expand1p50_center",
                    "region_mode": "gt_bbox_expand",
                    "expand_ratio": 1.50,
                    "filter_mode": "center",
                },
                idx=3,
            ),
            _normalize_protocol(
                {
                    "name": "S4_gt_bbox_expand1p50_iof30",
                    "region_mode": "gt_bbox_expand",
                    "expand_ratio": 1.50,
                    "filter_mode": "overlap",
                    "overlap_metric": "iof",
                    "overlap_threshold": 0.30,
                },
                idx=4,
            ),
            _normalize_protocol(
                {
                    "name": "S5_gt_bbox_expand1p50_clip_iof05",
                    "region_mode": "gt_bbox_expand",
                    "expand_ratio": 1.50,
                    "filter_mode": "clip",
                    "overlap_metric": "iof",
                    "overlap_threshold": 0.05,
                    "clip_segmentation": True,
                },
                idx=5,
            ),
            _normalize_protocol(
                {
                    "name": "S6_gt_bbox_square1p50_clip_iof05",
                    "region_mode": "gt_bbox_square_expand",
                    "expand_ratio": 1.50,
                    "filter_mode": "clip",
                    "overlap_metric": "iof",
                    "overlap_threshold": 0.05,
                    "clip_segmentation": True,
                },
                idx=6,
            ),
        ]

    if key in {"xray_partial_v2", "xray_incomplete_v2", "mask_focus"}:
        return [
            _normalize_protocol(
                {
                    "name": "M1_gt_mask_bbox_inside",
                    "region_mode": "gt_mask_bbox",
                    "filter_mode": "inside",
                },
                idx=1,
            ),
            _normalize_protocol(
                {
                    "name": "M2_gt_mask_dilate8_inside",
                    "region_mode": "gt_mask_dilate_bbox",
                    "dilate_px": 8,
                    "filter_mode": "inside",
                },
                idx=2,
            ),
            _normalize_protocol(
                {
                    "name": "M3_gt_mask_dilate8_clip_iof05",
                    "region_mode": "gt_mask_dilate_bbox",
                    "dilate_px": 8,
                    "filter_mode": "clip",
                    "overlap_metric": "iof",
                    "overlap_threshold": 0.05,
                    "clip_segmentation": True,
                },
                idx=3,
            ),
            _normalize_protocol(
                {
                    "name": "M4_gt_mask_dilate12_overlap_iof20",
                    "region_mode": "gt_mask_dilate_bbox",
                    "dilate_px": 12,
                    "filter_mode": "overlap",
                    "overlap_metric": "iof",
                    "overlap_threshold": 0.20,
                },
                idx=4,
            ),
        ]

    if key in {"xray_partiall_addi"}:
        return [
            _normalize_protocol(
                {
                    "name": "S3a_gt_bbox_expand1p25_center",
                    "region_mode": "gt_bbox_expand",
                    "expand_ratio": 1.25,
                    "filter_mode": "center",
                },
                idx=1,
            ),
            _normalize_protocol(
                {
                    "name": "S3_gt_bbox_expand1p50_center",
                    "region_mode": "gt_bbox_expand",
                    "expand_ratio": 1.50,
                    "filter_mode": "center",
                },
                idx=2,
            ),
            _normalize_protocol(
                {
                    "name": "S3b_gt_bbox_expand1p75_center",
                    "region_mode": "gt_bbox_expand",
                    "expand_ratio": 1.75,
                    "filter_mode": "center",
                },
                idx=3,
            ),
        ]

    if key in {"s3_r1p5_center", "s3_r1.5_center", "s3_center"}:
        return [
            _normalize_protocol(
                {
                    "name": "S3_gt_bbox_expand1p50_center",
                    "region_mode": "gt_bbox_expand",
                    "expand_ratio": 1.50,
                    "filter_mode": "center",
                },
                idx=1,
            )
        ]

    if key in {"xray_partial_all", "xray_incomplete_all"}:
        return _get_protocol_preset("xray_partial_v1") + _get_protocol_preset("xray_partial_v2")

    raise ValueError(
        f"Unsupported protocol_preset '{protocol_preset}'. "
        "Use one of: xray_partial_v1, xray_partial_v2, xray_partiall_addi, xray_partial_all, s3_r1p5_center, none."
    )


def _apply_region_protocol(coco_gt, coco_dt, protocol, eval_mode):
    filter_mode = str(protocol.get("filter_mode", "inside")).lower().strip()
    clip_segmentation = bool(protocol.get("clip_segmentation", True))
    should_clip_seg = (eval_mode == "segm") and clip_segmentation

    filtered = []
    meta = {
        "images_with_regions": 0,
        "num_dt_total": 0,
        "num_dt_kept": 0,
        "num_dt_clipped": 0,
    }

    for img in coco_gt.dataset.get("images", []):
        img_id = img.get("id")
        h = img.get("height")
        w = img.get("width")
        if img_id is None or h is None or w is None:
            continue

        gt_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns = coco_gt.loadAnns(gt_ids) if gt_ids else []
        regions = _build_regions_for_image(gt_anns, h, w, protocol)
        if not regions:
            continue
        meta["images_with_regions"] += 1

        dt_ids = coco_dt.getAnnIds(imgIds=[img_id])
        dt_anns = coco_dt.loadAnns(dt_ids) if dt_ids else []
        meta["num_dt_total"] += len(dt_anns)
        for ann in dt_anns:
            dt_box = ann.get("bbox")
            if not _is_valid_bbox_xywh(dt_box):
                continue
            keep, best_region, _best_score = _select_region_for_prediction(dt_box, regions, protocol)
            if not keep or best_region is None:
                continue

            new_ann = dict(ann)
            if filter_mode == "clip":
                new_ann = _clip_annotation_to_region(
                    new_ann,
                    best_region,
                    image_h=h,
                    image_w=w,
                    clip_segmentation=should_clip_seg,
                )
                if new_ann is None:
                    continue
                meta["num_dt_clipped"] += 1

            filtered.append(new_ann)
            meta["num_dt_kept"] += 1

    for idx, ann in enumerate(filtered, start=1):
        ann["id"] = idx

    return filtered, meta


def evaluate_output_eval_metrics(
    suffix,
    start_id=0,
    end_id=100,
    gras=None,
    datasets=None,
    output_eval_dir="./output_eval",
    output_path=None,
    skip_fusion=False,
    eval_mode="bbox",
    protocol_preset=None,
    region_protocols=None,
    protocol_region_modes=None,
    protocol_filter_modes=None,
    protocol_expand_ratios=None,
    protocol_overlap_metrics=None,
    protocol_overlap_thresholds=None,
    protocol_dilate_pxs=None,
    protocol_dilate_ratios=None,
    protocol_include_baseline=True,
    protocol_write_detail_sheet=True,
    write_abo=True,
):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from config import XrayRoot

    def _compute_segmentation_metrics(coco_gt, coco_dt):
        total_gt = 0
        sum_iou = 0.0
        sum_f1 = 0.0
        for img in coco_gt.dataset.get("images", []):
            img_id = img.get("id")
            h = img.get("height")
            w = img.get("width")
            if img_id is None or h is None or w is None:
                continue
            gt_ids = coco_gt.getAnnIds(imgIds=[img_id])
            gt_anns = coco_gt.loadAnns(gt_ids) if gt_ids else []
            gt_rles = []
            for ann in gt_anns:
                rle = _ann_to_rle(ann, h, w)
                if rle is not None:
                    gt_rles.append(rle)
            if not gt_rles:
                continue

            dt_ids = coco_dt.getAnnIds(imgIds=[img_id])
            dt_anns = coco_dt.loadAnns(dt_ids) if dt_ids else []
            dt_rles = []
            for ann in dt_anns:
                rle = _ann_to_rle(ann, h, w)
                if rle is not None:
                    dt_rles.append(rle)

            if dt_rles:
                ious = mask_utils.iou(gt_rles, dt_rles, [0] * len(dt_rles))
                max_ious = ious.max(axis=1) if ious.size else np.zeros(len(gt_rles))
            else:
                max_ious = np.zeros(len(gt_rles))

            sum_iou += float(max_ious.sum())
            sum_f1 += float((2 * max_ious / (1 + max_ious)).sum())
            total_gt += len(gt_rles)

        mean_iou = sum_iou / total_gt if total_gt else 0.0
        mean_f1 = sum_f1 / total_gt if total_gt else 0.0
        return mean_iou, mean_f1

    def _filter_dt_by_gt_overlap(coco_gt, coco_dt, min_iou=0.0):
        filtered = []
        for img in coco_gt.dataset.get("images", []):
            img_id = img.get("id")
            if img_id is None:
                continue
            gt_ids = coco_gt.getAnnIds(imgIds=[img_id])
            gt_anns = coco_gt.loadAnns(gt_ids) if gt_ids else []
            gt_boxes = [ann.get("bbox") for ann in gt_anns if ann.get("bbox") is not None]
            if not gt_boxes:
                continue

            dt_ids = coco_dt.getAnnIds(imgIds=[img_id])
            dt_anns = coco_dt.loadAnns(dt_ids) if dt_ids else []
            for ann in dt_anns:
                dt_box = ann.get("bbox")
                if dt_box is None:
                    continue
                max_iou = 0.0
                for gt_box in gt_boxes:
                    max_iou = max(max_iou, _bbox_iou(gt_box, dt_box))
                if max_iou > min_iou:
                    filtered.append(ann)
        return filtered

    def _compute_average_best_overlap(coco_gt, coco_dt, mode="bbox"):
        mode = str(mode).lower().strip()
        total_gt = 0
        sum_best = 0.0
        for img in coco_gt.dataset.get("images", []):
            img_id = img.get("id")
            if img_id is None:
                continue
            gt_ids = coco_gt.getAnnIds(imgIds=[img_id])
            gt_anns = coco_gt.loadAnns(gt_ids) if gt_ids else []
            if mode == "segm":
                h = img.get("height")
                w = img.get("width")
                if h is None or w is None:
                    continue
                gt_rles = []
                for ann in gt_anns:
                    rle = _ann_to_rle(ann, h, w)
                    if rle is not None:
                        gt_rles.append(rle)
                if not gt_rles:
                    continue
                dt_ids = coco_dt.getAnnIds(imgIds=[img_id])
                dt_anns = coco_dt.loadAnns(dt_ids) if dt_ids else []
                dt_rles = []
                for ann in dt_anns:
                    rle = _ann_to_rle(ann, h, w)
                    if rle is not None:
                        dt_rles.append(rle)
                if dt_rles:
                    ious = mask_utils.iou(gt_rles, dt_rles, [0] * len(dt_rles))
                    max_ious = ious.max(axis=1) if ious.size else np.zeros(len(gt_rles))
                else:
                    max_ious = np.zeros(len(gt_rles))
                sum_best += float(max_ious.sum())
                total_gt += len(gt_rles)
            else:
                gt_boxes = [ann.get("bbox") for ann in gt_anns if ann.get("bbox") is not None]
                if not gt_boxes:
                    continue
                dt_ids = coco_dt.getAnnIds(imgIds=[img_id])
                dt_anns = coco_dt.loadAnns(dt_ids) if dt_ids else []
                dt_boxes = [ann.get("bbox") for ann in dt_anns if ann.get("bbox") is not None]
                for gt_box in gt_boxes:
                    best_iou = 0.0
                    for dt_box in dt_boxes:
                        best_iou = max(best_iou, _bbox_iou(gt_box, dt_box))
                    sum_best += float(best_iou)
                total_gt += len(gt_boxes)
        return (sum_best / total_gt) if total_gt else 0.0

    eval_mode = str(eval_mode).lower().strip()
    if eval_mode not in {"bbox", "segm"}:
        raise ValueError("eval_mode must be 'bbox' or 'segm'.")

    if gras is None:
        gras = [f"{i/10:.1f}" for i in range(1, 11)]
    if suffix == "_sam2" or suffix == "_sam3":
        gras = ["0.1"]
    if datasets is None:
        datasets = list(XrayRoot.keys())

    resolved_protocols = []
    if region_protocols is not None:
        resolved_protocols = [_normalize_protocol(p, idx=i + 1) for i, p in enumerate(region_protocols)]
    elif (
        protocol_region_modes is not None
        or protocol_filter_modes is not None
        or protocol_expand_ratios is not None
        or protocol_overlap_metrics is not None
        or protocol_overlap_thresholds is not None
        or protocol_dilate_pxs is not None
        or protocol_dilate_ratios is not None
    ):
        resolved_protocols = build_region_eval_protocols(
            region_modes=protocol_region_modes,
            filter_modes=protocol_filter_modes,
            expand_ratios=protocol_expand_ratios,
            overlap_metrics=protocol_overlap_metrics,
            overlap_thresholds=protocol_overlap_thresholds,
            dilate_pxs=protocol_dilate_pxs,
            dilate_ratios=protocol_dilate_ratios,
            clip_segmentation=True,
        )
    elif protocol_preset:
        resolved_protocols = _get_protocol_preset(protocol_preset)

    use_region_protocols = len(resolved_protocols) > 0
    protocol_include_baseline = bool(protocol_include_baseline)
    protocol_write_detail_sheet = bool(protocol_write_detail_sheet)

    metric_labels = [
        "AP IoU=0.50:0.95",
        "AP IoU=0.50",
        "AP IoU=0.75",
        "AP small",
        "AP medium",
        "AP large",
        "AR maxDets=1",
        "AR maxDets=10",
        "AR maxDets=100",
        "AR small",
        "AR medium",
        "AR large",
    ]
    if eval_mode == "segm":
        metric_labels.extend(["Mean IoU", "Mean F1"])
    if (eval_mode == "bbox") and (not use_region_protocols):
        metric_labels.extend(["pAP IoU=0.50:0.95", "pAP IoU=0.50", "pAP IoU=0.75"])

    if output_path is None:
        safe_suffix = suffix.lstrip("_") if suffix else "metrics"
        preset_tag = str(protocol_preset or "").strip()
        safe_preset = ""
        if preset_tag:
            safe_preset = "_" + "".join(
                c if (c.isalnum() or c in {"-", "_"}) else "_" for c in preset_tag
            )
        if end_id in [100,1000]:
            output_path = os.path.join(output_eval_dir, f"{eval_mode}_metrics_{safe_suffix}_{end_id}{safe_preset}.xlsx")
        else:
            output_path = os.path.join(output_eval_dir, f"{eval_mode}_metrics_{safe_suffix}_full{safe_preset}.xlsx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer_mode = "a" if os.path.exists(output_path) else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": writer_mode}
    if writer_mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"
    # print("dataset {} end_id {}".format(dataset, end_id))
    try:
        writer = pd.ExcelWriter(output_path, **writer_kwargs)
    except Exception:
        writer = pd.ExcelWriter(output_path, engine="openpyxl", mode="w")
    with writer as writer:
        for dataset in datasets:
            dataset_root = XrayRoot.get(dataset)
            if not dataset_root:
                continue
            val_json = _resolve_val_json(dataset_root)
            if not val_json:
                continue

            sliced_json = os.path.join(
                output_eval_dir, f"temp_{dataset}_val_{start_id}_{end_id}.json"
            )
            sliced_json = _slice_coco_json(val_json, start_id, end_id, sliced_json)

            fused_cols = ["Fuse", "F_our"]
            base_cols = gras if skip_fusion else gras + fused_cols
            cols = []
            if use_region_protocols:
                if protocol_include_baseline:
                    cols.extend(base_cols)
                for base_col in base_cols:
                    for proto in resolved_protocols:
                        cols.append(f"{base_col}::{proto['name']}")
            else:
                cols = list(base_cols)
            df = pd.DataFrame(index=metric_labels, columns=cols, dtype=float)
            abo_values = {col: np.nan for col in cols}
            coco_gt = COCO(sliced_json)

            protocol_rows = []
            protocol_row_map = {}
            if use_region_protocols:
                for base_col in base_cols:
                    for proto in resolved_protocols:
                        proto_col = f"{base_col}::{proto['name']}"
                        row = {
                            "base_column": base_col,
                            "protocol_column": proto_col,
                            "protocol_name": proto["name"],
                            "description": _protocol_to_description(proto),
                            "region_mode": proto["region_mode"],
                            "filter_mode": proto["filter_mode"],
                            "expand_ratio": proto["expand_ratio"],
                            "dilate_px": proto["dilate_px"],
                            "dilate_ratio": proto["dilate_ratio"],
                            "overlap_metric": proto["overlap_metric"],
                            "overlap_threshold": proto["overlap_threshold"],
                            "clip_segmentation": int(bool(proto["clip_segmentation"])),
                            "images_with_regions": np.nan,
                            "num_dt_total": np.nan,
                            "num_dt_kept": np.nan,
                            "num_dt_clipped": np.nan,
                            "keep_ratio": np.nan,
                            "status": "pending_or_missing_prediction",
                        }
                        protocol_rows.append(row)
                        protocol_row_map[proto_col] = row

            def _fill_main_metrics_for_col(col_name, stats):
                for idx, label in enumerate(metric_labels):
                    if idx < len(stats):
                        df.loc[label, col_name] = float(stats[idx])

            def _run_coco_eval_to_col(coco_dt_obj, col_name):
                coco_eval = COCOeval(coco_gt, coco_dt_obj, eval_mode)
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                stats = coco_eval.stats
                _fill_main_metrics_for_col(col_name, stats)
                abo_values[col_name] = float(_compute_average_best_overlap(coco_gt, coco_dt_obj, mode=eval_mode))
                if eval_mode == "segm":
                    mean_iou, mean_f1 = _compute_segmentation_metrics(coco_gt, coco_dt_obj)
                    df.loc["Mean IoU", col_name] = float(mean_iou)
                    df.loc["Mean F1", col_name] = float(mean_f1)
                return stats

            def _fill_zero_for_col(col_name):
                for label in metric_labels:
                    df.loc[label, col_name] = 0.0
                abo_values[col_name] = 0.0

            def _evaluate_single_prediction(base_col, pred_path):
                if not pred_path or not os.path.exists(pred_path):
                    return
                coco_dt = coco_gt.loadRes(pred_path)

                if (not use_region_protocols) or protocol_include_baseline:
                    _run_coco_eval_to_col(coco_dt, base_col)

                if not use_region_protocols:
                    if eval_mode == "bbox":
                        filtered = _filter_dt_by_gt_overlap(coco_gt, coco_dt, min_iou=0.0)
                        if filtered:
                            coco_dt_partial = coco_gt.loadRes(filtered)
                            coco_eval_partial = COCOeval(coco_gt, coco_dt_partial, eval_mode)
                            coco_eval_partial.evaluate()
                            coco_eval_partial.accumulate()
                            coco_eval_partial.summarize()
                            pstats = coco_eval_partial.stats
                            df.loc["pAP IoU=0.50:0.95", base_col] = float(pstats[0])
                            df.loc["pAP IoU=0.50", base_col] = float(pstats[1])
                            df.loc["pAP IoU=0.75", base_col] = float(pstats[2])
                    return

                for proto in resolved_protocols:
                    proto_col = f"{base_col}::{proto['name']}"
                    filtered_dt, meta = _apply_region_protocol(coco_gt, coco_dt, proto, eval_mode)
                    if filtered_dt:
                        coco_dt_proto = coco_gt.loadRes(filtered_dt)
                        _run_coco_eval_to_col(coco_dt_proto, proto_col)
                    else:
                        _fill_zero_for_col(proto_col)
                    keep_ratio = (
                        float(meta["num_dt_kept"]) / float(meta["num_dt_total"])
                        if meta["num_dt_total"] > 0
                        else 0.0
                    )
                    row = protocol_row_map.get(proto_col)
                    if row is not None:
                        row["images_with_regions"] = meta["images_with_regions"]
                        row["num_dt_total"] = meta["num_dt_total"]
                        row["num_dt_kept"] = meta["num_dt_kept"]
                        row["num_dt_clipped"] = meta["num_dt_clipped"]
                        row["keep_ratio"] = keep_ratio
                        row["status"] = "evaluated"

            for gra in gras:
                if suffix == "_sam2":
                    output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}{suffix}"
                elif suffix == "_sam3":
                    output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}{suffix}_1.0"
                else:
                    output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}_unsamv2_{gra}{suffix}"
                pred_path = os.path.join(output_eval_dir, output_subdir, "coco_instances_results.json")
                _evaluate_single_prediction(gra, pred_path)

            if not skip_fusion:
                tag = "_".join(gras)
                try:
                    from gra_fusion import fuse_results
                except Exception:
                    fuse_results = None

                fusion_targets = {
                    "Fuse": "merge",
                    "F_our": "ori",
                }
                for col, mode in fusion_targets.items():
                    rel = f"gra_fusion_{dataset}_img{start_id}_{end_id}_unsamv2_{tag}{suffix}_{mode}/coco_instances_results.json"
                    pred_path = os.path.join(output_eval_dir, rel)
                    if not os.path.exists(pred_path) and fuse_results is not None:
                        pred_path = fuse_results(
                            gras=gras,
                            dataset=dataset,
                            start_id=start_id,
                            end_id=end_id,
                            method="unsamv2",
                            comment=suffix,
                            input_root=output_eval_dir,
                            output_root=output_eval_dir,
                            fusion_mode=mode,
                            no_save=False,
                        ) or pred_path
                    _evaluate_single_prediction(col, pred_path)

            df.to_excel(writer, sheet_name=dataset[:31])
            if bool(write_abo):
                abo_df = pd.DataFrame(index=["ABO"], columns=cols, dtype=float)
                for col in cols:
                    abo_df.loc["ABO", col] = abo_values.get(col, np.nan)
                abo_sheet = f"{dataset[:27]}_abo"
                abo_df.to_excel(writer, sheet_name=abo_sheet)
            if use_region_protocols and protocol_write_detail_sheet:
                proto_df = pd.DataFrame(protocol_rows)
                if proto_df.empty:
                    proto_df = pd.DataFrame(
                        [
                            {
                                "base_column": "",
                                "protocol_column": "",
                                "protocol_name": "",
                                "description": "No protocol result generated.",
                            }
                        ]
                    )
                preset_desc = protocol_preset if protocol_preset else "custom_or_direct"
                head_df = pd.DataFrame(
                    [
                        {
                            "dataset": dataset,
                            "eval_mode": eval_mode,
                            "suffix": suffix,
                            "start_id": start_id,
                            "end_id": end_id,
                            "protocol_preset": preset_desc,
                            "protocol_count": len(resolved_protocols),
                            "protocol_include_baseline": int(protocol_include_baseline),
                        }
                    ]
                )
                proto_sheet_name = f"{dataset[:20]}_proto"
                head_df.to_excel(writer, sheet_name=proto_sheet_name, index=False, startrow=0)
                proto_df.to_excel(writer, sheet_name=proto_sheet_name, index=False, startrow=3)

    print(f"Saved metrics to {output_path}")


def export_interactive_eval_metrics(
    dataset,
    stage,
    gras,
    comment_suffix="",
    output_dir="./evaluation/interactive/outputs/xray_eval",
    output_path=None,
    sam_type="unsamv2",
    ckpt_tag="",
):
    if isinstance(gras, str):
        gras = [gra.strip() for gra in gras.split(",") if gra.strip()]
    else:
        gras = [str(gra).strip() for gra in gras if str(gra).strip()]

    metric_labels = [
        "NoC80",
        "NoC90",
        "IoU@1",
        "over_max_80",
        "over_max_90",
        "num_images",
        "num_objects",
    ]
    df = pd.DataFrame(index=metric_labels, columns=gras, dtype=float)

    for gra in gras:
        file_tag = f"{dataset}{comment_suffix}_{stage}_gra{gra}"
        if ckpt_tag:
            file_tag = f"{file_tag}_{ckpt_tag}"
        json_path = os.path.join(output_dir, f"{file_tag}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        for label in metric_labels:
            value = metrics.get(label)
            if value is None:
                continue
            try:
                df.loc[label, gra] = float(value)
            except (TypeError, ValueError):
                continue

    if output_path is None:
        parts = ["interactive_metrics"]
        if comment_suffix:
            parts.append(comment_suffix.lstrip("_"))
        if stage:
            parts.append(stage)
        if sam_type:
            parts.append(sam_type)
        if ckpt_tag:
            parts.append(ckpt_tag)
        output_path = os.path.join(output_dir, "_".join(parts) + ".xlsx")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer_mode = "a" if os.path.exists(output_path) else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": writer_mode}
    if writer_mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"
    try:
        writer = pd.ExcelWriter(output_path, **writer_kwargs)
    except Exception:
        writer = pd.ExcelWriter(output_path, engine="openpyxl", mode="w")

    with writer as writer:
        df.to_excel(writer, sheet_name=dataset[:31])

    print(f"Saved interactive metrics to {output_path}")
    return output_path


def _resolve_single_region_protocol(region_protocol=None, region_protocol_preset=None):
    if region_protocol is not None:
        return _normalize_protocol(region_protocol, idx=1)
    if region_protocol_preset:
        protos = _get_protocol_preset(region_protocol_preset)
        if not protos:
            return None
        if len(protos) != 1:
            raise ValueError(
                f"region_protocol_preset '{region_protocol_preset}' resolves to {len(protos)} protocols; expected exactly 1."
            )
        return protos[0]
    return None


def evaluate_output_eval_bbox_image_ap(
    dataset,
    suffix,
    gra,
    model_name="unsamv2",
    start_id=0,
    end_id=None,
    output_eval_dir="./output_eval",
    output_path=None,
    region_protocol=None,
    region_protocol_preset=None,
):
    """Compute per-image AP for a specific dataset/suffix/gra."""
    from pycocotools.coco import COCO
    from config import XrayRoot

    dataset_root = XrayRoot.get(dataset)
    if not dataset_root:
        raise ValueError(f"Unknown dataset: {dataset}")
    val_json = _resolve_val_json(dataset_root)
    if not val_json:
        raise FileNotFoundError(f"No val JSON found for dataset: {dataset}")

    sliced_json = os.path.join(
        output_eval_dir, f"temp_{dataset}_val_{start_id}_{end_id}.json"
    )
    sliced_json = _slice_coco_json(val_json, start_id, end_id, sliced_json)
    coco_gt = COCO(sliced_json)

    if suffix == "_sam2":
        output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}{suffix}"
    elif suffix == "_sam3":
        output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}{suffix}_1.0"
    else:
        output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}_{model_name}_{gra}{suffix}"
    pred_path = os.path.join(output_eval_dir, output_subdir, "coco_instances_results.json")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    coco_dt = coco_gt.loadRes(pred_path)
    resolved_protocol = _resolve_single_region_protocol(
        region_protocol=region_protocol,
        region_protocol_preset=region_protocol_preset,
    )

    if resolved_protocol is not None:
        filtered_dt, _meta = _apply_region_protocol(coco_gt, coco_dt, resolved_protocol, eval_mode="bbox")
        dt_anns = filtered_dt
    else:
        dt_anns = coco_dt.dataset.get("annotations", [])

    dt_by_image = defaultdict(list)
    for ann in dt_anns:
        dt_by_image[ann.get("image_id")].append(ann)

    images = coco_gt.dataset.get("images", [])
    iou_thresholds = np.round(np.arange(0.5, 0.96, 0.05), 2)
    rows = []
    for img in images:
        img_id = img.get("id")
        if img_id is None:
            continue
        gt_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns = coco_gt.loadAnns(gt_ids) if gt_ids else []
        gt_boxes = [ann.get("bbox") for ann in gt_anns if ann.get("bbox") is not None]

        dt_img_anns = dt_by_image.get(img_id, [])
        valid_dt = [ann for ann in dt_img_anns if ann.get("bbox") is not None]
        dt_boxes = [ann.get("bbox") for ann in valid_dt]
        dt_scores = [float(ann.get("score", 1.0)) for ann in valid_dt]

        ap, ap50, ap75 = _compute_image_ap(
            gt_boxes, dt_boxes, dt_scores, iou_thresholds
        )
        rows.append(
            {
                "image_id": img_id,
                "file_name": img.get("file_name", ""),
                "AP": ap,
                "AP50": ap50,
                "AP75": ap75,
            }
        )

    df = pd.DataFrame(rows)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith(".csv"):
            df.to_csv(output_path, index=False)
        else:
            df.to_excel(output_path, index=False)
    return df


def merge_image_ap_by_suffix(
    dataset,
    gra,
    suffixes,
    model_names=None,
    start_id=0,
    end_id=None,
    output_eval_dir="./output_eval",
    output_eval_dirs=None,
    output_path=None,
    sort_suffix=None,
    region_protocol=None,
    region_protocol_preset=None,
):
    """Merge per-image AP across multiple suffixes and save for inspection."""
    if model_names is None:
        model_names = ["unsamv2"] * len(suffixes)
    if len(model_names) != len(suffixes):
        raise ValueError("model_names must match suffixes length.")
    if isinstance(gra, (list, tuple)):
        gra_list = list(gra)
        if len(gra_list) != len(suffixes):
            raise ValueError("gra list must match suffixes length.")
    else:
        gra_list = [gra] * len(suffixes)

    if output_eval_dirs is not None:
        if len(output_eval_dirs) != len(suffixes):
            raise ValueError("output_eval_dirs must match suffixes length.")
        eval_dirs = list(output_eval_dirs)
    else:
        eval_dirs = [output_eval_dir] * len(suffixes)

    merged = None
    col_map = {}
    for suffix, model_name, gra_val, eval_dir in zip(suffixes, model_names, gra_list, eval_dirs):
        df = evaluate_output_eval_bbox_image_ap(
            dataset=dataset,
            suffix=suffix,
            gra=gra_val,
            model_name=model_name,
            start_id=start_id,
            end_id=end_id,
            output_eval_dir=eval_dir,
            output_path=None,
            region_protocol=region_protocol,
            region_protocol_preset=region_protocol_preset,
        )
        col_name = f"{model_name}{suffix}"
        col_map[suffix] = col_name
        df = df[["image_id", "file_name", "AP"]].rename(columns={"AP": col_name})
        merged = df if merged is None else merged.merge(df, on=["image_id", "file_name"], how="outer")

    if merged is None:
        raise ValueError("No data merged.")

    ordered_cols = ["image_id", "file_name"]
    for suffix in suffixes:
        ordered_cols.append(col_map[suffix])
    merged = merged.reindex(columns=ordered_cols)

    if sort_suffix:
        sort_col = col_map.get(sort_suffix, sort_suffix)
        if sort_col in merged.columns:
            merged = merged.sort_values(by=sort_col, ascending=False)

    if output_path is None:
        safe_tag = "_".join([s.lstrip("_") for s in suffixes])
        output_path = os.path.join(
            output_eval_dir, f"per_image_ap_{dataset}_{gra}_{safe_tag}.xlsx"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".csv"):
        merged.to_csv(output_path, index=False)
    else:
        merged.to_excel(output_path, index=False)
    return merged


def select_image_ids_by_ap(
    dataset,
    suffix,
    gra,
    model_name="unsamv2",
    start_id=0,
    end_id=None,
    output_eval_dir="./output_eval",
    output_eval_dir_override=None,
    ap_value=1.0,
    tol=1e-6,
    region_protocol=None,
    region_protocol_preset=None,
):
    """Return image_ids whose AP is close to ap_value for the given model/suffix/gra."""
    eval_dir = output_eval_dir_override or output_eval_dir
    df = evaluate_output_eval_bbox_image_ap(
        dataset=dataset,
        suffix=suffix,
        gra=gra,
        model_name=model_name,
        start_id=start_id,
        end_id=end_id,
        output_eval_dir=eval_dir,
        output_path=None,
        region_protocol=region_protocol,
        region_protocol_preset=region_protocol_preset,
    )
    if "AP" not in df.columns:
        return []
    mask = df["AP"].notna() & (df["AP"] >= (ap_value - tol)) & (df["AP"] <= (ap_value + tol))
    return df.loc[mask, "image_id"].astype(int).tolist()


def select_image_ids_by_ap_gap(
    merged_df,
    first_method_col,
    other_method_cols,
    ap_value=1.0,
    tol=1e-6,
    top_n=100,
):
    """Select image_ids where first method AP≈ap_value, ranked by (first AP - max(other AP))."""
    if merged_df is None or merged_df.empty:
        return []
    if first_method_col not in merged_df.columns:
        raise ValueError(f"Missing first_method_col in merged_df: {first_method_col}")

    cols = [c for c in other_method_cols if c in merged_df.columns]
    if not cols:
        raise ValueError("No valid other_method_cols found in merged_df.")

    df = merged_df.copy()
    first_ap = pd.to_numeric(df[first_method_col], errors="coerce")
    others_ap = df[cols].apply(pd.to_numeric, errors="coerce")
    other_max = others_ap.max(axis=1, skipna=True).fillna(-1.0)
    gap = first_ap - other_max

    mask = first_ap.notna() & (first_ap >= (ap_value - tol)) & (first_ap <= (ap_value + tol))
    picked = df.loc[mask, ["image_id"]].copy()
    picked["_gap"] = gap.loc[mask]
    picked = picked.sort_values(by="_gap", ascending=False)

    if top_n is not None:
        top_n = int(top_n)
        if top_n > 0:
            picked = picked.head(top_n)
        else:
            picked = picked.iloc[0:0]

    return picked["image_id"].astype(int).tolist()


def _resolve_image_path_local(file_name, image_root, dataset_root):
    if os.path.isabs(file_name):
        return file_name
    cand = os.path.join(image_root, file_name)
    if os.path.exists(cand):
        return cand
    return os.path.join(dataset_root, file_name)


def _decode_rle_mask(segmentation, height, width):
    if segmentation is None:
        return None
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        mask = mask_utils.decode(mask_utils.merge(rles))
        return mask.astype(bool)
    if isinstance(segmentation, dict):
        seg = dict(segmentation)
        if "size" not in seg or not seg["size"]:
            seg["size"] = [int(height), int(width)]
        if isinstance(seg.get("counts"), list):
            rle = mask_utils.frPyObjects(seg, height, width)
            mask = mask_utils.decode(rle)
            return mask.astype(bool)
        mask = mask_utils.decode(seg)
        return mask.astype(bool)
    return None


def _overlay_mask(img_bgr, mask, color, alpha=0.4):
    if mask is None:
        return img_bgr
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.shape[:2] != img_bgr.shape[:2]:
        return img_bgr
    overlay = img_bgr.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + color_arr * alpha
    return overlay.astype(np.uint8)


def _mask_to_contours(mask):
    if mask is None:
        return []
    if mask.dtype != np.uint8:
        mask_u8 = mask.astype(np.uint8)
    else:
        mask_u8 = mask
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[:, :, 0]
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours if contours is not None else []


def _draw_mask_boundary(img_bgr, mask, color, thickness=3):
    if mask is None:
        return img_bgr
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.shape[:2] != img_bgr.shape[:2]:
        return img_bgr
    contours = _mask_to_contours(mask)
    if not contours:
        return img_bgr
    out = img_bgr.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _contrast_color_bgr(idx):
    h = int((int(idx) * 67) % 180)
    hsv = np.uint8([[[h, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _decode_base64_image_to_bgr(image_base64):
    if not image_base64:
        return None
    payload = str(image_base64).strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    image_bytes = base64.b64decode(payload)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    if image_array.size == 0:
        return None
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def _to_grayscale_bgr(img_bgr):
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _render_single_mask(img_bgr, mask, color, mode="edge", alpha=0.45, thickness=3):
    mode_key = str(mode or "edge").lower().strip()
    if mode_key in {"fill", "inside", "mask"}:
        return _overlay_mask(img_bgr, mask, color, alpha=float(alpha))
    return _draw_mask_boundary(img_bgr, mask, color, thickness=int(thickness))


def _lighten_color(color, factor=0.45):
    return tuple(int(c + (255 - c) * factor) for c in color)


def _score_vis_palette_bgr():
    return [
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
        (0, 255, 0),     # green
        (255, 128, 0),   # blue/orange
        (255, 255, 0),   # cyan
        (0, 128, 255),   # orange
        (128, 0, 255),   # purple
        (0, 255, 128),   # spring green
        (255, 0, 128),   # pink
        (128, 255, 0),   # lime
    ]


def _method_topk_palette_bgr():
    """Fixed bright colors for method-level top-k visualization."""
    return [
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
        (255, 255, 0),   # cyan
        (0, 255, 0),     # green
        (0, 128, 255),   # orange
        (255, 0, 128),   # pink
    ]


def _draw_text_panel_top_left(
    img_bgr,
    lines,
    x0=8,
    y0=8,
    font_scale=0.55,
    thickness=1,
    pad=6,
    row_gap=4,
    bg_alpha=0.45,
    bg_color=(0, 0, 0),
    text_color=(255, 255, 255),
):
    if isinstance(lines, str):
        lines = [lines]
    lines = [str(x) for x in lines if str(x)]
    if not lines:
        return img_bgr
    out = img_bgr.copy()
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for line in lines]
    max_w = max(w for w, _h in sizes)
    max_h = max(h for _w, h in sizes)
    line_h = max_h + row_gap
    panel_w = max_w + 2 * pad
    panel_h = line_h * len(lines) + 2 * pad

    h_img, w_img = out.shape[:2]
    x1 = min(w_img - 1, x0 + panel_w)
    y1 = min(h_img - 1, y0 + panel_h)
    if x1 <= x0 or y1 <= y0:
        return out

    # overlay = out.copy()
    # cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, thickness=-1)
    # roi_base = out[y0:y1, x0:x1]
    # roi_overlay = overlay[y0:y1, x0:x1]
    # out[y0:y1, x0:x1] = cv2.addWeighted(
    #     roi_overlay, float(bg_alpha), roi_base, float(1.0 - bg_alpha), 0.0
    # )

    # y = y0 + pad + max_h
    # for line in lines:
    #     cv2.putText(
    #         out,
    #         line,
    #         (x0 + pad, y),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         font_scale,
    #         text_color,
    #         thickness,
    #         cv2.LINE_AA,
    #     )
    #     y += line_h
    return out


def _compose_topk_mask_panel(
    base_bgr,
    det_records,
    topk=5,
    cols=3,
    alpha=0.45,
    show_score=False,
    score_bg_alpha=0.35,
    mask_color=None,
    single_mask_vis_mode="edge",
    single_mask_alpha=0.45,
    single_mask_thickness=4,
):
    """Compose top-k single-mask visualizations into one large panel."""
    records = sorted(det_records, key=lambda x: -float(x.get("score", 0.0)))
    if not records:
        return _draw_text_panel_top_left(
            base_bgr.copy(), ["no pred"], bg_alpha=score_bg_alpha
        )
    topk = max(1, int(topk))
    cols = max(1, int(cols))
    alpha = float(alpha)
    chosen = records[: min(topk, len(records))]
    tile_h, tile_w = base_bgr.shape[:2]
    panel_cols = min(cols, len(chosen))
    panel_rows = (len(chosen) + panel_cols - 1) // panel_cols
    gap = 8
    panel_h = panel_rows * tile_h + (panel_rows - 1) * gap
    panel_w = panel_cols * tile_w + (panel_cols - 1) * gap
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    for idx, rec in enumerate(chosen):
        r = idx // panel_cols
        c = idx % panel_cols
        y0 = r * (tile_h + gap)
        x0 = c * (tile_w + gap)
        color = mask_color if mask_color is not None else rec.get("color", (0, 255, 255))
        tile = _render_single_mask(
            base_bgr.copy(),
            rec["mask"],
            color,
            mode=single_mask_vis_mode,
            alpha=single_mask_alpha,
            thickness=single_mask_thickness,
        )
        if show_score:
            tile = _draw_text_panel_top_left(
                tile, [f"{float(rec.get('score', 0.0)):.2f}"], bg_alpha=score_bg_alpha
            )
        panel[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return panel


def _normalize_topk_values(topk_values, default_values=(10, 5, 3)):
    if topk_values is None:
        values = list(default_values)
    elif isinstance(topk_values, str):
        values = [x.strip() for x in topk_values.split(",") if x.strip()]
    else:
        values = list(topk_values)

    normalized = []
    seen = set()
    for value in values:
        try:
            topk = int(value)
        except Exception:
            continue
        if topk <= 0 or topk in seen:
            continue
        seen.add(topk)
        normalized.append(topk)
    return normalized


def _compose_ranked_mask_overlay(
    base_bgr,
    det_records,
    topk,
    alpha=0.45,
    palette=None,
    single_mask_vis_mode="fill",
    single_mask_thickness=2,
    show_rank_legend=True,
):
    records = sorted(det_records, key=lambda x: -float(x.get("score", 0.0)))
    chosen = records[: min(max(1, int(topk)), len(records))]
    out = base_bgr.copy()
    if not chosen:
        return _draw_text_panel_top_left(out, [f"Top{int(topk)}: no pred"], bg_alpha=0.35)

    if palette is None:
        palette = _score_vis_palette_bgr()

    legend_lines = [f"Top{int(topk)}"]
    for idx, rec in enumerate(chosen, start=1):
        color = palette[(idx - 1) % len(palette)]
        out = _render_single_mask(
            out,
            rec["mask"],
            color,
            mode=single_mask_vis_mode,
            alpha=alpha,
            thickness=single_mask_thickness,
        )
        if show_rank_legend:
            legend_lines.append(f"#{idx} {float(rec.get('score', 0.0)):.2f}")

    if show_rank_legend:
        out = _draw_text_panel_top_left(out, legend_lines, bg_alpha=0.35)
    return out


def visualization_select(
    dataset,
    image_ids,
    suffixes,
    gra,
    model_names=None,
    method_aliases=None,
    start_id=0,
    end_id=None,
    output_eval_dir="./output_eval",
    output_eval_dirs=None,
    output_dir=None,
    score_thresh=0.0,
    max_det=100,
    draw_gt=True,
    gt_alias="gt",
    skip_existing=True,
    region_protocol=None,
    region_protocol_preset=None,
    group_by_gt_region=False,
    score_sorted_vis=False,
    score_sorted_topk=5,
    score_sorted_lowk=10,
    score_sorted_alpha=0.45,
    compact_topk_panel=False,
    compact_topk_k=5,
    compact_topk_cols=3,
    compact_topk_alpha=0.45,
    compact_topk_show_score=False,
    single_mask_vis_mode="edge",
    grayscale_image=False,
):
    """Visualize selected image ids with GT and predictions from multiple suffixes."""
    from config import XrayRoot, XrayImgRoot

    if isinstance(image_ids, str):
        image_ids = [int(x) for x in image_ids.split(",") if x.strip()]
    if not image_ids:
        raise ValueError("image_ids is empty.")

    if model_names is None:
        model_names = ["unsamv2"] * len(suffixes)
    if len(model_names) != len(suffixes):
        raise ValueError("model_names must match suffixes length.")
    if method_aliases is None:
        method_aliases = [s.lstrip("_") or "method" for s in suffixes]
    if len(method_aliases) != len(suffixes):
        raise ValueError("method_aliases must match suffixes length.")

    if isinstance(gra, (list, tuple)):
        gra_list = list(gra)
        if len(gra_list) != len(suffixes):
            raise ValueError("gra list must match suffixes length.")
    else:
        gra_list = [gra] * len(suffixes)

    if output_eval_dirs is not None:
        if len(output_eval_dirs) != len(suffixes):
            raise ValueError("output_eval_dirs must match suffixes length.")
        eval_dirs = list(output_eval_dirs)
    else:
        eval_dirs = [output_eval_dir] * len(suffixes)

    dataset_root = XrayRoot.get(dataset)
    if not dataset_root:
        raise ValueError(f"Unknown dataset: {dataset}")
    val_json = _resolve_val_json(dataset_root)
    if not val_json:
        raise FileNotFoundError(f"No val JSON found for dataset: {dataset}")

    image_root = XrayImgRoot.get(dataset, os.path.join(dataset_root, "xray"))

    with open(val_json, "r", encoding="utf-8") as f:
        gt_coco = json.load(f)
    images = {img["id"]: img for img in gt_coco.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in gt_coco.get("annotations", []):
        anns_by_image[ann.get("image_id")].append(ann)

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output_visualization")
    os.makedirs(output_dir, exist_ok=True)
    resolved_region_protocol = _resolve_single_region_protocol(
        region_protocol=region_protocol,
        region_protocol_preset=region_protocol_preset,
    )

    pred_maps = []
    for suffix, model_name, gra_val, eval_dir in zip(
        suffixes, model_names, gra_list, eval_dirs
    ):
        if suffix == "_sam2":
            output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}{suffix}"
        elif suffix == "_sam3":
            output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}{suffix}_1.0"
        else:
            output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}_{model_name}_{gra_val}{suffix}"
        pred_path = os.path.join(eval_dir, output_subdir, "coco_instances_results.json")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        pred_by_image = defaultdict(list)
        for det in preds:
            pred_by_image[det.get("image_id")].append(det)
        pred_maps.append(pred_by_image)

    bbox_thickness = 3
    mask_thickness = 4
    thin_mask_thickness = 2
    score_sorted_vis = bool(score_sorted_vis)
    score_sorted_topk = max(1, int(score_sorted_topk))
    score_sorted_lowk = max(1, int(score_sorted_lowk))
    score_sorted_alpha = float(score_sorted_alpha)
    compact_topk_panel = bool(compact_topk_panel)
    compact_topk_k = max(1, int(compact_topk_k))
    compact_topk_cols = max(1, int(compact_topk_cols))
    compact_topk_alpha = float(compact_topk_alpha)
    compact_topk_show_score = bool(compact_topk_show_score)
    single_mask_vis_mode = str(single_mask_vis_mode or "edge").lower().strip()
    method_topk_palette = _method_topk_palette_bgr()

    for img_id in image_ids:
        info = images.get(img_id)
        if not info:
            continue
        file_name = info.get("file_name", "")
        if not file_name:
            continue
        img_path = _resolve_image_path_local(file_name, image_root, dataset_root)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        if grayscale_image:
            img = _to_grayscale_bgr(img)

        height, width = img.shape[:2]
        base = img.copy()

        if draw_gt:
            out_name = f"{dataset}_{img_id}_{gt_alias}_gt.png"
            out_path = os.path.join(output_dir, out_name)
            if not (skip_existing and os.path.exists(out_path)):
                gt_vis = base.copy()
                for ann in anns_by_image.get(img_id, []):
                    bbox = ann.get("bbox")
                    if bbox and len(bbox) == 4:
                        x, y, w, h = map(int, bbox)
                        cv2.rectangle(gt_vis, (x, y), (x + w, y + h), (0, 0, 255), bbox_thickness)
                cv2.imwrite(out_path, gt_vis)

        for method_idx, (pred_by_image, alias) in enumerate(zip(pred_maps, method_aliases)):
            method_topk_color = method_topk_palette[method_idx % len(method_topk_palette)]
            vis_mask_s3 = base.copy()
            dets = pred_by_image.get(img_id, [])
            dets = sorted(dets, key=lambda d: -float(d.get("score", 0.0)))
            det_records = []
            crop_idx = 1

            for det_idx, det in enumerate(dets[:max_det]):
                score = float(det.get("score", 0.0))
                if score < score_thresh:
                    continue
                bbox = det.get("bbox")
                if not (bbox and len(bbox) == 4):
                    continue
                mask = _decode_rle_mask(det.get("segmentation"), height, width)
                if mask is None:
                    continue

                contrast = _contrast_color_bgr(det_idx)
                vis_mask_s3 = _render_single_mask(
                    vis_mask_s3,
                    mask,
                    contrast,
                    mode=single_mask_vis_mode,
                    alpha=score_sorted_alpha,
                    thickness=thin_mask_thickness,
                )

                x, y, w, h = [int(v) for v in bbox]
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(width, x + w)
                y1 = min(height, y + h)
                if x1 <= x0 or y1 <= y0:
                    continue

                # Save per-mask crops only in non-compact mode.
                if not compact_topk_panel:
                    crop_img = base[y0:y1, x0:x1].copy()
                    crop_mask = mask[y0:y1, x0:x1]
                    crop_img = _render_single_mask(
                        crop_img,
                        crop_mask,
                        contrast,
                        mode=single_mask_vis_mode,
                        alpha=0.45,
                        thickness=thin_mask_thickness,
                    )
                    crop_name = f"{dataset}_{img_id}_{alias}_mask_b3_{crop_idx}.png"
                    crop_path = os.path.join(output_dir, crop_name)
                    if not (skip_existing and os.path.exists(crop_path)):
                        cv2.imwrite(crop_path, crop_img)
                    crop_idx += 1

                det_records.append(
                    {
                        "bbox": [float(v) for v in bbox],
                        "mask": mask,
                        "score": score,
                        "color": contrast,
                    }
                )

            if compact_topk_panel:
                panel = _compose_topk_mask_panel(
                    base_bgr=base,
                    det_records=det_records,
                    topk=compact_topk_k,
                    cols=compact_topk_cols,
                    alpha=compact_topk_alpha,
                    show_score=compact_topk_show_score,
                    mask_color=method_topk_color,
                    single_mask_vis_mode=single_mask_vis_mode,
                    single_mask_alpha=compact_topk_alpha,
                    single_mask_thickness=mask_thickness,
                )
                compact_name = f"{dataset}_{img_id}_{alias}_top{compact_topk_k}_panel.png"
                compact_path = os.path.join(output_dir, compact_name)
                if not (skip_existing and os.path.exists(compact_path)):
                    cv2.imwrite(compact_path, panel)
                continue

            out_mask_s3 = f"{dataset}_{img_id}_{alias}_mask_b3.png"
            out_mask_path_s3 = os.path.join(output_dir, out_mask_s3)
            if not (skip_existing and os.path.exists(out_mask_path_s3)):
                cv2.imwrite(out_mask_path_s3, vis_mask_s3)

            if score_sorted_vis:
                sorted_records = sorted(det_records, key=lambda x: -float(x["score"]))
                num_preds = len(sorted_records)
                top_k = min(score_sorted_topk, num_preds)

                for i in range(top_k):
                    rec = sorted_records[i]
                    color = method_topk_color
                    vis_single = _render_single_mask(
                        base.copy(),
                        rec["mask"],
                        color,
                        mode=single_mask_vis_mode,
                        alpha=score_sorted_alpha,
                        thickness=mask_thickness,
                    )
                    vis_single = _draw_text_panel_top_left(
                        vis_single,
                        [f"{float(rec['score']):.2f}"],
                    )
                    out_name = f"{dataset}_{img_id}_{alias}_score_top{i+1}.png"
                    out_path = os.path.join(output_dir, out_name)
                    if not (skip_existing and os.path.exists(out_path)):
                        cv2.imwrite(out_path, vis_single)

                group_records = []
                if num_preds < score_sorted_topk:
                    group_records = []
                    group_tag = ""
                elif num_preds < (score_sorted_topk + score_sorted_lowk):
                    group_records = sorted_records[score_sorted_topk:]
                    group_tag = "score_rest_group"
                else:
                    group_records = sorted_records[-score_sorted_lowk:]
                    group_tag = f"score_low{score_sorted_lowk}_group"

                if group_records:
                    vis_group = base.copy()
                    for rec in group_records:
                        vis_group = _render_single_mask(
                            vis_group,
                            rec["mask"],
                            rec["color"],
                            mode=single_mask_vis_mode,
                            alpha=score_sorted_alpha,
                            thickness=thin_mask_thickness,
                        )
                    group_name = f"{dataset}_{img_id}_{alias}_{group_tag}.png"
                    group_path = os.path.join(output_dir, group_name)
                    if not (skip_existing and os.path.exists(group_path)):
                        cv2.imwrite(group_path, vis_group)

            if not (group_by_gt_region and resolved_region_protocol is not None):
                continue

            gt_anns = anns_by_image.get(img_id, [])
            for gt_idx, gt_ann in enumerate(gt_anns, start=1):
                region = _build_region_from_gt_ann(
                    gt_ann, image_h=height, image_w=width, protocol=resolved_region_protocol
                )
                if not _is_valid_bbox_xywh(region):
                    continue
                rx, ry, rw, rh = [int(round(v)) for v in region]
                rx0 = max(0, rx)
                ry0 = max(0, ry)
                rx1 = min(width, rx + rw)
                ry1 = min(height, ry + rh)
                if rx1 <= rx0 or ry1 <= ry0:
                    continue

                # Draw matched masks first (no bbox), then derive two outputs:
                # 1) center_region: with GT + region bbox on full image
                # 2) center_region_crop: cropped from no-bbox image
                group_vis_no_box = base.copy()
                matched = []
                for rec in det_records:
                    cx, cy = _bbox_center_xywh(rec["bbox"])
                    if _point_inside_xywh(cx, cy, region):
                        matched.append(rec)
                        group_vis_no_box = _render_single_mask(
                            group_vis_no_box,
                            rec["mask"],
                            rec["color"],
                            mode=single_mask_vis_mode,
                            alpha=score_sorted_alpha,
                            thickness=mask_thickness,
                        )

                group_vis = group_vis_no_box.copy()
                cv2.rectangle(group_vis, (rx0, ry0), (rx1, ry1), (0, 255, 255), bbox_thickness)
                gb = gt_ann.get("bbox")
                if gb and len(gb) == 4:
                    gx, gy, gw, gh = [int(v) for v in gb]
                    cv2.rectangle(
                        group_vis,
                        (max(0, gx), max(0, gy)),
                        (min(width, gx + gw), min(height, gy + gh)),
                        (0, 0, 255),
                        bbox_thickness,
                    )

                group_name = (
                    f"{dataset}_{img_id}_{alias}_gtg{gt_idx:02d}_{resolved_region_protocol['name']}_region.png"
                )
                group_path = os.path.join(output_dir, group_name)
                if not (skip_existing and os.path.exists(group_path)):
                    cv2.imwrite(group_path, group_vis)

                region_crop = group_vis_no_box[ry0:ry1, rx0:rx1].copy()
                region_crop_name = (
                    f"{dataset}_{img_id}_{alias}_gtg{gt_idx:02d}_{resolved_region_protocol['name']}_region_crop.png"
                )
                region_crop_path = os.path.join(output_dir, region_crop_name)
                if not (skip_existing and os.path.exists(region_crop_path)):
                    cv2.imwrite(region_crop_path, region_crop)

                for local_idx, rec in enumerate(matched, start=1):
                    bx, by, bw, bh = [int(v) for v in rec["bbox"]]
                    x0 = max(0, bx)
                    y0 = max(0, by)
                    x1 = min(width, bx + bw)
                    y1 = min(height, by + bh)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    grouped_crop = base[y0:y1, x0:x1].copy()
                    grouped_mask = rec["mask"][y0:y1, x0:x1]
                    grouped_crop = _render_single_mask(
                        grouped_crop,
                        grouped_mask,
                        rec["color"],
                        mode=single_mask_vis_mode,
                        alpha=0.45,
                        thickness=mask_thickness,
                    )
                    grouped_name = (
                        f"{dataset}_{img_id}_{alias}_gtg{gt_idx:02d}_{resolved_region_protocol['name']}_mask_{local_idx}.png"
                    )
                    grouped_path = os.path.join(output_dir, grouped_name)
                    if not (skip_existing and os.path.exists(grouped_path)):
                        cv2.imwrite(grouped_path, grouped_crop)


def visualization_from_bundle_json(
    json_path,
    output_dir=None,
    score_thresh=0.0,
    max_det=100,
    score_sorted_vis=False,
    score_sorted_topk=5,
    score_sorted_lowk=10,
    score_sorted_alpha=0.45,
    compact_topk_panel=False,
    compact_topk_k=5,
    compact_topk_cols=3,
    compact_topk_alpha=0.45,
    compact_topk_show_score=False,
    single_mask_vis_mode="edge",
    grayscale_image=True,
    skip_existing=True,
    save_original=True,
    combined_topk_values=(10, 5, 3),
    combined_topk_alpha=0.45,
):
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    image_info = payload.get("image") or {}
    meta = payload.get("meta") or {}
    methods = payload.get("methods") or []
    if not methods:
        raise ValueError("No methods found in visualization bundle JSON.")

    image_bgr = _decode_base64_image_to_bgr(image_info.get("base64"))
    if image_bgr is None:
        raise ValueError("Failed to decode image.base64 from visualization bundle JSON.")
    if grayscale_image:
        image_bgr = _to_grayscale_bgr(image_bgr)

    image_coco = image_info.get("coco") or {}
    dataset = str(meta.get("dataset") or "dataset")
    image_id = image_coco.get("id", meta.get("image_id", "unknown"))

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output_visualization")
    os.makedirs(output_dir, exist_ok=True)

    height, width = image_bgr.shape[:2]
    base = image_bgr.copy()
    bbox_thickness = 3
    mask_thickness = 4
    thin_mask_thickness = 2
    written_files = []

    score_sorted_vis = bool(score_sorted_vis)
    score_sorted_topk = max(1, int(score_sorted_topk))
    score_sorted_lowk = max(1, int(score_sorted_lowk))
    score_sorted_alpha = float(score_sorted_alpha)
    compact_topk_panel = bool(compact_topk_panel)
    compact_topk_k = max(1, int(compact_topk_k))
    compact_topk_cols = max(1, int(compact_topk_cols))
    compact_topk_alpha = float(compact_topk_alpha)
    compact_topk_show_score = bool(compact_topk_show_score)
    single_mask_vis_mode = str(single_mask_vis_mode or "edge").lower().strip()
    method_topk_palette = _method_topk_palette_bgr()
    combined_topk_values = _normalize_topk_values(combined_topk_values)
    combined_topk_alpha = float(combined_topk_alpha)

    if save_original:
        origin_name = f"{dataset}_{image_id}_image.png"
        origin_path = os.path.join(output_dir, origin_name)
        if not (skip_existing and os.path.exists(origin_path)):
            cv2.imwrite(origin_path, base)
        written_files.append(origin_path)

    for method_idx, method in enumerate(methods):
        alias = str(method.get("alias") or f"method_{method_idx + 1}")
        method_topk_color = method_topk_palette[method_idx % len(method_topk_palette)]
        vis_mask = base.copy()
        dets = sorted(
            method.get("predictions_coco") or [],
            key=lambda d: -float(d.get("score", 0.0)),
        )
        det_records = []
        crop_idx = 1

        for det_idx, det in enumerate(dets[: max(1, int(max_det))]):
            score = float(det.get("score", 0.0))
            if score < float(score_thresh):
                continue
            bbox = det.get("bbox")
            if not (bbox and len(bbox) == 4):
                continue
            mask = _decode_rle_mask(det.get("segmentation"), height, width)
            if mask is None:
                continue

            contrast = _contrast_color_bgr(det_idx)
            vis_mask = _render_single_mask(
                vis_mask,
                mask,
                contrast,
                mode=single_mask_vis_mode,
                alpha=score_sorted_alpha,
                thickness=thin_mask_thickness,
            )

            x, y, w, h = [int(round(v)) for v in bbox]
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(width, x + w)
            y1 = min(height, y + h)
            if x1 <= x0 or y1 <= y0:
                continue

            if not compact_topk_panel:
                crop_img = base[y0:y1, x0:x1].copy()
                crop_mask = mask[y0:y1, x0:x1]
                crop_img = _render_single_mask(
                    crop_img,
                    crop_mask,
                    contrast,
                    mode=single_mask_vis_mode,
                    alpha=0.45,
                    thickness=thin_mask_thickness,
                )
                crop_name = f"{dataset}_{image_id}_{alias}_mask_b3_{crop_idx}.png"
                crop_path = os.path.join(output_dir, crop_name)
                if not (skip_existing and os.path.exists(crop_path)):
                    cv2.imwrite(crop_path, crop_img)
                written_files.append(crop_path)
                crop_idx += 1

            det_records.append(
                {
                    "bbox": [float(v) for v in bbox],
                    "mask": mask,
                    "score": score,
                    "color": contrast,
                }
            )

        for topk in combined_topk_values:
            topk_overlay = _compose_ranked_mask_overlay(
                base_bgr=base,
                det_records=det_records,
                topk=topk,
                alpha=combined_topk_alpha,
                palette=_score_vis_palette_bgr(),
                single_mask_vis_mode=single_mask_vis_mode,
                single_mask_thickness=thin_mask_thickness,
                show_rank_legend=True,
            )
            topk_overlay_name = f"{dataset}_{image_id}_{alias}_top{topk}_overlay.png"
            topk_overlay_path = os.path.join(output_dir, topk_overlay_name)
            if not (skip_existing and os.path.exists(topk_overlay_path)):
                cv2.imwrite(topk_overlay_path, topk_overlay)
            written_files.append(topk_overlay_path)

        if compact_topk_panel:
            panel = _compose_topk_mask_panel(
                base_bgr=base,
                det_records=det_records,
                topk=compact_topk_k,
                cols=compact_topk_cols,
                alpha=compact_topk_alpha,
                show_score=compact_topk_show_score,
                mask_color=method_topk_color,
                single_mask_vis_mode=single_mask_vis_mode,
                single_mask_alpha=compact_topk_alpha,
                single_mask_thickness=mask_thickness,
            )
            compact_name = f"{dataset}_{image_id}_{alias}_top{compact_topk_k}_panel.png"
            compact_path = os.path.join(output_dir, compact_name)
            if not (skip_existing and os.path.exists(compact_path)):
                cv2.imwrite(compact_path, panel)
            written_files.append(compact_path)
            continue

        out_mask_name = f"{dataset}_{image_id}_{alias}_mask_b3.png"
        out_mask_path = os.path.join(output_dir, out_mask_name)
        if not (skip_existing and os.path.exists(out_mask_path)):
            cv2.imwrite(out_mask_path, vis_mask)
        written_files.append(out_mask_path)

        if not score_sorted_vis:
            continue

        sorted_records = sorted(det_records, key=lambda x: -float(x["score"]))
        num_preds = len(sorted_records)
        top_k = min(score_sorted_topk, num_preds)

        for i in range(top_k):
            rec = sorted_records[i]
            vis_single = _render_single_mask(
                base.copy(),
                rec["mask"],
                method_topk_color,
                mode=single_mask_vis_mode,
                alpha=score_sorted_alpha,
                thickness=mask_thickness,
            )
            vis_single = _draw_text_panel_top_left(
                vis_single,
                [f"{float(rec['score']):.2f}"],
            )
            out_name = f"{dataset}_{image_id}_{alias}_score_top{i+1}.png"
            out_path = os.path.join(output_dir, out_name)
            if not (skip_existing and os.path.exists(out_path)):
                cv2.imwrite(out_path, vis_single)
            written_files.append(out_path)

        if num_preds < score_sorted_topk:
            group_records = []
            group_tag = ""
        elif num_preds < (score_sorted_topk + score_sorted_lowk):
            group_records = sorted_records[score_sorted_topk:]
            group_tag = "score_rest_group"
        else:
            group_records = sorted_records[-score_sorted_lowk:]
            group_tag = f"score_low{score_sorted_lowk}_group"

        if not group_records:
            continue

        vis_group = base.copy()
        for rec in group_records:
            vis_group = _render_single_mask(
                vis_group,
                rec["mask"],
                rec["color"],
                mode=single_mask_vis_mode,
                alpha=score_sorted_alpha,
                thickness=thin_mask_thickness,
            )
        group_name = f"{dataset}_{image_id}_{alias}_{group_tag}.png"
        group_path = os.path.join(output_dir, group_name)
        if not (skip_existing and os.path.exists(group_path)):
            cv2.imwrite(group_path, vis_group)
        written_files.append(group_path)

    return written_files


def compare_coco_annotations(
    ori_json_path,
    musc_json_path,
    min_iou=0.0,
    output_path=None,
    max_area=150000.0,
    only_best=True,
    max_musc_area=150000.0,
):
    with open(ori_json_path, "r", encoding="utf-8") as f:
        coco_ori = json.load(f)
    with open(musc_json_path, "r", encoding="utf-8") as f:
        coco_musc = json.load(f)

    images_ori = {img["id"]: img for img in coco_ori.get("images", [])}
    images_musc = {img["id"]: img for img in coco_musc.get("images", [])}

    ann_musc_by_image = {}
    for ann in coco_musc.get("annotations", []):
        ann_musc_by_image.setdefault(ann.get("image_id"), []).append(ann)

    header = [
        "image_id",
        "ori_ann_id",
        "best_musc_id",
        "mask_iou",
        "bbox_iou",
        "area_ori",
        "area_musc",
        "area_ratio",
    ]
    lines = ["\t".join(header)]

    stats = {"total": 0, "matched": 0}
    best_by_image = {}

    for ann in tqdm(coco_ori.get("annotations", []), desc="compare_coco"):
        image_id = ann.get("image_id")
        image = images_ori.get(image_id) or images_musc.get(image_id)
        if image is None:
            continue
        height = image.get("height")
        width = image.get("width")
        if height is None or width is None:
            continue

        rle_ori = _ann_to_rle(ann, height, width)
        if rle_ori is None:
            continue
        area_ori = float(ann.get("area", mask_utils.area(rle_ori)))
        if area_ori > max_area:
            continue
        musc_anns = ann_musc_by_image.get(image_id, [])
        if not musc_anns:
            continue

        rles_musc = []
        valid_musc = []
        for m_ann in musc_anns:
            rle_musc = _ann_to_rle(m_ann, height, width)
            if rle_musc is None:
                continue
            area_musc = float(m_ann.get("area", mask_utils.area(rle_musc)))
            if area_musc > max_musc_area:
                continue
            rles_musc.append(rle_musc)
            valid_musc.append(m_ann)

        if not rles_musc:
            continue

        ious = mask_utils.iou([rle_ori], rles_musc, [0] * len(rles_musc))[0]
        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])
        if best_iou < min_iou:
            continue

        best_musc = valid_musc[best_idx]
        area_musc = float(best_musc.get("area", mask_utils.area(rles_musc[best_idx])))
        bbox_ori = ann.get("bbox", mask_utils.toBbox(rle_ori).tolist())
        bbox_musc = best_musc.get("bbox", mask_utils.toBbox(rles_musc[best_idx]).tolist())
        bbox_iou = _bbox_iou(bbox_ori, bbox_musc)
        area_ratio = area_ori / area_musc if area_musc > 0 else 0.0

        row = [
            str(image_id),
            str(ann.get("id")),
            str(best_musc.get("id")),
            f"{best_iou:.4f}",
            f"{bbox_iou:.4f}",
            f"{area_ori:.1f}",
            f"{area_musc:.1f}",
            f"{area_ratio:.3f}",
        ]
        if only_best:
            existing = best_by_image.get(image_id)
            if existing is None or best_iou > existing["iou"]:
                best_by_image[image_id] = {"iou": best_iou, "row": row}
        else:
            lines.append("\t".join(row))
        stats["total"] += 1
        stats["matched"] += 1

    if only_best:
        for image_id in sorted(best_by_image.keys()):
            lines.append("\t".join(best_by_image[image_id]["row"]))

    if output_path is None:
        output_path = os.path.join(os.getcwd(), "compare_coco_annotations.log")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
        f.write(f"total={stats['total']} matched={stats['matched']}\n")
    print(f"Saved comparison log to {output_path}")


def _ann_area(ann, image_info=None):
    area = ann.get("area")
    if area is not None:
        return float(area)
    if image_info is None:
        return None
    height = image_info.get("height")
    width = image_info.get("width")
    if height is None or width is None:
        return None
    rle = _ann_to_rle(ann, height, width)
    if rle is None:
        return None
    return float(mask_utils.area(rle))


def plot_area_histogram(
    json_path_a,
    json_path_b,
    output_path=None,
    bins=50,
    log_scale=False,
    min_area=100000.0,
    max_area=250000.0,
):
    with open(json_path_a, "r", encoding="utf-8") as f:
        coco_a = json.load(f)
    with open(json_path_b, "r", encoding="utf-8") as f:
        coco_b = json.load(f)

    images_a = {img["id"]: img for img in coco_a.get("images", [])}
    images_b = {img["id"]: img for img in coco_b.get("images", [])}

    def collect_areas(coco, images_by_id):
        areas = []
        for ann in coco.get("annotations", []):
            image_info = images_by_id.get(ann.get("image_id"))
            area = _ann_area(ann, image_info=image_info)
            if area is None:
                continue
            if area < min_area or area > max_area:
                continue
            if area is not None:
                areas.append(area)
        return areas

    areas_a = collect_areas(coco_a, images_a)
    areas_b = collect_areas(coco_b, images_b)

    if not areas_a or not areas_b:
        raise ValueError("No valid areas found in one of the JSON files.")

    label_a = os.path.basename(json_path_a)
    label_b = os.path.basename(json_path_b)

    plt.figure(figsize=(10, 6))
    plt.hist(areas_a, bins=bins, alpha=0.6, label=label_a)
    plt.hist(areas_b, bins=bins, alpha=0.6, label=label_b)
    plt.xlabel(f"Area (pixels, {min_area:.0f}-{max_area:.0f})")
    plt.ylabel("Count")
    plt.title("COCO Annotation Area Histogram")
    plt.legend()
    if log_scale:
        plt.yscale("log")

    if output_path is None:
        output_path = os.path.join(os.getcwd(), "area_histogram.png")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved area histogram to {output_path}")


def plot_compare_log_metrics(log_path, output_path=None, bins=50):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = None
    rows = []
    for line in lines:
        if line.startswith("total="):
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        if len(parts) != len(header):
            continue
        rows.append(parts)

    if not header or not rows:
        raise ValueError("No valid data rows found in log file.")

    id_cols = {"image_id", "ori_ann_id", "best_musc_id"}
    metric_cols = [i for i, name in enumerate(header) if name not in id_cols]
    metric_names = [header[i] for i in metric_cols]

    data = {name: [] for name in metric_names}
    for parts in rows:
        for idx, name in zip(metric_cols, metric_names):
            try:
                data[name].append(float(parts[idx]))
            except ValueError:
                continue

    num_metrics = len(metric_names)
    if num_metrics == 0:
        raise ValueError("No metric columns found to plot.")

    cols = 3
    rows_n = int(np.ceil(num_metrics / cols))
    plt.figure(figsize=(5 * cols, 4 * rows_n))

    for i, name in enumerate(metric_names, start=1):
        values = data.get(name, [])
        ax = plt.subplot(rows_n, cols, i)
        ax.hist(values, bins=bins, alpha=0.8)
        ax.set_title(name)
        ax.set_ylabel("Count")
        ax.set_xlabel(name)

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "compare_metrics.png")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved metrics plot to {output_path}")


def merge_curriculum_dataset(
    dataset_dir: str,
    stage_counts: list[int],
    stages: tuple[str, ...] = ("easy", "hard", "extreme"),
    output_subdir: str = "merge",
) -> str:
    """
    Merge easy/hard/extreme data into a single folder with renumbered ids.

    Args:
        dataset_dir: Root containing {easy,hard,extreme}/xray and /anno.
        stage_counts: Number of images to take from each stage
            (e.g. [4000, 6000, 8000]).
        stages: Stage names in order.
        output_subdir: Name of merged output directory under dataset_dir.

    Returns:
        Path to the merged dataset directory.
    """
    if len(stage_counts) != len(stages):
        raise ValueError("stage_counts length must match stages length.")
    if any(c < 0 for c in stage_counts):
        raise ValueError("stage_counts must be non-negative.")

    offsets = []
    running = 0
    for count in stage_counts:
        offsets.append(running)
        running += count
    out_root = os.path.join(dataset_dir, output_subdir)
    out_img_dir = os.path.join(out_root, "xray")
    out_anno_dir = os.path.join(out_root, "anno")
    out_anno_fusion_dir = os.path.join(out_root, "anno_fusion")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_anno_dir, exist_ok=True)
    os.makedirs(out_anno_fusion_dir, exist_ok=True)

    stage_maps = []
    for stage, offset, max_count in zip(stages, offsets, stage_counts):
        stage_img_dir = os.path.join(dataset_dir, stage, "xray")
        stage_anno_dir = os.path.join(dataset_dir, stage, "anno")
        stage_anno_fusion_dir = os.path.join(dataset_dir, stage, "anno_fusion")
        if not os.path.isdir(stage_img_dir) or not os.path.isdir(stage_anno_dir):
            continue

        id_map = {}
        file_map = {}
        copied = 0
        for fname in sorted(os.listdir(stage_img_dir)):
            if max_count and copied >= max_count:
                break
            base, ext = os.path.splitext(fname)
            if not base.isdigit():
                continue
            img_path = os.path.join(stage_img_dir, fname)
            anno_path = os.path.join(stage_anno_dir, f"{base}.json")
            if not os.path.isfile(img_path) or not os.path.isfile(anno_path):
                continue

            new_id = offset + int(base)
            new_base = f"{new_id:0{len(base)}d}"
            dst_img = os.path.join(out_img_dir, f"{new_base}{ext}")
            dst_anno = os.path.join(out_anno_dir, f"{new_base}.json")
            if not os.path.exists(dst_img):
                shutil.copy2(img_path, dst_img)
            if not os.path.exists(dst_anno):
                shutil.copy2(anno_path, dst_anno)

            if os.path.isdir(stage_anno_fusion_dir):
                anno_fusion_src = os.path.join(stage_anno_fusion_dir, f"{base}.json")
                if os.path.isfile(anno_fusion_src):
                    dst_anno_fusion = os.path.join(out_anno_fusion_dir, f"{new_base}.json")
                    if not os.path.exists(dst_anno_fusion):
                        shutil.copy2(anno_fusion_src, dst_anno_fusion)
            id_map[int(base)] = new_id
            file_map[fname] = f"{new_base}{ext}"
            copied += 1

        stage_maps.append(
            {
                "stage": stage,
                "id_map": id_map,
                "file_map": file_map,
            }
        )

    def _rewrite_file_name(old_name: str, new_base: str) -> str:
        folder, fname = os.path.split(old_name)
        ext = os.path.splitext(fname)[1]
        new_name = f"{new_base}{ext}"
        return os.path.join(folder, new_name) if folder else new_name

    def _merge_coco(filename: str) -> None:
        merged = {
            "info": None,
            "licenses": None,
            "categories": None,
            "images": [],
            "annotations": [],
        }
        next_ann_id = 1
        for item in stage_maps:
            stage = item["stage"]
            id_map = item["id_map"]
            if not id_map:
                continue
            src_path = os.path.join(dataset_dir, stage, filename)
            if not os.path.exists(src_path):
                continue
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if merged["info"] is None:
                merged["info"] = data.get("info", DEFAULT_INFO)
                merged["licenses"] = data.get("licenses", DEFAULT_LICENSES)
                merged["categories"] = data.get("categories", DEFAULT_CATEGORIES)

            for image in data.get("images", []):
                old_id = image.get("id")
                if old_id is None or old_id not in id_map:
                    continue
                new_id = id_map[old_id]
                new_base = f"{new_id:06d}"
                new_image = dict(image)
                new_image["id"] = new_id
                if "file_name" in new_image:
                    new_image["file_name"] = _rewrite_file_name(
                        new_image["file_name"], new_base
                    )
                merged["images"].append(new_image)

            for ann in data.get("annotations", []):
                old_image_id = ann.get("image_id")
                if old_image_id is None or old_image_id not in id_map:
                    continue
                new_ann = dict(ann)
                new_ann["image_id"] = id_map[old_image_id]
                new_ann["id"] = next_ann_id
                next_ann_id += 1
                merged["annotations"].append(new_ann)

        if merged["info"] is None:
            return

        out_path = os.path.join(out_root, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

    _merge_coco("coco.json")
    _merge_coco("coco_agnostic.json")

    return out_root

if __name__ == "__main__":
    evaluate_output_eval_metrics(
    suffix="_augmentation_restart_fusion_095_v2_easy",
    start_id=0,
    end_id=100,
    output_eval_dir="/path/to/LAO-X/output_eval",
    )
    convert_to_class_agnostic("/path/to/datasets/det_compass_syn_100")
