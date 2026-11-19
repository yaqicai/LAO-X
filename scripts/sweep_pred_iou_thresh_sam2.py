#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = Path(os.environ.get("LAOX_CKPT_ROOT", REPO_ROOT / "checkpoints"))
RESULTS_ROOT = Path(os.environ.get("LAOX_RESULTS_ROOT", REPO_ROOT / "results"))


def _add_sam2_to_sys_path() -> None:
    sam2_dir = REPO_ROOT / "sam2"
    if str(sam2_dir) not in sys.path:
        sys.path.insert(0, str(sam2_dir))


def load_unsam_config(path: str):
    spec = importlib.util.spec_from_file_location("unsam_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _class_agnostic_categories():
    return [{"id": 1, "name": "object"}]


def _minimal_coco(images: list[dict], annotations: list[dict]) -> dict:
    return {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": _class_agnostic_categories(),
    }


def build_subset_coco(coco_json: str, max_images: int, out_path: str) -> str:
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])[:max_images]
    selected_ids = {img["id"] for img in images}
    annotations = []
    for ann in data.get("annotations", []):
        if ann.get("image_id") not in selected_ids:
            continue
        ann = dict(ann)
        ann["category_id"] = 1
        annotations.append(ann)

    out = _minimal_coco(images, annotations)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out_path


def build_dvxray_coco(image_root: str, anno_json: str, max_images: int, out_path: str) -> str:
    with open(anno_json, "r", encoding="utf-8") as f:
        anno_data = json.load(f)
    excluded = set()
    for img in anno_data.get("images", []):
        file_name = img.get("file_name", "")
        excluded.add(file_name)
        excluded.add(os.path.basename(file_name))

    json_files = sorted(Path(image_root).glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No per-image json annotations found in {image_root}")

    images = []
    annotations = []
    img_to_id = {}
    ann_id = 1

    def add_image(file_name: str) -> int:
        if file_name in img_to_id:
            return img_to_id[file_name]
        img_path = file_name if os.path.isabs(file_name) else os.path.join(image_root, file_name)
        with Image.open(img_path) as img:
            width, height = img.size
        img_id = len(img_to_id) + 1
        img_to_id[file_name] = img_id
        images.append({"id": img_id, "file_name": file_name, "width": width, "height": height})
        return img_id

    def add_box(img_id: int, bbox: list[float]):
        nonlocal ann_id
        if len(bbox) != 4:
            return
        x_min, y_min, x_max, y_max = map(float, bbox)
        bw = max(0.0, x_max - x_min)
        bh = max(0.0, y_max - y_min)
        if bw <= 0 or bh <= 0:
            return
        annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_min, y_min, bw, bh],
                "segmentation": [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]],
                "area": bw * bh,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("name") or json_path.stem
        objects = data.get("objects", [])
        if not objects:
            continue

        for suffix, key in [("_OL.png", "ol_bb"), ("_SD.png", "sd_bb")]:
            file_name = f"{name}{suffix}"
            if file_name in excluded or os.path.basename(file_name) in excluded:
                continue
            if len(images) >= max_images:
                break
            img_id = add_image(file_name)
            for obj in objects:
                bbox = obj.get(key)
                if bbox:
                    add_box(img_id, bbox)
        if len(images) >= max_images:
            break

    out = _minimal_coco(images, annotations)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out_path


def build_hixray_coco(image_root: str, anno_dir: str, max_images: int, out_path: str) -> str:
    txt_files = sorted(Path(anno_dir).glob("*.txt"))[:max_images]
    if not txt_files:
        raise FileNotFoundError(f"No .txt annotations found in {anno_dir}")

    images = []
    annotations = []
    img_to_id = {}
    ann_id = 1

    for txt_path in txt_files:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) < 6:
                continue
            file_name = parts[0]
            x_min, y_min, x_max, y_max = map(float, parts[2:6])
            if file_name not in img_to_id:
                img_path = file_name if os.path.isabs(file_name) else os.path.join(image_root, file_name)
                with Image.open(img_path) as img:
                    width, height = img.size
                img_id = len(img_to_id) + 1
                img_to_id[file_name] = img_id
                images.append({"id": img_id, "file_name": file_name, "width": width, "height": height})
            img_id = img_to_id[file_name]
            bw = max(0.0, x_max - x_min)
            bh = max(0.0, y_max - y_min)
            if bw <= 0 or bh <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x_min, y_min, bw, bh],
                    "segmentation": [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        if len(images) >= max_images:
            break

    out = _minimal_coco(images[:max_images], annotations)
    keep_ids = {img["id"] for img in out["images"]}
    out["annotations"] = [ann for ann in out["annotations"] if ann["image_id"] in keep_ids]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out_path


def compute_coco_ap(coco_json: str, predictions: list[dict], iou_type: str = "bbox"):
    if not predictions:
        return 0.0, np.array([])
    coco_gt = COCO(coco_json)
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0]), coco_eval.stats


def bbox_iou_matrix(boxes: np.ndarray, gt_box: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=np.float32)
    x0 = np.maximum(boxes[:, 0], gt_box[0])
    y0 = np.maximum(boxes[:, 1], gt_box[1])
    x1 = np.minimum(boxes[:, 2], gt_box[2])
    y1 = np.minimum(boxes[:, 3], gt_box[3])
    inter = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    area_a = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    area_b = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def build_mask_generator(args):
    _add_sam2_to_sys_path()
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    sam2_model = build_sam2(args.model_config, args.checkpoint, device=device, apply_postprocessing=True)
    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=args.stability_score_offset,
        mask_threshold=args.mask_threshold,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="coco_rle",
        use_m2m=args.use_m2m,
        multimask_output=args.multimask_output,
    )


def generate_predictions(generator, image_root: str, images: list[dict], gra: float):
    predictions = []
    boxes_by_image: dict[int, list[list[float]]] = {}
    for index, info in enumerate(images):
        img_id = int(info["id"])
        file_name = info.get("file_name")
        if not file_name:
            continue
        img_path = file_name if os.path.isabs(file_name) else os.path.join(image_root, file_name)
        if not os.path.exists(img_path):
            print(f"[warn] missing image: {img_path}")
            continue
        with Image.open(img_path) as img:
            image_np = np.array(img.convert("RGB"))
        anns = generator.generate(image_np, gra=gra)
        boxes = []
        for ann in anns:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 0 or h <= 0:
                continue
            score = float(ann.get("predicted_iou", 1.0))
            pred = {
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "score": score,
            }
            predictions.append(pred)
            boxes.append([x, y, x + w, y + h])
        boxes_by_image[img_id] = boxes
        if (index + 1) % 10 == 0:
            print(f"  processed {index + 1}/{len(images)} images")
    return predictions, boxes_by_image


def mean_best_bbox_iou(coco_json: str, boxes_by_image: dict[int, list[list[float]]]) -> float:
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    vals = []
    for ann in data.get("annotations", []):
        img_id = int(ann["image_id"])
        x, y, w, h = [float(v) for v in ann["bbox"]]
        gt = np.array([x, y, x + w, y + h], dtype=np.float32)
        boxes = np.asarray(boxes_by_image.get(img_id, []), dtype=np.float32)
        ious = bbox_iou_matrix(boxes, gt)
        vals.append(float(ious.max()) if ious.size else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep pred_iou_thresh for SAM2 automatic-mask X-ray calibration.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.py"))
    parser.add_argument("--datasets", default="det_compass_xp")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--pred-iou-values", default="")
    parser.add_argument("--pred-iou-min", type=float, default=0.3)
    parser.add_argument("--pred-iou-max", type=float, default=0.95)
    parser.add_argument("--pred-iou-step", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=str(CKPT_ROOT / "sam2.1_hiera_small.pt"))
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--points-per-batch", type=int, default=512)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--stability-score-offset", type=float, default=1.0)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    parser.add_argument("--use-m2m", action="store_true")
    parser.add_argument("--no-multimask-output", action="store_true")
    parser.add_argument("--gra", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "pred_iou_thresh_sam2"))
    parser.add_argument("--pred-iou-thresh", type=float, default=0.77)
    return parser.parse_args()

def _thresholds(args):
    if args.pred_iou_values:
        return [float(v) for v in args.pred_iou_values.split(",") if v.strip()]
    return np.round(
        np.arange(args.pred_iou_min, args.pred_iou_max + 1e-9, args.pred_iou_step),
        4,
    ).tolist()


def main():
    args = parse_args()
    args.multimask_output = not args.no_multimask_output
    cfg = load_unsam_config(args.config)
    img_roots = cfg.XrayImgRootVal
    anno_paths = cfg.XrayImgRootValAnno

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    thresholds = _thresholds(args)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    generator = build_mask_generator(args)

    for dataset in datasets:
        image_root = img_roots[dataset]
        anno_path = anno_paths[dataset]
        dataset_dir = output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if dataset == "HiXray":
            val_json = build_hixray_coco(image_root, anno_path, args.max_images, str(dataset_dir / "val_coco.json"))
        elif dataset == "DvXray":
            val_json = build_dvxray_coco(image_root, anno_path, args.max_images, str(dataset_dir / "val_coco.json"))
        else:
            val_json = build_subset_coco(anno_path, args.max_images, str(dataset_dir / "val_coco.json"))

        with open(val_json, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        images = val_data.get("images", [])
        results = []
        best = {"pred_iou_thresh": None, "ap": -1.0}
        print(f"==> Dataset: {dataset} ({len(images)} images)")

        for th in thresholds:
            generator.pred_iou_thresh = float(th)
            predictions, boxes_by_image = generate_predictions(generator, image_root, images, args.gra)
            mean_iou = mean_best_bbox_iou(val_json, boxes_by_image)
            ap, stats = compute_coco_ap(val_json, predictions, iou_type="bbox")
            print(f"[{dataset}] pred_iou_thresh={th} AP={ap:.4f} mean_iou={mean_iou:.4f}")
            stats_list = stats.tolist() if isinstance(stats, np.ndarray) and stats.size else []
            results.append(
                {
                    "pred_iou_thresh": th,
                    "ap": ap,
                    "mean_iou": mean_iou,
                    "num_predictions": len(predictions),
                    "stats": stats_list,
                }
            )
            pred_path = dataset_dir / f"pred_iou_{str(th).replace('.', 'p')}_predictions.json"
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f)
            if not np.isnan(ap) and ap > best["ap"]:
                best = {"pred_iou_thresh": th, "ap": ap}

        out_path = dataset_dir / "pred_iou_thresh_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"dataset": dataset, "results": results, "best": best}, f, indent=2)
        print(f"[{dataset}] Best pred_iou_thresh: {best['pred_iou_thresh']} (AP={best['ap']:.4f})")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
