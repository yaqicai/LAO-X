import argparse
import copy
import datetime
import itertools
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog, build_detection_test_loader
from detectron2.data import detection_utils as d2_utils
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.coco_evaluation import COCOeval, COCOevalMaxDets, COCOeval_opt
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.utils.file_io import PathManager

from config import XrayRoot, XrayImgRoot
try:
    import wandb
except ImportError:
    wandb = None


def init_seed_torch(seed=1, deterministic=True):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def _write_evaluation_provenance(output_dir, args):
    payload = {
        "checkpoint": os.path.abspath(args.sam2_ckpt),
        "checkpoint_suffix": args.ckpt_suffix,
        "model_seed": args.model_seed,
        "evaluation_seed": args.seed,
        "train_data_profile": args.train_data_profile,
        "dataset": args.dataset,
        "start_id": args.start_id,
        "end_id": args.end_id,
        "granularity": args.gra,
        "mask_threshold": args.mask_threshold,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score": args.stability_score,
        "deterministic": args.deterministic,
        "amp": args.amp,
        "command": sys.argv,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    path = os.path.join(output_dir, "evaluation_provenance.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


COCO_METRIC_ORDER = [
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

COCO_METRIC_KEYS = {
    "AP IoU=0.50:0.95": "AP",
    "AP IoU=0.50": "AP50",
    "AP IoU=0.75": "AP75",
    "AP small": "APs",
    "AP medium": "APm",
    "AP large": "APl",
    "AR maxDets=1": "AR@1",
    "AR maxDets=10": "AR@10",
    "AR maxDets=100": "AR@100",
    "AR small": "ARs",
    "AR medium": "ARm",
    "AR large": "ARl",
}


def _ensure_coco_meta(data, dataset=None):
    desc = dataset or "dataset"
    print(desc)
    if "info" not in data:
        data["info"] = {
            "year": 2025,
            "version": 1.0,
            "description": desc,
            "contributor": "None",
        }
    else:
        if not isinstance(data["info"], dict):
            data["info"] = {
                "year": 2025,
                "version": 1.0,
                "description": desc,
                "contributor": "None",
            }
        elif "description" not in data["info"]:
            data["info"]["description"] = desc
    if "licenses" not in data:
        data["licenses"] = []
    if "categories" not in data:
        data["categories"] = [
            {
                "id": 1,
                "name": "object",
            }
        ]
    return data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def _add_sam2_to_sys_path() -> None:
    if os.environ.get("UNSAMV2_USE_INSTALLED_SAM2") == "1":
        return
    repo_root = Path(__file__).resolve().parent
    sam2_dir = repo_root / "sam2"
    if str(sam2_dir) not in sys.path:
        sys.path.insert(0, str(sam2_dir))


def _overlay_masks(base_image: Image.Image, masks, colors, alpha=0.45):
    image = np.array(base_image).astype(np.float32)
    for mask, color in zip(masks, colors):
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        image[mask] = image[mask] * (1 - alpha) + color_arr * alpha
    return Image.fromarray(image.astype(np.uint8))


def _draw_mask_boundaries(base_image: Image.Image, masks, colors, thickness=4, darken=0.6):
    image = np.array(base_image).copy()
    for mask, color in zip(masks, colors):
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.ndim == 3:
            mask_u8 = mask_u8[:, :, 0]
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        boundary_color = tuple(int(c * darken) for c in color)
        cv2.drawContours(image, contours, -1, boundary_color, thickness)
    return Image.fromarray(image)


def _overlay_mask_bgr(img_bgr, mask, color, alpha=0.4):
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


def _draw_mask_boundary_bgr(img_bgr, mask, color, thickness=3):
    if mask is None:
        return img_bgr
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.shape[:2] != img_bgr.shape[:2]:
        return img_bgr
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[:, :, 0]
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    out = img_bgr.copy()
    cv2.drawContours(out, contours, -1, color, int(thickness))
    return out


def _lighten_color_bgr(color, factor=0.45):
    return tuple(int(c + (255 - c) * factor) for c in color)


def _render_mask_edge_with_light_fill_bgr(
    img_bgr,
    mask,
    color,
    edge_thickness=2,
    fill_alpha=0.30,
    fill_lighten_factor=0.22,
):
    fill_color = _lighten_color_bgr(color, factor=fill_lighten_factor)
    out = _overlay_mask_bgr(img_bgr, mask, fill_color, alpha=float(fill_alpha))
    out = _draw_mask_boundary_bgr(out, mask, color, thickness=int(edge_thickness))
    return out


def _score_vis_palette_bgr():
    return [
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 128, 255),
        (128, 0, 255),
        (0, 255, 128),
        (255, 0, 128),
        (128, 255, 0),
    ]


def _build_shared_topn_palette(topn):
    topn = max(1, int(topn))
    base = list(_score_vis_palette_bgr())
    if topn <= len(base):
        return base[:topn]
    extra = []
    start = len(base)
    for idx in range(topn - len(base)):
        h = int(((start + idx) * 67) % 180)
        hsv = np.uint8([[[h, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        extra.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return base + extra


def _is_target_image(image_path, target_image):
    target = str(target_image or "").strip()
    if not target:
        return False
    image_norm = os.path.normpath(str(image_path))
    target_norm = os.path.normpath(target)
    if image_norm == target_norm:
        return True
    image_base = os.path.basename(image_norm)
    target_base = os.path.basename(target_norm)
    if image_base == target_base:
        return True
    return Path(image_base).stem == Path(target_base).stem


def _render_selected_mask_ids_overlay(
    image_rgb,
    det_records,
    selected_mask_ids,
    mask_id_base=1,
    alpha=0.45,
    thickness=2,
):
    if not det_records or not selected_mask_ids:
        return None, []
    ranked_records = sorted(det_records, key=lambda x: -float(x.get("score", 0.0)))
    chosen = []
    valid_ids = []
    for mask_id in selected_mask_ids:
        rank_idx = int(mask_id) - int(mask_id_base)
        if 0 <= rank_idx < len(ranked_records):
            chosen.append(ranked_records[rank_idx])
            valid_ids.append(int(mask_id))
    if not chosen:
        return None, []

    palette = _build_shared_topn_palette(len(chosen))
    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for idx, rec in enumerate(chosen):
        color = palette[idx % len(palette)]
        fill_color = _lighten_color_bgr(color, factor=0.24)
        out = _overlay_mask_bgr(
            out,
            rec["mask"],
            fill_color,
            alpha=min(0.48, max(0.18, float(alpha) * 0.80)),
        )
    return out, valid_ids


def _render_selected_mask_points_overlay(
    image_rgb,
    det_records,
    selected_mask_ids,
    mask_id_base=1,
    radius=5,
    point_color=(0, 255, 0),
):
    if not det_records or not selected_mask_ids:
        return None, []
    ranked_records = sorted(det_records, key=lambda x: -float(x.get("score", 0.0)))
    chosen = []
    valid_ids = []
    for mask_id in selected_mask_ids:
        rank_idx = int(mask_id) - int(mask_id_base)
        if 0 <= rank_idx < len(ranked_records):
            rec = ranked_records[rank_idx]
            if rec.get("point") is None:
                continue
            chosen.append(rec)
            valid_ids.append(int(mask_id))
    if not chosen:
        return None, []

    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for rec in chosen:
        px, py = rec["point"]
        px = int(px)
        py = int(py)
        cv2.circle(
            out,
            (px, py),
            int(radius),
            point_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return out, valid_ids


def _draw_boxes(image: Image.Image, boxes, colors, width=2):
    drawer = ImageDraw.Draw(image)
    for box, color in zip(boxes, colors):
        x1, y1, x2, y2 = [float(v) for v in box]
        drawer.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return image


def _draw_xywh_boxes(image: Image.Image, boxes, colors, width=2):
    drawer = ImageDraw.Draw(image)
    for box, color in zip(boxes, colors):
        if not box:
            continue
        x, y, w, h = [float(v) for v in box]
        drawer.rectangle([x, y, x + w, y + h], outline=color, width=width)
    return image


def _save_png_fast(image: Image.Image, path: str) -> None:
    image.save(path, compress_level=1)


def _to_grayscale_bgr(img_bgr):
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _to_grayscale_pil(image_pil: Image.Image) -> Image.Image:
    img_rgb = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray_bgr = _to_grayscale_bgr(img_bgr)
    gray_rgb = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(gray_rgb)


def _mask_to_xyxy_bbox(mask):
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2, y2]


def _decode_coco_mask(seg, height, width):
    if seg is None:
        return None
    if isinstance(seg, list):
        if len(seg) == 0:
            return None
        try:
            rles = mask_utils.frPyObjects(seg, height, width)
            rle = mask_utils.merge(rles)
        except Exception:
            return None
    elif isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):
            try:
                rle = mask_utils.frPyObjects(seg, height, width)
            except Exception:
                return None
        else:
            rle = seg
    else:
        return None
    mask = mask_utils.decode(rle)
    if mask.shape[0] != height or mask.shape[1] != width:
        mask = mask[:height, :width]
    return mask.astype(bool)


def _collect_image_paths(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = []
    for root, _dirs, files in os.walk(image_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                paths.append(os.path.join(root, name))
    return sorted(paths)


def _predict_instances_for_image(image_path, predictor, score_thresh=0.0):
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    outputs = predictor([{"file_name": image_path, "image_id": 0}])
    instances = outputs[0]["instances"].to("cpu")
    if instances.has("scores") and score_thresh > 0:
        keep = instances.scores >= float(score_thresh)
        instances = instances[keep]
    return image_pil, image_np, instances


def _extract_vis_records_from_instances(instances):
    vis_masks = []
    vis_boxes = []
    vis_points = []
    vis_items = []

    if instances.has("pred_masks"):
        masks = instances.pred_masks.tensor.cpu().numpy().astype(bool)
    else:
        masks = []
    if instances.has("scores"):
        scores = instances.scores.cpu().numpy().astype(float)
    else:
        scores = np.ones((len(masks),), dtype=np.float32)
    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor.cpu().numpy()
    else:
        boxes = []

    for idx_mask, mask in enumerate(masks):
        if mask is None or not np.any(mask):
            continue
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        point_xy = (cx, cy)
        box = boxes[idx_mask] if idx_mask < len(boxes) else _mask_to_xyxy_bbox(mask)
        score_val = float(scores[idx_mask]) if idx_mask < len(scores) else 0.0
        vis_masks.append(mask)
        if box is not None:
            vis_boxes.append(box)
        vis_points.append(point_xy)
        vis_items.append({"mask": mask, "box": box, "point": point_xy, "score": score_val})
    return vis_masks, vis_boxes, vis_points, vis_items


def _save_selected_mask_map_outputs(
    image_np,
    image_gray_np,
    vis_items,
    selected_mask_ids,
    selected_mask_id_base,
    output_dir,
    stem,
    gra_suffix,
):
    selected_overlay_bgr, valid_ids = _render_selected_mask_ids_overlay(
        image_np,
        vis_items,
        selected_mask_ids,
        mask_id_base=selected_mask_id_base,
        alpha=0.45,
        thickness=2,
    )
    if selected_overlay_bgr is None:
        return False

    ids_tag = "_".join(str(x) for x in valid_ids)
    out_sel = os.path.join(
        output_dir,
        f"{stem}_maskids_{ids_tag}_overlay{gra_suffix}.png",
    )
    cv2.imwrite(out_sel, selected_overlay_bgr)

    selected_overlay_gray_bgr, _ = _render_selected_mask_ids_overlay(
        image_gray_np,
        vis_items,
        selected_mask_ids,
        mask_id_base=selected_mask_id_base,
        alpha=0.45,
        thickness=2,
    )
    if selected_overlay_gray_bgr is not None:
        out_sel_gray = os.path.join(
            output_dir,
            f"{stem}_maskids_{ids_tag}_overlay_gray{gra_suffix}.png",
        )
        cv2.imwrite(out_sel_gray, selected_overlay_gray_bgr)

    selected_points_bgr, point_ids = _render_selected_mask_points_overlay(
        image_np,
        vis_items,
        selected_mask_ids,
        mask_id_base=selected_mask_id_base,
        radius=4,
        point_color=(0, 255, 0),
    )
    if selected_points_bgr is not None:
        point_ids_tag = "_".join(str(x) for x in point_ids)
        out_sel_points = os.path.join(
            output_dir,
            f"{stem}_maskids_{point_ids_tag}_points_overlay{gra_suffix}.png",
        )
        cv2.imwrite(out_sel_points, selected_points_bgr)
        selected_points_gray_bgr, _ = _render_selected_mask_points_overlay(
            image_gray_np,
            vis_items,
            selected_mask_ids,
            mask_id_base=selected_mask_id_base,
            radius=4,
            point_color=(0, 255, 0),
        )
        if selected_points_gray_bgr is not None:
            out_sel_points_gray = os.path.join(
                output_dir,
                f"{stem}_maskids_{point_ids_tag}_points_overlay_gray{gra_suffix}.png",
            )
            cv2.imwrite(out_sel_points_gray, selected_points_gray_bgr)
    return True


def _visualize_instances(image, instances, output_prefix, seed=7, gra=None):
    num_instances = len(instances)
    colors = []
    for i in range(num_instances):
        intensity = max(80, 255 - (i * 25) % 175)
        colors.append((intensity, 0, 0))  # RGB red scale

    base = image
    vis_mask = base
    if instances.has("pred_masks"):
        masks = instances.pred_masks.tensor.cpu().numpy().astype(bool)
        vis_mask = _overlay_masks(base, masks, colors, alpha=0.35)

    vis_bbox = base.copy()
    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        vis_bbox = _draw_boxes(vis_bbox, boxes, colors, width=3)

    vis_full = vis_mask.copy()
    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        vis_full = _draw_boxes(vis_full, boxes, colors, width=3)

    base_gray = _to_grayscale_pil(base)
    vis_mask_gray = base_gray
    if instances.has("pred_masks"):
        masks = instances.pred_masks.tensor.cpu().numpy().astype(bool)
        vis_mask_gray = _overlay_masks(base_gray, masks, colors, alpha=0.35)

    vis_bbox_gray = base_gray.copy()
    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        vis_bbox_gray = _draw_boxes(vis_bbox_gray, boxes, colors, width=3)

    vis_full_gray = vis_mask_gray.copy()
    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        vis_full_gray = _draw_boxes(vis_full_gray, boxes, colors, width=3)

    vis_bbox.save(f"{output_prefix}_bbox_{gra}.png")
    vis_mask.save(f"{output_prefix}_mask_{gra}.png")
    vis_full.save(f"{output_prefix}_full_{gra}.png")
    vis_bbox_gray.save(f"{output_prefix}_bbox_gray_{gra}.png")
    vis_mask_gray.save(f"{output_prefix}_mask_gray_{gra}.png")
    vis_full_gray.save(f"{output_prefix}_full_gray_{gra}.png")


def visualize_directory_images(
    image_dir,
    predictor,
    output_dir,
    score_thresh=0.0,
    seed=7,
    gra=None,
):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = _collect_image_paths(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    for image_path in tqdm(image_paths, desc="visualize_dir"):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        if isinstance(predictor, UnSAMv2Predictor):
            outputs = predictor([{"file_name": image_path, "image_id": 0}])
            instances = outputs[0]["instances"].to("cpu")
        else:
            outputs = predictor(image_np[:, :, ::-1])
            instances = outputs["instances"].to("cpu")

        if instances.has("scores") and score_thresh > 0:
            keep = instances.scores >= float(score_thresh)
            instances = instances[keep]

        stem = Path(image_path).stem
        output_prefix = os.path.join(output_dir, stem)
        _visualize_instances(image, instances, output_prefix, seed=seed, gra=gra)


def visualize_directory_interactive(
    image_dir,
    predictor,
    output_dir,
    score_thresh=0.0,
    gra_tag=None,
    selected_mask_map=None,
    selected_mask_id_base=1,
):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = _collect_image_paths(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    selected_mask_map = dict(selected_mask_map or {})
    selected_mask_map = {
        str(k).strip(): [int(v) for v in vals]
        for k, vals in selected_mask_map.items()
        if str(k).strip() and vals
    }
    selected_mask_id_base = int(selected_mask_id_base)

    for image_path in tqdm(image_paths, desc="interactive_dir"):
        image_pil, image_np, instances = _predict_instances_for_image(
            image_path, predictor, score_thresh=score_thresh
        )
        image_gray_pil = _to_grayscale_pil(image_pil)
        vis_masks, vis_boxes, vis_points, vis_items = _extract_vis_records_from_instances(instances)

        if not vis_masks:
            continue

        colors = [(0, 255, 0)] * len(vis_masks)
        bbox_colors = [(0, 255, 0)] * len(vis_masks)

        vis_mask = _overlay_masks(image_pil, vis_masks, colors, alpha=0.4)
        vis_bbox = image_pil.copy()
        if vis_boxes:
            vis_bbox = _draw_boxes(vis_bbox, vis_boxes, bbox_colors, width=3)
        vis_full = vis_mask.copy()
        if vis_boxes:
            vis_full = _draw_boxes(vis_full, vis_boxes, bbox_colors, width=3)
        vis_point = image_pil.copy()

        vis_mask_gray = _overlay_masks(image_gray_pil, vis_masks, colors, alpha=0.4)
        vis_bbox_gray = image_gray_pil.copy()
        if vis_boxes:
            vis_bbox_gray = _draw_boxes(vis_bbox_gray, vis_boxes, bbox_colors, width=3)
        vis_full_gray = vis_mask_gray.copy()
        if vis_boxes:
            vis_full_gray = _draw_boxes(vis_full_gray, vis_boxes, bbox_colors, width=3)
        vis_point_gray = image_gray_pil.copy()

        if vis_points:
            for target in (
                vis_bbox,
                vis_mask,
                vis_full,
                vis_point,
                vis_bbox_gray,
                vis_mask_gray,
                vis_full_gray,
                vis_point_gray,
            ):
                drawer = ImageDraw.Draw(target)
                for px, py in vis_points:
                    r = 4
                    drawer.ellipse(
                        (px - r, py - r, px + r, py + r),
                        outline=(0, 255, 0),
                        fill=(0, 255, 0),
                        width=2,
                    )

        stem = Path(image_path).stem
        gra_suffix = f"_gra{gra_tag}" if gra_tag is not None else ""
        _save_png_fast(image_pil, os.path.join(output_dir, f"{stem}_orig{gra_suffix}.png"))
        _save_png_fast(vis_bbox, os.path.join(output_dir, f"{stem}_bbox{gra_suffix}.png"))
        _save_png_fast(vis_mask, os.path.join(output_dir, f"{stem}_mask{gra_suffix}.png"))
        _save_png_fast(vis_full, os.path.join(output_dir, f"{stem}_full{gra_suffix}.png"))
        _save_png_fast(vis_point, os.path.join(output_dir, f"{stem}_point{gra_suffix}.png"))
        _save_png_fast(image_gray_pil, os.path.join(output_dir, f"{stem}_orig_gray{gra_suffix}.png"))
        _save_png_fast(vis_bbox_gray, os.path.join(output_dir, f"{stem}_bbox_gray{gra_suffix}.png"))
        _save_png_fast(vis_mask_gray, os.path.join(output_dir, f"{stem}_mask_gray{gra_suffix}.png"))
        _save_png_fast(vis_full_gray, os.path.join(output_dir, f"{stem}_full_gray{gra_suffix}.png"))
        _save_png_fast(vis_point_gray, os.path.join(output_dir, f"{stem}_point_gray{gra_suffix}.png"))

        if len(vis_items) > 1:
            for idx_item, item in enumerate(vis_items, start=1):
                per_mask = item["mask"]
                per_box = item["box"]
                per_point = item["point"]
                per_mask_img = _overlay_masks(image_pil, [per_mask], [(0, 255, 0)], alpha=0.4)
                per_mask_img_gray = _overlay_masks(
                    image_gray_pil, [per_mask], [(0, 255, 0)], alpha=0.4
                )
                per_bbox_img = image_pil.copy()
                per_bbox_img_gray = image_gray_pil.copy()
                if per_box is not None:
                    per_bbox_img = _draw_boxes(per_bbox_img, [per_box], [(0, 255, 0)], width=3)
                    per_bbox_img_gray = _draw_boxes(
                        per_bbox_img_gray, [per_box], [(0, 255, 0)], width=3
                    )
                per_full_img = per_mask_img.copy()
                per_full_img_gray = per_mask_img_gray.copy()
                if per_box is not None:
                    per_full_img = _draw_boxes(per_full_img, [per_box], [(0, 255, 0)], width=3)
                    per_full_img_gray = _draw_boxes(
                        per_full_img_gray, [per_box], [(0, 255, 0)], width=3
                    )
                per_point_img = image_pil.copy()
                per_point_img_gray = image_gray_pil.copy()

                if per_point is not None:
                    for target in (
                        per_bbox_img,
                        per_mask_img,
                        per_full_img,
                        per_point_img,
                        per_bbox_img_gray,
                        per_mask_img_gray,
                        per_full_img_gray,
                        per_point_img_gray,
                    ):
                        drawer = ImageDraw.Draw(target)
                        px, py = per_point
                        r = 4
                        drawer.ellipse(
                            (px - r, py - r, px + r, py + r),
                            outline=(0, 255, 0),
                            fill=(0, 255, 0),
                            width=2,
                        )

                per_suffix = f"_{idx_item}"
                _save_png_fast(
                    per_bbox_img, os.path.join(output_dir, f"{stem}{per_suffix}_bbox{gra_suffix}.png")
                )
                _save_png_fast(
                    per_mask_img, os.path.join(output_dir, f"{stem}{per_suffix}_mask{gra_suffix}.png")
                )
                _save_png_fast(
                    per_full_img, os.path.join(output_dir, f"{stem}{per_suffix}_full{gra_suffix}.png")
                )
                _save_png_fast(
                    per_point_img, os.path.join(output_dir, f"{stem}{per_suffix}_point{gra_suffix}.png")
                )
                _save_png_fast(
                    per_bbox_img_gray,
                    os.path.join(output_dir, f"{stem}{per_suffix}_bbox_gray{gra_suffix}.png"),
                )
                _save_png_fast(
                    per_mask_img_gray,
                    os.path.join(output_dir, f"{stem}{per_suffix}_mask_gray{gra_suffix}.png"),
                )
                _save_png_fast(
                    per_full_img_gray,
                    os.path.join(output_dir, f"{stem}{per_suffix}_full_gray{gra_suffix}.png"),
                )
                _save_png_fast(
                    per_point_img_gray,
                    os.path.join(output_dir, f"{stem}{per_suffix}_point_gray{gra_suffix}.png"),
                )

    if not selected_mask_map:
        return

    print("Running selected mask-id fusion visualization stage...")
    for image_key, selected_mask_ids in selected_mask_map.items():
        matches = [path for path in image_paths if _is_target_image(path, image_key)]
        if not matches:
            print(f"[selected-mask-map] Skip missing image: {image_key}")
            continue
        if len(matches) > 1:
            print(
                f"[selected-mask-map] Multiple matches for {image_key}, use first: {matches[0]}"
            )
        target_path = matches[0]
        image_pil, image_np, instances = _predict_instances_for_image(
            target_path, predictor, score_thresh=score_thresh
        )
        image_gray_pil = _to_grayscale_pil(image_pil)
        image_gray_np = np.array(image_gray_pil)
        _vis_masks, _vis_boxes, _vis_points, vis_items = _extract_vis_records_from_instances(instances)
        if not vis_items:
            print(f"[selected-mask-map] No valid prediction masks for {target_path}")
            continue
        stem = Path(target_path).stem
        gra_suffix = f"_gra{gra_tag}" if gra_tag is not None else ""
        ok = _save_selected_mask_map_outputs(
            image_np=image_np,
            image_gray_np=image_gray_np,
            vis_items=vis_items,
            selected_mask_ids=selected_mask_ids,
            selected_mask_id_base=selected_mask_id_base,
            output_dir=output_dir,
            stem=stem,
            gra_suffix=gra_suffix,
        )
        if not ok:
            print(
                f"[selected-mask-map] No valid mask ids for {target_path}: {selected_mask_ids}"
            )


def interactive_one_click_eval(
    coco_json,
    image_root,
    sam2_ckpt,
    sam2_cfg,
    device,
    start_id=0,
    end_id=None,
    max_images=0,
    output_dir=None,
    gra_tag=None,
    stability_score=0.9,
    pred_iou_thresh=0.77,
    mask_threshold=0.0,
    model_scale="small",
):
    isegm_utils = None
    Clicker = None
    get_sam_input = None
    try:
        from isegm.inference import utils as isegm_utils
        from isegm.inference.clicker import Clicker
        from isegm.inference.evaluation import get_sam_input
    except ImportError:
        print("isegm not installed; falling back to centroid click.")

    def _mask_iou(a, b):
        if a is None or b is None:
            return 0.0
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    def _centroid_click(mask):
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            return None, None
        cx = int(xs.mean())
        cy = int(ys.mean())
        return np.array([[cx, cy]], dtype=np.float32), np.array([1], dtype=np.int64)

    _add_sam2_to_sys_path()

    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann.get("image_id"), []).append(ann)

    start_id = max(start_id, 0)
    end_id = len(images) if end_id is None else min(end_id, len(images))

    mask_generator = build_unsamv2_mask_generator(
        sam2_ckpt,
        stability_score=stability_score,
        pred_iou_thresh=pred_iou_thresh,
        mask_threshold=mask_threshold,
        device=device,
        model_scale=model_scale,
    )
    gra_value = float(gra_tag) if gra_tag is not None else 1.0
    print(gra_value)
    predictor = UnSAMv2Predictor(
        mask_generator,
        gra=gra_value,
        amp=True,
        profile_time=False,
    )

    total_ious = []
    total_objects = 0
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for idx in range(start_id, end_id):
        if max_images and (idx - start_id) >= max_images:
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

        outputs = predictor([{"image": image}])
        instances = outputs[0]["instances"]
        if hasattr(instances, "to"):
            instances = instances.to("cpu")
        pred_masks_all = []
        if instances.has("pred_masks"):
            pred_masks_all = instances.pred_masks.tensor.numpy().astype(bool)
        anns = ann_by_image.get(img_id, [])
        vis_items = []
        vis_masks = []
        vis_boxes = []
        vis_points = []
        for ann in anns:
            gt_mask = None
            if ann.get("segmentation") is not None:
                gt_mask = _decode_coco_mask(ann.get("segmentation"), height, width)
            if gt_mask is None:
                bbox = ann.get("bbox")
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(width, int(x + w))
                    y2 = min(height, int(y + h))
                    if x2 > x1 and y2 > y1:
                        gt_mask = np.zeros((height, width), dtype=bool)
                        gt_mask[y1:y2, x1:x2] = True
            if gt_mask is None or gt_mask.sum() == 0:
                continue

            if Clicker is not None and get_sam_input is not None:
                clicker = Clicker(gt_mask=gt_mask)
                pred_mask = np.zeros_like(gt_mask, dtype=bool)
                clicker.make_next_click(pred_mask)
                point_coords, point_labels = get_sam_input(clicker)
            else:
                point_coords, point_labels = _centroid_click(gt_mask)
                if point_coords is None:
                    continue
            point_xy = None
            if output_dir and point_coords is not None and len(point_coords) > 0:
                px, py = point_coords[0]
                point_xy = (int(px), int(py))
                vis_points.append(point_xy)

            pred_mask = None
            if len(pred_masks_all) > 0:
                best_iou = -1.0
                best_mask = None
                for cand_mask in pred_masks_all:
                    iou = _mask_iou(gt_mask, cand_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = cand_mask
                pred_mask = best_mask
            if pred_mask is None:
                pred_mask = np.zeros_like(gt_mask, dtype=bool)
            if isegm_utils is not None:
                iou = isegm_utils.get_iou(gt_mask, pred_mask)
            else:
                iou = _mask_iou(gt_mask, pred_mask)
            total_ious.append(iou)
            total_objects += 1
            if output_dir:
                vis_mask = pred_mask.astype(bool)
                vis_masks.append(vis_mask)
                box = _mask_to_xyxy_bbox(pred_mask)
                if box is not None:
                    vis_boxes.append(box)
                vis_items.append(
                    {
                        "mask": vis_mask,
                        "box": box,
                        "point": point_xy,
                    }
                )

        if output_dir and vis_masks:
            colors = [(0, 255, 0)] * len(vis_masks)  # strong green for masks
            bbox_colors = [(0, 255, 0)] * len(vis_masks)

            image_pil = Image.fromarray(image)
            vis_mask = _overlay_masks(image_pil, vis_masks, colors, alpha=0.7)
            vis_mask = _draw_mask_boundaries(vis_mask, vis_masks, colors, thickness=4, darken=0.6)
            vis_bbox = image_pil.copy()
            if vis_boxes:
                vis_bbox = _draw_boxes(vis_bbox, vis_boxes, bbox_colors, width=3)
            vis_full = vis_mask.copy()
            if vis_boxes:
                vis_full = _draw_boxes(vis_full, vis_boxes, bbox_colors, width=3)
            vis_point = image_pil.copy()

            if vis_points:
                for target in (vis_bbox, vis_mask, vis_full, vis_point):
                    drawer = ImageDraw.Draw(target)
                    for px, py in vis_points:
                        r = 4
                        drawer.ellipse(
                            (px - r, py - r, px + r, py + r),
                            outline=(0, 255, 0),
                            fill=(0, 255, 0),
                            width=2,
                        )

            stem = Path(file_name).stem
            gra_suffix = f"_gra{gra_tag}" if gra_tag is not None else ""
            _save_png_fast(image_pil, os.path.join(output_dir, f"{stem}_orig{gra_suffix}.png"))
            _save_png_fast(vis_bbox, os.path.join(output_dir, f"{stem}_bbox{gra_suffix}.png"))
            _save_png_fast(vis_mask, os.path.join(output_dir, f"{stem}_mask{gra_suffix}.png"))
            _save_png_fast(vis_full, os.path.join(output_dir, f"{stem}_full{gra_suffix}.png"))
            _save_png_fast(vis_point, os.path.join(output_dir, f"{stem}_point{gra_suffix}.png"))

            if len(vis_items) > 1:
                for idx_item, item in enumerate(vis_items, start=1):
                    per_mask = item["mask"]
                    per_box = item["box"]
                    per_point = item["point"]
                    per_mask_img = _overlay_masks(image_pil, [per_mask], [(0, 255, 0)], alpha=0.7)
                    per_mask_img = _draw_mask_boundaries(
                        per_mask_img, [per_mask], [(0, 255, 0)], thickness=4, darken=0.6
                    )
                    per_bbox_img = image_pil.copy()
                    if per_box is not None:
                        per_bbox_img = _draw_boxes(per_bbox_img, [per_box], [(0, 255, 0)], width=3)
                    per_full_img = per_mask_img.copy()
                    if per_box is not None:
                        per_full_img = _draw_boxes(per_full_img, [per_box], [(0, 255, 0)], width=3)

                    per_point_img = image_pil.copy()
                    if per_point is not None:
                        for target in (per_bbox_img, per_mask_img, per_full_img, per_point_img):
                            drawer = ImageDraw.Draw(target)
                            px, py = per_point
                            r = 4
                            drawer.ellipse(
                                (px - r, py - r, px + r, py + r),
                                outline=(0, 255, 0),
                                fill=(0, 255, 0),
                                width=2,
                            )

                    per_suffix = f"_{idx_item}"
                    _save_png_fast(
                        per_bbox_img, os.path.join(output_dir, f"{stem}{per_suffix}_bbox{gra_suffix}.png")
                    )
                    _save_png_fast(
                        per_mask_img, os.path.join(output_dir, f"{stem}{per_suffix}_mask{gra_suffix}.png")
                    )
                    _save_png_fast(
                        per_full_img, os.path.join(output_dir, f"{stem}{per_suffix}_full{gra_suffix}.png")
                    )
                    _save_png_fast(
                        per_point_img, os.path.join(output_dir, f"{stem}{per_suffix}_point{gra_suffix}.png")
                    )

    if total_objects == 0:
        print("No valid objects found for evaluation.")
        return

    mean_iou = float(np.mean(total_ious))
    print(f"One-click mIoU: {mean_iou:.4f} over {total_objects} objects")


def visualize_single_image(image_path, predictor, output_path, seed=7):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    outputs = predictor(image_np[:, :, ::-1])
    instances = outputs["instances"].to("cpu")
    rng = random.Random(seed)

    num_instances = len(instances)
    colors = [
        (rng.randint(30, 225), rng.randint(30, 225), rng.randint(30, 225))
        for _ in range(num_instances)
    ]

    overlay = image
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()
        overlay = _overlay_masks(overlay, masks, colors, alpha=0.35)

    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor.numpy()
        overlay = _draw_boxes(overlay, boxes, colors, width=2)

    overlay.save(output_path)


def build_unsamv2_mask_generator(
    ckpt_path,
    stability_score=0.9,
    pred_iou_thresh=0.77,
    mask_threshold=0,
    device="cuda",
    model_scale="small",
):
    _add_sam2_to_sys_path()
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    model_cfgs = {
        "small": "configs/unsamv2_small.yaml",
        "large": "configs/unsamv2_large.yaml",
    }
    if model_scale not in model_cfgs:
        raise ValueError(f"Unsupported model_scale: {model_scale}")
    model_cfg = model_cfgs[model_scale]
    sam2 = build_sam2(model_cfg, ckpt_path, device=device, apply_postprocessing=True)
    return SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=512,
        mask_threshold=mask_threshold,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=True,
    )


class UnSAMv2Predictor:
    def __init__(self, mask_generator, gra, amp=True, profile_time=False):
        self.mask_generator = mask_generator
        self.gra = gra
        self.amp = amp
        self.profile_time = profile_time

    def eval(self):
        return self

    def __call__(self, batched_inputs):
        # self.mask_generator.eval()
        outputs = []
        for entry in batched_inputs:
            t0 = time.perf_counter()
            if "image" in entry:
                t_read_start = time.perf_counter()
                image = entry["image"]
                if torch.is_tensor(image):
                    image_np = image.permute(1, 2, 0).cpu().numpy()
                else:
                    image_np = np.asarray(image)
                t_read_end = time.perf_counter()
            else:
                t_read_start = time.perf_counter()
                image_path = entry["file_name"]
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image)
                t_read_end = time.perf_counter()
            with torch.no_grad():
                t_model_start = time.perf_counter()
                if torch.cuda.is_available() and self.amp:
                    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        raw_masks = self.mask_generator.generate(image_np, gra=self.gra)
                else:
                    raw_masks = self.mask_generator.generate(image_np, gra=self.gra)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_model_end = time.perf_counter()
            t_post_start = time.perf_counter()
            height, width = image_np.shape[:2]
            if raw_masks:
                masks = [m["segmentation"] for m in raw_masks]
                scores = [
                    float(m.get("predicted_iou", m.get("stability_score", 1.0)))
                    for m in raw_masks
                ]
                boxes_xyxy = []
                filtered_masks = []
                filtered_scores = []
                for mask, score in zip(masks, scores):
                    box = _mask_to_xyxy_bbox(mask)
                    if box is None:
                        continue
                    boxes_xyxy.append(box)
                    filtered_masks.append(mask)
                    filtered_scores.append(score)
                if boxes_xyxy:
                    mask_tensor = torch.stack(
                        [torch.from_numpy(m.astype(np.uint8)) for m in filtered_masks]
                    )
                    instances = Instances((height, width))
                    instances.pred_boxes = Boxes(torch.tensor(boxes_xyxy, dtype=torch.float32))
                    instances.scores = torch.tensor(filtered_scores, dtype=torch.float32)
                    instances.pred_classes = torch.zeros(len(boxes_xyxy), dtype=torch.int64)
                    instances.pred_masks = BitMasks(mask_tensor)
                else:
                    instances = Instances((height, width))
                    instances.pred_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
                    instances.scores = torch.zeros((0,), dtype=torch.float32)
                    instances.pred_classes = torch.zeros((0,), dtype=torch.int64)
                    instances.pred_masks = BitMasks(
                        torch.zeros((0, height, width), dtype=torch.uint8)
                    )
            else:
                instances = Instances((height, width))
                instances.pred_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
                instances.scores = torch.zeros((0,), dtype=torch.float32)
                instances.pred_classes = torch.zeros((0,), dtype=torch.int64)
                instances.pred_masks = BitMasks(
                    torch.zeros((0, height, width), dtype=torch.uint8)
                )
            outputs.append({"instances": instances})
            t_post_end = time.perf_counter()
            if self.profile_time:
                t_total = t_post_end - t0
                t_read = t_read_end - t_read_start
                t_model = t_model_end - t_model_start
                t_post = t_post_end - t_post_start
                print(
                    f"[profile] image_id={entry.get('image_id', 'N/A')} "
                    f"read={t_read:.4f}s model={t_model:.4f}s post={t_post:.4f}s total={t_total:.4f}s",
                    flush=True,
                )
        return outputs


def _evaluate_predictions_on_coco_agnostic(
    coco_gt,
    coco_results,
    iou_type,
    kpt_oks_sigmas=None,
    use_fast_impl=True,
    img_ids=None,
    max_dets_per_image=None,
):
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(coco_gt, coco_dt, iou_type)
    coco_eval.params.useCats = 0
    if max_dets_per_image is None:
        max_dets_per_image = [1, 10, 100]
    else:
        assert (
            len(max_dets_per_image) >= 3
        ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
        if max_dets_per_image[2] != 100:
            coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)
            coco_eval.params.useCats = 0
    if iou_type != "keypoints":
        coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    if iou_type == "keypoints":
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params, "kpt_oks_sigmas"), "pycocotools is too old!"
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
        num_keypoints_gt = len(next(iter(coco_gt.anns.values()))["keypoints"]) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
            f"[COCOEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
            f"COCO ground truth contains {num_keypoints_gt} keypoints. "
            f"They have to agree with each other. For meaning of OKS, please refer to "
            "http://cocodataset.org/#keypoints-eval."
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


class COCOEvaluatorAgnostic(COCOEvaluator):
    def __init__(self, *args, save_every=100, resume_from_results=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._save_every = max(int(save_every), 0)
        self._resume_from_results = bool(resume_from_results)
        self._results_file = None
        self._progress_file = None
        self._existing_coco_results = []
        self._existing_processed_image_ids = []
        self._processed_image_ids = []
        self._processed_since_last_flush = 0
        if self._output_dir:
            self._results_file = os.path.join(self._output_dir, "coco_instances_results.json")
            self._progress_file = os.path.join(self._output_dir, "coco_instances_progress.json")
        self._load_existing_state()

    def _load_existing_state(self):
        self._existing_coco_results = []
        self._existing_processed_image_ids = []
        if not self._resume_from_results or not self._results_file or not os.path.exists(self._results_file):
            return
        try:
            with PathManager.open(self._results_file, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                self._existing_coco_results = payload
        except Exception as exc:
            self._logger.warning("Failed to load existing results from %s: %s", self._results_file, exc)
            self._existing_coco_results = []

        progress_ids = []
        if self._progress_file and os.path.exists(self._progress_file):
            try:
                with PathManager.open(self._progress_file, "r") as f:
                    progress_payload = json.load(f)
                if isinstance(progress_payload, dict):
                    raw_ids = progress_payload.get("processed_image_ids", [])
                    if isinstance(raw_ids, list):
                        for image_id in raw_ids:
                            try:
                                progress_ids.append(int(image_id))
                            except (TypeError, ValueError):
                                continue
            except Exception as exc:
                self._logger.warning(
                    "Failed to load progress file %s, fallback to inferring from results: %s",
                    self._progress_file,
                    exc,
                )

        if progress_ids:
            self._existing_processed_image_ids = progress_ids
        else:
            seen = set()
            ordered_ids = []
            for item in self._existing_coco_results:
                image_id = item.get("image_id")
                try:
                    image_id = int(image_id)
                except (TypeError, ValueError):
                    continue
                if image_id in seen:
                    continue
                seen.add(image_id)
                ordered_ids.append(image_id)
            self._existing_processed_image_ids = ordered_ids

        if self._existing_coco_results:
            self._logger.info(
                "Loaded %d existing coco results and %d processed image ids from %s",
                len(self._existing_coco_results),
                len(self._existing_processed_image_ids),
                self._results_file,
            )

    def get_resume_image_ids(self):
        return list(self._existing_processed_image_ids)

    def _flatten_predictions(self):
        return list(itertools.chain(*[x.get("instances", []) for x in self._predictions]))

    def _combined_processed_image_ids(self):
        seen = set()
        combined = []
        for image_id in self._existing_processed_image_ids + self._processed_image_ids:
            if image_id in seen:
                continue
            seen.add(image_id)
            combined.append(image_id)
        return combined

    def _combined_coco_results(self):
        return self._existing_coco_results + self._flatten_predictions()

    def _normalize_category_ids(self, coco_results):
        if not hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            return
        dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        contiguous_ids = set(reverse_id_mapping.keys())
        dataset_ids = set(dataset_id_to_contiguous_id.keys())
        for result in coco_results:
            category_id = result.get("category_id")
            if category_id in contiguous_ids:
                result["category_id"] = reverse_id_mapping[category_id]
            elif category_id in dataset_ids:
                continue
            else:
                raise AssertionError(
                    f"Unexpected category_id={category_id}. "
                    f"Expected contiguous ids {sorted(contiguous_ids)} or dataset ids {sorted(dataset_ids)}."
                )

    def _flush_partial_results(self, force=False):
        if not self._results_file:
            return
        if not force:
            if self._save_every <= 0:
                return
            if self._processed_since_last_flush < self._save_every:
                return
        coco_results = self._combined_coco_results()
        self._normalize_category_ids(coco_results)
        os.makedirs(self._output_dir, exist_ok=True)
        with PathManager.open(self._results_file, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()
        if self._progress_file:
            progress_payload = {
                "processed_image_ids": self._combined_processed_image_ids(),
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
            with PathManager.open(self._progress_file, "w") as f:
                f.write(json.dumps(progress_payload))
                f.flush()
        self._processed_since_last_flush = 0

    def reset(self):
        super().reset()
        self._processed_image_ids = []
        self._processed_since_last_flush = 0
        self._load_existing_state()

    def process(self, inputs, outputs):
        super().process(inputs, outputs)
        for entry in inputs:
            image_id = entry.get("image_id")
            try:
                image_id = int(image_id)
            except (TypeError, ValueError):
                continue
            self._processed_image_ids.append(image_id)
            self._processed_since_last_flush += 1
        self._flush_partial_results(force=False)

    def evaluate(self, img_ids=None):
        if len(getattr(self, "_predictions", [])) == 0 and self._existing_coco_results:
            self._predictions = [{"instances": []}]
        try:
            return super().evaluate(img_ids=img_ids)
        except TypeError:
            return super().evaluate()

    def _eval_predictions(self, predictions, img_ids=None):
        self._logger.info("Preparing results for COCO format ...")
        coco_results = self._combined_coco_results()
        self._normalize_category_ids(coco_results)
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        self._flush_partial_results(force=True)
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco_agnostic(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None
            )

            res = self._derive_coco_results(coco_eval, task, class_names=None)
            self._results[task] = res


def _skip_processed_prefix(dataset_dicts, processed_image_ids):
    if not processed_image_ids:
        return dataset_dicts, 0
    processed_set = set(processed_image_ids)
    skip_count = 0
    for item in dataset_dicts:
        image_id = item.get("image_id")
        try:
            image_id = int(image_id)
        except (TypeError, ValueError):
            break
        if image_id in processed_set:
            skip_count += 1
            continue
        break
    return dataset_dicts[skip_count:], skip_count


def load_categories(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [cat.get("name", str(cat.get("id"))) for cat in data.get("categories", [])]


# with open("/path/to/datasets/pidray/pidray/annotations/xray_train.json", "r", encoding="utf-8") as f:
#         data = json.load(f)
def register_dataset(name, json_path, image_root):
    # if name in DatasetCatalog.list():
    #     return
    register_coco_instances(name, {}, json_path, image_root)
    classes = load_categories(json_path)
    if classes:
        MetadataCatalog.get(name).thing_classes = classes


def build_cfg(weights_path, cfg_path, score_thresh, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def resolve_dataset_paths(dataset, val_json, image_root, start_id=0, end_id=1):
    if image_root is None:
        image_root = XrayImgRoot[dataset]
    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"Image root for dataset '{dataset}' does not exist: {image_root}"
        )

    if val_json is None:
        dataset_root = XrayRoot[dataset]
        dataset_specific_candidates = {
            "pidray": [
                os.path.join(
                    dataset_root,
                    "annotations",
                    "instances_test2017_agnostic.json",
                ),
            ],
        }
        candidates = [
            *dataset_specific_candidates.get(dataset, []),
            os.path.join(dataset_root, "val.json"),
            os.path.join(dataset_root, "default_coco_xray_v2.json"),
            os.path.join(dataset_root, "coco_agnostic.json"),
            os.path.join(dataset_root, "annotations", "coco_agnostic.json"),
        ]
        val_json = next((path for path in candidates if os.path.isfile(path)), None)
        if val_json is None:
            candidate_list = "\n  - ".join(candidates)
            raise FileNotFoundError(
                f"No annotation JSON found for dataset '{dataset}'. Tried:\n"
                f"  - {candidate_list}"
            )
        print(f"origin val_json: {val_json}")
        with open(val_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = _ensure_coco_meta(data, dataset=dataset)
        num_images = len(data["images"])
        effective_end = num_images if end_id is None else min(end_id, num_images)
        if start_id != 0 or effective_end != num_images:
            data["images"] = data["images"][start_id:end_id]
            valid_image_ids = {img["id"] for img in data["images"]}
            data["annotations"] = [
                ann for ann in data["annotations"] if ann["image_id"] in valid_image_ids
            ]
            slice_end = start_id + len(data["images"])
            sliced_json_path = os.path.join(
                "annotations",
                f"temp_{dataset}_val_{start_id}_{slice_end}.json",
            )
            os.makedirs(os.path.dirname(sliced_json_path), exist_ok=True)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=os.path.dirname(sliced_json_path),
                    prefix=f".{Path(sliced_json_path).name}.",
                    suffix=".tmp",
                    delete=False,
                ) as sf:
                    json.dump(data, sf)
                    temp_path = sf.name
                os.replace(temp_path, sliced_json_path)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            val_json = sliced_json_path
        print(val_json)
    elif not os.path.isfile(val_json):
        raise FileNotFoundError(
            f"Annotation JSON for dataset '{dataset}' does not exist: {val_json}"
        )
    return val_json, image_root


def run_evaluation(cfg, dataset_name, output_dir, batch_size, num_workers):
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=output_dir)
    data_loader = build_detection_test_loader(
        cfg,
        dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = DefaultPredictor(cfg).model
    data_iter = tqdm(data_loader, total=len(data_loader), desc="Evaluating", unit="batch")
    return inference_on_dataset(model, data_iter, evaluator)


def build_coco_evaluator(
    dataset_name,
    cfg,
    output_dir,
    tasks,
    task_agnostic=False,
    save_every=100,
    resume_from_results=True,
):
    try:
        evaluator_cls = COCOEvaluatorAgnostic if task_agnostic else COCOEvaluator
        if task_agnostic:
            return evaluator_cls(
                dataset_name,
                cfg,
                False,
                output_dir=output_dir,
                tasks=tuple(tasks),
                save_every=save_every,
                resume_from_results=resume_from_results,
            )
        return evaluator_cls(dataset_name, cfg, False, output_dir=output_dir, tasks=tuple(tasks))
    except TypeError:
        evaluator_cls = COCOEvaluatorAgnostic if task_agnostic else COCOEvaluator
        if task_agnostic:
            return evaluator_cls(
                dataset_name,
                tuple(tasks),
                False,
                output_dir=output_dir,
                save_every=save_every,
                resume_from_results=resume_from_results,
            )
        return evaluator_cls(dataset_name, tuple(tasks), False, output_dir=output_dir)


def unsamv2_mapper(dataset_dict):
    dataset_dict = dataset_dict.copy()
    dataset_dict["image"] = d2_utils.read_image(dataset_dict["file_name"], format="RGB")
    return dataset_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="det_compass_syn_100",
        choices=["det_compass_xp", "det_compass_syn_100", "det_compass_syn_312", "PIXray", "CLCXray", "DvXray", "HiXray", "pidray", "crop_aug_easy", "crop_aug_hard", "crop_aug_extreme"],
    )
    parser.add_argument("--method", default="unsamv2", choices=["detectron2", "unsamv2"])
    parser.add_argument("--weights", default=None, help="Path to model weights")
    parser.add_argument(
        "--config",
        default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        help="Detectron2 model zoo config",
    )
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--val_json", default=None, help="COCO val json path")
    parser.add_argument("--image-root", default=None, help="Image root for val dataset")
    parser.add_argument("--output-dir", default="./output_eval")
    parser.add_argument("--visualize-image", default=None, help="Single image path for visualization")
    parser.add_argument("--visualize-out", default="vis_single.png")
    parser.add_argument("--image-dir", default=None, help="Directory of images for visualization")
    parser.add_argument(
        "--selected-mask-image",
        default="",
        help="Target image (path, filename, or stem) for selected mask-id overlay visualization.",
    )
    parser.add_argument(
        "--selected-mask-ids",
        default="",
        help="Comma-separated mask ids for one-image overlay, based on score-ranked predictions.",
    )
    parser.add_argument(
        "--selected-mask-id-base",
        type=int,
        default=1,
        choices=[0, 1],
        help="Index base for --selected-mask-ids (1 or 0).",
    )
    parser.add_argument(
        "--selected-mask-map",
        default="",
        help=(
            "Image-to-mask-id mapping for fusion overlays. "
            "Format: 'imgA.png:1,3,5;imgB.png:2,4'. "
            "Image key supports absolute path, filename, or stem."
        ),
    )
    parser.add_argument("--sam2-ckpt", default=None, help="Path to trained UnSAMv2 checkpoint")
    parser.add_argument("--sam2-cfg", default="configs/unsamv2_small.yaml", help="SAM2 config path")
    parser.add_argument(
        "--ckpt-suffix",
        default="",
        help="Optional checkpoint suffix for output naming. If empty, it can be inferred from --sam2-ckpt.",
    )
    parser.add_argument("--gra", default="1.0", help="Granularity for UnSAMv2")
    parser.add_argument(
        "--eval-mode",
        default="seg",
        help="Evaluation mode for UnSAMv2: bbox, seg, or bbox,seg",
    )
    parser.add_argument(
        "--task-agnostic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate without category matching (class-agnostic COCO eval)",
    )
    parser.add_argument("--test_batch", type=int, default=1, help="IMS_PER_BATCH for evaluation")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AMP for UnSAMv2 inference",
    )
    parser.add_argument("--stability-score", type=float, default=0.7)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.77) # 0.5
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    
    parser.add_argument(
        "--model-scale",
        default="small",
        choices=["small", "large"],
        help="UnSAMv2 model scale for SAM2 backbone.",
    )
    parser.add_argument(
        "--profile-time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-image timing breakdown during inference.",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=1, help="Evaluation random seed")
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Training seed of the evaluated checkpoint (provenance only)",
    )
    parser.add_argument(
        "--train-data-profile",
        default="",
        help="Training-data scale/profile of the evaluated checkpoint (provenance only)",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic PyTorch/CUDA algorithms",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="CUDA device index for single-process evaluation.",
    )
    parser.add_argument("--start_id", type=int, default=0, help="Start index (0-based) for dataset slice")
    parser.add_argument("--end_id", type=int, default=None, help="End index (exclusive) for dataset slice")
    parser.add_argument(
        "--merge-gra",
        default=None,
        help="Comma-separated gra values to merge saved coco_instances_results.json for eval",
    )
    parser.add_argument(
        "--merge-results-files",
        default=None,
        help="Comma-separated paths to coco_instances_results.json (overrides --merge-gra)",
    )
    parser.add_argument(
        "--merge-output-subdir",
        default=None,
        help="Override output subdir for merged evaluation results",
    )
    parser.add_argument("--exist_result",type=str2bool,default=False,help="train or test")
    parser.add_argument("--comment",type=str,default="",help="train or test")
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save partial coco_instances_results.json every N processed samples (0 disables periodic save).",
    )
    parser.add_argument(
        "--resume-from-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume evaluation by loading existing results/progress files in the output directory.",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run interactive one-click evaluation.",
    )
    parser.add_argument(
        "--interactive-max-images",
        type=int,
        default=0,
        help="Limit number of images for interactive eval (0 = all).",
    )
    parser.add_argument(
        "--interactive-vis-dir",
        default="",
        help="Output directory for interactive visualizations.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--wandb-project", default="unsamv2-eval")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    
    return parser.parse_args()


def _parse_csv_values(value):
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_int_values(value):
    tokens = _parse_csv_values(value)
    out = []
    for token in tokens:
        try:
            out.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer value: {token}") from exc
    return out


def _parse_selected_mask_map(value):
    raw = str(value or "").strip()
    if not raw:
        return {}
    out = {}
    entries = [entry.strip() for entry in raw.split(";") if entry.strip()]
    for entry in entries:
        if ":" in entry:
            image_key, ids_part = entry.split(":", 1)
        elif "=" in entry:
            image_key, ids_part = entry.split("=", 1)
        else:
            raise ValueError(
                f"Invalid --selected-mask-map entry: {entry}. "
                "Expected '<image>:id1,id2,...'"
            )
        image_key = image_key.strip()
        ids_part = ids_part.strip().replace("，", ",")
        if not image_key:
            raise ValueError(f"Empty image key in --selected-mask-map entry: {entry}")
        ids = _parse_csv_int_values(ids_part)
        if not ids:
            continue
        out[image_key] = ids
    return out


def _metrics_from_coco_dict(metrics):
    out = {}
    if not isinstance(metrics, dict):
        return out
    for label in COCO_METRIC_ORDER:
        key = COCO_METRIC_KEYS.get(label)
        if key in metrics:
            try:
                out[label] = float(metrics[key])
            except (TypeError, ValueError):
                continue
    return out


def _metrics_from_coco_stats(stats, prefixes=None):
    if stats is None or len(stats) < len(COCO_METRIC_ORDER):
        return {}
    out = {}
    for idx, label in enumerate(COCO_METRIC_ORDER):
        if prefixes and not any(label.startswith(prefix) for prefix in prefixes):
            continue
        try:
            out[label] = float(stats[idx])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _wandb_init(args, extra_config=None):
    if not args.wandb or wandb is None or not hasattr(wandb, "init"):
        return
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
    )
    cfg = vars(args).copy()
    if extra_config:
        cfg.update(extra_config)
    wandb.config.update(cfg, allow_val_change=True)


def _wandb_log_metrics(task, metrics):
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    payload = {f"{task}/{label}": value for label, value in metrics.items()}
    if payload:
        wandb.log(payload)


def _wandb_log_coco_eval(evaluator, prefixes=None):
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    coco_eval = getattr(evaluator, "_coco_eval", None)
    if not coco_eval:
        return
    for task_name, eval_obj in coco_eval.items():
        stats = getattr(eval_obj, "stats", None) if eval_obj is not None else None
        metrics = _metrics_from_coco_stats(stats, prefixes=prefixes)
        _wandb_log_metrics(task_name, metrics)


def _eval_merged_coco_results(evaluator, coco_results, img_ids=None):
    evaluator.reset()
    evaluator._eval_predictions([{"instances": coco_results}], img_ids=img_ids)
    return evaluator._results


def _resolve_ckpt_suffix(ckpt_suffix, sam2_ckpt):
    suffix = str(ckpt_suffix or "").strip()
    if suffix:
        return suffix
    ckpt_path = str(sam2_ckpt or "").strip()
    if not ckpt_path:
        return ""
    stem = Path(ckpt_path).stem.strip()
    if not stem:
        return ""
    stem_lower = stem.lower()
    if stem_lower == "checkpoint":
        return ""
    if stem_lower.startswith("checkpoint_"):
        return stem[len("checkpoint_") :]
    if "unsamv2" in stem_lower:
        return stem
    return ""


def main():
    args = parse_args()
    init_seed_torch(args.seed, deterministic=args.deterministic)
    if args.device == "cuda" and args.device_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)

    if args.wandb and wandb is None:
        print("wandb not installed, skipping wandb logging.")
        args.wandb = False

    if args.image_dir:
        selected_mask_map = _parse_selected_mask_map(args.selected_mask_map)
        selected_mask_ids = _parse_csv_int_values(args.selected_mask_ids)
        selected_mask_image = str(args.selected_mask_image or "").strip()
        if selected_mask_ids and not selected_mask_image:
            raise ValueError("--selected-mask-image is required when --selected-mask-ids is set")
        if selected_mask_ids and selected_mask_image:
            selected_mask_map[selected_mask_image] = selected_mask_ids
        if args.method == "unsamv2":
            if not args.sam2_ckpt:
                raise ValueError("--sam2-ckpt is required when method is unsamv2")
            mask_generator = build_unsamv2_mask_generator(
                args.sam2_ckpt,
                stability_score=args.stability_score,
                pred_iou_thresh=args.pred_iou_thresh,
                mask_threshold=args.mask_threshold,
                device=args.device,
                model_scale=args.model_scale,
            )
            predictor = UnSAMv2Predictor(
                mask_generator,
                gra=float(args.gra),
                amp=args.amp,
                profile_time=args.profile_time,
            )
        else:
            if not args.weights:
                raise ValueError("--weights is required when method is detectron2")
            classes = ["object"]
            cfg = build_cfg(args.weights, args.config, args.score_thresh, len(classes))
            cfg.TEST.IMS_PER_BATCH = args.test_batch
            cfg.DATALOADER.NUM_WORKERS = args.num_workers
            predictor = DefaultPredictor(cfg)

        out_dir = os.path.join(
            args.output_dir,
            f"vis_{Path(args.image_dir).name}{args.comment}",
        )
        if args.method == "unsamv2":
            visualize_directory_interactive(
                args.image_dir,
                predictor,
                out_dir,
                score_thresh=args.score_thresh,
                gra_tag=args.gra,
                selected_mask_map=selected_mask_map,
                selected_mask_id_base=args.selected_mask_id_base,
            )
        else:
            visualize_directory_images(
                args.image_dir,
                predictor,
                out_dir,
                score_thresh=args.score_thresh,
                gra=args.gra,
            )
        print(f"Saved directory visualizations to {out_dir}")
        return

    val_json, image_root = resolve_dataset_paths(args.dataset, args.val_json, args.image_root, args.start_id, args.end_id)
    if args.interactive:
        if not args.sam2_ckpt:
            raise ValueError("--sam2-ckpt is required for interactive evaluation")
        vis_dir = args.interactive_vis_dir
        if not vis_dir:
            vis_dir = os.path.join(
                args.output_dir,
                f"interactive_vis_{args.dataset}_img{args.start_id}_{args.end_id}",
            )
        vis_dir = os.path.join(vis_dir, args.dataset)
        interactive_one_click_eval(
            coco_json=val_json,
            image_root=image_root,
            sam2_ckpt=args.sam2_ckpt,
            sam2_cfg=args.sam2_cfg,
            device=args.device,
            start_id=args.start_id,
            end_id=args.end_id,
            max_images=args.interactive_max_images,
            output_dir=vis_dir,
            gra_tag=args.gra,
            stability_score=args.stability_score,
            pred_iou_thresh=args.pred_iou_thresh,
            mask_threshold=args.mask_threshold,
            model_scale=args.model_scale,
        )
        return
    dataset_name = f"{args.dataset}_val"
    register_dataset(dataset_name, val_json, image_root)
    print(f"argsments: {args}")
    merge_enabled = bool(args.merge_gra or args.merge_results_files)
    if merge_enabled:
        gra_values = _parse_csv_values(args.merge_gra)
        merged_path = None
        if args.merge_results_files:
            merge_paths = _parse_csv_values(args.merge_results_files)
            if len(merge_paths) != 1:
                raise ValueError("--merge-results-files must contain exactly one fused result file")
            merged_path = merge_paths[0]
        else:
            tag = "_".join(gra_values) 
            output_subdir = f"gra_fusion_{args.dataset}_img{args.start_id}_{args.end_id}_{args.method}_{tag}{args.comment}"
            merged_path = os.path.join(args.output_dir, output_subdir, "coco_instances_results.json")
        if not os.path.exists(merged_path):
            raise FileNotFoundError(f"Missing coco results file: {merged_path}")
        _wandb_init(args, extra_config={"merge_results": merged_path})
        eval_mode = {m.strip().lower() for m in args.eval_mode.split(",") if m.strip()}
        tasks = []
        if "bbox" in eval_mode:
            tasks.append("bbox")
        if "seg" in eval_mode or "segm" in eval_mode:
            tasks.append("segm")
        if not tasks:
            raise ValueError("--eval-mode must include bbox, seg, or both")
        classes = MetadataCatalog.get(dataset_name).thing_classes
        weights_path = args.weights or ""
        cfg = build_cfg(weights_path, args.config, args.score_thresh, len(classes))
        cfg.TEST.IMS_PER_BATCH = args.test_batch
        cfg.DATALOADER.NUM_WORKERS = args.num_workers
        cfg.DATASETS.TEST = (dataset_name,)
        if args.merge_output_subdir:
            output_dir = os.path.join(args.output_dir, args.merge_output_subdir)
        else:
            output_dir = os.path.dirname(merged_path)
        os.makedirs(output_dir, exist_ok=True)
        evaluator = build_coco_evaluator(
            dataset_name,
            cfg,
            output_dir,
            tasks=tasks,
            task_agnostic=False,
        )
        # merged_path = "/path/to/LAO-X/output_eval/pixray_img0_5047_unsamv2_0.7/coco_instances_results.json"
        print("evaluating merged results from:", merged_path)
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        cocoGt = COCO(args.val_json)
        cocoDt = cocoGt.loadRes(merged_path)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        _wandb_log_metrics("bbox", _metrics_from_coco_stats(cocoEval.stats))
        # results = _eval_merged_coco_results(evaluator, coco_results)
        # print(results)
        return
    # if args.exist_result:
    #     from pycocotools.coco import COCO
    #     from pycocotools.cocoeval import COCOeval
    #     with open("/path/to/LAO-X/temp_det_compass_syn_100_val_0_5000.json", "r", encoding="utf-8") as f:
    #         coco_gt = json.load(f)
    #     test_loader = build_detection_test_loader(cfg, dataset_name)
    #     evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output_eval/{}_unsamv2_gra_fusion/".format(args.dataset))
    #     evaluator.reset()
    #     print("test_loader image:{}, object:{}".format(len(test_loader.dataset), len(coco_gt['annotations'])))
    #     cocoGt = COCO(os.path.join(XrayRoot[args.dataset], "coco_agnostic.json"))

    #     # 模型原结果
    #     cocoDt = cocoGt.loadRes("/path/to/ovod_xray/detectron2_series/xray_ovod_det2/sam2_eval_det_compass_syn_promptrcnn_05/coco_instances_results.json")

    #     # 按 bbox 评估
    #     cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    #     cocoEval.evaluate()
    #     cocoEval.accumulate()
    #     cocoEval.summarize()
    #     return
    if args.method == "unsamv2":
        if not args.sam2_ckpt:
            raise ValueError("--sam2-ckpt is required when method is unsamv2")
        _wandb_init(args, extra_config={"task": "unsamv2"})
        if torch.cuda.is_available() and not args.deterministic:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass
        gra_value = float(args.gra)
        ckpt_suffix_value = _resolve_ckpt_suffix(args.ckpt_suffix, args.sam2_ckpt)
        ckpt_suffix = f"_{ckpt_suffix_value}" if ckpt_suffix_value else ""
        output_subdir = (
            f"{args.dataset.lower()}_img{args.start_id}_{args.end_id}_{args.method}_{args.gra}"
            f"{args.comment}{ckpt_suffix}"
        )
        output_dir = os.path.join(args.output_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        _write_evaluation_provenance(output_dir, args)
        eval_mode = {m.strip().lower() for m in args.eval_mode.split(",") if m.strip()}
        tasks = []
        if "bbox" in eval_mode:
            tasks.append("bbox")
        if "seg" in eval_mode or "segm" in eval_mode:
            tasks.append("segm")
        if not tasks:
            raise ValueError("--eval-mode must include bbox, seg, or both")
        mask_generator = build_unsamv2_mask_generator(
            args.sam2_ckpt,
            stability_score=args.stability_score,
            device=args.device,
            model_scale=args.model_scale,
            mask_threshold=args.mask_threshold,

        )
        classes = MetadataCatalog.get(dataset_name).thing_classes
        # print(len(classes))
        cfg = build_cfg("", args.config, args.score_thresh, len(classes))
        cfg.TEST.IMS_PER_BATCH = args.test_batch
        cfg.DATALOADER.NUM_WORKERS = args.num_workers
        cfg.DATASETS.TEST = (dataset_name,)
        evaluator = build_coco_evaluator(
            dataset_name,
            cfg,
            output_dir,
            tasks,
            task_agnostic=args.task_agnostic,
            save_every=args.save_every,
            resume_from_results=args.resume_from_results,
        )
        dataset_dicts = DatasetCatalog.get(dataset_name)
        resume_image_ids = []
        if args.resume_from_results and hasattr(evaluator, "get_resume_image_ids"):
            resume_image_ids = evaluator.get_resume_image_ids()
            dataset_dicts, skip_count = _skip_processed_prefix(dataset_dicts, resume_image_ids)
            if skip_count > 0:
                print(
                    f"[resume] skip first {skip_count} samples, remaining {len(dataset_dicts)} "
                    f"(resume ids: {len(resume_image_ids)})"
                )
            else:
                print("[resume] no matching processed prefix, evaluate from current start_id.")

        if len(dataset_dicts) == 0 and resume_image_ids:
            print("[resume] all samples in range are already processed. Reusing saved results for evaluation.")
            evaluator.reset()
            results = evaluator.evaluate()
            print(results)
            if isinstance(results, dict):
                for task_name, metrics in results.items():
                    _wandb_log_metrics(task_name, _metrics_from_coco_dict(metrics))
            _wandb_log_coco_eval(evaluator, prefixes=("AR",))
            return

        data_loader = build_detection_test_loader(
            dataset_dicts,
            mapper=unsamv2_mapper,
            batch_size=args.test_batch,
            num_workers=args.num_workers,
        )
        data_iter = tqdm(data_loader, total=len(data_loader), desc="Evaluating", unit="batch")
        predictor = UnSAMv2Predictor(
            mask_generator,
            gra=gra_value,
            amp=args.amp,
            profile_time=args.profile_time,
        )
        results = inference_on_dataset(predictor, data_iter, evaluator)
        print(results)
        if isinstance(results, dict):
            for task_name, metrics in results.items():
                _wandb_log_metrics(task_name, _metrics_from_coco_dict(metrics))
        _wandb_log_coco_eval(evaluator, prefixes=("AR",))
        return

    if not args.weights:
        raise ValueError("--weights is required when method is detectron2")

    classes = MetadataCatalog.get(dataset_name).thing_classes
    cfg = build_cfg(args.weights, args.config, args.score_thresh, len(classes))
    cfg.TEST.IMS_PER_BATCH = args.test_batch
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATASETS.TEST = (dataset_name,)

    if args.visualize_image:
        predictor = DefaultPredictor(cfg)
        visualize_single_image(args.visualize_image, predictor, args.visualize_out)
        print(f"Saved visualization to {args.visualize_out}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    _wandb_init(args, extra_config={"task": "detectron2"})
    evaluator = build_coco_evaluator(
        dataset_name,
        cfg,
        args.output_dir,
        tasks=("bbox", "segm"),
        task_agnostic=args.task_agnostic,
    )
    dataset_dicts = DatasetCatalog.get(dataset_name)
    mapper = DatasetMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(
        dataset_dicts,
        mapper=mapper,
        batch_size=args.test_batch,
        num_workers=args.num_workers,
    )
    data_iter = tqdm(data_loader, total=len(data_loader), desc="Evaluating", unit="batch")
    model = DefaultPredictor(cfg).model
    results = inference_on_dataset(model, data_iter, evaluator)
    print(results)
    if isinstance(results, dict):
        for task_name, metrics in results.items():
            _wandb_log_metrics(task_name, _metrics_from_coco_dict(metrics))
    _wandb_log_coco_eval(evaluator, prefixes=("AR",))

    


if __name__ == "__main__":
    main()
