import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

from config import Root as DATA_ROOT

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.environ.get("LAOX_RESULTS_ROOT", os.path.join(REPO_ROOT, "results"))


def _add_sam2_to_sys_path() -> None:
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


def _load_culter_masks(json_path):
    data = json.loads(Path(json_path).read_text())
    annotations = data.get("annotations", [])
    masks = []
    metas = []
    for ann in annotations:
        seg = ann.get("segmentation")
        if not seg:
            continue
        mask = mask_utils.decode(seg).astype(bool)
        masks.append(mask)
        metas.append(
            {
                "granularity": ann.get("granularity"),
                "bbox": ann.get("bbox"),
            }
        )
    return masks, metas


def _draw_granularity_labels(image: Image.Image, metas):
    drawer = ImageDraw.Draw(image)
    for meta in metas:
        granularity = meta.get("granularity")
        bbox = meta.get("bbox")
        if granularity is None or not bbox:
            continue
        x, y = int(bbox[0]), int(bbox[1])
        text = f"granularity: {granularity:.3f}"
        drawer.text((x, y), text, fill=(255, 255, 255))
    return image


def _mask_to_xywh_bbox(mask):
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _decode_instance_mask(segmentation, height, width):
    if not segmentation:
        return None
    if isinstance(segmentation, dict):
        return mask_utils.decode(segmentation).astype(bool)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        return mask_utils.decode(rle).astype(bool)
    return None


def visualize_fusion(image_path, fusion_path, image_id, output_path, seed=7):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    data = json.loads(Path(fusion_path).read_text())
    image_id_int = int(image_id)
    dets = [d for d in data if d.get("image_id") == image_id_int]
    masks = []
    bboxes = []
    for det in dets:
        mask = _decode_instance_mask(det.get("segmentation"), height, width)
        if mask is None:
            continue
        masks.append(mask)
        bboxes.append(det.get("bbox"))
    rng = random.Random(seed)
    colors = [(rng.randint(30, 225), rng.randint(30, 225), rng.randint(30, 225)) for _ in masks]
    overlay = _overlay_masks(image, masks, colors, alpha=0.35)
    drawer = ImageDraw.Draw(overlay)
    for bbox, color in zip(bboxes, colors):
        if not bbox:
            continue
        x, y, w, h = bbox
        drawer.rectangle([x, y, x + w, y + h], outline=color, width=2)
    overlay.save(output_path)
    print("save to", output_path)


def visualize_gt(image_path, gt_json_path, image_id, output_path, seed=7):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    data = json.loads(Path(gt_json_path).read_text())
    annotations = data.get("annotations", [])
    image_id_int = int(image_id)
    masks = []
    bboxes = []
    for ann in annotations:
        if ann.get("image_id") != image_id_int:
            continue
        mask = _decode_instance_mask(ann.get("segmentation"), height, width)
        bboxes.append(ann.get("bbox"))
        if mask is None:
            continue
        masks.append(mask)
    rng = random.Random(seed)
    colors = [(rng.randint(30, 225), rng.randint(30, 225), rng.randint(30, 225)) for _ in bboxes]
    overlay = _overlay_masks(image, masks, colors, alpha=0.35)
    drawer = ImageDraw.Draw(overlay)
    for bbox, color in zip(bboxes, colors):
        if not bbox:
            continue
        x, y, w, h = bbox
        drawer.rectangle([x, y, x + w, y + h], outline=color, width=2)
    overlay.save(output_path, overwrite=True)
    print("save to", output_path)


def visualize_culter(image_path, json_path, output_path, seed=7):
    image = Image.open(image_path).convert("RGB")
    masks, metas = _load_culter_masks(json_path)
    rng = random.Random(seed)
    colors = [(rng.randint(30, 225), rng.randint(30, 225), rng.randint(30, 225)) for _ in masks]
    bboxes = [_mask_to_xywh_bbox(mask) for mask in masks]
    overlay = _overlay_masks(image, masks, colors)
    overlay = _draw_granularity_labels(overlay, metas)
    drawer = ImageDraw.Draw(overlay)
    for bbox, color in zip(bboxes, colors):
        if not bbox:
            continue
        x, y, w, h = bbox
        drawer.rectangle([x, y, x + w, y + h], outline=color, width=2)
    overlay.save(output_path)


def visualize_unsamv2(
    image_path,
    ckpt_path,
    output_path,
    gra=0.15,
    stability_score=0.9,
    device="cuda",
):
    _add_sam2_to_sys_path()
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    model_cfg = "configs/unsamv2_small.yaml"

    sam2 = build_sam2(model_cfg, ckpt_path, device=device, apply_postprocessing=True)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,  # 128
        mask_threshold=-1,
        pred_iou_thresh=0.77,
        stability_score_thresh=stability_score,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=True,
    )

    raw_masks = mask_generator.generate(image_np, gra=gra)
    masks = [m["segmentation"] for m in raw_masks]
    bboxes = [_mask_to_xywh_bbox(m) for m in masks]
    rng = random.Random(7)
    colors = [(rng.randint(30, 225), rng.randint(30, 225), rng.randint(30, 225)) for _ in masks]
    overlay = _overlay_masks(image, masks, colors, alpha=0.35)
    drawer = ImageDraw.Draw(overlay)
    for bbox, color in zip(bboxes, colors):
        if not bbox:
            continue
        x, y, w, h = bbox
        drawer.rectangle([x, y, x + w, y + h], outline=color, width=2)
    label = f"stability_score: {stability_score:.2f}"
    text_width = drawer.textlength(label)
    x = max(0, overlay.width - text_width - 6)
    y = 4
    drawer.text((x, y), label, fill=(255, 255, 255))
    overlay.save(output_path)
    print('save to', output_path)


def _collect_image_paths(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = []
    for root, _dirs, files in os.walk(image_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                paths.append(os.path.join(root, name))
    return sorted(paths)


def visualize_sam2_dir(
    image_dir,
    output_dir,
    sam2_ckpt,
    conf=0.25,
    device="cuda",
):
    from ultralytics import SAM

    os.makedirs(output_dir, exist_ok=True)
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    model = SAM(sam2_ckpt).to(device).eval()

    image_paths = _collect_image_paths(image_dir)
    if not image_paths:
        print("No images found in", image_dir)
        return

    rng = random.Random(7)
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        with torch.inference_mode():
            preds = model(image_path, conf=conf)
        if not preds or preds[0].masks is None:
            continue
        pred = preds[0]
        masks = pred.masks.data
        if masks is None:
            continue
        masks = [m.detach().cpu().numpy().astype(bool) for m in masks]
        bboxes = []
        scores = []
        if pred.boxes is not None:
            if getattr(pred.boxes, "xyxy", None) is not None:
                bboxes = pred.boxes.xyxy.detach().cpu().tolist()
            if getattr(pred.boxes, "conf", None) is not None:
                scores = pred.boxes.conf.detach().cpu().tolist()

        colors = [
            (rng.randint(30, 225), rng.randint(30, 225), rng.randint(30, 225))
            for _ in masks
        ]
        vis_mask = _overlay_masks(image, masks, colors, alpha=0.35)
        vis_bbox = image.copy()
        vis_full = vis_mask.copy()
        drawer_bbox = ImageDraw.Draw(vis_bbox)
        drawer_full = ImageDraw.Draw(vis_full)
        for i, (bbox, color) in enumerate(zip(bboxes, colors)):
            if not bbox:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            drawer_bbox.rectangle([x1, y1, x2, y2], outline=color, width=2)
            drawer_full.rectangle([x1, y1, x2, y2], outline=color, width=2)
            if i < len(scores):
                drawer_bbox.text((x1, max(0, y1 - 8)), f"{scores[i]:.2f}", fill=color)
                drawer_full.text((x1, max(0, y1 - 8)), f"{scores[i]:.2f}", fill=color)

        stem = Path(image_path).stem
        vis_bbox.save(os.path.join(output_dir, f"{stem}_bbox.png"))
        vis_mask.save(os.path.join(output_dir, f"{stem}_mask.png"))
        vis_full.save(os.path.join(output_dir, f"{stem}_full.png"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default=os.path.join(RESULTS_ROOT, "granularity_divide_and_conquer/vis/000010.json"),
        help="Path to granularity divide-and-conquer JSON",
    )
    parser.add_argument(
        "--ckpt",
        default=os.path.join(RESULTS_ROOT, "train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt"),
        help="Path to trained UnSAMv2 checkpoint",
    )
    parser.add_argument(
        "--gra",
        type=float,
        default=None,
        help="Granularity for UnSAMv2 (optional).",
    )
    parser.add_argument(
        "--stability-score",
        type=float,
        default=0.9,
        help="stability_score_thresh for UnSAMv2",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--image_id", default="000000", type=str)
    parser.add_argument("--start_id", default=0, type=int)
    parser.add_argument("--end_id", default=5000, type=int)
    parser.add_argument(
        "--dataset",
        default="det_compass_syn_100",
        choices=["det_compass", "det_compass_syn_100", "det_compass_syn_312", "PIXray", "det_compass_xp"],
    )
    parser.add_argument("--gras", required=True, help="Comma-separated list of gra values, e.g. 0.15,0.3,0.5")
    parser.add_argument("--method", default="unsamv2", choices=["detectron2", "unsamv2"])
    parser.add_argument("--comment",type=str,default="",help="train or test")
    parser.add_argument("--predict-dir", default="", help="Run SAM2 on all images in a directory.")
    parser.add_argument("--predict-out", default="./output_instances_vis", help="Output dir for predictions.")
    parser.add_argument("--sam2-ckpt", default="", help="SAM2 checkpoint for directory prediction.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for SAM2.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.predict_dir:
        if not args.sam2_ckpt:
            raise ValueError("--sam2-ckpt is required when using --predict-dir")
        visualize_sam2_dir(
            image_dir=args.predict_dir,
            output_dir=args.predict_out,
            sam2_ckpt=args.sam2_ckpt,
            conf=args.conf,
            device=args.device,
        )
        return
    gt_path = os.path.join(DATA_ROOT, args.dataset.lower(), "coco.json")
    args.image_id_idx = args.image_id
    if args.dataset == "PIXray":
        with open(gt_path, 'r') as f:
            gt_anno = json.load(f)
        args.image_id = gt_anno['images'][int(args.image_id)]['file_name'].split('/')[-1].split('.')[0]
    if args.dataset == "det_compass_syn_100":
        args.image = os.path.join(DATA_ROOT, args.dataset.lower(), "xray", f"{args.image_id}.png")
    elif args.dataset == "PIXray":
        args.image = os.path.join(DATA_ROOT, args.dataset.lower(), "PIXray", f"{args.image_id}.png")
    elif args.dataset == "det_compass_xp":
        args.image = os.path.join(DATA_ROOT, args.dataset.lower(), f"{args.image_id}.png")
    
    gt_out = f"./results/UnSAMv2_vis/{args.image_id}_gt.png"
    if os.path.exists(gt_path):
        os.makedirs(os.path.dirname(gt_out), exist_ok=True)
        visualize_gt(args.image, gt_path, args.image_id_idx, gt_out)
    else:
        print("gt_path not found:", gt_path)
    
    args.json = os.path.join(
        RESULTS_ROOT, "cutler", args.dataset.lower(), "granularity_divide_and_conquer/vis", f"{args.image_id}.json"
    )
    args.cutler_out = os.path.join(RESULTS_ROOT, "UnSAMv2_vis", f"{args.image_id}_culter.png")
    gra_list = [g.strip() for g in args.gras.split(",") if g.strip()]
    if len(gra_list) > 1:
        fusion_path = (
            "./output_eval/gra_fusion_{}_img{}_{}_{}_{}{}/coco_instances_results.json"
        ).format(args.dataset, args.start_id, args.end_id, args.method, '_'.join(gra_list), args.comment)
        fusion_out = f"./results/UnSAMv2_vis/{args.image_id}_fusion_gra_{'_'.join(gra_list)}{args.comment}.png"
    else:
        fusion_path = (
            "./output_eval/{}_img{}_{}_{}_{}{}/coco_instances_results.json"
        ).format(args.dataset.lower(), args.start_id, args.end_id, args.method, '_'.join(gra_list), args.comment)
        fusion_out = f"./results/UnSAMv2_vis/{args.image_id}_gra_{'_'.join(gra_list)}{args.comment}.png"
    os.makedirs(os.path.dirname(fusion_out), exist_ok=True)
    if os.path.exists(fusion_path):
        visualize_fusion(args.image, fusion_path, args.image_id_idx, fusion_out)
    else:
        print("fusion_path not found:", fusion_path)
if __name__ == "__main__":
    main()
