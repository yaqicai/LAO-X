#!/usr/bin/env python3
import os
import cv2
import sys
import json
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from pycocotools import mask as mask_utils
from util_xray import (
    DEFAULT_CATEGORIES,
    DEFAULT_INFO,
    DEFAULT_LICENSES,
    _bbox_iou_xyxy,
    generate_anno_per_image,
)


def parse_range_float(s):
    """解析 'min,max' 字符串为浮点数元组，如果是单个数值则返回 (val, val)"""
    try:
        parts = list(map(float, s.split(",")))
        if len(parts) == 1:
            return parts[0], parts[0]
        elif len(parts) == 2:
            return min(parts), max(parts)
        else:
            raise ValueError
    except:
        raise argparse.ArgumentTypeError(
            f"参数格式错误: '{s}'。应为 '值' 或 '最小值,最大值'"
        )


def parse_range_int(s):
    """解析 'min,max' 字符串为整数元组"""
    try:
        parts = list(map(int, s.split(",")))
        if len(parts) == 1:
            return parts[0], parts[0]
        elif len(parts) == 2:
            return min(parts), max(parts)
        else:
            raise ValueError
    except:
        raise argparse.ArgumentTypeError(
            f"参数格式错误: '{s}'。应为 '值' 或 '最小值,最大值'"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a COCO json into per-image annotation json files."
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
    parser.add_argument(
        "--img-root",
        default="",
        help="Optional root for image paths when COCO file_name is relative.",
    )
    parser.add_argument(
        "--dataset-gen", default="", help="Generated dataset name under config.Root."
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
        "--data-augmentation-output-dir",
        default="",
        help="Output directory for generated augmentation images.",
    )
    parser.add_argument(
        "--num-generated", type=int, default=0, help="Number of augmented images."
    )
    parser.add_argument(
        "--num-objects",
        type=parse_range_int,
        default="3,5",
        help="每张图物体数量范围，例如 '3,8' 或 '5'",
    )
    parser.add_argument(
        "--alpha",
        type=parse_range_float,
        default="0.1,1.0",
        help="相对密度范围 (0-1)，例如 '0.1,0.5'",
    )
    parser.add_argument(
        "--beta",
        type=parse_range_float,
        default="0.0,0.5",
        help="背景复杂度范围 (0-1)，例如 '0.2,0.8'",
    )
    parser.add_argument(
        "--gamma",
        type=parse_range_float,
        default="0.0,0.3",
        help="遮挡重叠率范围 (0-1)，例如 '0.0,0.4'",
    )
    parser.add_argument(
        "--iou-log",
        default="",
        help="Path to compare_coco_annotations.log for filtering.",
    )
    parser.add_argument(
        "--alpha-iou",
        type=float,
        default=None,
        help="Mask IoU threshold for filtering objects.",
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


def _resolve_image_path(file_name, img_root):
    if os.path.isabs(file_name):
        return file_name
    if img_root:
        return os.path.join(img_root, file_name)
    return file_name


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
    from config import Root, XrayRoot, XrayImgRoot

    return Root, XrayRoot, XrayImgRoot


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


def _resolve_input_paths(args, root, xray_root, xray_img_root):
    dataset_root = ""
    if args.dataset:
        dataset_root = xray_root.get(args.dataset) or _resolve_base_dir(
            root, args.dataset, args.stage
        )
    coco_json = _resolve_coco_json(dataset_root, args.coco_json, args.coco_name)
    img_root = args.img_root or xray_img_root.get(args.dataset) or dataset_root
    return coco_json, img_root


def load_iou_log(log_path):
    iou_map = {}
    if not log_path:
        return iou_map
    with open(log_path, "r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("total="):
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            try:
                image_id = int(row.get("image_id"))
                iou = float(row.get("mask_iou"))
            except (TypeError, ValueError):
                continue
            key = image_id
            prev = iou_map.get(key)
            if prev is None or iou > prev:
                iou_map[key] = iou
    return iou_map


def object_bank(coco, img_root="", iou_map=None, alpha_iou=None):
    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann.get("image_id")].append(ann)

    bank = []
    # 增加 padding，防止边缘羽化时被截断
    pad = 10

    for image_id, image in tqdm(images.items(), desc="object_bank"):
        anns = ann_by_image.get(image_id, [])
        if not anns:
            continue
        iou_val = None
        if iou_map is not None:
            iou_val = iou_map.get(image_id)

        file_name = image.get("file_name")
        if not file_name:
            continue
        img_path = _resolve_image_path(file_name, img_root)
        if not os.path.isfile(img_path):
            continue

        # 使用 CV2 读取，方便后续处理，最后转回 PIL
        # 注意：cv2 读取的是 BGR，需要转 RGB
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w, _ = img_cv.shape

        for ann in anns:
            if iou_map is not None and alpha_iou is not None:
                if iou_val is None or iou_val < alpha_iou:
                    continue
            seg = ann.get("segmentation")
            mask = _decode_mask(seg, h, w)  # Boolean mask
            if mask is None:
                continue

            # 转为 uint8
            mask_uint8 = mask.astype(np.uint8) * 255

            # --- 优化点 1: 计算包围盒 (带 Padding) ---
            ys, xs = np.where(mask)
            if ys.size == 0 or xs.size == 0:
                continue
            x1 = max(0, int(xs.min()) - pad)
            y1 = max(0, int(ys.min()) - pad)
            x2 = min(w, int(xs.max()) + 1 + pad)
            y2 = min(h, int(ys.max()) + 1 + pad)

            if x2 <= x1 or y2 <= y1:
                continue

            # 裁剪图像和 Mask
            crop_rgb = img_cv[y1:y2, x1:x2]
            crop_mask = mask_uint8[y1:y2, x1:x2]

            # --- 优化点 2: 边缘羽化 (Feathering) ---
            # 模拟 X-ray 物体边缘厚度逐渐变薄的物理特性
            # 1. 稍微腐蚀一点点，去掉标注时的边缘噪声
            # kernel = np.ones((3, 3), np.uint8)
            # crop_mask = cv2.erode(crop_mask, kernel, iterations=1)

            # 2. 高斯模糊，使边缘变软
            # Blur 半径取决于图像分辨率，通常 3-7 之间
            crop_mask_soft = cv2.GaussianBlur(crop_mask, (5, 5), 0)

            # --- 优化点 3: 背景清洗 (Background Cleaning) ---
            # 这一步非常关键！
            # 我们希望 Mask 之外的区域是纯黑/纯白，不要保留原图的背景噪声。
            # 将 RGB 图像中 Mask 为 0 的地方强制置为 0 (或者 255，取决于你的背景底色，这里设为0无所谓，因为Alpha通道会控制它不可见)
            # 但为了防止半透明区域颜色异常，最好保持原色，全靠 Alpha 通道控制。

            # 转换回 PIL RGBA
            crop_pil = Image.fromarray(crop_rgb)
            crop_alpha = Image.fromarray(crop_mask_soft)  # 使用模糊后的 Mask 作为 Alpha

            crop_rgba = crop_pil.convert("RGBA")
            crop_rgba.putalpha(crop_alpha)

            difficulty = 1.0 - iou_val if iou_val is not None else 0.5
            bank.append(
                {
                    "rgba": crop_rgba,
                    "size": crop_rgba.size,
                    "category_id": ann.get("category_id", 1),
                    "original_area": float(
                        mask.sum()
                    ),  # 记录原始面积，可用于过滤太小的碎片
                    "iou": iou_val,
                    "difficulty": difficulty,
                }
            )

    # # 可以在这里加一个过滤，去掉面积过小的碎片
    # bank = [b for b in bank if b["original_area"] > 100]

    return bank


def _xray_curve(x, k=3.0):
    x = np.clip(x, 0.0, 1.0)
    return 1.0 - np.exp(-k * x)


def data_augmentation(
    object_bank,
    image_paths,
    output_dir,
    num_generated,
    num_objects_range,  # 接收范围元组 (min, max)
    alpha_range,  # 接收范围元组 (min, max)
    beta_range,  # 接收范围元组 (min, max)
    gamma_range,  # 接收范围元组 (min, max)
    info,
    licenses,
    categories,
):
    if not object_bank or num_generated <= 0:
        return None
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(7)  # 保持随机种子以复现，或者移除以获得完全随机

    coco_out = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    ann_id = 1

    print(f"开始生成: 数量={num_generated}")
    print(
        f"参数范围: Objects={num_objects_range}, Density(α)={alpha_range}, Clutter(β)={beta_range}, Overlap(γ)={gamma_range}"
    )

    for idx in tqdm(range(num_generated), desc="data_augmentation"):
        # ==========================================
        # 1. 动态难度采样 (Dynamic Difficulty Sampling)
        # ==========================================
        # 针对当前这一张图，随机决定其难度参数
        # 这样生成的数据集既包含简单的图，也包含极难的图，符合真实分布
        current_alpha = rng.uniform(alpha_range[0], alpha_range[1])
        current_beta = rng.uniform(beta_range[0], beta_range[1])
        current_gamma = rng.uniform(gamma_range[0], gamma_range[1])
        current_num_objs = rng.randint(num_objects_range[0], num_objects_range[1])

        # ==========================================
        # 2. 背景合成
        # ==========================================
        if image_paths:
            bg_path = rng.choice(image_paths)
            bg = Image.open(bg_path).convert("RGBA")
        else:
            bg = Image.new("RGBA", (512, 512), (255, 255, 255, 255))

        bg_w, bg_h = bg.size

        # 使用采样到的 current_beta 控制背景杂波
        if current_beta > 0.1:
            noise_level = int(current_beta * 50)
            noise_arr = np.random.normal(0, noise_level, (bg_h, bg_w, 3)).astype(
                np.uint8
            )
            noise_img = Image.fromarray(noise_arr, mode="RGB").convert("RGBA")
            bg = Image.blend(bg, noise_img, alpha=0.2 * current_beta)

        bg_np = np.array(bg.convert("RGB"), dtype=np.float32) / 255.0

        image_masks = []
        placed_boxes = []
        occupancy = np.zeros((bg_h, bg_w), dtype=np.float32)
        occupancy_difficulty = np.zeros((bg_h, bg_w), dtype=np.float32)

        # ==========================================
        # 3. 物体放置
        # ==========================================
        for _ in range(current_num_objs):
            obj_data = rng.choice(object_bank)
            obj_img = obj_data["rgba"].copy()
            obj_difficulty = float(obj_data.get("difficulty", 0.5))
            obj_difficulty = max(0.0, min(1.0, obj_difficulty))

            # --- 密度模拟 (基于 current_alpha) ---
            r, g, b, a = obj_img.split()
            a_np = np.array(a).astype(float)

            # 基础因子 + 采样到的当前图密度因子
            # 如果 current_alpha 很小，物体就很透
            density_factor = 0.2 + 0.8 * current_alpha

            # 在图级别参数的基础上，给每个物体微小的个体差异 (+-10%)
            obj_specific_density = density_factor * rng.uniform(0.9, 1.1)
            obj_specific_density = min(1.0, max(0.05, obj_specific_density))

            a_np = a_np * obj_specific_density
            obj_img.putalpha(Image.fromarray(a_np.astype(np.uint8)))

            # 随机缩放
            scale = rng.uniform(0.8, 1.2)
            new_w = max(1, int(obj_img.width * scale))
            new_h = max(1, int(obj_img.height * scale))
            if new_w >= bg_w or new_h >= bg_h:
                continue
            obj_resized = obj_img.resize((new_w, new_h), Image.BILINEAR)

            # --- 遮挡判断 (基于 current_gamma) ---
            placed_bbox = None
            for _try in range(50):
                x = rng.randint(0, bg_w - new_w)
                y = rng.randint(0, bg_h - new_h)
                candidate_bbox = (x, y, x + new_w, y + new_h)

                collision = False
                for existing_bbox in placed_boxes:
                    iou = _bbox_iou_xyxy(candidate_bbox, existing_bbox)
                    # 使用当前图的 gamma 阈值
                    if iou > current_gamma:
                        collision = True
                        break

                if not collision:
                    placed_bbox = candidate_bbox
                    break

            if placed_bbox is None:
                continue

            # 融合：仅对重叠区域做深蓝色映射，难度系数走非线性曲线
            alpha_mask = np.array(obj_resized.split()[-1], dtype=np.float32) / 255.0
            obj_rgb = np.array(obj_resized.convert("RGB"), dtype=np.float32) / 255.0
            x1, y1, x2, y2 = placed_bbox
            region = bg_np[y1:y2, x1:x2]
            base = (
                region * (1.0 - alpha_mask[..., None]) + obj_rgb * alpha_mask[..., None]
            )

            occ_region = occupancy[y1:y2, x1:x2]
            overlap_strength = alpha_mask * occ_region
            if overlap_strength.max() > 0.0:
                occ_diff_region = occupancy_difficulty[y1:y2, x1:x2]
                combined_diff = 0.5 * (occ_diff_region + obj_difficulty)
                blue_strength = overlap_strength * _xray_curve(combined_diff, k=3.0)
                base_gray = base.mean(axis=2)
                blue_base = np.array([0.08, 0.18, 0.65], dtype=np.float32)
                blue_tint = (
                    blue_base[None, None, :]
                    * (0.35 + 0.65 * (1.0 - base_gray))[..., None]
                )
                base = (
                    base * (1.0 - blue_strength[..., None])
                    + blue_tint * blue_strength[..., None]
                )

            bg_np[y1:y2, x1:x2] = base
            occupancy[y1:y2, x1:x2] = np.maximum(occ_region, alpha_mask)
            occ_diff_region = occupancy_difficulty[y1:y2, x1:x2]
            diff_mask = alpha_mask > 0.05
            if diff_mask.any():
                updated_diff = occ_diff_region.copy()
                updated_diff[diff_mask] = np.maximum(
                    occ_diff_region[diff_mask], obj_difficulty
                )
                occupancy_difficulty[y1:y2, x1:x2] = updated_diff
            placed_boxes.append(placed_bbox)

            # 生成 Mask (强监督)
            alpha_mask_u8 = (alpha_mask * 255.0).astype(np.uint8)
            # 阈值设低一点，因为低 alpha 下像素值很小，但mask需要存在
            binary_mask = (alpha_mask_u8 > 5).astype(np.uint8)

            full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
            full_mask[
                placed_bbox[1] : placed_bbox[3], placed_bbox[0] : placed_bbox[2]
            ] = binary_mask
            image_masks.append((full_mask, obj_data.get("category_id", 1)))

        # ==========================================
        # 4. 保存与标注
        # ==========================================
        file_name = f"{idx:06d}.png"
        out_path = os.path.join(output_dir, "xray", file_name)
        Image.fromarray((bg_np * 255.0).clip(0, 255).astype(np.uint8)).save(out_path)

        coco_out["images"].append(
            {
                "id": idx,
                "file_name": file_name,
                "width": bg.width,
                "height": bg.height,
                # 记录每张图具体的随机参数，这对分析模型在不同难度下的表现非常有用！
                "difficulty_params": {
                    "alpha": round(current_alpha, 2),
                    "beta": round(current_beta, 2),
                    "gamma": round(current_gamma, 2),
                    "num_objects": current_num_objs,
                },
            }
        )

        for mask, cat_id in image_masks:
            if mask.max() == 0:
                continue
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("ascii")
            area = float(mask_utils.area(rle))
            bbox = mask_utils.toBbox(rle).tolist()
            coco_out["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": cat_id,
                    "area": area,
                    "bbox": bbox,
                    "segmentation": rle,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return coco_out


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
    root_dir, xray_root, xray_img_root = _load_config()

    if args.fusion:
        from util_xray import convert_to_class_agnostic
        convert_to_class_agnostic("/path/to/datasets/{}/{}/coco.json".format(args.dataset, args.stage))
        base_dir = _resolve_base_dir(root_dir, args.dataset, args.stage)
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

    coco_json, img_root = _resolve_input_paths(args, root_dir, xray_root, xray_img_root)
    if not coco_json:
        raise SystemExit("Missing --coco-json.")
    with open(coco_json, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if not images:
        raise SystemExit("No images found in COCO JSON.")

    if args.num_generated > 0:
        base_dir = (
            args.data_augmentation_output_dir
            or _resolve_base_dir(root_dir, args.dataset_gen or args.dataset, args.stage)
            or os.getcwd()
        )
        aug_dir = base_dir
        anno_dir = os.path.join(base_dir, "anno")
        iou_map = load_iou_log(args.iou_log) if args.iou_log else None
        bank = object_bank(
            coco, img_root=img_root, iou_map=iou_map, alpha_iou=args.alpha_iou
        )
        image_paths = [
            _resolve_image_path(img.get("file_name", ""), img_root)
            for img in images
            if img.get("file_name")
        ]
        info = coco.get("info") or DEFAULT_INFO
        licenses = coco.get("licenses") or DEFAULT_LICENSES
        categories = coco.get("categories") or DEFAULT_CATEGORIES
        coco_aug = data_augmentation(
            bank,
            image_paths,
            aug_dir,
            args.num_generated,
            args.num_objects,
            args.alpha,
            args.beta,
            args.gamma,
            info,
            licenses,
            categories,
        )
        if coco_aug is not None:
            aug_json_path = os.path.join(base_dir, "coco_agnostic.json")
            with open(aug_json_path, "w") as f:
                json.dump(coco_aug, f, indent=2, ensure_ascii=True)
            print(f"Saved augmented COCO JSON to {aug_json_path}")
            generate_anno_per_image(
                coco_aug,
                anno_dir,
                name_mode=args.name_mode,
                granularity_mode=args.granularity_mode,
                default_info=DEFAULT_INFO,
                default_licenses=DEFAULT_LICENSES,
                default_categories=DEFAULT_CATEGORIES,
            )
        print(f"Saved {args.num_generated} augmented images to {aug_dir}")
    else:
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
