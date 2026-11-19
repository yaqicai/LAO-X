import argparse
import datetime
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util


# <LAO-X>/third_party/CutLER/cutler/demo/export_musc_heat_to_coco.py
LAOX_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = Path(os.environ.get("LAOX_DATA_ROOT", LAOX_ROOT / "datasets")) / "det_compass_xp"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert MUSC heat maps to a standard COCO instance annotation file by "
            "thresholding the recovered 0~1 heat map."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "xray",
        help="Directory that stores the original images.",
    )
    parser.add_argument(
        "--musc-dir",
        type=Path,
        required=True,
        help="Directory that stores the colorized MUSC heat maps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_ROOT / "annotations" / "coco_musc.json",
        help="Path to save the generated COCO annotation json.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert the recovered 0~1 heat map into a binary mask.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1.0,
        help="Masks smaller than this area are dropped.",
    )
    parser.add_argument(
        "--heatmap-pattern",
        dest="heatmap_patterns",
        action="append",
        default=None,
        help=(
            "Filename pattern used to locate a heat map for each image. "
            "Use {stem} and {name}. Repeat this argument to provide multiple patterns. "
            "Defaults to {stem}.png, {stem}.jpg and {stem}.jpeg."
        ),
    )
    parser.add_argument(
        "--image-ext",
        action="append",
        default=None,
        help="Image suffix to scan. Repeat for multiple suffixes. Defaults to .png/.jpg/.jpeg/.bmp.",
    )
    parser.add_argument(
        "--skip-missing-heatmap",
        action="store_true",
        help="Skip images without a matching heat map instead of raising an error.",
    )
    parser.add_argument(
        "--append-threshold-to-output",
        action="store_true",
        help="Append the threshold value to the output filename stem.",
    )
    return parser.parse_args()


def build_jet_lut():
    gradient_values = np.arange(256, dtype=np.uint8).reshape(256, 1)
    jet_colors = cv2.applyColorMap(gradient_values, cv2.COLORMAP_JET).reshape(256, 3)
    packed = (
        jet_colors[:, 0].astype(np.uint32)
        | (jet_colors[:, 1].astype(np.uint32) << 8)
        | (jet_colors[:, 2].astype(np.uint32) << 16)
    )
    order = np.argsort(packed)
    return {
        "colors_bgr": jet_colors.astype(np.int32),
        "packed_sorted": packed[order],
        "values_sorted": np.arange(256, dtype=np.uint8)[order],
    }


def recover_heatmap_from_image(image, jet_lut):
    h, w, c = image.shape
    if c != 3:
        raise ValueError(f"Expected a 3-channel heat map image, but got shape {image.shape}.")

    image_flat = image.reshape(-1, 3)
    packed = (
        image_flat[:, 0].astype(np.uint32)
        | (image_flat[:, 1].astype(np.uint32) << 8)
        | (image_flat[:, 2].astype(np.uint32) << 16)
    )
    packed_sorted = jet_lut["packed_sorted"]
    values_sorted = jet_lut["values_sorted"]

    lookup_idx = np.searchsorted(packed_sorted, packed)
    recovered = np.zeros(image_flat.shape[0], dtype=np.uint8)
    valid = lookup_idx < packed_sorted.shape[0]
    valid[valid] &= packed_sorted[lookup_idx[valid]] == packed[valid]
    recovered[valid] = values_sorted[lookup_idx[valid]]

    if not np.all(valid):
        invalid_pixels = image_flat[~valid].astype(np.int32)
        colors_bgr = jet_lut["colors_bgr"]
        dists = np.sum((invalid_pixels[:, None, :] - colors_bgr[None, :, :]) ** 2, axis=2)
        recovered[~valid] = np.argmin(dists, axis=1).astype(np.uint8)

    return recovered.reshape(h, w).astype(np.float32) / 255.0


def load_heatmap(heatmap_path, target_size, jet_lut):
    heatmap_bgr = cv2.imread(str(heatmap_path), cv2.IMREAD_COLOR)
    if heatmap_bgr is None:
        raise ValueError(f"Failed to read heat map: {heatmap_path}")

    target_w, target_h = target_size
    if heatmap_bgr.shape[1] != target_w or heatmap_bgr.shape[0] != target_h:
        heatmap_bgr = cv2.resize(heatmap_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return recover_heatmap_from_image(heatmap_bgr, jet_lut)


def encode_binary_mask(binary_mask):
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    area = float(mask_util.area(rle))
    bbox = mask_util.toBbox(rle).tolist()
    return rle, area, bbox


def find_heatmap_path(musc_dir, image_path, patterns):
    stem = image_path.stem
    name = image_path.name
    for pattern in patterns:
        candidate = musc_dir / pattern.format(stem=stem, name=name)
        if candidate.exists():
            return candidate
    return None


def iter_image_paths(image_dir, image_exts):
    normalized_exts = {ext if ext.startswith(".") else f".{ext}" for ext in image_exts}
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in normalized_exts
    )


def format_threshold_suffix(threshold):
    threshold_text = f"{threshold:.6f}".rstrip("0").rstrip(".")
    return threshold_text.replace(".", "p")


def main():
    args = parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"--threshold must be in [0, 1], but got {args.threshold}.")
    if args.min_area < 0:
        raise ValueError(f"--min-area must be non-negative, but got {args.min_area}.")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.image_dir}")
    if not args.musc_dir.is_dir():
        raise FileNotFoundError(f"Heat map directory does not exist: {args.musc_dir}")

    heatmap_patterns = args.heatmap_patterns or ["{stem}.png", "{stem}.jpg", "{stem}.jpeg"]
    image_exts = args.image_ext or [".png", ".jpg", ".jpeg", ".bmp"]
    image_paths = iter_image_paths(args.image_dir, image_exts)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.image_dir}")

    jet_lut = build_jet_lut()
    now = datetime.datetime.utcnow().isoformat(" ")
    coco_images = []
    coco_annotations = []
    ann_id = 0
    missing_heatmaps = []

    for image_id, image_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        height, width = image_bgr.shape[:2]
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "date_captured": now,
                "license": 1,
                "coco_url": "",
                "flickr_url": "",
            }
        )

        heatmap_path = find_heatmap_path(args.musc_dir, image_path, heatmap_patterns)
        if heatmap_path is None:
            missing_heatmaps.append(image_path.name)
            if args.skip_missing_heatmap:
                continue
            raise FileNotFoundError(
                f"No heat map found for {image_path.name} under {args.musc_dir}. "
                f"Tried patterns: {heatmap_patterns}"
            )

        musc_heat = load_heatmap(heatmap_path, (width, height), jet_lut)
        binary_mask = musc_heat >= args.threshold
        if not np.any(binary_mask):
            continue

        rle, area, bbox = encode_binary_mask(binary_mask)
        if area < args.min_area:
            continue

        coco_annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "iscrowd": 0,
                "area": area,
                "bbox": bbox,
                "segmentation": rle,
                "width": width,
                "height": height,
            }
        )
        ann_id += 1

    output_path = args.output
    if args.append_threshold_to_output:
        threshold_suffix = format_threshold_suffix(args.threshold)
        output_path = output_path.with_name(
            f"{output_path.stem}_thr{threshold_suffix}{output_path.suffix}"
        )

    coco = {
        "info": {
            "description": "MUSC heat map annotations converted to COCO format",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.utcnow().year,
            "contributor": "",
            "date_created": now,
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": [{"id": 1, "name": "fg", "supercategory": "fg"}],
        "images": coco_images,
        "annotations": coco_annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(coco, f)

    print(
        "Finished: "
        f"{len(coco_images)} images, "
        f"{len(coco_annotations)} annotations, "
        f"{len(missing_heatmaps)} missing heat maps. "
        f"Output: {output_path}"
    )
    if missing_heatmaps:
        preview = ", ".join(missing_heatmaps[:10])
        suffix = " ..." if len(missing_heatmaps) > 10 else ""
        print(f"Missing heat map samples: {preview}{suffix}")


if __name__ == "__main__":
    main()
