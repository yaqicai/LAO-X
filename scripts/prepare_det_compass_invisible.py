#!/usr/bin/env python3
"""Build COCO annotations for the DET-COMPASS invisible split."""

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import sys

from datasets import Dataset, concatenate_datasets
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import XrayImgRoot, XrayRoot  # noqa: E402

# Hugging Face datasets cache: $HF_DATASETS_CACHE, else $HF_HOME/datasets,
# else ~/.cache/huggingface/datasets (the library defaults).
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE") or os.path.join(
    os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    ),
    "datasets",
)
DEFAULT_CACHE_ROOT = os.path.join(
    HF_DATASETS_CACHE, "PAGF___det-compass", "default", "0.0.0"
)
DEFAULT_MATCHES = os.path.join(
    XrayRoot["det_compass_xp"], "annotations", "huggingface_matches.json"
)
DEFAULT_IMAGE_ROOT = XrayImgRoot["det_compass_xp"]
DEFAULT_OUTPUT = (
    "annotations/det_compass_xp_coco_invisible_agnostic.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--matches", default=DEFAULT_MATCHES)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--category-counts-output",
        default="",
        help="Optional JSON output for the source-category distribution.",
    )
    return parser.parse_args()


def find_invisible_arrows(cache_root):
    pattern = os.path.join(
        cache_root, "*", "det-compass-invisible-*.arrow"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No cached invisible Arrow shards matched: {pattern}"
        )
    build_dirs = {str(Path(path).parent) for path in paths}
    if len(build_dirs) != 1:
        raise RuntimeError(
            "Multiple DET-COMPASS cache builds found. Pass the exact "
            f"--cache-root. Found: {sorted(build_dirs)}"
        )
    return paths


def load_index_to_filename(matches_path):
    with open(matches_path, "r", encoding="utf-8") as handle:
        matches = json.load(handle)["matches"]

    index_to_filename = {}
    for filename, records in matches.items():
        for record in records:
            if record.get("split") != "invisible":
                continue
            index = int(record["index"])
            if index in index_to_filename:
                raise ValueError(f"Duplicate invisible index: {index}")
            index_to_filename[index] = filename
    return index_to_filename


def clipped_xywh(box, width, height):
    if len(box) != 4:
        raise ValueError(f"Expected xywh box, got: {box}")
    x, y, w, h = (float(value) for value in box)
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + w, 0.0), float(width))
    y2 = min(max(y + h, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid clipped box {box} for {width}x{height}")
    return [x1, y1, x2 - x1, y2 - y1]


def main():
    args = parse_args()
    arrow_paths = find_invisible_arrows(args.cache_root)
    dataset = concatenate_datasets(
        [Dataset.from_file(path) for path in arrow_paths]
    ).select_columns(
        ["xray_width", "xray_height", "objects"]
    )
    index_to_filename = load_index_to_filename(args.matches)

    if len(index_to_filename) != len(dataset):
        raise ValueError(
            "HuggingFace/local match count mismatch: "
            f"{len(dataset)} rows vs {len(index_to_filename)} matches"
        )

    images = []
    annotations = []
    category_counts = Counter()
    category_files = {}
    annotation_id = 1

    for index, row in enumerate(dataset):
        filename = index_to_filename[index]
        image_path = os.path.join(args.image_root, filename)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        width = int(row["xray_width"])
        height = int(row["xray_height"])
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"Image-size mismatch for {filename}: "
                    f"metadata={(width, height)}, file={image.size}"
                )

        image_id = index + 1
        images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
                "hf_split": "invisible",
                "hf_index": index,
            }
        )

        objects = row["objects"]
        fields = [
            "xray_bbox",
            "categories",
            "xray_visible",
            "dangerous",
            "IN_ids",
            "WN_ids",
        ]
        lengths = {field: len(objects[field]) for field in fields}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Inconsistent object fields at invisible index {index}: {lengths}"
            )

        for object_index, source_category in enumerate(objects["categories"]):
            bbox = clipped_xywh(
                objects["xray_bbox"][object_index], width, height
            )
            category_counts[source_category] += 1
            category_files.setdefault(source_category, set()).add(filename)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "source_category": source_category,
                    "xray_visible": bool(
                        objects["xray_visible"][object_index]
                    ),
                    "dangerous": bool(objects["dangerous"][object_index]),
                    "IN_id": objects["IN_ids"][object_index],
                    "WN_id": objects["WN_ids"][object_index],
                }
            )
            annotation_id += 1

    coco = {
        "info": {
            "description": (
                "DET-COMPASS invisible split. Category-agnostic evaluation "
                "annotations reconstructed from the cached HuggingFace source."
            ),
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, indent=2)

    counts_path = Path(
        args.category_counts_output
        or output_path.with_name("invisible_category_counts.json")
    )
    counts_payload = {
        "num_images": len(images),
        "num_objects": len(annotations),
        "num_categories": len(category_counts),
        "all_xray_visible_false": all(
            not annotation["xray_visible"] for annotation in annotations
        ),
        "categories": [
            {
                "name": name,
                "objects": count,
                "images": len(category_files[name]),
                "files": sorted(category_files[name]),
            }
            for name, count in sorted(
                category_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }
    with counts_path.open("w", encoding="utf-8") as handle:
        json.dump(counts_payload, handle, indent=2)

    print(
        f"Wrote {output_path}: {len(images)} images, "
        f"{len(annotations)} objects, {len(category_counts)} source categories"
    )
    print(f"Wrote {counts_path}")


if __name__ == "__main__":
    main()
