# Derived from DET-COMPASS-CROPS by Lorenzo Vaquero (used with permission);
# modified for LAO-X.
"""LAO-X X-ray synthesis.

Subcommands
-----------
match    Map local DET-COMPASS X-ray files to HuggingFace (split, index)
         entries by pixel hash; writes the matches JSON used below.
augment  Generate one shard of a synthesis stage: objects from the object
         bank are composed (Beer-Lambert in the high/low-energy domain, or
         copy-paste with --composition-mode rgb) onto exactly one container
         from the DET-COMPASS 'invisible' split, and the result is rendered
         with the fixed DET-COMPASS lookup tables. Writes xray/<id>.png and
         coco_<start>_<end>.json (and coco_train.json for the shard with
         --start-id 0).
merge    Merge the coco_<start>_<end>.json shards of a stage into coco.json.

Every image draws from its own random stream derived from (seed, stage,
image id), so the output does not depend on how images are split into shards.
Use run_stage.sh to generate a full stage.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Dict, List, Tuple

import cv2
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from datasets import config as hf_datasets_config
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from compass_editor.colour import (
    center_pad_unpad,
    combine_highlow_images,
    get_bbox_from_mask,
    highlow_to_colour,
    highlow_to_density,
    highlow_to_grey,
    sample_to_highlow,
)
from compass_editor.dataset import set_seed, transform_xray_crops


HF_DATASET = "PAGF/DET-COMPASS"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Container canvas (height, width) and object transformation limits.
IMAGE_SIZE = (580, 880)
MAX_ANGLE = 180.0
MAX_SCALE = 1.2
MAX_TRANSLATION = 0.75
PLACEMENT_TRIES = 50

# Namespace of the per-image seed; changing it changes every generated image.
SEED_NAMESPACE = "laox-single-legacy-v1"


# ---------------------------------------------------------------------------
# DET-COMPASS loading
# ---------------------------------------------------------------------------

def _hf_datasets_cache_root() -> str:
    """HuggingFace datasets cache (honours HF_DATASETS_CACHE / HF_HOME)."""
    return str(hf_datasets_config.HF_DATASETS_CACHE)


def _load_det_compass_dataset_from_cache() -> DatasetDict:
    """Load the 'test' and 'invisible' splits straight from cached Arrow shards.

    Looks for <cache>/*___det-compass/<config>/<version>/<revision>/ directories
    that hold dataset_info.json and *-test-*.arrow shards (both the standard
    HuggingFace layout and a local reconstruction of it); the most recently
    modified one is used.
    """
    cache_root = _hf_datasets_cache_root()
    candidates = []
    for cache_dir in glob.glob(
        os.path.join(cache_root, "*___det-compass", "*", "*", "*")
    ):
        if not os.path.isfile(os.path.join(cache_dir, "dataset_info.json")):
            continue
        if glob.glob(os.path.join(cache_dir, "*-test-*.arrow")):
            candidates.append(cache_dir)

    if not candidates:
        raise FileNotFoundError(
            f"No complete cached {HF_DATASET} dataset was found under "
            f"{cache_root}. Run once with --online (network access) first."
        )

    cache_dir = max(candidates, key=os.path.getmtime)
    splits = {}
    for split in ("test", "invisible"):
        shard_paths = sorted(
            glob.glob(os.path.join(cache_dir, f"*-{split}-*.arrow"))
        )
        if not shard_paths:
            continue
        shards = [Dataset.from_file(path) for path in shard_paths]
        splits[split] = (
            shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        )

    if "test" not in splits:
        raise FileNotFoundError(
            f"Cached {HF_DATASET} test split is missing from {cache_dir}."
        )
    print(f"Loading {HF_DATASET} directly from cache: {cache_dir}")
    return DatasetDict(splits)


def _load_det_compass_dataset(offline: bool):
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        return _load_det_compass_dataset_from_cache()
    with _dataset_lock():
        return load_dataset(HF_DATASET, download_mode="reuse_cache_if_exists")


@contextmanager
def _dataset_lock(timeout_s: int = 1800):
    """
    Serialize HuggingFace dataset preparation to avoid concurrent .incomplete writes.
    """
    import fcntl

    lock_path = os.path.join(_hf_datasets_cache_root(), ".det_compass_lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    start = time.time()
    with open(lock_path, "w") as f:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start > timeout_s:
                    raise TimeoutError(
                        f"Timeout waiting for dataset lock: {lock_path}"
                    )
                time.sleep(1.0)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def parse_range_float(s: str) -> Tuple[float, float]:
    """
    Parse 'min,max' or 'val' into a float tuple.
    """
    try:
        parts = list(map(float, s.split(",")))
        if len(parts) == 1:
            return parts[0], parts[0]
        if len(parts) == 2:
            return min(parts), max(parts)
        raise ValueError
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid float range '{s}'. Use 'val' or 'min,max'."
        ) from exc


def parse_range_int(s: str) -> Tuple[int, int]:
    """
    Parse 'min,max' or 'val' into an int tuple.
    """
    try:
        parts = list(map(int, s.split(",")))
        if len(parts) == 1:
            return parts[0], parts[0]
        if len(parts) == 2:
            return min(parts), max(parts)
        raise ValueError
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid int range '{s}'. Use 'val' or 'min,max'."
        ) from exc


# ---------------------------------------------------------------------------
# Small array / COCO helpers
# ---------------------------------------------------------------------------

def _ensure_mask_size(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask is None:
        return None
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    h = min(mask.shape[0], height)
    w = min(mask.shape[1], width)
    out = np.zeros((height, width), dtype=mask.dtype)
    out[:h, :w] = mask[:h, :w]
    return out


def _decode_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        mask = mask_utils.decode(mask_utils.merge(rles)).astype(bool)
        return _ensure_mask_size(mask, height, width)
    if isinstance(segmentation, dict):
        seg = dict(segmentation)
        if "counts" in seg and isinstance(seg.get("counts"), list):
            rle = mask_utils.frPyObjects(seg, height, width)
            mask = mask_utils.decode(rle).astype(bool)
            return _ensure_mask_size(mask, height, width)
        seg["size"] = [int(height), int(width)]
        mask = mask_utils.decode(seg).astype(bool)
        return _ensure_mask_size(mask, height, width)
    return None


def _bbox_iou_xyxy(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _hash_image_array(arr: np.ndarray) -> str:
    """
    Hash image content with shape to make collisions very unlikely.
    """
    h = hashlib.md5(arr.tobytes()).hexdigest()
    return f"{arr.shape}-{h}"


def _save_png(path: str, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _load_image_as_rgb_array(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


# ---------------------------------------------------------------------------
# match: local X-ray files -> HuggingFace (split, index)
# ---------------------------------------------------------------------------

def _strip_other_images(sample: Dict) -> Image.Image:
    """
    Keep only xray_images[0] (colour) and drop rgb_image and other xray images.
    Returns the remaining xray PIL image.
    """
    if "rgb_image" in sample:
        sample.pop("rgb_image", None)
    if "xray_images" in sample:
        xray_images = sample.get("xray_images") or []
        if not xray_images:
            return None
        keep = xray_images[0]
        sample["xray_images"] = [keep]
        return keep
    return None


def build_local_hash_map(xray_dir: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Returns:
        hash_to_files: hash -> [filenames]
        all_files: list of all filenames considered
    """
    hash_to_files: Dict[str, List[str]] = {}
    all_files: List[str] = []
    for fname in tqdm(sorted(os.listdir(xray_dir)), desc="Indexing local xray files"):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS:
            continue
        all_files.append(fname)
        arr = _load_image_as_rgb_array(os.path.join(xray_dir, fname))
        h = _hash_image_array(arr)
        hash_to_files.setdefault(h, []).append(fname)
    return hash_to_files, all_files


def match_dataset_to_local_xray(
    xray_dir: str,
    output_path: str,
    splits: Tuple[str, ...] = ("test", "invisible"),
    limit: int = None,
    offline: bool = False,
) -> Dict:
    print(f"Loading dataset {HF_DATASET}...")
    dataset_xp = _load_det_compass_dataset(offline=offline)

    print(f"Indexing local xray images in {xray_dir}...")
    hash_to_files, all_files = build_local_hash_map(xray_dir)

    matches: Dict[str, List[Dict]] = {}
    unmatched_dataset: List[Dict] = []
    matched_files = set()

    for split in splits:
        if split not in dataset_xp:
            continue
        ds = dataset_xp[split]
        total = len(ds) if limit is None else min(len(ds), limit)
        print(f"Matching split '{split}' ({total} samples)...")
        for i in tqdm(range(total), desc=f"Matching {split}"):
            sample = ds[i]
            xray_img = _strip_other_images(sample)
            if xray_img is None:
                unmatched_dataset.append({"split": split, "index": i, "reason": "missing_xray"})
                continue

            arr = np.asarray(xray_img.convert("RGB"))
            h = _hash_image_array(arr)
            file_matches = hash_to_files.get(h)
            if not file_matches:
                unmatched_dataset.append({"split": split, "index": i, "reason": "no_local_match"})
                continue

            for fname in file_matches:
                matches.setdefault(fname, []).append({"split": split, "index": i})
                matched_files.add(fname)

    unmatched_files = [f for f in all_files if f not in matched_files]

    result = {
        "xray_dir": xray_dir,
        "splits": list(splits),
        "matches": matches,
        "unmatched_dataset": unmatched_dataset,
        "unmatched_files": unmatched_files,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# augment: object bank and container preparation
# ---------------------------------------------------------------------------

def _load_matches_split_map(matches_path: str) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Returns:
        file_to_split: filename -> split ("test" or "invisible")
        ambiguous: list of files with multiple splits
    """
    with open(matches_path, "r") as f:
        data = json.load(f)

    file_to_split: Dict[str, str] = {}
    ambiguous: List[Dict] = []
    matches = data.get("matches", {})
    for fname, entries in matches.items():
        splits = sorted({e.get("split") for e in entries if e.get("split")})
        if not splits:
            continue
        if len(splits) > 1:
            # Keep a stable split for files that appear in several splits.
            ambiguous.append({"file": fname, "splits": splits})
        file_to_split[fname] = splits[0]
    return file_to_split, ambiguous


def _load_matches_index_map(matches_path: str, split: str) -> Dict[str, int]:
    with open(matches_path, "r") as f:
        data = json.load(f)
    file_to_index: Dict[str, int] = {}
    matches = data.get("matches", {})
    for fname, entries in matches.items():
        for entry in entries:
            if entry.get("split") == split:
                file_to_index[fname] = int(entry.get("index"))
                break
    return file_to_index


def filter_coco_for_test_split(
    coco_path: str,
    matches_path: str,
    test_coco_path: str,
    max_area_ratio: float = 0.5,
) -> Tuple[Dict, Dict]:
    """Object-bank COCO restricted to DET-COMPASS 'test' images.

    Images listed in the held-out test COCO are removed, as are images with
    any annotation covering more than max_area_ratio of the image.
    """
    print(f"Loading COCO annotations from {coco_path}...")
    with open(coco_path, "r") as f:
        coco = json.load(f)

    original_image_count = len(coco.get("images", []))
    print(f"Loading test annotations from {test_coco_path}...")
    with open(test_coco_path, "r") as f:
        test_coco = json.load(f)
    test_files = {
        img.get("file_name")
        for img in test_coco.get("images", [])
        if img.get("file_name")
    }
    if test_files:
        keep_images = [
            img for img in coco.get("images", [])
            if img.get("file_name") not in test_files
        ]
        keep_image_ids = {img["id"] for img in keep_images}
        keep_annotations = [
            ann for ann in coco.get("annotations", [])
            if ann.get("image_id") in keep_image_ids
        ]
        coco["images"] = keep_images
        coco["annotations"] = keep_annotations
    print(f"Original images: {original_image_count}")
    print(f"After removing test images: {len(coco.get('images', []))}")

    print(f"Loading matches from {matches_path}...")
    file_to_split, ambiguous = _load_matches_split_map(matches_path)
    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in tqdm(coco.get("annotations", []), desc="Indexing annotations"):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    kept_images = []
    kept_annotations = []
    dropped_images = 0
    dropped_annotations = 0

    for image_id, img in tqdm(images_by_id.items(), desc="Filtering images"):
        if file_to_split.get(img["file_name"]) != "test":
            continue
        img_area = float(img.get("width", 0) * img.get("height", 0))
        if img_area <= 0:
            continue
        ann_list = anns_by_image.get(image_id, [])
        has_large = any(
            (ann.get("area", 0) / img_area) > max_area_ratio for ann in ann_list
        )
        if has_large:
            dropped_images += 1
            dropped_annotations += len(ann_list)
            continue
        kept_images.append(img)
        kept_annotations.extend(ann_list)

    print(f"After removing large masks: {len(kept_images)}")

    filtered_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": coco.get("categories", []),
    }

    stats = {
        "num_images": len(coco.get("images", [])),
        "num_annotations": len(coco.get("annotations", [])),
        "kept_images": len(kept_images),
        "kept_annotations": len(kept_annotations),
        "dropped_images": dropped_images,
        "dropped_annotations": dropped_annotations,
        "num_ambiguous_files": len(ambiguous),
    }
    return filtered_coco, stats


def _build_foreground_samples(
    filtered_coco: Dict,
    dataset_test,
    matches_path: str,
    output_dir: str = None,
) -> List[Dict]:
    """Crop every object-bank mask out of its DET-COMPASS test image.

    If output_dir is given, also writes coco_train.json there: the
    DET-COMPASS boxes of the source images used by the object bank.
    """
    print("Building foreground samples from filtered COCO annotations...")
    file_to_index = _load_matches_index_map(matches_path, split="test")
    image_meta = {img["id"]: img for img in filtered_coco.get("images", [])}
    image_id_to_index: Dict[int, int] = {}
    for image_id, img in image_meta.items():
        idx = file_to_index.get(img["file_name"])
        if idx is not None:
            image_id_to_index[image_id] = idx

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        coco_train = {
            "info": filtered_coco.get("info", {}),
            "licenses": filtered_coco.get("licenses", []),
            "images": [],
            "annotations": [],
            "categories": filtered_coco.get("categories", []),
        }
        ann_id = 0
        for image_id, img in image_meta.items():
            idx = image_id_to_index.get(image_id)
            if idx is None:
                continue
            sample = dataset_test[idx]
            width = sample.get("xray_width")
            height = sample.get("xray_height")
            if width is None or height is None:
                xray_images = sample.get("xray_images") or []
                if xray_images:
                    arr = np.asarray(xray_images[0])
                    height, width = arr.shape[0], arr.shape[1]
            if width is None or height is None:
                continue
            coco_train["images"].append(
                {
                    "id": image_id,
                    "file_name": img.get("file_name"),
                    "width": int(width),
                    "height": int(height),
                }
            )
            objects = sample.get("objects", {})
            bboxes = objects.get("xray_bbox") or []
            categories = objects.get("categories") or []
            for bbox, category_id in zip(bboxes, categories):
                if not bbox:
                    continue
                try:
                    category_id = int(category_id)
                except (TypeError, ValueError):
                    category_id = 0
                area = float(bbox[2] * bbox[3]) if bbox else 0.0
                coco_train["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1
        output_path = os.path.join(output_dir, "coco_train.json")
        with open(output_path, "w") as f:
            json.dump(coco_train, f)
        print(f"Saved dataset_xp train annotations to {output_path}")

    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    foreground_samples: List[Dict] = []
    for ann in tqdm(filtered_coco.get("annotations", []), desc="Cropping foregrounds"):
        image_id = ann.get("image_id")
        idx = image_id_to_index.get(image_id)
        if idx is None:
            continue
        img = image_meta.get(image_id)
        if img is None:
            continue
        height = int(img.get("height", 0))
        width = int(img.get("width", 0))
        if height <= 0 or width <= 0:
            continue

        if idx not in cache:
            sample = dataset_test[idx]
            xray_images = sample.get("xray_images") or []
            if len(xray_images) < 4:
                continue
            high = np.asarray(xray_images[2])
            low = np.asarray(xray_images[3])
            cache[idx] = (high, low)
        high, low = cache[idx]

        segmentation = ann.get("segmentation")
        mask = _decode_mask(segmentation, height, width)
        if mask is None and ann.get("bbox"):
            x, y, w, h = [int(v) for v in ann["bbox"]]
            mask = np.zeros((height, width), dtype=bool)
            mask[y : y + h, x : x + w] = True
        if mask is None:
            continue
        mask = _ensure_mask_size(mask, height, width)
        if mask.shape[0] != high.shape[0] or mask.shape[1] != high.shape[1]:
            mask = _ensure_mask_size(mask, high.shape[0], high.shape[1])
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        high_crop = high[y_min : y_max + 1, x_min : x_max + 1]
        low_crop = low[y_min : y_max + 1, x_min : x_max + 1]
        mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1].astype(np.uint8) * 255

        foreground_samples.append(
            {
                "raw_xray_image_high": high_crop.astype(np.uint8),
                "raw_xray_image_low": low_crop.astype(np.uint8),
                "xray_image_mask": Image.fromarray(mask_crop, mode="L"),
                "category": int(ann.get("category_id", 0)),
                "source_file_name": img.get("file_name"),
                "annotation_id": ann.get("id"),
            }
        )

    foreground_samples.sort(key=lambda item: (item["source_file_name"], item["annotation_id"]))
    return foreground_samples


def _prepare_foreground_cache(foreground_samples: List[Dict]) -> None:
    print("Caching foreground samples (highlow + mask arrays)...")
    for sample in tqdm(foreground_samples, desc="Cache foreground"):
        cache = sample.setdefault("_cached_arrays", {})
        if "highlow" not in cache:
            cache["highlow"] = sample_to_highlow(sample, as_image=True)
        if "xray_mask" not in cache:
            cache["xray_mask"] = np.array(sample["xray_image_mask"], dtype=np.uint8)
        if "_mask_area" not in sample:
            sample["_mask_area"] = int((cache["xray_mask"] > 0).sum())


def _padded_background(bg_sample: Dict, image_size: Tuple[int, int]) -> np.ndarray:
    """High/low container image centre-padded (with 255) or cropped to image_size."""
    cache = bg_sample.setdefault("_cached_arrays", {})
    if (
        "highlow_padded" not in cache
        or cache.get("highlow_padded_size") != image_size
    ):
        bg_img_original = sample_to_highlow(bg_sample, as_image=True)
        bg_img, _ = center_pad_unpad(
            bg_img_original, target_size=image_size, constant_values=255
        )
        cache["highlow_padded"] = bg_img
        cache["highlow_padded_size"] = image_size
    return cache["highlow_padded"]


def _prepare_background_cache(
    background_samples: List[Dict],
    image_size: Tuple[int, int],
) -> None:
    print("Caching background samples (padded highlow arrays)...")
    for sample in tqdm(background_samples, desc="Cache background"):
        _padded_background(sample, image_size)


# ---------------------------------------------------------------------------
# augment: composition and rendering
# ---------------------------------------------------------------------------

def _sample_transformed_object(
    fg_sample: Dict,
    image_size: Tuple[int, int],
    max_angle: float,
    max_scale: float,
    max_t: float,
    max_perspective: float,
    existing_boxes: List[Tuple[float, float, float, float]],
    gamma: float,
    max_tries: int = PLACEMENT_TRIES,
):
    """Draw transformations until the object's box overlaps every already
    placed box with IoU <= gamma; gives up after max_tries."""
    for _ in range(max_tries):
        objs = transform_xray_crops(
            fg_samples=[fg_sample],
            image_size=image_size,
            max_angle=max_angle,
            max_scale=max_scale,
            max_t=max_t,
            max_perspective=max_perspective,
        )
        if not objs:
            continue
        obj = objs[0]
        bbox = get_bbox_from_mask(obj["fg_mask_t"])
        if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        x1 = float(bbox[0])
        y1 = float(bbox[1])
        x2 = float(bbox[0] + bbox[2])
        y2 = float(bbox[1] + bbox[3])
        candidate = (x1, y1, x2, y2)
        if any(_bbox_iou_xyxy(candidate, b) > gamma for b in existing_boxes):
            continue
        return obj, candidate
    return None, None


def render_xray(composite_img: np.ndarray):
    """Render a high/low composite with the fixed DET-COMPASS lookup tables.

    Returns (colour, grey, high, low, density); only colour is written to disk.
    """
    composite_img = composite_img[..., :2]
    xray_colour = highlow_to_colour(composite_img)
    xray_density = highlow_to_density(composite_img)
    xray_grey = highlow_to_grey(composite_img)
    xray_high = Image.fromarray(composite_img[..., 0], mode="L")
    xray_low = Image.fromarray(composite_img[..., 1], mode="L")
    return xray_colour, xray_grey, xray_high, xray_low, xray_density


def create_synthetic_sample(
    foreground_samples: List[Dict],
    background_samples: List[Dict],
    num_objs: int,
    gamma: float,
    composition_mode: str,
    max_perspective: float,
) -> Dict:
    """One synthetic image: one random container plus num_objs bank objects."""
    sample = {
        "xray_images": [],
        "xray_width": None,
        "xray_height": None,
        "objects": {
            "xray_bbox": [],
            "xray_mask": [],
            "categories": [],
        },
    }

    bg_img = _padded_background(random.choice(background_samples), IMAGE_SIZE)

    if not foreground_samples or num_objs <= 0:
        selected = []
    else:
        selected = np.random.choice(foreground_samples, size=num_objs, replace=True).tolist()
        # Place large objects first.
        selected.sort(key=lambda x: x.get("_mask_area", 0), reverse=True)

    transformed_objs = []
    placed_boxes: List[Tuple[float, float, float, float]] = []
    for fg_sample in selected:
        obj, bbox_xyxy = _sample_transformed_object(
            fg_sample=fg_sample,
            image_size=bg_img.shape[0:2],
            max_angle=MAX_ANGLE,
            max_scale=MAX_SCALE,
            max_t=MAX_TRANSLATION,
            max_perspective=max_perspective,
            existing_boxes=placed_boxes,
            gamma=gamma,
        )
        if obj is None:
            continue
        placed_boxes.append(bbox_xyxy)
        transformed_objs.append(obj)

    composite_img = bg_img
    for obj in transformed_objs:
        if composition_mode == "xray":
            composite_img = combine_highlow_images(
                composite_img, obj["fg_img_t"], substract=False, maskB=obj["fg_mask_t"]
            )
        elif composition_mode == "rgb":
            mask = obj["fg_mask_t"] > 0
            composite_img = composite_img.copy()
            composite_img[mask] = obj["fg_img_t"][mask]
        else:
            raise ValueError(f"Unknown composition_mode '{composition_mode}'")
        bbox = get_bbox_from_mask(obj["fg_mask_t"])
        sample["objects"]["xray_bbox"].append(bbox)
        sample["objects"]["xray_mask"].append(
            Image.fromarray(obj["fg_mask_t"] > 0).convert("1")
        )
        sample["objects"]["categories"].append(obj["fg_sample"].get("category", "unknown"))

    xray_colour, xray_grey, xray_high, xray_low, xray_density = render_xray(composite_img)

    sample["xray_images"] = [xray_colour, xray_grey, xray_high, xray_low, xray_density]
    sample["xray_height"] = xray_colour.shape[0]
    sample["xray_width"] = xray_colour.shape[1]

    return sample


def seed_image(base_seed: int, stage: str, image_id: int) -> int:
    """Independent per-image stream; independent of worker count and chunking."""
    payload = f"{SEED_NAMESPACE}:{base_seed}:{stage}:{image_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed & 0x7fffffff)
    import torch
    torch.manual_seed(seed)
    return seed


def generate_synthetic_dataset(
    foreground_samples: List[Dict],
    background_samples: List[Dict],
    output_dir: str,
    num_generated: int,
    num_objects_range: Tuple[int, int],
    gamma_range: Tuple[float, float],
    max_perspective: float,
    composition_mode: str,
    io_workers: int,
    start_id: int,
    info: Dict,
    licenses: List[Dict],
    categories: List[Dict],
    base_seed: int,
    stage: str,
):
    if not background_samples:
        raise ValueError("No background samples available from the 'invisible' split.")
    print(f"Generating {num_generated} synthetic images into {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    xray_dir = os.path.join(output_dir, "xray")
    os.makedirs(xray_dir, exist_ok=True)
    coco_output = {
        "info": info,
        "licenses": licenses,
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    ann_id = 0
    io_executor = ThreadPoolExecutor(max_workers=io_workers if io_workers > 0 else 1)
    io_futures: deque = deque()
    max_pending = max(1, (io_workers if io_workers > 0 else 1) * 4)

    num_min, num_max = num_objects_range
    gamma_min, gamma_max = gamma_range

    for i in tqdm(range(num_generated), desc="synthesis"):
        global_id = start_id + i
        seed_image(base_seed, stage, global_id)
        current_num_objs = np.random.randint(num_min, num_max + 1)
        current_gamma = np.random.uniform(gamma_min, gamma_max)
        sample = create_synthetic_sample(
            foreground_samples=foreground_samples,
            background_samples=background_samples,
            num_objs=current_num_objs,
            gamma=current_gamma,
            composition_mode=composition_mode,
            max_perspective=max_perspective,
        )

        png_filename = f"{global_id:06d}.png"
        png_path = os.path.join(xray_dir, png_filename)
        xray_colour = np.clip(sample["xray_images"][0], 0, 255).astype(np.uint8)
        io_futures.append(io_executor.submit(_save_png, png_path, xray_colour))
        if len(io_futures) >= max_pending:
            io_futures.popleft().result()

        coco_output["images"].append(
            {
                "id": global_id,
                "file_name": png_filename,
                "width": sample["xray_width"],
                "height": sample["xray_height"],
            }
        )

        for bbox, mask, category_id in zip(
            sample["objects"]["xray_bbox"],
            sample["objects"]["xray_mask"],
            sample["objects"]["categories"],
        ):
            try:
                category_id = int(category_id)
            except (TypeError, ValueError):
                category_id = 0

            # COCO polygons from the external contours of the object mask.
            segmentation = []
            mask_arr = np.array(mask, dtype=np.uint8)
            mask_arr = (mask_arr > 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.shape[0] < 3:
                    continue
                flattened = contour.reshape(-1, 2).ravel().tolist()
                if len(flattened) >= 6:
                    segmentation.append(flattened)
            mask_area = int(mask_arr.sum())

            coco_output["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": global_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": mask_area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )
            ann_id += 1

    end_id = start_id + num_generated
    coco_json_path = os.path.join(output_dir, f"coco_{start_id}_{end_id}.json")
    with open(coco_json_path, "w") as f:
        json.dump(coco_output, f)

    while io_futures:
        io_futures.popleft().result()
    io_executor.shutdown(wait=True)

    print(f"Saved synthetic COCO annotations to {coco_json_path}")
    return coco_json_path


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def merge_coco_jsons(
    output_dir: str,
    output_path: str,
    pattern: str,
):
    json_paths = sorted(glob.glob(os.path.join(output_dir, pattern)))
    if not json_paths:
        raise FileNotFoundError(f"No COCO json files found in {output_dir} matching {pattern}")

    merged = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    images_seen = set()
    for idx, path in enumerate(json_paths):
        with open(path, "r") as f:
            data = json.load(f)
        if idx == 0:
            merged["info"] = data.get("info", {})
            merged["licenses"] = data.get("licenses", [])
            merged["categories"] = data.get("categories", [])

        for img in data.get("images", []):
            img_id = img.get("id")
            if img_id in images_seen:
                continue
            merged["images"].append(img)
            images_seen.add(img_id)

        for ann in data.get("annotations", []):
            merged["annotations"].append(dict(ann))

    merged["images"].sort(key=lambda item: item["id"])
    merged["annotations"].sort(key=lambda item: (item["image_id"], item["id"]))
    for ann_id, ann in enumerate(merged["annotations"]):
        ann["id"] = ann_id

    with open(output_path, "w") as f:
        json.dump(merged, f)

    print(f"Merged {len(json_paths)} files into {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_offline_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        default=True,
        help="Load DET-COMPASS from the local HuggingFace cache only (default).",
    )
    group.add_argument(
        "--online",
        dest="offline",
        action="store_false",
        help="Allow HuggingFace downloads (needed on the first run).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LAO-X X-ray synthesis (single container, fixed DET-COMPASS lookup tables).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("match", help="Map local DET-COMPASS X-ray files to HuggingFace split/index.")
    p.add_argument("--xray-dir", required=True,
                   help="Directory with the local DET-COMPASS colour X-ray images.")
    p.add_argument("--output", default="huggingface_matches.json",
                   help="Output matches JSON (default: huggingface_matches.json).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional limit per split, for debugging.")
    _add_offline_flags(p)

    p = sub.add_parser("augment", help="Generate one shard of a synthesis stage.")
    p.add_argument("--object-bank-json", required=True,
                   help="COCO JSON with the object-bank masks (on DET-COMPASS test images).")
    p.add_argument("--test-coco", required=True,
                   help="Held-out labelled COCO JSON; its images are excluded from the object bank.")
    p.add_argument("--matches-json", required=True,
                   help="Matches JSON written by the 'match' subcommand.")
    p.add_argument("--output-dir", required=True,
                   help="Stage output directory (xray/ and COCO JSONs are written here).")
    p.add_argument("--stage", required=True,
                   help="Stage name (e.g. easy, hard, extreme); part of every image seed.")
    p.add_argument("--num-generated", type=int, required=True,
                   help="Number of images in this shard.")
    p.add_argument("--start-id", type=int, default=0,
                   help="Id of the first image of this shard (default: 0).")
    p.add_argument("--num-objects", type=parse_range_int, required=True,
                   help="Objects per image, 'min,max' or 'n'.")
    p.add_argument("--gamma", type=parse_range_float, required=True,
                   help="Maximum box IoU between placed objects, 'min,max' or 'g'; drawn per image.")
    p.add_argument("--composition-mode", choices=["xray", "rgb"], default="xray",
                   help="xray: Beer-Lambert composition (LAO-X); rgb: copy-paste overwrite (ablation).")
    p.add_argument("--max-area-ratio", type=float, default=0.5,
                   help="Drop object-bank images with any mask larger than this image fraction (default: 0.5).")
    p.add_argument("--max-perspective", type=float, default=0.1,
                   help="Max perspective jitter per corner, as an image-size fraction (default: 0.1).")
    p.add_argument("--seed", type=int, default=40,
                   help="Base seed (default: 40); per-image streams do not depend on sharding.")
    p.add_argument("--io-workers", type=int, default=4,
                   help="Threads writing PNG files (default: 4).")
    _add_offline_flags(p)

    p = sub.add_parser("merge", help="Merge the COCO shards of a stage into one file.")
    p.add_argument("--output-dir", required=True, help="Stage output directory.")
    p.add_argument("--pattern", default="coco_[0-9]*_[0-9]*.json",
                   help="Glob for the shard files (default: 'coco_[0-9]*_[0-9]*.json').")
    p.add_argument("--output", default="coco.json",
                   help="Merged file name, relative to --output-dir (default: coco.json).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "augment" and not 0 <= args.seed < 2**32:
        parser.error("seed must be an unsigned 32-bit integer")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    if getattr(args, "offline", False):
        print("Offline mode enabled: using local HuggingFace cache only.")

    if args.command == "match":
        match_dataset_to_local_xray(
            xray_dir=args.xray_dir,
            output_path=args.output,
            splits=("test", "invisible"),
            limit=args.limit,
            offline=args.offline,
        )
        return

    if args.command == "merge":
        merge_coco_jsons(
            output_dir=args.output_dir,
            output_path=os.path.join(args.output_dir, args.output),
            pattern=args.pattern,
        )
        return

    filtered_coco, stats = filter_coco_for_test_split(
        coco_path=args.object_bank_json,
        matches_path=args.matches_json,
        test_coco_path=args.test_coco,
        max_area_ratio=args.max_area_ratio,
    )
    print(json.dumps(stats, indent=2))

    print(f"Loading dataset {HF_DATASET} (offline={args.offline})...")
    dataset_xp = _load_det_compass_dataset(offline=args.offline)
    for split in ("test", "invisible"):
        if split in dataset_xp and "rgb_image" in dataset_xp[split].column_names:
            dataset_xp[split] = dataset_xp[split].remove_columns("rgb_image")

    set_seed(args.seed)

    foreground_samples = _build_foreground_samples(
        filtered_coco=filtered_coco,
        dataset_test=dataset_xp["test"],
        matches_path=args.matches_json,
        output_dir=args.output_dir if args.start_id == 0 else None,
    )
    background_samples = list(dataset_xp["invisible"])

    _prepare_foreground_cache(foreground_samples)
    _prepare_background_cache(background_samples, image_size=IMAGE_SIZE)

    print(
        json.dumps(
            {
                "foreground_samples": len(foreground_samples),
                "background_samples": len(background_samples),
            },
            indent=2,
        )
    )

    generate_synthetic_dataset(
        foreground_samples=foreground_samples,
        background_samples=background_samples,
        output_dir=args.output_dir,
        num_generated=args.num_generated,
        num_objects_range=args.num_objects,
        gamma_range=args.gamma,
        max_perspective=args.max_perspective,
        composition_mode=args.composition_mode,
        io_workers=args.io_workers,
        start_id=args.start_id,
        info=filtered_coco.get("info", {}),
        licenses=filtered_coco.get("licenses", []),
        categories=filtered_coco.get("categories", []),
        base_seed=args.seed,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
