import os
import json
import pickle
import argparse
from typing import List, Optional

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import cv2
from compass_editor.dataset import split_dataset, create_synthetic_sample, set_seed
   
os.environ["HF_HOME"] = "/path/to/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/path/to/hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/path/to/hf_cache/hub"
# export HF_HOME=/path/to/hf_cache
# export HF_DATASETS_CACHE=/path/to/hf_cache/datasets
# export HUGGINGFACE_HUB_CACHE=/path/to/hf_cache/hub
def _parse_categories(raw: Optional[List[str]]) -> Optional[List[str]]:
    """
    Support space or comma separated category lists.
    """
    if not raw:
        return None

    categories = []
    for item in raw:
        categories.extend([token for token in item.split(",") if token])

    return categories or None


def _save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _save_png(path, arr):
    Image.fromarray(arr).save(path)

def vis_sample(sample, output_dir):
    xray_colour = np.clip(sample["xray_images"][0], 0, 255).astype(np.uint8)
    Image.fromarray(xray_colour).save(os.path.join(output_dir, "temp_ori.png"))

    bbox_img = xray_colour.copy()
    for bbox in sample["objects"]["xray_bbox"]:
        if not bbox:
            continue
        x, y, w, h = [int(round(v)) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(bbox_img, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
    Image.fromarray(bbox_img).save(os.path.join(output_dir, "temp_bbox.png"))

    mask_img = xray_colour.copy()
    for idx, (mask, bbox) in enumerate(zip(sample["objects"]["xray_mask"], sample["objects"]["xray_bbox"])):
        mask_arr = np.array(mask, dtype=np.uint8)
        if mask_arr.max() == 0:
            continue
        color = (
            int((37 * (idx + 1)) % 255),
            int((91 * (idx + 1)) % 255),
            int((173 * (idx + 1)) % 255),
        )
        mask_bool = mask_arr > 0
        mask_img[mask_bool] = (0.6 * mask_img[mask_bool] + 0.4 * np.array(color)).astype(np.uint8)

        if bbox:
            x, y, w, h = [int(round(v)) for v in bbox]
            if w > 0 and h > 0:
                cv2.rectangle(mask_img, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 2)

    Image.fromarray(mask_img).save(os.path.join(output_dir, "temp_mask.png"))


def main():
    parser = argparse.ArgumentParser(description="Create synthetic DET-COMPASS samples with optional class filtering and homography jitter.")
    parser.add_argument("--output_dir", default="/path/to/xray/datasets/det_compass_syn_debug", help="Directory where synthetic samples are written.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of synthetic samples to generate.")
    parser.add_argument("--min_objs", type=int, default=3, help="Minimum number of objects per image.")
    parser.add_argument("--max_objs", type=int, default=15, help="Maximum number of objects per image.")
    parser.add_argument("--allowed_category_num", type=int, default=109, help="Minimum number of objects per image.")
    parser.add_argument("--max_perspective", type=float, default=0.1, help="Maximum perspective jitter ratio per corner for homography.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument("--dataset_json_path", type=str, default="/path/to/xray/datasets/det_compass_syn/det_compass_syn/default_coco_xray_v2.json")
    parser.add_argument("--save_png", action="store_true", default=True, help="Save xray PNGs; disable to speed up generation.")
    parser.add_argument("--save_pickle", action="store_true", default=True, help="Save pickle files; disable to speed up generation.")
    parser.add_argument("--build_segmentation", action="store_true", default=True, help="Build COCO polygon segmentation from masks; disable for bbox-only speed.")
    parser.add_argument("--io_workers", type=int, default=4, help="Thread workers for async IO (PNG/pickle saves).")


    
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # allowed_categories = _parse_categories(args.categories)

    print('Loading "lorenzovaquero/DET-COMPASS-CROPS" dataset...')
    dataset = load_dataset(
        "lorenzovaquero/DET-COMPASS-CROPS",
        split="train",
        # cache_dir="/path/to/hf_cache"
    )
    dataset_xp = load_dataset(
        "PAGF/DET-COMPASS",
        split="test",
    )

    print("Splitting dataset...")
    foreground_samples, background_samples = split_dataset(dataset, only_xray_visible=True, only_rgb_visible=True)

    print('Loading information of DET_COMPASS_SYN from {}'.format(args.dataset_json_path))
    with open(args.dataset_json_path) as f:
        info_syn = json.load(f)

    dangerous_categories = set([10,28,31,39,44,71,76,101,105,110,132,142,150,181,190,204,224,227,229,310])
    allowed_categories = [info_syn['categories'][idx]['name'] for idx in (set(range(0, args.allowed_category_num)) - dangerous_categories)]
    print("allowed category num:{}".format(len(allowed_categories)))
    filtered_foreground = [s for s in foreground_samples if s.get('category', 'unknown') in allowed_categories]
    category_name_to_id = {c["name"]: c["id"] for c in info_syn.get("categories", [])}

    if args.seed is not None:
        set_seed(args.seed)

    coco_output = {
        "info": info_syn.get("info", {}),
        "licenses": info_syn.get("licenses", []),
        "images": [],
        "annotations": [],
        "categories": info_syn.get("categories", []),
    }
    annotation_id = 0
    coco_json_path = os.path.join(output_dir, "coco.json")
    xray_dir = os.path.join(output_dir, "xray")
    if args.save_png:
        os.makedirs(xray_dir, exist_ok=True)

    io_executor = ThreadPoolExecutor(max_workers=args.io_workers if args.io_workers > 0 else 1)
    io_futures = []
    coco_bbox_count = 0
    coco_seg_count = 0
    sample_bbox_count_sum = 0
    sample_mask_count_sum = 0
    sample_bbox_ori_count_sum = 0
    sample_mask_ori_count_sum = 0
    for i in tqdm(range(args.num_samples)):
        sample = create_synthetic_sample(
            foreground_samples=filtered_foreground,
            background_samples=background_samples,
            generate_rgb=False,
            min_objs=args.min_objs,
            max_objs=args.max_objs,
            max_perspective=args.max_perspective,
        )
        vis_sample(sample, output_dir)

        if args.save_pickle:
            filename = f"{i:06d}_sample.pkl"
            filepath = os.path.join(output_dir, filename)
            io_futures.append(io_executor.submit(_save_pickle, filepath, sample))

        png_filename = f"{i:06d}.png"
        if args.save_png:
            png_path = os.path.join(xray_dir, png_filename)
            xray_colour = np.clip(sample['xray_images'][0], 0, 255).astype(np.uint8)
            io_futures.append(io_executor.submit(_save_png, png_path, xray_colour))

        coco_output["images"].append(
            {
                "id": i,
                "file_name": png_filename,
                "width": sample["xray_width"],
                "height": sample["xray_height"],
            }
        )

        sample_bbox_count = 0
        sample_mask_count = 0
        sample_bbox_ori_count = len(sample["objects"]["xray_bbox"])
        sample_mask_ori_count = len(sample["objects"]["xray_mask"])
        for bbox, mask, category_name in zip(sample["objects"]["xray_bbox"], sample["objects"]["xray_mask"], sample["objects"]["categories"]):
            category_id = category_name_to_id.get(category_name)
            segmentation = []
            mask_area = bbox[2] * bbox[3] if bbox else 0

            if args.build_segmentation:
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
                if mask_area > 0:
                    sample_mask_count += 1
            else:
                mask_arr = np.array(mask, dtype=np.uint8)
                if mask_arr.max() > 0:
                    sample_mask_count += 1

            if bbox and bbox[2] > 0 and bbox[3] > 0:
                sample_bbox_count += 1
                coco_bbox_count += 1
            if segmentation:
                coco_seg_count += 1

            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": i,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": mask_area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )
            annotation_id += 1
            
        sample_bbox_count_sum += sample_bbox_count
        sample_mask_count_sum += sample_mask_count
        sample_bbox_ori_count_sum += sample_bbox_ori_count
        sample_mask_ori_count_sum += sample_mask_ori_count
        print(
            f"[{i + 1}/{args.num_samples}] "
            f"sample bbox/mask: {sample_bbox_count}/{sample_mask_count} | "
            f"coco bbox/seg: {coco_bbox_count}/{coco_seg_count} | "
            f"sample bbox ori/mask ori: {sample_bbox_ori_count}/{sample_mask_ori_count} | "
            f"sample bbox all/mask all: {sample_bbox_count_sum}/{sample_mask_count_sum} | "
            f"sample bbox ori all/mask ori all: {sample_bbox_ori_count_sum}/{sample_mask_ori_count_sum}"
        )

    with open(coco_json_path, "w") as f:
        json.dump(coco_output, f)
    for future in io_futures:
        future.result()
    io_executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
