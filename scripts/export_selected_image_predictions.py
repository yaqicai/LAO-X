#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import XrayImgRoot, XrayRoot


DEFAULT_OUTPUT_EVAL_DIR = "/path/to/results/output_eval"
DEFAULT_OUTPUT_EVAL_DIR_SAM3 = "/path/to/sam3/output_eval"

DEFAULT_DATASETS = ["CLCXray", "DvXray", "PIXray", "HiXray", "pidray"]
DEFAULT_VIS_IMAGE_IDS_MATRIX = (
    "78,204,225,271,428;"
    "259,439,593,1011,1090,1159,1630,2162,2204,2224,2228,2242,2342,2405,2410,2763,2941,3135,3406,3577,3615,3652,3738,3740,3786,3790,3802,3959,3992,4078,4186,4264,4604,5244,5278,5296,5364,5645,5661;"
    "419,459,471,532,544,732,963,1007,1019,1134;"
    "7331,7536,7573,8128;"
    "170,919,1232,1500,1506,1570,2310,2775,2866,3089,3171,3559,3713,4015,5853,6310,6565,8105,8115,13518,14472,15294,15295,25747,18079"
)

DEFAULT_SUFFIXES = [
    "_augmentation_restart_crop_musc_aug_fusion_bs16_extreme_m0.5_48",
    "_det_compass_xp_divide_and_conquer_1000_m0.5",
    "_sam2",
    "_sam3",
]
DEFAULT_MODEL_NAMES = ["unsamv2", "unsamv2", "sam2", "sam3"]
DEFAULT_METHOD_ALIASES = ["ours", "unsamv2", "sam2", "sam3"]
DEFAULT_GRAS_BY_METHOD = [
    "1.0,1.0,1.0,0.3,1.0",
    "0.5,0.3,0.3,0.3,0.5",
    "0.0,0.0,0.0,0.0,0.0",
    "0.0,0.0,0.0,0.0,0.0",
]


def resolve_val_json(dataset_root):
    candidates = [
        os.path.join(dataset_root, "val.json"),
        os.path.join(dataset_root, "default_coco_xray_v2.json"),
        os.path.join(dataset_root, "coco_agnostic.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_image_path(file_name, image_root, dataset_root):
    if os.path.isabs(file_name):
        return file_name
    cand = os.path.join(image_root, file_name)
    if os.path.exists(cand):
        return cand
    return os.path.join(dataset_root, file_name)


def get_end_id(dataset):
    mapping = {
        "det_compass_xp": 307,
        "CLCXray": 956,
        "DvXray": 9155,
        "PIXray": 5047,
        "HiXray": 9069,
        "pidray": 18220,
    }
    return mapping.get(dataset, 1928)


def parse_matrix(matrix_str):
    rows = []
    for row in str(matrix_str or "").split(";"):
        row = row.strip()
        if not row:
            rows.append([])
            continue
        values = []
        for token in row.replace("，", ",").split(","):
            token = token.strip()
            if not token:
                continue
            values.append(int(token))
        rows.append(values)
    return rows


def resolve_gra_list_for_dataset(dataset):
    if dataset not in DEFAULT_DATASETS:
        raise ValueError(f"Dataset not in default run_visualization list: {dataset}")
    dataset_idx = DEFAULT_DATASETS.index(dataset)
    gra_list = []
    for row in DEFAULT_GRAS_BY_METHOD:
        parts = [x.strip() for x in row.split(",")]
        if len(parts) != len(DEFAULT_DATASETS):
            raise ValueError("Invalid DEFAULT_GRAS_BY_METHOD configuration.")
        gra_list.append(parts[dataset_idx])
    return gra_list


def build_prediction_path(dataset, start_id, end_id, suffix, model_name, gra, eval_root):
    if suffix == "_sam2":
        output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}{suffix}"
    elif suffix == "_sam3":
        output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}{suffix}_1.0"
    else:
        output_subdir = f"{dataset.lower()}_img{start_id}_{end_id}_{model_name}_{gra}{suffix}"
    return os.path.join(eval_root, output_subdir, "coco_instances_results.json")


def choose_default_image_id(dataset):
    rows = parse_matrix(DEFAULT_VIS_IMAGE_IDS_MATRIX)
    if dataset not in DEFAULT_DATASETS:
        return 78
    di = DEFAULT_DATASETS.index(dataset)
    if di >= len(rows) or not rows[di]:
        return 78
    return int(rows[di][0])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export one image and all methods' COCO predictions to a single JSON."
    )
    parser.add_argument("--dataset", default="CLCXray")
    parser.add_argument("--image-id", type=int, default=None)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--output-eval-dir", default=DEFAULT_OUTPUT_EVAL_DIR)
    parser.add_argument("--output-eval-dir-sam3", default=DEFAULT_OUTPUT_EVAL_DIR_SAM3)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--embed-image-base64", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    image_id = args.image_id if args.image_id is not None else choose_default_image_id(dataset)
    end_id = args.end_id if args.end_id is not None else get_end_id(dataset)

    dataset_root = XrayRoot.get(dataset)
    if not dataset_root:
        raise ValueError(f"Unknown dataset: {dataset}")
    val_json = resolve_val_json(dataset_root)
    if not val_json:
        raise FileNotFoundError(f"No val json found for dataset: {dataset}")

    with open(val_json, "r", encoding="utf-8") as f:
        coco_gt = json.load(f)
    image_map = {int(img.get("id")): img for img in coco_gt.get("images", [])}
    if int(image_id) not in image_map:
        raise ValueError(f"image_id={image_id} not found in {val_json}")
    image_info = image_map[int(image_id)]

    image_root = XrayImgRoot.get(dataset, os.path.join(dataset_root, "xray"))
    image_path = resolve_image_path(image_info.get("file_name", ""), image_root, dataset_root)
    image_exists = os.path.exists(image_path)

    image_payload = {
        "coco": image_info,
        "resolved_path": image_path,
        "exists": bool(image_exists),
    }
    if args.embed_image_base64 and image_exists:
        with open(image_path, "rb") as f:
            image_payload["base64"] = base64.b64encode(f.read()).decode("utf-8")

    gra_list = resolve_gra_list_for_dataset(dataset)
    methods = []
    for suffix, model_name, alias, gra in zip(
        DEFAULT_SUFFIXES, DEFAULT_MODEL_NAMES, DEFAULT_METHOD_ALIASES, gra_list
    ):
        eval_root = args.output_eval_dir_sam3 if suffix == "_sam3" else args.output_eval_dir
        pred_path = build_prediction_path(
            dataset=dataset,
            start_id=args.start_id,
            end_id=end_id,
            suffix=suffix,
            model_name=model_name,
            gra=gra,
            eval_root=eval_root,
        )
        method_payload = {
            "alias": alias,
            "model_name": model_name,
            "suffix": suffix,
            "gra": gra,
            "prediction_file": pred_path,
            "prediction_file_exists": os.path.exists(pred_path),
            "predictions_coco": [],
            "num_predictions": 0,
        }
        if os.path.exists(pred_path):
            with open(pred_path, "r", encoding="utf-8") as f:
                preds = json.load(f)
            selected = [ann for ann in preds if int(ann.get("image_id", -1)) == int(image_id)]
            selected = sorted(selected, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            method_payload["predictions_coco"] = selected
            method_payload["num_predictions"] = len(selected)
        methods.append(method_payload)

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_script": "scripts/run_visualization.sh",
            "dataset": dataset,
            "image_id": int(image_id),
            "start_id": int(args.start_id),
            "end_id": int(end_id),
            "method_aliases": DEFAULT_METHOD_ALIASES,
        },
        "image": image_payload,
        "methods": methods,
    }

    output_json = args.output_json
    if output_json is None:
        output_json = os.path.join(
            "output_visualization_reorganized_full_select",
            f"{dataset}_img{image_id}_all_methods_predictions.json",
        )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
