#!/usr/bin/env python3
"""Evaluate saved any-object predictions on the PIDray hidden subset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path("/path/to/LAO-X")
DATA_ROOT = Path("/path/to/datasets/pidray/annotations")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util_xray import _apply_region_protocol, _get_protocol_preset

DEFAULT_PREDICTIONS = {
    "SAM2": Path(
        "/path/to/results/output_eval/"
        "pidray_img0_18220_sam2/coco_instances_results.json"
    ),
    "SAM3": Path(
        "/path/to/sam3/output_eval/"
        "pidray_img0_18220_sam3_1.0/coco_instances_results.json"
    ),
    "LAO-X": Path(
        "/path/to/results/output_eval/"
        "pidray_img0_18220_unsamv2_0.5_"
        "det_compass_xp_divide_and_conquer_1000_m0.5/"
        "coco_instances_results.json"
    ),
    "UnSAMv2": PROJECT_ROOT
    / "output_eval"
    / (
        "pidray_img0_18220_unsamv2_1.0_augmentation_restart_"
        "crop_musc_aug_fusion_bs16_xp_extreme_m0.5_48"
    )
    / "coco_instances_results.json",
}

GRANULARITIES = {
    "SAM2": "0.1",
    "SAM3": "0.1",
    "LAO-X": "0.5",
    "UnSAMv2": "1.0",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_hidden_gt(hidden_json: Path, full_json: Path) -> tuple[dict, set[int]]:
    hidden = load_json(hidden_json)
    full = load_json(full_json)

    hidden_names = {image["file_name"] for image in hidden["images"]}
    selected_images = [
        deepcopy(image)
        for image in full["images"]
        if image["file_name"] in hidden_names
    ]
    selected_names = {image["file_name"] for image in selected_images}
    missing = sorted(hidden_names - selected_names)
    if missing:
        raise ValueError(
            f"{len(missing)} hidden images are missing from the full-test annotation; "
            f"examples: {missing[:5]}"
        )

    image_ids = {image["id"] for image in selected_images}
    selected_annotations = [
        deepcopy(annotation)
        for annotation in full["annotations"]
        if annotation["image_id"] in image_ids
    ]
    for annotation in selected_annotations:
        annotation["category_id"] = 1

    subset = {
        "info": deepcopy(full.get("info", {})),
        "licenses": deepcopy(full.get("licenses", full.get("license", []))),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    return subset, image_ids


def coco_from_dataset(dataset: dict) -> COCO:
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def evaluate_prediction(
    coco_gt: COCO,
    image_ids: set[int],
    prediction_path: Path,
    protocol: dict,
) -> tuple[dict, int, int]:
    predictions = load_json(prediction_path)
    subset_predictions = [
        prediction
        for prediction in predictions
        if prediction.get("image_id") in image_ids
    ]
    for prediction in subset_predictions:
        prediction["category_id"] = 1

    coco_dt = coco_gt.loadRes(subset_predictions)
    filtered_predictions, _ = _apply_region_protocol(
        coco_gt, coco_dt, protocol, eval_mode="bbox"
    )
    filtered_dt = coco_gt.loadRes(filtered_predictions)

    evaluator = COCOeval(coco_gt, filtered_dt, "bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.catIds = [1]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    metrics = {
        "AP50:95": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
    }
    return metrics, len(subset_predictions), len(filtered_predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden-json",
        type=Path,
        default=DATA_ROOT / "xray_test_hidden.json",
    )
    parser.add_argument(
        "--full-json",
        type=Path,
        default=PROJECT_ROOT / "output_eval" / "temp_pidray_val_0_18220.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output_eval" / "pidray_hidden_eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hidden_gt, image_ids = build_hidden_gt(args.hidden_json, args.full_json)
    hidden_gt_path = args.output_dir / "pidray_hidden_agnostic_full_ids.json"
    with hidden_gt_path.open("w", encoding="utf-8") as handle:
        json.dump(hidden_gt, handle)

    coco_gt = coco_from_dataset(hidden_gt)
    protocol = _get_protocol_preset("s3_r1p5_center")[0]
    rows = []
    for method, prediction_path in DEFAULT_PREDICTIONS.items():
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing {method} prediction: {prediction_path}")
        metrics, num_before, num_after = evaluate_prediction(
            coco_gt, image_ids, prediction_path, protocol
        )
        rows.append(
            {
                "method": method,
                "granularity": GRANULARITIES[method],
                **metrics,
                "num_predictions_before_s3": num_before,
                "num_predictions_after_s3": num_after,
                "prediction_json": str(prediction_path),
            }
        )

    metadata = {
        "protocol": "S3_gt_bbox_expand1p50_center",
        "num_images": len(hidden_gt["images"]),
        "num_annotations": len(hidden_gt["annotations"]),
        "min_image_id": min(image_ids),
        "max_image_id": max(image_ids),
        "hidden_source": str(args.hidden_json),
        "full_annotation_source": str(args.full_json),
        "filtered_annotation": str(hidden_gt_path),
        "results": rows,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    with (args.output_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
