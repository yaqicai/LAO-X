#!/usr/bin/env python3
"""Evaluate DET-COMPASS invisible predictions with the paper's S3 protocol."""

import argparse
import csv
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xray_utils import _apply_region_protocol, _get_protocol_preset


METRICS = {
    "AP50:95": 0,
    "AP50": 1,
    "AP75": 2,
    "AR1": 6,
    "AR10": 7,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--output-prefix",
        default="output_eval/det_compass_invisible_s3_1p5",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_coco(dataset):
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def normalize_predictions(predictions, image_ids):
    valid_ids = set(image_ids)
    normalized = []
    for prediction in predictions:
        if prediction.get("image_id") not in valid_ids:
            continue
        item = deepcopy(prediction)
        item["category_id"] = 1
        normalized.append(item)
    return normalized


def evaluate_dataset(dataset, predictions, protocol):
    coco_gt = build_coco(dataset)
    normalized = normalize_predictions(
        predictions, [image["id"] for image in dataset["images"]]
    )
    if normalized:
        with redirect_stdout(StringIO()):
            coco_dt = coco_gt.loadRes(normalized)
        filtered, protocol_meta = _apply_region_protocol(
            coco_gt, coco_dt, protocol, "bbox"
        )
    else:
        filtered = []
        protocol_meta = {
            "images_with_regions": len(dataset["images"]),
            "num_dt_total": 0,
            "num_dt_kept": 0,
            "num_dt_clipped": 0,
        }

    if not filtered:
        return {name: 0.0 for name in METRICS}, protocol_meta

    with redirect_stdout(StringIO()):
        coco_dt = coco_gt.loadRes(filtered)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    metrics = {
        name: max(0.0, float(evaluator.stats[index])) * 100.0
        for name, index in METRICS.items()
    }
    return metrics, protocol_meta


def subset_for_category(dataset, category):
    annotations = [
        deepcopy(annotation)
        for annotation in dataset["annotations"]
        if annotation.get("source_category") == category
    ]
    image_ids = {annotation["image_id"] for annotation in annotations}
    images = [
        deepcopy(image)
        for image in dataset["images"]
        if image["id"] in image_ids
    ]
    return {
        "info": deepcopy(dataset.get("info", {})),
        "licenses": deepcopy(dataset.get("licenses", [])),
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }


def main():
    args = parse_args()
    ground_truth = load_json(args.ground_truth)
    predictions = load_json(args.predictions)
    if isinstance(predictions, dict):
        predictions = predictions.get("annotations", predictions.get("results"))
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a COCO result list")

    protocols = _get_protocol_preset("s3_r1p5_center")
    if len(protocols) != 1:
        raise RuntimeError("S3 preset must resolve to exactly one protocol")
    protocol = protocols[0]

    counts = Counter(
        annotation["source_category"]
        for annotation in ground_truth["annotations"]
    )
    image_name_by_id = {
        image["id"]: image["file_name"] for image in ground_truth["images"]
    }
    overall_metrics, overall_meta = evaluate_dataset(
        ground_truth, predictions, protocol
    )
    rows = [
        {
            "category": "ALL",
            "images": len(ground_truth["images"]),
            "objects": len(ground_truth["annotations"]),
            "image_files": f"{len(ground_truth['images'])} files",
            **overall_metrics,
        }
    ]
    for category in sorted(counts):
        subset = subset_for_category(ground_truth, category)
        metrics, _ = evaluate_dataset(subset, predictions, protocol)
        rows.append(
            {
                "category": category,
                "images": len(subset["images"]),
                "objects": counts[category],
                "image_files": "; ".join(
                    sorted(
                        {
                            image_name_by_id[annotation["image_id"]]
                            for annotation in subset["annotations"]
                        }
                    )
                ),
                **metrics,
            }
        )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{output_prefix}.csv")
    json_path = Path(f"{output_prefix}.json")
    markdown_path = Path(f"{output_prefix}.md")
    fieldnames = [
        "category",
        "images",
        "objects",
        "image_files",
        "AP50",
        "AP75",
        "AP50:95",
        "AR1",
        "AR10",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "protocol": protocol,
        "ground_truth": str(Path(args.ground_truth).resolve()),
        "predictions": str(Path(args.predictions).resolve()),
        "protocol_meta": overall_meta,
        "results": rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    table_lines = [
        "| Category | Images | Objects | Image files | AP50 | AP75 | AP50:95 | AR1 | AR10 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['category']} | {row['images']} | {row['objects']} | "
            f"{row['image_files']} | "
            f"{row['AP50']:.2f} | {row['AP75']:.2f} | "
            f"{row['AP50:95']:.2f} | {row['AR1']:.2f} | "
            f"{row['AR10']:.2f} |"
        )
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(table_lines) + "\n")
    print("\n".join(table_lines))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
