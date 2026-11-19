#!/usr/bin/env python3
"""Compare standard COCO and S3 protocols using saved bbox predictions.

The script reports TP, FP, FN, precision, recall, F1, and false discovery
rate (FP / (TP + FP)). Classical FPR is intentionally not reported because
object detection has no finite, well-defined set of true-negative windows.

Paths: annotations are read from ``$LAOX_DATA_ROOT`` (default ``<repo>/datasets``);
saved predictions from ``$LAOX_OUTPUT_EVAL`` (default ``<repo>/output_eval``,
where ``test_det_v2.py`` writes them); SAM 3 predictions from
``$SAM3_OUTPUT_EVAL`` (default: same as ``$LAOX_OUTPUT_EVAL``). Reports go to
``$LAOX_RESULTS_ROOT/analysis/`` (default ``<repo>/results``) unless
``--output-dir`` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("LAOX_DATA_ROOT", PROJECT_ROOT / "datasets"))
RESULT_ROOT = Path(os.environ.get("LAOX_OUTPUT_EVAL", PROJECT_ROOT / "output_eval"))
SAM3_ROOT = Path(os.environ.get("SAM3_OUTPUT_EVAL", RESULT_ROOT))
RESULTS_ROOT = Path(os.environ.get("LAOX_RESULTS_ROOT", PROJECT_ROOT / "results"))

DATASETS = {
    "DET-COMPASS": {
        "annotation": DATA_ROOT
        / "det_compass_xp"
        / "annotations"
        / "coco_agnostic.json",
        "slug": "det_compass_xp",
        "end_id": 307,
    },
    "CLCXray": {
        "annotation": DATA_ROOT / "CLCXray" / "annotations" / "coco_agnostic.json",
        "slug": "clcxray",
        "end_id": 956,
    },
}

# Granularity is fixed once per dataset using the S3-1.5 AP50:95 operating
# point selected in the accompanying protocol comparison.
EXPERIMENTS = {
    ("DET-COMPASS", "SAM2"): {
        "granularity": "0.1",
        "prediction": RESULT_ROOT
        / "det_compass_xp_img0_307_sam2"
        / "coco_instances_results.json",
    },
    ("CLCXray", "SAM2"): {
        "granularity": "0.1",
        "prediction": RESULT_ROOT
        / "clcxray_img0_956_sam2"
        / "coco_instances_results.json",
    },
    ("DET-COMPASS", "SAM3"): {
        "granularity": "0.1",
        "prediction": SAM3_ROOT
        / "det_compass_xp_img0_307_sam3_1.0"
        / "coco_instances_results.json",
    },
    ("CLCXray", "SAM3"): {
        "granularity": "0.1",
        "prediction": SAM3_ROOT
        / "clcxray_img0_956_sam3_1.0"
        / "coco_instances_results.json",
    },
    ("DET-COMPASS", "LAO-X"): {
        "granularity": "0.4",
        "prediction": RESULT_ROOT
        / (
            "det_compass_xp_img0_307_unsamv2_0.4_"
            "det_compass_xp_divide_and_conquer_1000_m0.5"
        )
        / "coco_instances_results.json",
    },
    ("CLCXray", "LAO-X"): {
        "granularity": "0.5",
        "prediction": RESULT_ROOT
        / (
            "clcxray_img0_956_unsamv2_0.5_"
            "det_compass_xp_divide_and_conquer_1000_m0.5"
        )
        / "coco_instances_results.json",
    },
    ("DET-COMPASS", "UnSAMv2"): {
        "granularity": "1.0",
        "prediction": RESULT_ROOT
        / (
            "det_compass_xp_img0_307_unsamv2_1.0_augmentation_restart_"
            "crop_musc_aug_fusion_bs16_xp_extreme_m0.5_48"
        )
        / "coco_instances_results.json",
    },
    ("CLCXray", "UnSAMv2"): {
        "granularity": "1.0",
        "prediction": RESULT_ROOT
        / (
            "clcxray_img0_956_unsamv2_1.0_augmentation_restart_"
            "crop_musc_aug_fusion_bs16_xp_extreme_m0.5_48"
        )
        / "coco_instances_results.json",
    },
}

PROTOCOLS = {
    "Standard COCO": None,
    "S3 1.25": 1.25,
    "S3 1.5": 1.5,
    "S3 1.75": 1.75,
}

CSV_FIELDS = [
    "dataset",
    "method",
    "granularity",
    "protocol",
    "expand_ratio",
    "iou_threshold",
    "operating_point",
    "score_threshold",
    "max_dets",
    "num_gt",
    "num_detections",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "false_discovery_rate",
    "annotation_json",
    "prediction_json",
]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _load_prediction_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if "annotations" in data:
            data = data["annotations"]
        elif "results" in data:
            data = data["results"]
    if not isinstance(data, list):
        raise TypeError(f"Prediction JSON must contain a list: {path}")
    return data


def _expand_bbox(
    bbox: list[float],
    ratio: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in bbox)
    center_x = x + width * 0.5
    center_y = y + height * 0.5
    half_width = width * ratio * 0.5
    half_height = height * ratio * 0.5
    x1 = max(0.0, center_x - half_width)
    y1 = max(0.0, center_y - half_height)
    x2 = min(float(image_width), center_x + half_width)
    y2 = min(float(image_height), center_y + half_height)
    return x1, y1, x2, y2


def _filter_s3_predictions(
    coco_gt: COCO,
    predictions: list[dict[str, Any]],
    expand_ratio: float | None,
) -> list[dict[str, Any]]:
    if expand_ratio is None:
        return predictions

    regions_by_image: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for image_id, image in coco_gt.imgs.items():
        width = float(image["width"])
        height = float(image["height"])
        for annotation in coco_gt.imgToAnns.get(image_id, []):
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                continue
            regions_by_image[image_id].append(
                _expand_bbox(bbox, expand_ratio, width, height)
            )

    filtered = []
    epsilon = 1e-6
    for prediction in predictions:
        bbox = prediction.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, width, height = (float(value) for value in bbox)
        center_x = x + width * 0.5
        center_y = y + height * 0.5
        regions = regions_by_image.get(int(prediction["image_id"]), [])
        if any(
            center_x >= x1 - epsilon
            and center_x <= x2 + epsilon
            and center_y >= y1 - epsilon
            and center_y <= y2 + epsilon
            for x1, y1, x2, y2 in regions
        ):
            filtered.append(prediction)
    return filtered


def _collect_coco_flags(
    coco_gt: COCO,
    predictions: list[dict[str, Any]],
    iou_thresholds: list[float],
    max_dets: int,
) -> dict[float, dict[str, Any]]:
    if not predictions:
        num_gt = sum(not annotation.get("iscrowd", 0) for annotation in coco_gt.anns.values())
        return {
            threshold: {
                "scores": np.empty(0, dtype=float),
                "tp_flags": np.empty(0, dtype=bool),
                "fp_flags": np.empty(0, dtype=bool),
                "num_gt": num_gt,
            }
            for threshold in iou_thresholds
        }

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.useCats = 0
    evaluator.params.iouThrs = np.asarray(iou_thresholds, dtype=float)
    evaluator.params.maxDets = [1, min(10, max_dets), max_dets]
    evaluator.evaluate()

    area_all = np.asarray(evaluator.params.areaRng[0], dtype=float)
    per_threshold = {
        threshold: {
            "scores": [],
            "tp_flags": [],
            "fp_flags": [],
            "num_gt": 0,
        }
        for threshold in iou_thresholds
    }

    for eval_image in evaluator.evalImgs:
        if eval_image is None:
            continue
        if eval_image["maxDet"] != max_dets:
            continue
        if not np.allclose(np.asarray(eval_image["aRng"], dtype=float), area_all):
            continue

        gt_ignore = np.asarray(eval_image["gtIgnore"], dtype=bool)
        scores = np.asarray(eval_image["dtScores"], dtype=float)
        for threshold_index, threshold in enumerate(iou_thresholds):
            dt_ignore = np.asarray(
                eval_image["dtIgnore"][threshold_index],
                dtype=bool,
            )
            dt_matches = np.asarray(
                eval_image["dtMatches"][threshold_index],
                dtype=float,
            )
            valid = ~dt_ignore
            record = per_threshold[threshold]
            record["num_gt"] += int(np.count_nonzero(~gt_ignore))
            record["scores"].extend(scores[valid].tolist())
            record["tp_flags"].extend((dt_matches[valid] > 0).tolist())
            record["fp_flags"].extend((dt_matches[valid] == 0).tolist())

    for record in per_threshold.values():
        scores = np.asarray(record["scores"], dtype=float)
        tp_flags = np.asarray(record["tp_flags"], dtype=bool)
        fp_flags = np.asarray(record["fp_flags"], dtype=bool)
        order = np.argsort(-scores, kind="mergesort")
        record["scores"] = scores[order]
        record["tp_flags"] = tp_flags[order]
        record["fp_flags"] = fp_flags[order]
    return per_threshold


def _metrics_at_mask(
    record: dict[str, Any],
    selected: np.ndarray,
) -> dict[str, float | int]:
    tp = int(np.count_nonzero(record["tp_flags"][selected]))
    fp = int(np.count_nonzero(record["fp_flags"][selected]))
    fn = max(0, int(record["num_gt"]) - tp)
    num_detections = tp + fp
    precision = tp / num_detections if num_detections else 0.0
    recall = tp / int(record["num_gt"]) if record["num_gt"] else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_discovery_rate = fp / num_detections if num_detections else 0.0
    return {
        "num_gt": int(record["num_gt"]),
        "num_detections": num_detections,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_discovery_rate": false_discovery_rate,
    }


def _evaluate_operating_points(
    record: dict[str, Any],
    fixed_score_thresholds: list[float],
) -> list[tuple[str, float, dict[str, float | int]]]:
    scores = record["scores"]
    rows = [
        (
            "all_saved",
            float("-inf"),
            _metrics_at_mask(record, np.ones(scores.shape, dtype=bool)),
        )
    ]
    for threshold in fixed_score_thresholds:
        rows.append(
            (
                f"score>={threshold:g}",
                threshold,
                _metrics_at_mask(record, scores >= threshold),
            )
        )

    if scores.size == 0:
        rows.append(
            (
                "max_f1_test_oracle",
                math.nan,
                _metrics_at_mask(record, np.zeros(scores.shape, dtype=bool)),
            )
        )
        return rows

    cumulative_tp = np.cumsum(record["tp_flags"], dtype=float)
    cumulative_fp = np.cumsum(record["fp_flags"], dtype=float)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    recall = cumulative_tp / max(float(record["num_gt"]), 1.0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    # A score threshold must include every prediction tied at that score.
    # Evaluate only the final item in each equal-score group.
    group_ends = np.r_[scores[:-1] != scores[1:], True]
    candidate_indices = np.flatnonzero(group_ends)
    candidate_f1 = f1[candidate_indices]
    best_index = int(candidate_indices[np.flatnonzero(candidate_f1 == candidate_f1.max())[0]])
    best_threshold = float(scores[best_index])
    selected = scores >= best_threshold
    rows.append(
        (
            "max_f1_test_oracle",
            best_threshold,
            _metrics_at_mask(record, selected),
        )
    )
    return rows


def _serialize_threshold(value: float) -> str | float:
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "all"
    return value


def run_analysis(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for (dataset_name, method), experiment in EXPERIMENTS.items():
        dataset = DATASETS[dataset_name]
        annotation_path = Path(dataset["annotation"])
        prediction_path = Path(experiment["prediction"])
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation JSON: {annotation_path}")
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing prediction JSON: {prediction_path}")

        coco_gt = COCO(str(annotation_path))
        predictions = _load_prediction_list(prediction_path)
        valid_image_ids = set(coco_gt.getImgIds())
        unexpected_image_ids = {
            int(prediction["image_id"])
            for prediction in predictions
            if int(prediction["image_id"]) not in valid_image_ids
        }
        if unexpected_image_ids:
            raise ValueError(
                f"{method}/{dataset_name} has predictions for unknown image ids: "
                f"{sorted(unexpected_image_ids)[:10]}"
            )

        for protocol_name, expand_ratio in PROTOCOLS.items():
            filtered_predictions = _filter_s3_predictions(
                coco_gt,
                predictions,
                expand_ratio,
            )
            records = _collect_coco_flags(
                coco_gt,
                filtered_predictions,
                args.iou_thresholds,
                args.max_dets,
            )
            for iou_threshold, record in records.items():
                for operating_point, score_threshold, metrics in _evaluate_operating_points(
                    record,
                    args.score_thresholds,
                ):
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "method": method,
                            "granularity": experiment["granularity"],
                            "protocol": protocol_name,
                            "expand_ratio": expand_ratio or "",
                            "iou_threshold": iou_threshold,
                            "operating_point": operating_point,
                            "score_threshold": _serialize_threshold(score_threshold),
                            "max_dets": args.max_dets,
                            **metrics,
                            "annotation_json": str(annotation_path),
                            "prediction_json": str(prediction_path),
                        }
                    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze TP/FP/FN under standard COCO and S3 protocols.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "analysis" / "protocol_false_positives",
    )
    parser.add_argument(
        "--iou-thresholds",
        type=_parse_csv_floats,
        default=[0.5, 0.75],
        help="Comma-separated IoU thresholds (default: 0.5,0.75).",
    )
    parser.add_argument(
        "--score-thresholds",
        type=_parse_csv_floats,
        default=[0.9],
        help="Comma-separated fixed score thresholds (default: 0.9).",
    )
    parser.add_argument(
        "--max-dets",
        type=int,
        default=100,
        help="COCO per-image maximum detections (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_dets < 10:
        raise ValueError("--max-dets must be at least 10")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_analysis(args)

    csv_path = args.output_dir / "false_positive_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / "false_positive_metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "definitions": {
                    "precision": "TP / (TP + FP)",
                    "recall": "TP / (TP + FN)",
                    "f1": "2 * precision * recall / (precision + recall)",
                    "false_discovery_rate": "FP / (TP + FP) = 1 - precision",
                    "classical_fpr": (
                        "Not reported: FP / (FP + TN) is not well-defined for "
                        "object detection because the set of background windows is unbounded."
                    ),
                    "max_f1_test_oracle": (
                        "Best F1 selected on the evaluated test predictions; diagnostic only."
                    ),
                },
                "rows": rows,
            },
            handle,
            indent=2,
        )
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote detailed metadata to {json_path}")


if __name__ == "__main__":
    main()
