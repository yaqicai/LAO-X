#!/usr/bin/env python3
"""Evaluate three training seeds under the fixed S3-1.5 detection protocol."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from analyze_protocol_false_positives import (
    _filter_s3_predictions,
    _load_prediction_list,
)


PROJECT_ROOT = Path("/path/to/LAO-X")
RESULT_ROOT = Path("/path/to/results/output_eval")
PROJECT_OUTPUT = PROJECT_ROOT / "output_eval"
DATA_ROOT = Path("/path/to/datasets")

DATASETS = {
    "DET-COMPASS": {
        "slug": "det_compass_xp",
        "end_id": 307,
        "annotation": DATA_ROOT
        / "det_compass_xp"
        / "annotations"
        / "coco_agnostic.json",
    },
    "CLCXray": {
        "slug": "clcxray",
        "end_id": 956,
        "annotation": DATA_ROOT / "CLCXray" / "annotations" / "coco_agnostic.json",
    },
}

EXPERIMENTS = {
    1: {
        "run": "original",
        "training_seed": 1234,
        "deterministic_controlled": False,
        "root": RESULT_ROOT,
        "suffix": "augmentation_restart_crop_musc_aug_fusion_bs16_extreme_m0.5_48",
    },
    2: {
        "run": "seed2",
        "training_seed": 2,
        "deterministic_controlled": True,
        "root": PROJECT_OUTPUT,
        "suffix": (
            "augmentation_restart_crop_musc_aug_fusion_bs16_seed2_extreme_m0.5_48"
        ),
    },
    3: {
        "run": "seed3",
        "training_seed": 3,
        "deterministic_controlled": True,
        "root": PROJECT_OUTPUT,
        "suffix": (
            "augmentation_restart_crop_musc_aug_fusion_bs16_seed3_extreme_m0.5_48"
        ),
    },
}

METRICS = {
    "AP50": 1,
    "AP75": 2,
    "AP50:95": 0,
    "AR1": 6,
    "AR10": 7,
}


def prediction_path(dataset: dict, seed: int) -> Path:
    experiment = EXPERIMENTS[seed]
    directory = (
        f"{dataset['slug']}_img0_{dataset['end_id']}_unsamv2_1.0_"
        f"{experiment['suffix']}"
    )
    return experiment["root"] / directory / "coco_instances_results.json"


def evaluate(annotation_path: Path, result_path: Path) -> dict[str, float]:
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation_path))
        predictions = _load_prediction_list(result_path)
        predictions = _filter_s3_predictions(coco_gt, predictions, 1.5)
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.useCats = 0
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {metric: float(evaluator.stats[index]) for metric, index in METRICS.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "seed_robustness",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    for dataset_name, dataset in DATASETS.items():
        for seed in EXPERIMENTS:
            result_path = prediction_path(dataset, seed)
            seed_rows.append(
                {
                    "dataset": dataset_name,
                    "run": EXPERIMENTS[seed]["run"],
                    "training_seed": EXPERIMENTS[seed]["training_seed"],
                    "deterministic_controlled": EXPERIMENTS[seed][
                        "deterministic_controlled"
                    ],
                    "granularity": 1.0,
                    "protocol": "S3_gt_bbox_expand1p50_center",
                    **evaluate(Path(dataset["annotation"]), result_path),
                    "annotation_json": str(dataset["annotation"]),
                    "prediction_json": str(result_path),
                }
            )

    summary_rows = []
    for dataset_name in DATASETS:
        dataset_rows = [row for row in seed_rows if row["dataset"] == dataset_name]
        for metric in METRICS:
            values = np.asarray([row[metric] for row in dataset_rows], dtype=float)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "n": values.size,
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "sample_variance": float(values.var(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

    seed_csv = args.output_dir / "seed_metrics.csv"
    with seed_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_rows[0].keys())
        writer.writeheader()
        writer.writerows(seed_rows)

    summary_csv = args.output_dir / "seed_metrics_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    output_json = args.output_dir / "seed_metrics.json"
    output_json.write_text(
        json.dumps(
            {
                "protocol": {
                    "name": "S3_gt_bbox_expand1p50_center",
                    "expand_ratio": 1.5,
                    "filter": "prediction bbox center inside expanded GT bbox",
                    "granularity": 1.0,
                    "class_agnostic": True,
                    "max_dets": [1, 10, 100],
                },
                "seed_rows": seed_rows,
                "summary_rows": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {seed_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
