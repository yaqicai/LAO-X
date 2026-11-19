"""Evaluate object-bank COCO files against a labelled COCO file (bbox AP/AR).

Images are matched by file name; prediction scores default to 1.0.

    python eval_coco_cutler_against_gt.py --gt_path coco_agnostic.json \
        --pred_paths coco_cutler_musc.json coco_cutler_ori.json [--output_json metrics.json]
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO_STAT_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR@1",
    "AR@10",
    "AR@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _ensure_coco_meta(coco_data: Dict) -> Dict:
    coco_data.setdefault("info", {})
    coco_data.setdefault("licenses", [])
    coco_data.setdefault("images", [])
    coco_data.setdefault("annotations", [])
    coco_data.setdefault("categories", [])
    return coco_data


def _build_gt_filename_maps(gt_coco: Dict) -> Tuple[Dict[str, int], List[int]]:
    filename_to_gt_id = {}
    img_ids = []
    for img in gt_coco.get("images", []):
        fname = os.path.basename(img["file_name"])
        filename_to_gt_id[fname] = img["id"]
        img_ids.append(img["id"])
    return filename_to_gt_id, sorted(img_ids)


def _prepare_detection_results(gt_coco: Dict, pred_coco: Dict) -> Tuple[List[Dict], Dict]:
    filename_to_gt_id, gt_img_ids = _build_gt_filename_maps(gt_coco)
    pred_images_by_id = {img["id"]: img for img in pred_coco.get("images", [])}

    results = []
    matched_pred_image_ids = set()
    skipped_annotations = 0

    for ann in pred_coco.get("annotations", []):
        pred_img = pred_images_by_id.get(ann.get("image_id"))
        if pred_img is None:
            skipped_annotations += 1
            continue
        fname = os.path.basename(pred_img.get("file_name", ""))
        gt_image_id = filename_to_gt_id.get(fname)
        if gt_image_id is None:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            skipped_annotations += 1
            continue
        category_id = int(ann.get("category_id", 1))
        results.append(
            {
                "image_id": gt_image_id,
                "category_id": category_id,
                "bbox": [float(v) for v in bbox],
                "score": float(ann.get("score", 1.0)),
            }
        )
        matched_pred_image_ids.add(pred_img["id"])

    summary = {
        "gt_images": len(gt_coco.get("images", [])),
        "gt_annotations": len(gt_coco.get("annotations", [])),
        "pred_images_total": len(pred_coco.get("images", [])),
        "pred_annotations_total": len(pred_coco.get("annotations", [])),
        "pred_images_matched": len(matched_pred_image_ids),
        "pred_annotations_matched": len(results),
        "pred_annotations_skipped": skipped_annotations,
        "eval_img_ids": gt_img_ids,
    }
    return results, summary


def evaluate_prediction(gt_path: str, pred_path: str) -> Dict:
    gt_data = _ensure_coco_meta(_load_json(gt_path))
    pred_data = _ensure_coco_meta(_load_json(pred_path))
    results, prep = _prepare_detection_results(gt_data, pred_data)

    gt = COCO(gt_path)
    gt.dataset = _ensure_coco_meta(gt.dataset)
    gt.createIndex()
    if results:
        dt = gt.loadRes(results)
    else:
        dt = gt.loadRes([])

    coco_eval = COCOeval(gt, dt, iouType="bbox")
    coco_eval.params.imgIds = prep["eval_img_ids"]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        name: float(value)
        for name, value in zip(COCO_STAT_NAMES, coco_eval.stats.tolist())
    }
    return {
        "prediction_file": pred_path,
        "ground_truth_file": gt_path,
        "prep": {k: v for k, v in prep.items() if k != "eval_img_ids"},
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate COCO annotation files against a GT COCO file with standard COCO bbox metrics."
    )
    parser.add_argument(
        "--gt_path",
        required=True,
        help="Ground-truth COCO json path.",
    )
    parser.add_argument(
        "--pred_paths",
        nargs="+",
        required=True,
        help="Prediction COCO json paths to evaluate.",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Optional path to save aggregated metrics as JSON.",
    )
    args = parser.parse_args()

    all_results = []
    for pred_path in args.pred_paths:
        print(f"\nEvaluating {pred_path}")
        result = evaluate_prediction(args.gt_path, pred_path)
        prep = result["prep"]
        metrics = result["metrics"]
        print(
            json.dumps(
                {
                    "prediction_file": result["prediction_file"],
                    "ground_truth_file": result["ground_truth_file"],
                    "prep": prep,
                    "metrics": metrics,
                },
                indent=2,
            )
        )
        all_results.append(result)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
