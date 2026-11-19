#!/usr/bin/env python3
"""Summarize quick CLCXray sanity-check predictions with the S3 protocol."""

import argparse
import csv
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import XrayRoot
from util_xray import _apply_region_protocol, _get_protocol_preset, _slice_coco_json


DEFAULT_ANNOTATION = Path(XrayRoot["CLCXray"]) / "annotations" / "coco_agnostic.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=32)
    parser.add_argument("--protocol", default="s3_r1p5_center")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def evaluate_prediction(coco_gt, prediction_path, protocol):
    with prediction_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    if not predictions:
        return [0.0] * 12, {
            "num_dt_total": 0,
            "num_dt_kept": 0,
            "images_with_regions": 0,
        }

    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
        filtered, meta = _apply_region_protocol(coco_gt, coco_dt, protocol, "bbox")
        if not filtered:
            return [0.0] * 12, meta
        filtered_dt = coco_gt.loadRes(filtered)
        evaluator = COCOeval(coco_gt, filtered_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return [float(value) for value in evaluator.stats], meta


def markdown_table(rows, start_id, end_id, protocol_name):
    lines = [
        f"CLCXray quick sanity check: images [{start_id}, {end_id}), protocol `{protocol_name}`.",
        "",
        "| Object bank | Checkpoint | AP50 | AP75 | AP50:95 | AR1 | AR10 | Kept predictions | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        checkpoint = f"{row['stage']} / epoch {row['epoch']}"
        if row["status"] == "ok":
            metric_values = [
                f"{row['ap50']:.2f}",
                f"{row['ap75']:.2f}",
                f"{row['ap50_95']:.2f}",
                f"{row['ar1']:.2f}",
                f"{row['ar10']:.2f}",
                f"{row['num_dt_kept']}/{row['num_dt_total']}",
            ]
        else:
            metric_values = ["-", "-", "-", "-", "-", "-"]
        lines.append(
            "| {method} | {checkpoint} | {values} | {status} |".format(
                method=row["method"],
                checkpoint=checkpoint,
                values=" | ".join(metric_values),
                status=row["status"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if not args.annotation.is_file():
        raise FileNotFoundError(f"CLCXray annotation not found: {args.annotation}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    subset_path = args.output_json.parent / (
        f"clcxray_subset_{args.start_id}_{args.end_id}.json"
    )
    _slice_coco_json(
        str(args.annotation), args.start_id, args.end_id, str(subset_path)
    )
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(subset_path))

    protocols = _get_protocol_preset(args.protocol)
    if len(protocols) != 1:
        raise ValueError(
            f"Quick evaluation requires exactly one protocol, got {len(protocols)}"
        )
    protocol = protocols[0]

    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        manifest_rows = list(csv.DictReader(f, delimiter="\t"))

    results = []
    for manifest_row in manifest_rows:
        prediction_path = Path(manifest_row["prediction"])
        row = dict(manifest_row)
        row["epoch"] = int(row["epoch"])
        row["seed"] = int(row["seed"])
        if manifest_row.get("status") != "ok" or not prediction_path.is_file():
            row["status"] = manifest_row.get("status") or "missing_prediction"
            results.append(row)
            continue
        try:
            stats, meta = evaluate_prediction(coco_gt, prediction_path, protocol)
            row.update(
                {
                    "status": "ok",
                    "ap50_95": 100.0 * stats[0],
                    "ap50": 100.0 * stats[1],
                    "ap75": 100.0 * stats[2],
                    "ar1": 100.0 * stats[6],
                    "ar10": 100.0 * stats[7],
                    "num_dt_total": int(meta["num_dt_total"]),
                    "num_dt_kept": int(meta["num_dt_kept"]),
                }
            )
        except Exception as exc:  # Keep the other checkpoints reportable.
            row["status"] = f"evaluation_failed: {exc}"
        results.append(row)

    payload = {
        "dataset": "CLCXray",
        "start_id": args.start_id,
        "end_id": args.end_id,
        "num_images": args.end_id - args.start_id,
        "protocol": protocol["name"],
        "annotation": str(args.annotation),
        "results": results,
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    table = markdown_table(results, args.start_id, args.end_id, protocol["name"])
    args.output_markdown.write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()
