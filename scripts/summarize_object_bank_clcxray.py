#!/usr/bin/env python3
"""Summarize full CLCXray object-bank evaluations under the paper's S3 protocol."""

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import XrayRoot
from xray_utils import _apply_region_protocol, _get_protocol_preset


DEFAULT_ANNOTATION = Path(XrayRoot["CLCXray"]) / "annotations" / "coco_agnostic.json"
GRANULARITIES = tuple(f"{value / 10:.1f}" for value in range(1, 11))
METHODS = {
    "cutler_n1": "CutLER N1 object bank",
    "cutler_n3": "CutLER N3 object bank",
    "musc_thr0p8": "MuSC (thr=0.8) object bank",
}
STAGES = {"easy": 80, "hard": 64}

# Values reported for CLCXray in Table 4. The selected granularities were
# recovered from the corresponding saved full-dataset evaluation workbooks.
TABLE4 = {
    "easy": {
        "granularity": "0.6",
        "ap50": 35.29117726768965,
        "ap75": 24.843115464722725,
        "ap50_95": 22.91977410485684,
        "ar1": 26.84025334271639,
        "ar10": 38.923293455313164,
    },
    "hard": {
        "granularity": "0.7",
        "ap50": 36.85299100105971,
        "ap75": 27.073077820928404,
        "ap50_95": 24.157883749919122,
        "ar1": 27.62139338494018,
        "ar10": 36.82617874736101,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("LAOX_OUTPUT_EVAL", REPO_ROOT / "output_eval")),
    )
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("LAOX_RESULTS_ROOT", REPO_ROOT / "results"))
        / "analysis"
        / "object_bank_clcxray",
    )
    return parser.parse_args()


def prediction_dir(output_root, method, stage, epoch, granularity):
    name = (
        f"clcxray_img0_956_unsamv2_{granularity}_augmentation_restart_"
        f"crop_objbank_{method}_fusion_bs16_seed1_{stage}_m0.5_{epoch}"
    )
    return output_root / name


def require_complete(output_root, method, stage, epoch, granularity):
    directory = prediction_dir(output_root, method, stage, epoch, granularity)
    progress_path = directory / "coco_instances_progress.json"
    prediction_path = directory / "coco_instances_results.json"
    final_path = directory / "instances_predictions.pth"
    missing = [
        str(path) for path in (progress_path, prediction_path, final_path) if not path.is_file()
    ]
    if missing:
        raise RuntimeError("Incomplete evaluation; missing: " + ", ".join(missing))
    with progress_path.open("r", encoding="utf-8") as f:
        num_processed = len(json.load(f).get("processed_image_ids", []))
    if num_processed != 956:
        raise RuntimeError(
            f"Incomplete evaluation: {directory.name} has {num_processed}/956 images"
        )
    return prediction_path


def evaluate(coco_gt, prediction_path, protocol):
    with prediction_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
        filtered, _ = _apply_region_protocol(coco_gt, coco_dt, protocol, "bbox")
        filtered_dt = coco_gt.loadRes(filtered)
        evaluator = COCOeval(coco_gt, filtered_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return {
        "ap50": 100.0 * float(stats[1]),
        "ap75": 100.0 * float(stats[2]),
        "ap50_95": 100.0 * float(stats[0]),
        "ar1": 100.0 * float(stats[6]),
        "ar10": 100.0 * float(stats[7]),
    }


def render_table(stage, rows):
    lines = [
        f"### {stage.capitalize()} checkpoint on CLCXray (S3, 1.5x)",
        "",
        "| Object bank | Granularity | AP50 | AP75 | AP50:95 | AR1 | AR10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {granularity} | {ap50:.2f} | {ap75:.2f} | "
            "{ap50_95:.2f} | {ar1:.2f} | {ar10:.2f} |".format(**row)
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    if not args.annotation.is_file():
        raise FileNotFoundError(args.annotation)
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(args.annotation))
    coco_gt.dataset.setdefault("info", {})
    protocol = _get_protocol_preset("s3_r1p5_center")[0]

    payload = {
        "dataset": "CLCXray",
        "num_images": len(coco_gt.getImgIds()),
        "annotation": str(args.annotation),
        "protocol": protocol["name"],
        "selection_metric": "AP50:95",
        "stages": {},
    }
    tables = []
    for stage in args.stages:
        epoch = STAGES[stage]
        table4_row = {"method": "Original object bank (Table 4)", **TABLE4[stage]}
        rows = [table4_row]
        stage_results = []
        for method, label in METHODS.items():
            candidates = []
            for granularity in GRANULARITIES:
                prediction_path = require_complete(
                    args.output_root, method, stage, epoch, granularity
                )
                metrics = evaluate(coco_gt, prediction_path, protocol)
                candidates.append({"granularity": granularity, **metrics})
            best = max(candidates, key=lambda row: row["ap50_95"])
            rows.append({"method": label, **best})
            stage_results.append(
                {"method": method, "label": label, "best": best, "all": candidates}
            )
        payload["stages"][stage] = {
            "epoch": epoch,
            "table4": TABLE4[stage],
            "object_banks": stage_results,
        }
        tables.append(render_table(stage, rows))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "clcxray_easy_hard_s3_1p5.json"
    markdown_path = args.output_dir / "clcxray_easy_hard_s3_1p5.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown = "\n\n".join(tables) + "\n"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
