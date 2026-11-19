import argparse
import contextlib
import csv
import datetime
import io
import itertools
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# <LAO-X>/third_party/CutLER/cutler/demo/run_musc_fusion_ablation.py
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # third_party/CutLER
DEFAULT_CUTLER_ROOT = DEFAULT_PROJECT_ROOT / "cutler"
LAOX_ROOT = DEFAULT_PROJECT_ROOT.parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("LAOX_DATA_ROOT", LAOX_ROOT / "datasets")) / "det_compass_xp"
DEFAULT_ANN_ROOT = DEFAULT_DATA_ROOT / "annotations"
DEFAULT_RESULTS_ROOT = Path(os.environ.get("LAOX_RESULTS_ROOT", LAOX_ROOT / "results"))
DEFAULT_CKPT_ROOT = Path(os.environ.get("LAOX_CKPT_ROOT", LAOX_ROOT / "checkpoints"))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a small MuSC fusion ablation for CutLER demo outputs and "
            "evaluate the predicted COCO jsons with COCOeval."
        )
    )
    parser.add_argument("--demo-cwd", type=Path, default=DEFAULT_CUTLER_ROOT)
    parser.add_argument("--demo-script", default="demo/demo.py")
    parser.add_argument(
        "--config-file",
        default="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml",
    )
    parser.add_argument("--xray-dir", type=Path, default=DEFAULT_DATA_ROOT / "xray")
    parser.add_argument(
        "--musc-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "object_bank" / "musc_heatmaps",
        help="Flat dir of MuSc forward heatmaps named like the X-ray images "
        "(output of third_party/MuSc/scripts/laox_heatmaps.sh).",
    )
    parser.add_argument(
        "--model-weights",
        default=str(DEFAULT_CKPT_ROOT / "cutler_cascade_final.pth"),
    )
    parser.add_argument(
        "--demo-opts",
        nargs="*",
        default=[],
        help="Additional Detectron2 opts appended after MODEL.WEIGHTS.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--gt-json", type=Path, default=DEFAULT_ANN_ROOT / "coco_agnostic.json")
    parser.add_argument(
        "--full-gt-json",
        type=Path,
        default=DEFAULT_ANN_ROOT / "coco_agnostic_1928.json",
        help="Full DET-COMPASS GT. Useful when the 1623-image synthetic-source subset is desired.",
    )
    parser.add_argument(
        "--matches-json",
        type=Path,
        default=DEFAULT_ANN_ROOT / "huggingface_matches.json",
    )
    parser.add_argument(
        "--heldout-json",
        type=Path,
        default=DEFAULT_ANN_ROOT / "coco_agnostic.json",
        help="Images excluded by create_synthetic_dataset_musc.py before building synthetic sources.",
    )
    parser.add_argument(
        "--synthetic-source-coco",
        type=Path,
        default=DEFAULT_ANN_ROOT / "coco_cutler_musc.json",
        help="COCO json used to reproduce the foreground filtering path from create_synthetic_dataset_musc.py.",
    )
    parser.add_argument(
        "--subset-mode",
        choices=[
            "synthetic_candidates",
            "synthetic_foreground",
            "all_gt",
            "all_input",
            "file_list",
        ],
        default="synthetic_candidates",
        help=(
            "synthetic_candidates keeps matched files except heldout GT files; "
            "synthetic_foreground reproduces the foreground image filtering in create_synthetic_dataset_musc.py."
        ),
    )
    parser.add_argument("--file-list", type=Path, default=None)
    parser.add_argument("--max-area-ratio", type=float, default=0.5)
    parser.add_argument("--expected-subset-size", type=int, default=1623)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "object_bank" / "det_compass_xp_cutler_N1_musc_ablation",
    )
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-empty-gt", action="store_true")
    parser.add_argument("--eval-existing-only", action="store_true")
    parser.add_argument("--include-baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sweep-mode",
        choices=["one_factor", "full_grid"],
        default="one_factor",
        help="one_factor is the intended small ablation around the default setting.",
    )
    parser.add_argument("--coeff-values", default="-5,-4,-3,-2,-1,0,1,2,3,4,5")
    parser.add_argument("--temp-values", default="0.1:1.0:0.1")
    parser.add_argument("--power-values", default="1,2,3,4,5")
    parser.add_argument("--default-coeff", type=float, default=-5.0)
    parser.add_argument("--default-sharp-temp", type=float, default=0.8)
    parser.add_argument("--default-soft-temp", type=float, default=0.6)
    parser.add_argument("--default-power", type=float, default=3.0)
    parser.add_argument(
        "--baseline-ori-json",
        type=Path,
        default=DEFAULT_ANN_ROOT / "coco_cutler_ori.json",
    )
    parser.add_argument(
        "--baseline-musc-json",
        type=Path,
        default=DEFAULT_ANN_ROOT / "coco_cutler_musc.json",
    )
    return parser.parse_args()


def parse_number_list(spec, cast=float):
    values = []
    if ":" in spec and "," not in spec:
        start_s, stop_s, step_s = spec.split(":")
        start = float(start_s)
        stop = float(stop_s)
        step = float(step_s)
        if step == 0:
            raise ValueError("Step cannot be 0")
        cur = start
        eps = abs(step) / 1000.0
        cmp = (lambda x: x <= stop + eps) if step > 0 else (lambda x: x >= stop - eps)
        while cmp(cur):
            values.append(cast(round(cur, 10)))
            cur += step
    else:
        values = [cast(v.strip()) for v in spec.split(",") if v.strip()]
    return values


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def numeric_file_key(name):
    stem = Path(name).stem
    try:
        return (0, int(stem), name)
    except ValueError:
        return (1, stem, name)


def image_files_from_coco(path):
    data = load_json(path)
    return {img["file_name"] for img in data.get("images", []) if img.get("file_name")}


def build_subset(args, logger):
    if args.subset_mode == "file_list":
        if args.file_list is None:
            raise ValueError("--file-list is required with --subset-mode file_list")
        files = [line.strip() for line in args.file_list.read_text().splitlines() if line.strip()]
        files = [Path(f).name for f in files]
    elif args.subset_mode == "all_input":
        files = [p.name for p in args.xray_dir.glob("*.png")]
    elif args.subset_mode == "all_gt":
        files = list(image_files_from_coco(args.gt_json))
    elif args.subset_mode == "synthetic_candidates":
        matches = load_json(args.matches_json).get("matches", {})
        heldout_files = image_files_from_coco(args.heldout_json) if args.heldout_json else set()
        files = [fname for fname, entries in matches.items() if entries and fname not in heldout_files]
    elif args.subset_mode == "synthetic_foreground":
        source = load_json(args.synthetic_source_coco)
        heldout_files = image_files_from_coco(args.heldout_json) if args.heldout_json else set()
        matches = load_json(args.matches_json).get("matches", {})
        anns_by_image = {}
        for ann in source.get("annotations", []):
            anns_by_image.setdefault(ann.get("image_id"), []).append(ann)
        files = []
        for img in source.get("images", []):
            fname = img.get("file_name")
            if not fname or fname in heldout_files:
                continue
            entries = matches.get(fname, [])
            splits = {entry.get("split") for entry in entries}
            if "test" not in splits:
                continue
            img_area = float(img.get("width", 0) * img.get("height", 0))
            if img_area <= 0:
                continue
            ann_list = anns_by_image.get(img.get("id"), [])
            has_large = any(
                (float(ann.get("area", 0)) / img_area) > args.max_area_ratio
                for ann in ann_list
            )
            if not has_large:
                files.append(fname)
    else:
        raise ValueError(f"Unknown subset mode: {args.subset_mode}")

    files = sorted(set(files), key=numeric_file_key)
    if args.limit is not None:
        files = files[: args.limit]

    missing_xray = [f for f in files if not (args.xray_dir / f).is_file()]
    missing_musc = [f for f in files if not (args.musc_dir / f).is_file()]
    if missing_xray:
        raise FileNotFoundError(f"Missing {len(missing_xray)} xray files, e.g. {missing_xray[:5]}")
    if missing_musc:
        raise FileNotFoundError(f"Missing {len(missing_musc)} MuSC files, e.g. {missing_musc[:5]}")

    if args.expected_subset_size and len(files) != args.expected_subset_size:
        logger.warning(
            "Subset size is %d, not expected %d. mode=%s",
            len(files),
            args.expected_subset_size,
            args.subset_mode,
        )
    return files


def build_param_grid(args):
    coeffs = parse_number_list(args.coeff_values, float)
    temps = parse_number_list(args.temp_values, float)
    powers = parse_number_list(args.power_values, float)
    default = (
        args.default_coeff,
        args.default_sharp_temp,
        args.default_soft_temp,
        args.default_power,
    )
    combos = [default]
    if args.sweep_mode == "full_grid":
        combos = list(itertools.product(coeffs, temps, temps, powers))
    else:
        combos.extend((c, default[1], default[2], default[3]) for c in coeffs if c != default[0])
        combos.extend((default[0], t, default[2], default[3]) for t in temps if t != default[1])
        combos.extend((default[0], default[1], t, default[3]) for t in temps if t != default[2])
        combos.extend((default[0], default[1], default[2], p) for p in powers if p != default[3])
    return list(dict.fromkeys(combos))


def fmt_num(value):
    if float(value).is_integer():
        text = str(int(value))
    else:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def combo_label(coeff, sharp_temp, soft_temp, power):
    return (
        f"coeff_{fmt_num(coeff)}"
        f"__sharp_{fmt_num(sharp_temp)}"
        f"__soft_{fmt_num(soft_temp)}"
        f"__pow_{fmt_num(power)}"
    )


def setup_logging(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "ablation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("musc_ablation"), log_path


def write_subset_files(args, files):
    meta_dir = args.output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    input_list = meta_dir / "subset_inputs.txt"
    input_list.write_text("\n".join(str(args.xray_dir / f) for f in files) + "\n")

    gt_data = load_json(args.gt_json)
    keep_files = set(files)
    images = [img for img in gt_data.get("images", []) if img.get("file_name") in keep_files]
    keep_ids = {img["id"] for img in images}
    anns = [ann for ann in gt_data.get("annotations", []) if ann.get("image_id") in keep_ids]
    gt_subset = {
        "info": gt_data.get("info", {}),
        "licenses": gt_data.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": gt_data.get("categories", [{"id": 1, "name": "object"}]),
    }
    gt_subset_path = meta_dir / "gt_subset.json"
    with gt_subset_path.open("w") as f:
        json.dump(gt_subset, f)
    return input_list, gt_subset_path, len(images), len(anns)


def coco_predictions_to_detections(pred_json, gt_json):
    pred = load_json(pred_json)
    gt = load_json(gt_json)
    gt_ids = {img["id"] for img in gt.get("images", [])}
    gt_file_to_id = {img["file_name"]: img["id"] for img in gt.get("images", [])}
    pred_id_to_file = {
        img["id"]: img.get("file_name") for img in pred.get("images", []) if "id" in img
    }

    detections = []
    for ann in pred.get("annotations", []):
        image_id = ann.get("image_id")
        fname = pred_id_to_file.get(image_id)
        if fname in gt_file_to_id:
            image_id = gt_file_to_id[fname]
        if image_id not in gt_ids:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        detections.append(
            {
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(v) for v in bbox],
                "score": float(ann.get("score", 1.0)),
            }
        )
    return detections


def evaluate_bbox(pred_json, gt_json):
    detections = coco_predictions_to_detections(pred_json, gt_json)
    coco_gt = COCO(str(gt_json))
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        empty_path = Path(pred_json).with_suffix(".empty_detections.json")
        empty_path.write_text("[]")
        coco_dt = coco_gt.loadRes(str(empty_path))

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
    coco_eval.params.catIds = [1]
    eval_stdout = io.StringIO()
    with contextlib.redirect_stdout(eval_stdout):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats.tolist()
    keys = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return dict(zip(keys, stats)), len(detections), eval_stdout.getvalue()


def run_demo(args, files, pred_json, output_dir, coeff, sharp_temp, soft_temp, power, logger):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        args.demo_script,
        "--config-file",
        args.config_file,
        "--input",
        *[str(args.xray_dir / f) for f in files],
        "--output",
        str(output_dir),
        "--musc_dir",
        str(args.musc_dir),
        "--fusion_mode",
        "mul_blend_mul_v2",
        "--coco_output",
        str(pred_json),
        "--musc_sigmoid_coeff",
        str(coeff),
        "--musc_sharp_temp",
        str(sharp_temp),
        "--musc_soft_temp",
        str(soft_temp),
        "--musc_power",
        str(power),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--opts",
        "MODEL.WEIGHTS",
        args.model_weights,
        *args.demo_opts,
    ]
    if args.save_vis:
        cmd.insert(cmd.index("--opts"), "--vis_demo")

    logger.info("Running demo: %s", " ".join(cmd[:18]) + " ...")
    if args.dry_run:
        return
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    result = subprocess.run(
        cmd,
        cwd=str(args.demo_cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    run_log = pred_json.with_suffix(".run.log")
    run_log.write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Demo failed for {pred_json}; see {run_log}")


def append_csv(csv_path, row):
    exists = csv_path.is_file()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def eval_and_log(label, pred_json, gt_subset, row_base, csv_path, jsonl_path, logger):
    metrics, num_dt, eval_text = evaluate_bbox(pred_json, gt_subset)
    row = {
        **row_base,
        "label": label,
        "pred_json": str(pred_json),
        "num_detections": num_dt,
        **{k: float(v) for k, v in metrics.items()},
    }
    append_csv(csv_path, row)
    with jsonl_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info(
        "%s AP=%.4f AP50=%.4f AP75=%.4f detections=%d",
        label,
        row["AP"],
        row["AP50"],
        row["AP75"],
        num_dt,
    )
    logger.info("COCOeval summary for %s:\n%s", label, eval_text.strip())
    return row


def main():
    args = parse_args()
    logger, log_path = setup_logging(args.output_root)
    files = build_subset(args, logger)
    input_list, gt_subset, gt_images, gt_annotations = write_subset_files(args, files)
    logger.info("Output root: %s", args.output_root)
    logger.info("Log path: %s", log_path)
    logger.info("Subset mode: %s", args.subset_mode)
    logger.info("Subset inputs: %d files (%s)", len(files), input_list)
    logger.info("Eval GT subset: %d images, %d annotations (%s)", gt_images, gt_annotations, gt_subset)
    if gt_images == 0:
        logger.warning(
            "The selected subset has no overlap with --gt-json. "
            "For the synthetic-source subset, consider --gt-json %s.",
            args.full_gt_json,
        )
        if not args.dry_run and not args.allow_empty_gt:
            raise RuntimeError("No GT images remain after subsetting; pass a matching --gt-json.")

    csv_path = args.output_root / "ablation_metrics.csv"
    jsonl_path = args.output_root / "ablation_metrics.jsonl"
    row_base = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "subset_mode": args.subset_mode,
        "subset_size": len(files),
        "gt_images": gt_images,
        "gt_annotations": gt_annotations,
    }

    if args.include_baselines:
        for label, pred_json in (
            ("cutler_ori_existing", args.baseline_ori_json),
            ("musc_default_existing", args.baseline_musc_json),
        ):
            if pred_json.is_file() and not args.dry_run:
                eval_and_log(
                    label,
                    pred_json,
                    gt_subset,
                    {**row_base, "coeff": "", "sharp_temp": "", "soft_temp": "", "power": ""},
                    csv_path,
                    jsonl_path,
                    logger,
                )
            else:
                logger.info("Skip missing/dry-run baseline: %s", pred_json)

    if args.eval_existing_only:
        return

    pred_dir = args.output_root / "predictions"
    vis_root = args.output_root / "outputs"
    pred_dir.mkdir(parents=True, exist_ok=True)
    combos = build_param_grid(args)
    logger.info("Parameter combinations: %d (%s)", len(combos), args.sweep_mode)

    for coeff, sharp_temp, soft_temp, power in combos:
        label = combo_label(coeff, sharp_temp, soft_temp, power)
        pred_json = pred_dir / f"{label}.json"
        output_dir = vis_root / label
        if pred_json.is_file() and args.skip_existing:
            logger.info("Reuse existing prediction: %s", pred_json)
        else:
            run_demo(args, files, pred_json, output_dir, coeff, sharp_temp, soft_temp, power, logger)
        if args.dry_run:
            continue
        eval_and_log(
            label,
            pred_json,
            gt_subset,
            {
                **row_base,
                "coeff": coeff,
                "sharp_temp": sharp_temp,
                "soft_temp": soft_temp,
                "power": power,
            },
            csv_path,
            jsonl_path,
            logger,
        )


if __name__ == "__main__":
    main()
