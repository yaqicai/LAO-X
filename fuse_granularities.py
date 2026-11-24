#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

from pycocotools import mask as mask_utils


def parse_gras(gra_str):
    return [g.strip() for g in gra_str.split(",") if g.strip()]


def bbox_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def has_rle(det):
    seg = det.get("segmentation")
    return isinstance(seg, dict) and "counts" in seg and "size" in seg


def mask_iou(candidate, kept):
    cand_rle = candidate["segmentation"]
    kept_rles = [d["segmentation"] for d in kept]
    if not kept_rles:
        return []
    ious = mask_utils.iou([cand_rle], kept_rles, [0] * len(kept_rles))
    return ious[0].tolist()


def nms(dets, iou_thresh, use_mask_nms):
    kept = []
    for det in sorted(dets, key=lambda d: -d.get("score", 0.0)):
        if not kept:
            kept.append(det)
            continue
        suppress = False
        if use_mask_nms and has_rle(det) and all(has_rle(d) for d in kept):
            ious = mask_iou(det, kept)
            if ious and max(ious) > iou_thresh:
                suppress = True
        else:
            for k in kept:
                if bbox_iou(det["bbox"], k["bbox"]) > iou_thresh:
                    suppress = True
                    break
        if not suppress:
            kept.append(det)
    return kept


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def strong_filter_for_gra(dets, min_score, max_per_image):
    dets = [d for d in dets if d.get("score", 0.0) >= min_score]
    if max_per_image is None:
        return dets
    by_image = defaultdict(list)
    for d in dets:
        by_image[d["image_id"]].append(d)
    out = []
    for img_id, items in by_image.items():
        items = sorted(items, key=lambda d: -d.get("score", 0.0))[:max_per_image]
        out.extend(items)
    return out


def build_parser():
    parser = argparse.ArgumentParser(description="Fuse coco_instances_results.json across multiple granularity values.")
    parser.add_argument("--gras", required=True, help="Comma-separated list of gra values, e.g. 0.15,0.3,0.5")
    parser.add_argument("--input-root", default="output_eval", help="Root folder containing per-gra eval outputs.")
    parser.add_argument(
        "--input-template",
        default="{dataset}_img{start_id}_{end_id}_{method}_{gra}{comment}/coco_instances_results.json",
        help="Template for per-gra result file (relative to input-root).",
    )
    parser.add_argument("--output-root", default="output_eval", help="Root folder for fused output.")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="IoU threshold for NMS.")
    parser.add_argument("--use-mask-nms", action="store_true", help="Use mask IoU when RLE masks exist.")
    parser.add_argument("--min-score", type=float, default=0.05, help="Score threshold for general filtering.")
    parser.add_argument("--max-per-image", type=int, default=100, help="Max detections per image after fusion.")
    parser.add_argument(
        "--fusion-mode",
        default="ori",
        choices=["ori", "merge"],
        help="Fusion filtering mode. 'merge' skips all filtering/NMS and directly concatenates results.",
    )
    parser.add_argument(
        "--strong-gra",
        type=float,
        default=0.15,
        help="Apply strong filtering for gra <= this value.",
    )
    parser.add_argument(
        "--strong-min-score",
        type=float,
        default=0.7,
        help="Score threshold for strong filtering on the strong-gra results.",
    )
    parser.add_argument(
        "--strong-max-per-image",
        type=int,
        default=200,
        help="Max detections per image after strong filtering for strong-gra.",
    )
    parser.add_argument(
        "--dataset",
        default="det_compass_syn_100",
        choices=["det_compass_xp", "det_compass_syn_100", "det_compass_syn_312", "PIXray"],
    )
    parser.add_argument("--start_id", type=int, default=0, help="Start index (0-based) for dataset slice")
    parser.add_argument("--end_id", type=int, default=None, help="End index (exclusive) for dataset slice")
    parser.add_argument("--method", default="unsamv2", choices=["detectron2", "unsamv2"])
    parser.add_argument("--comment",type=str,default="",help="train or test")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not create fused output directory or json file.",
    )
    return parser


def fuse_results(
    gras,
    dataset="det_compass_syn_100",
    start_id=0,
    end_id=None,
    method="unsamv2",
    comment="",
    input_root="output_eval",
    input_template="{dataset}_img{start_id}_{end_id}_{method}_{gra}{comment}/coco_instances_results.json",
    output_root="output_eval",
    fusion_mode="ori",
    nms_iou=0.5,
    use_mask_nms=False,
    min_score=0.05,
    max_per_image=100,
    strong_gra=0.15,
    strong_min_score=0.7,
    strong_max_per_image=200,
    no_save=False,
):
    gra_list = parse_gras(gras) if isinstance(gras, str) else list(gras)
    if not gra_list:
        raise SystemExit("No gra values provided.")

    all_dets = []
    for gra in gra_list:
        rel_path = input_template.format(
            dataset=dataset.lower(),
            method=method,
            start_id=start_id,
            end_id=end_id,
            gra=gra,
            comment=comment,
        )
        path = os.path.join(input_root, rel_path)
        dets = load_results(path)
        if fusion_mode != "merge":
            dets = [d for d in dets if d.get("score", 0.0) >= min_score]
            try:
                gra_val = float(gra)
            except ValueError:
                gra_val = None
            if gra_val is not None and gra_val <= strong_gra + 1e-6:
                dets = strong_filter_for_gra(dets, strong_min_score, strong_max_per_image)
        all_dets.extend(dets)

    if fusion_mode == "merge":
        fused = all_dets
    else:
        by_image_cat = defaultdict(list)
        for d in all_dets:
            key = (d["image_id"], d["category_id"])
            by_image_cat[key].append(d)

        fused = []
        by_image = defaultdict(list)
        for (img_id, _cat_id), dets in by_image_cat.items():
            kept = nms(dets, nms_iou, use_mask_nms)
            by_image[img_id].extend(kept)

        for img_id, items in by_image.items():
            items = sorted(items, key=lambda d: -d.get("score", 0.0))[:max_per_image]
            fused.extend(items)

    if no_save:
        print(f"Fusion done (detections: {len(fused)}). Output not saved (--no-save).")
        return None

    out_dir = os.path.join(
        output_root,
        f"gra_fusion_{dataset}_img{start_id}_{end_id}_{method}_{'_'.join(gra_list)}{comment}_{fusion_mode}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "coco_instances_results.json")
    with open(out_path, "w") as f:
        json.dump(fused, f)

    print(f"Saved fused results: {out_path} (detections: {len(fused)})")
    return out_path


def main():
    parser = build_parser()
    args = parser.parse_args()
    fuse_results(
        gras=args.gras,
        dataset=args.dataset,
        start_id=args.start_id,
        end_id=args.end_id,
        method=args.method,
        comment=args.comment,
        input_root=args.input_root,
        input_template=args.input_template,
        output_root=args.output_root,
        fusion_mode=args.fusion_mode,
        nms_iou=args.nms_iou,
        use_mask_nms=args.use_mask_nms,
        min_score=args.min_score,
        max_per_image=args.max_per_image,
        strong_gra=args.strong_gra,
        strong_min_score=args.strong_min_score,
        strong_max_per_image=args.strong_max_per_image,
        no_save=args.no_save,
    )


if __name__ == "__main__":
    main()
