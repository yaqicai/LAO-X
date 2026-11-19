#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import shutil

# LAO-X repository root (sam2/training/scripts/ -> repo).
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = os.environ.get("LAOX_DATA_ROOT", str(REPO_ROOT / "datasets"))
RESULTS_ROOT = os.environ.get("LAOX_RESULTS_ROOT", str(REPO_ROOT / "results"))


def load_exclude_stems(coco_json_path: Path) -> set[str]:
    with coco_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "images" in data and isinstance(data["images"], list):
        items = data["images"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Unsupported COCO JSON format: expected dict with 'images' or list.")

    stems = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        file_name = item.get("file_name")
        if not file_name:
            continue
        stems.add(Path(file_name).stem)
    return stems


def copy_non_test_annotations(src_dir: Path, dst_dir: Path, exclude_stems: set[str], ann_ext: str) -> None:
    if not ann_ext.startswith("."):
        ann_ext = f".{ann_ext}"

    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for ann_path in sorted(src_dir.glob(f"*{ann_ext}")):
        if ann_path.stem in exclude_stems:
            skipped += 1
            continue
        shutil.copy2(ann_path, dst_dir / ann_path.name)
        copied += 1

    print(
        f"Done. Copied {copied} annotations to {dst_dir}. "
        f"Skipped {skipped} annotations from test set."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy annotation files from a source directory to a destination directory, "
            "excluding those whose basename matches test-set file_name stems in a COCO JSON."
        )
    )
    parser.add_argument(
        "--src-dir",
        default=os.path.join(
            RESULTS_ROOT, "cutler/det_compass_xp/granularity_divide_and_conquer_1000/vis"
        ),
        help="Source directory that contains annotation files.",
    )
    parser.add_argument(
        "--dst-dir",
        default=os.path.join(
            RESULTS_ROOT, "cutler/det_compass_xp/granularity_divide_and_conquer_1000_clean/vis"
        ),
        help="Destination directory for filtered annotation files.",
    )
    parser.add_argument(
        "--coco-json",
        default=os.path.join(DATA_ROOT, "det_compass_xp/coco_agnostic.json"),
        help="COCO JSON annotation file containing test-set images.",
    )
    parser.add_argument(
        "--ann-ext",
        default=".json",
        help="Annotation file extension to copy (default: .json).",
    )

    args = parser.parse_args()

    src_dir = Path(os.path.expanduser(args.src_dir))
    dst_dir = Path(os.path.expanduser(args.dst_dir))
    coco_json_path = Path(os.path.expanduser(args.coco_json))

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if not coco_json_path.is_file():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json_path}")

    exclude_stems = load_exclude_stems(coco_json_path)
    copy_non_test_annotations(src_dir, dst_dir, exclude_stems, args.ann_ext)


if __name__ == "__main__":
    main()
