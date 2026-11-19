# Derived from DET-COMPASS-CROPS by Lorenzo Vaquero (used with permission);
# modified for LAO-X.
"""Build the raw high/low -> colour / grey / density lookup tables.

Each table counts, over DET-COMPASS (HuggingFace ``PAGF/DET-COMPASS``,
``default`` configuration), how often every (high, low) attenuation pair maps
to each colour / grey / density value, and is pickled as
``xray_highlow_to_<table>_dict_RAW.pkl``. The generator denoises and densifies
these counts when it loads them (``compass_editor/colour.py``).

    python build_luts.py all                 # colour, grey and density
    python build_luts.py colour|grey|density [--out-dir DIR] [--force]
    python build_luts.py colour-slow         # per-pixel reference colour builder

Tables are written to ``--out-dir`` (default: ``$LAOX_LUT_DIR`` or
``synthesis/luts/``), which is where the generator reads them from.

``colour`` (vectorised) and ``colour-slow`` (per-pixel) accumulate identical
counts, but insert keys in a different order; ties in the later denoising and
k-NN interpolation may therefore resolve differently, so the two pickles are
not guaranteed to give identical rendered colours.
"""
import argparse
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from compass_editor.colour import lut_dir, quantize_color


HF_DATASET = "PAGF/DET-COMPASS"


def accumulate_colour(conversion_accumulator, highlow, colour):
    # highlow: HxWx2 uint8, colour: HxWx3 uint8
    grey_code = (highlow[...,0].astype(np.uint32) << 8) | highlow[...,1].astype(np.uint32)  # 0..65535
    colour_code = (
        (colour[...,0].astype(np.uint32) << 16) |
        (colour[...,1].astype(np.uint32) << 8) |
        colour[...,2].astype(np.uint32)
    )
    combo = (grey_code.astype(np.uint64) << 24) | colour_code  # unique (highlow, colour) key
    flat, counts = np.unique(combo.reshape(-1), return_counts=True)
    grey_flat = (flat >> 24).astype(np.uint32)
    colour_flat = (flat & 0xFFFFFF).astype(np.uint32)

    for g, c, cnt in zip(grey_flat, colour_flat, counts):
        c_tuple = ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
        conversion_accumulator[(int(g >> 8), int(g & 0xFF))][c_tuple] += int(cnt)


def build_colour():
    """Vectorised colour table (xray_images[0]) over the 'test' split."""
    ds = load_dataset(HF_DATASET, name="default", split="test")

    # Dictionary: grey_key -> count of each RGB color
    conversion_accumulator = defaultdict(Counter)

    for sample in tqdm(ds, desc="Building colour dictionary"):
        xray_colour_img = np.array(sample["xray_images"][0])  # shape: (H, W, 3)
        xray_high_img   = np.array(sample["xray_images"][2])  # shape: (H, W)
        xray_low_img    = np.array(sample["xray_images"][3])  # shape: (H, W)
        xray_grey_img   = np.stack([xray_high_img, xray_low_img], axis=-1)

        accumulate_colour(conversion_accumulator, xray_grey_img, xray_colour_img)

    return conversion_accumulator


def build_colour_slow():
    """Per-pixel colour table (xray_images[0]) over the 'test' split."""
    ds = load_dataset(HF_DATASET, name="default", split="test")

    # Dictionary: grey_key -> count of each RGB color
    conversion_accumulator = defaultdict(Counter)

    for sample in tqdm(ds, desc="Building colour dictionary (per pixel)"):
        xray_colour_img = np.array(sample["xray_images"][0])  # shape: (H, W, 3)
        xray_high_img   = np.array(sample["xray_images"][2])  # shape: (H, W)
        xray_low_img    = np.array(sample["xray_images"][3])  # shape: (H, W)
        xray_grey_img   = np.stack([xray_high_img, xray_low_img], axis=-1)

        # Flatten both images
        grey_flat = xray_grey_img.reshape(-1, 2)
        colour_flat = xray_colour_img.reshape(-1, 3)

        for grey_rgb, colour_rgb in zip(grey_flat, colour_flat):
            grey_key = quantize_color(grey_rgb)
            rgb_tuple = tuple(colour_rgb.astype(int))
            conversion_accumulator[grey_key][rgb_tuple] += 1

    return conversion_accumulator


def build_grey():
    """Grey table (xray_images[1]) over the 'test' split."""
    ds = load_dataset(HF_DATASET, name="default", split="test")

    # Dictionary: highlow_key -> count of each grey (RGB) value
    conversion_accumulator = defaultdict(Counter)

    for sample in tqdm(ds, desc="Building grey dictionary"):
        xray_colour_img = np.array(sample["xray_images"][1])  # shape: (H, W, 3)
        xray_high_img   = np.array(sample["xray_images"][2])  # shape: (H, W)
        xray_low_img    = np.array(sample["xray_images"][3])  # shape: (H, W)
        xray_highlow_img   = np.stack([xray_high_img, xray_low_img], axis=-1)

        # Flatten both images
        highlow_flat = xray_highlow_img.reshape(-1, 2)
        colour_flat = xray_colour_img.reshape(-1, 3)

        for highlow_rgb, colour_rgb in zip(highlow_flat, colour_flat):
            highlow_key = quantize_color(highlow_rgb)
            rgb_tuple = tuple(colour_rgb.astype(int))
            conversion_accumulator[highlow_key][rgb_tuple] += 1

    return conversion_accumulator


def build_density():
    """Density table (xray_images[4]) over all splits."""
    ds = load_dataset(HF_DATASET, name="default", split="all")

    # Dictionary: grey_key -> count of each density value
    conversion_accumulator = defaultdict(Counter)

    for sample in tqdm(ds, desc="Building density dictionary"):
        xray_density_img = np.array(sample["xray_images"][4])  # shape: (H, W)
        xray_high_img   = np.array(sample["xray_images"][2])  # shape: (H, W)
        xray_low_img    = np.array(sample["xray_images"][3])  # shape: (H, W)
        xray_grey_img   = np.stack([xray_high_img, xray_low_img], axis=-1)

        # Flatten both images
        grey_flat = xray_grey_img.reshape(-1, 2)
        density_flat = xray_density_img.reshape(-1, 1)

        for grey_rgb, density_rgb in zip(grey_flat, density_flat):
            grey_key = quantize_color(grey_rgb)
            rgb_tuple = tuple(density_rgb.astype(int))
            conversion_accumulator[grey_key][rgb_tuple] += 1

    return conversion_accumulator


# table name -> (output table, builder)
BUILDERS = {
    "colour": ("colour", build_colour),
    "colour-slow": ("colour", build_colour_slow),
    "grey": ("grey", build_grey),
    "density": ("density", build_density),
}


def main():
    parser = argparse.ArgumentParser(
        description="Build the raw DET-COMPASS high/low lookup tables used by the generator."
    )
    parser.add_argument("table", choices=sorted(BUILDERS) + ["all"],
                        help="Table to build; 'all' = colour, grey and density.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: $LAOX_LUT_DIR or synthesis/luts/).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing pickles.")
    args = parser.parse_args()

    out_dir = args.out_dir or lut_dir()
    names = ["colour", "grey", "density"] if args.table == "all" else [args.table]

    jobs = []
    for name in names:
        table, builder = BUILDERS[name]
        output_file = os.path.join(out_dir, f"xray_highlow_to_{table}_dict_RAW.pkl")
        if os.path.exists(output_file) and not args.force:
            parser.error(f"{output_file} already exists; pass --force to overwrite.")
        jobs.append((table, builder, output_file))

    os.makedirs(out_dir, exist_ok=True)
    for table, builder, output_file in jobs:
        conversion_accumulator = builder()
        print(f"Saving RAW {table} dict to {output_file}")
        with open(output_file, "wb") as f:
            pickle.dump(conversion_accumulator, f)


if __name__ == "__main__":
    main()
