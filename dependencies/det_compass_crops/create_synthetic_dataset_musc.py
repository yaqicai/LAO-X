import argparse
import hashlib
import json
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import glob
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm
import fcntl
from contextlib import contextmanager

from compass_editor.colour import (
    center_pad_unpad,
    combine_highlow_images,
    get_bbox_from_mask,
    highlow_to_colour,
    highlow_to_density,
    highlow_to_grey,
    sample_to_highlow,
)
from compass_editor.dataset import set_seed, transform_xray_crops


os.environ["HF_HOME"] = "/path/to/laox_repro/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/path/to/laox_repro/hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/path/to/laox_repro/hf_cache/hub"


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


COLOR_MAP_PALETTES = {
    # kept from previous sets
    "ivory_teal_crimson": ((232, 220, 172), (70, 176, 176), (196, 62, 78)),
    "silver_blue_emerald": ((206, 206, 192), (92, 146, 228), (64, 168, 114)),
    "lavender_gold_navy": ((194, 170, 220), (224, 178, 88), (52, 78, 148)),
    "clcxray_g2": ((234, 226, 184), (98, 190, 188), (40, 120, 210)),
    "dvxray_g3": ((246, 240, 226), (206, 212, 228), (138, 156, 206)),
    "hixray_g2": ((246, 226, 158), (220, 178, 112), (156, 98, 74)),

    # new designs inspired by legacy readability (light low-density, clear mid/high separation)
    "legacyplus_honey_mint_cobalt": ((246, 228, 170), (142, 206, 154), (44, 116, 216)),
    "legacyplus_sand_cyan_blue": ((238, 220, 176), (126, 202, 210), (36, 114, 206)),
    "legacyplus_pearl_lime_navy": ((236, 232, 198), (156, 206, 122), (54, 90, 170)),
    "legacyplus_amber_aqua_indigo": ((244, 214, 150), (116, 194, 198), (72, 86, 188)),
    "legacyplus_blush_teal_royal": ((244, 224, 210), (108, 188, 180), (50, 108, 214)),

    # white-zero modes: density==0 pure white, then smoothly transition to light low-density colours
    "whitezero_amber_teal_blue": ((250, 238, 196), (144, 204, 182), (48, 120, 214)),
    "whitezero_peach_mint_navy": ((250, 234, 214), (154, 206, 176), (58, 96, 182)),
    "whitezero_cream_cyan_violet": ((246, 240, 222), (150, 202, 220), (116, 88, 186)),

    # fully new 2D-focused schemes (material ratio controls warm/cool branch)
    "twod_graphite_coral_teal": ((244, 236, 222), (124, 182, 176), (52, 108, 124)),
    "twod_sand_violet_cyan": ((246, 232, 206), (156, 124, 198), (74, 184, 210)),
    "twod_ice_amber_ink": ((242, 246, 250), (214, 168, 106), (58, 78, 116)),
}


COLOR_MAP_LUMA_RANGE = {
    "ivory_teal_crimson": (224.0, 76.0),
    "silver_blue_emerald": (214.0, 70.0),
    "lavender_gold_navy": (210.0, 60.0),
    "clcxray_g2": (242.0, 66.0),
    "dvxray_g3": (244.0, 86.0),
    "hixray_g2": (234.0, 52.0),
    "legacyplus_honey_mint_cobalt": (236.0, 64.0),
    "legacyplus_sand_cyan_blue": (234.0, 66.0),
    "legacyplus_pearl_lime_navy": (238.0, 62.0),
    "legacyplus_amber_aqua_indigo": (232.0, 60.0),
    "legacyplus_blush_teal_royal": (236.0, 63.0),
    "whitezero_amber_teal_blue": (240.0, 68.0),
    "whitezero_peach_mint_navy": (240.0, 64.0),
    "whitezero_cream_cyan_violet": (242.0, 62.0),
    "twod_graphite_coral_teal": (238.0, 58.0),
    "twod_sand_violet_cyan": (236.0, 60.0),
    "twod_ice_amber_ink": (244.0, 56.0),
}


COLOR_MAP_PROFILE_PALETTES = {
    "kept": (
        "clcxray_g2",
        "dvxray_g3",
        "hixray_g2",
        "ivory_teal_crimson",
        "lavender_gold_navy",
        "silver_blue_emerald",
    ),
    "legacyplus": (
        "legacyplus_honey_mint_cobalt",
        "legacyplus_sand_cyan_blue",
        "legacyplus_pearl_lime_navy",
        "legacyplus_amber_aqua_indigo",
        "legacyplus_blush_teal_royal",
    ),
    "whitezero": (
        "whitezero_amber_teal_blue",
        "whitezero_peach_mint_navy",
        "whitezero_cream_cyan_violet",
    ),
    "twod_new": (
        "twod_graphite_coral_teal",
        "twod_sand_violet_cyan",
        "twod_ice_amber_ink",
    ),
}


WHITE_ZERO_DENSITY_PALETTES = set(COLOR_MAP_PROFILE_PALETTES["whitezero"])


def _ordered_unique(values: List[str]) -> Tuple[str, ...]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return tuple(out)


ALL_DENSITY_PALETTES = _ordered_unique(
    [
        *COLOR_MAP_PROFILE_PALETTES["kept"],
        *COLOR_MAP_PROFILE_PALETTES["legacyplus"],
        *COLOR_MAP_PROFILE_PALETTES["whitezero"],
        *COLOR_MAP_PROFILE_PALETTES["twod_new"],
    ]
)


def parse_palette_list(s: str) -> Tuple[str, ...]:
    """
    Parse comma-separated palette keys.
    Supports direct keys, profile names, and "all".
    """
    parts = tuple(p.strip().lower() for p in s.split(",") if p.strip())
    if not parts:
        raise argparse.ArgumentTypeError(
            "Palette list cannot be empty. Use values like 'all', 'legacyplus', or 'twod_new'."
        )

    expanded: List[str] = []
    invalid: List[str] = []

    for token in parts:
        if token in {"all", "*"}:
            expanded.extend(ALL_DENSITY_PALETTES)
        elif token in COLOR_MAP_PROFILE_PALETTES:
            expanded.extend(COLOR_MAP_PROFILE_PALETTES[token])
        elif token in COLOR_MAP_PALETTES:
            expanded.append(token)
        else:
            invalid.append(token)

    if invalid:
        raise argparse.ArgumentTypeError(
            "Unknown palette/profile token(s): "
            f"{invalid}. Valid keys: {sorted(COLOR_MAP_PALETTES.keys())}; "
            f"profiles: {sorted(COLOR_MAP_PROFILE_PALETTES.keys()) + ['all']}"
        )

    return _ordered_unique(expanded)


def _normalize_by_percentile(arr: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    lo, hi = np.percentile(arr, [q_low, q_high])
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _jitter_colour(colour: Tuple[int, int, int], jitter: float, rng: np.random.Generator) -> np.ndarray:
    c = np.array(colour, dtype=np.float32)
    if jitter <= 0:
        return c
    delta = rng.uniform(-jitter, jitter, size=3).astype(np.float32)
    return np.clip(c + delta, 0.0, 255.0)


def _colourize_density_with_triplet(
    density_norm: np.ndarray,
    low_rgb: np.ndarray,
    mid_rgb: np.ndarray,
    high_rgb: np.ndarray,
) -> np.ndarray:
    t = np.clip(density_norm, 0.0, 1.0).astype(np.float32)
    out = np.empty((t.shape[0], t.shape[1], 3), dtype=np.float32)

    low_to_mid = t <= 0.5
    a = (t[low_to_mid] * 2.0)[:, None]
    out[low_to_mid] = (1.0 - a) * low_rgb + a * mid_rgb

    mid_to_high = ~low_to_mid
    b = ((t[mid_to_high] - 0.5) * 2.0)[:, None]
    out[mid_to_high] = (1.0 - b) * mid_rgb + b * high_rgb

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _compute_density_features(high_low: np.ndarray):
    density_img = highlow_to_density(high_low).astype(np.uint8)

    density_base = _normalize_by_percentile(density_img.astype(np.float32), 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    density_local = clahe.apply(density_img)
    density_local_norm = _normalize_by_percentile(density_local.astype(np.float32), 1.0, 99.0)

    high = high_low[..., 0].astype(np.float32)
    low = high_low[..., 1].astype(np.float32)
    detail_raw = 255.0 - 0.5 * (high + low)
    detail_norm = _normalize_by_percentile(detail_raw, 1.0, 99.0)

    density_norm = np.clip(
        0.62 * density_base + 0.26 * density_local_norm + 0.12 * detail_norm,
        0.0,
        1.0,
    )

    ratio_raw = (low - high) / np.maximum(low + high, 1e-3)
    ratio_center = float(np.median(ratio_raw))
    ratio_dev = ratio_raw - ratio_center
    ratio_scale = float(np.percentile(np.abs(ratio_dev), 97.0))
    ratio_scale = max(ratio_scale, 1e-4)
    ratio_norm = np.clip(ratio_dev / ratio_scale, -1.0, 1.0).astype(np.float32)

    return density_img, density_norm.astype(np.float32), detail_norm, high, low, ratio_norm


def _get_2d_palette_spec(
    palette_key: str,
    low_rgb: np.ndarray,
    mid_rgb: np.ndarray,
    high_rgb: np.ndarray,
):
    # Generic 2D branch: density controls low-mid-high; energy-ratio controls warm/cool split.
    warm_rgb = np.clip(0.64 * low_rgb + 0.36 * mid_rgb, 0.0, 255.0)
    cool_rgb = np.clip(0.22 * mid_rgb + 0.78 * high_rgb, 0.0, 255.0)
    spec = {
        "warm_rgb": warm_rgb,
        "cool_rgb": cool_rgb,
        "alpha_min": 0.16,
        "alpha_max": 0.86,
        "density_start": 0.16,
        "ratio_power": 0.92,
        "legacy_keep": 0.28,
    }

    # Whitezero: keep tray/background close to legacy while recoloring object internals.
    if palette_key in WHITE_ZERO_DENSITY_PALETTES:
        spec.update(
            {
                "alpha_min": 0.08,
                "alpha_max": 0.80,
                "density_start": 0.22,
                "ratio_power": 0.90,
                "legacy_keep": 0.44,
            }
        )

    # Fully redesigned 2D schemes with explicit warm/cool anchors.
    if palette_key == "twod_graphite_coral_teal":
        spec.update(
            {
                "warm_rgb": np.array((224.0, 146.0, 118.0), dtype=np.float32),
                "cool_rgb": np.array((64.0, 168.0, 172.0), dtype=np.float32),
                "alpha_min": 0.20,
                "alpha_max": 0.92,
                "density_start": 0.12,
                "ratio_power": 0.86,
                "legacy_keep": 0.22,
            }
        )
    elif palette_key == "twod_sand_violet_cyan":
        spec.update(
            {
                "warm_rgb": np.array((214.0, 166.0, 106.0), dtype=np.float32),
                "cool_rgb": np.array((86.0, 190.0, 214.0), dtype=np.float32),
                "alpha_min": 0.18,
                "alpha_max": 0.90,
                "density_start": 0.14,
                "ratio_power": 0.84,
                "legacy_keep": 0.20,
            }
        )
    elif palette_key == "twod_ice_amber_ink":
        spec.update(
            {
                "warm_rgb": np.array((226.0, 170.0, 88.0), dtype=np.float32),
                "cool_rgb": np.array((62.0, 86.0, 130.0), dtype=np.float32),
                "alpha_min": 0.22,
                "alpha_max": 0.94,
                "density_start": 0.10,
                "ratio_power": 0.82,
                "legacy_keep": 0.16,
            }
        )

    return spec


def highlow_to_colour_density_palette_2d(
    high_low_img: np.ndarray,
    palette_key: str,
    jitter: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D color mapping: density controls depth branch, high/low ratio controls material branch.
    Preserves legacy structural readability while giving stronger cross-domain color variation.
    """
    high_low = np.asarray(high_low_img, dtype=np.uint8)
    density_img, density_norm, detail_norm, high, low, ratio_norm = _compute_density_features(high_low)
    density_gamma = float(rng.uniform(0.97, 1.03))
    density_norm = np.power(density_norm, density_gamma).astype(np.float32)

    # 2D low-density axis: rely more on physical attenuation than on compressed density LUT bins.
    atten_raw = 255.0 - 0.5 * (high + low)
    atten_norm = _normalize_by_percentile(atten_raw, 0.5, 99.5)
    density_2d = np.clip(
        0.28 * density_norm + 0.58 * atten_norm + 0.14 * detail_norm,
        0.0,
        1.0,
    ).astype(np.float32)

    low_ref, mid_ref, high_ref = COLOR_MAP_PALETTES[palette_key]
    low_rgb = _jitter_colour(low_ref, jitter * 0.30, rng)
    mid_rgb = _jitter_colour(mid_ref, jitter * 0.30, rng)
    high_rgb = _jitter_colour(high_ref, jitter * 0.30, rng)

    base_img = _colourize_density_with_triplet(density_2d, low_rgb, mid_rgb, high_rgb).astype(np.float32)

    spec = _get_2d_palette_spec(palette_key, low_rgb, mid_rgb, high_rgb)
    ratio_shaped = np.sign(ratio_norm) * (np.abs(ratio_norm) ** float(spec["ratio_power"]))
    ratio_pos = np.clip(ratio_shaped, 0.0, 1.0)
    ratio_neg = np.clip(-ratio_shaped, 0.0, 1.0)

    mat_img = base_img.copy()
    mat_img = (1.0 - ratio_pos[..., None]) * mat_img + ratio_pos[..., None] * spec["cool_rgb"]
    mat_img = (1.0 - ratio_neg[..., None]) * mat_img + ratio_neg[..., None] * spec["warm_rgb"]

    low_luma, high_luma = COLOR_MAP_LUMA_RANGE[palette_key]
    target_luma = low_luma * (1.0 - density_2d) + high_luma * density_2d

    # Enforce stronger separation between air(0-ish) and tray(low density) in 2D mode.
    low_band = np.clip((density_2d - 0.07) / 0.18, 0.0, 1.0)
    low_separation = low_band * (1.0 - np.clip((density_2d - 0.30) / 0.22, 0.0, 1.0))
    sep_strength = 34.0 if palette_key.startswith("twod_") else 26.0
    target_luma = target_luma - sep_strength * low_separation
    cur_luma = (
        0.2126 * mat_img[..., 0]
        + 0.7152 * mat_img[..., 1]
        + 0.0722 * mat_img[..., 2]
    )
    luma_scale = np.clip(target_luma / np.maximum(cur_luma, 1e-3), 0.72, 1.55)
    mat_img = mat_img * luma_scale[..., None]

    detail_gain = 0.90 + 0.20 * detail_norm
    mat_img = mat_img * detail_gain[..., None]

    # Density-dependent recolor strength: weak at low density, strong at medium/high density.
    density_gate = np.clip(
        (density_2d - float(spec["density_start"])) / max(1.0 - float(spec["density_start"]), 1e-3),
        0.0,
        1.0,
    )
    recolor_alpha = float(spec["alpha_min"]) + (float(spec["alpha_max"]) - float(spec["alpha_min"])) * density_gate

    # Keep legacy texture cues so tray/inner structure remain readable.
    legacy_img = highlow_to_colour(high_low).astype(np.float32)
    legacy_keep = float(spec["legacy_keep"])
    recolor_alpha = recolor_alpha * (1.0 - legacy_keep)
    colour_img = (1.0 - recolor_alpha[..., None]) * legacy_img + recolor_alpha[..., None] * mat_img

    # Global air anchor: keep true background near white in all 2D palettes.
    air_from_density = np.clip((0.09 - density_2d) / 0.09, 0.0, 1.0)
    air_from_signal = np.clip((np.minimum(high, low) - 248.0) / 7.0, 0.0, 1.0)
    air_whiten = np.maximum(air_from_density, 0.90 * air_from_signal)
    white = np.full_like(colour_img, 255.0)
    colour_img = (1.0 - air_whiten[..., None]) * colour_img + air_whiten[..., None] * white

    # For whitezero family, keep 0-density / near-air pixels even closer to white.
    if palette_key in WHITE_ZERO_DENSITY_PALETTES:
        near_zero = np.clip((0.12 - density_2d) / 0.12, 0.0, 1.0)
        air_whiten = np.clip((np.minimum(high, low) - 246.0) / 9.0, 0.0, 1.0)
        whiten = np.maximum(near_zero, air_whiten)
        white = np.full_like(colour_img, 255.0)
        colour_img = (1.0 - whiten[..., None]) * colour_img + whiten[..., None] * white
        pure_white_mask = (high >= 251.0) & (low >= 251.0)
        if np.any(pure_white_mask):
            colour_img[pure_white_mask] = 255.0

    colour_img = np.clip(colour_img, 0.0, 255.0).astype(np.uint8)
    return colour_img, density_img


def highlow_to_colour_density_palette(
    high_low_img: np.ndarray,
    palette_key: str,
    jitter: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Density-driven pseudo colorization with explicit luminance control and edge-preserving detail.
    Returns color image and original density image.
    """
    high_low = np.asarray(high_low_img, dtype=np.uint8)
    density_img = highlow_to_density(high_low).astype(np.uint8)

    # 1) Global + local density normalization.
    density_base = _normalize_by_percentile(density_img.astype(np.float32), 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    density_local = clahe.apply(density_img)
    density_local_norm = _normalize_by_percentile(density_local.astype(np.float32), 1.0, 99.0)

    # 2) Structure proxy from high/low channels.
    high = high_low[..., 0].astype(np.float32)
    low = high_low[..., 1].astype(np.float32)
    detail_raw = 255.0 - 0.5 * (high + low)
    detail_norm = _normalize_by_percentile(detail_raw, 1.0, 99.0)

    density_norm = np.clip(
        0.62 * density_base + 0.26 * density_local_norm + 0.12 * detail_norm,
        0.0,
        1.0,
    )
    density_gamma = float(rng.uniform(0.97, 1.03))
    density_norm = np.power(density_norm, density_gamma)

    low_ref, mid_ref, high_ref = COLOR_MAP_PALETTES[palette_key]
    low_rgb = _jitter_colour(low_ref, jitter * 0.30, rng)
    mid_rgb = _jitter_colour(mid_ref, jitter * 0.30, rng)
    high_rgb = _jitter_colour(high_ref, jitter * 0.30, rng)

    colour_img = _colourize_density_with_triplet(density_norm, low_rgb, mid_rgb, high_rgb).astype(np.float32)

    # 3) Force luminance ordering: low density brighter, high density darker.
    low_luma, high_luma = COLOR_MAP_LUMA_RANGE[palette_key]
    target_luma = low_luma * (1.0 - density_norm) + high_luma * density_norm
    cur_luma = (
        0.2126 * colour_img[..., 0]
        + 0.7152 * colour_img[..., 1]
        + 0.0722 * colour_img[..., 2]
    )
    scale = np.clip(target_luma / np.maximum(cur_luma, 1e-3), 0.72, 1.55)
    colour_img = colour_img * scale[..., None]

    # 4) Gentle detail modulation without darkening flat background.
    detail_gain = 0.92 + 0.16 * detail_norm
    colour_img = colour_img * detail_gain[..., None]

    # White-zero palettes: legacy-anchored rendering.
    # Keep legacy structure/texture in low density, recolor mainly mid/high densities.
    if palette_key in WHITE_ZERO_DENSITY_PALETTES:
        legacy_img = highlow_to_colour(high_low).astype(np.float32)
        legacy_luma = (
            0.2126 * legacy_img[..., 0]
            + 0.7152 * legacy_img[..., 1]
            + 0.0722 * legacy_img[..., 2]
        )

        # Keep low-density close to legacy, push colour change into medium/high densities.
        recolor_alpha = np.clip((density_norm - 0.28) / 0.56, 0.0, 1.0)
        recolor_alpha = np.power(recolor_alpha, 0.85)

        # Preserve legacy local luminance cues (tray structure, edges),
        # while injecting controlled depth contrast for higher density.
        desired_luma = low_luma * (1.0 - density_norm) + high_luma * density_norm
        mixed_luma = 0.82 * legacy_luma + 0.18 * desired_luma
        cur_palette_luma = (
            0.2126 * colour_img[..., 0]
            + 0.7152 * colour_img[..., 1]
            + 0.0722 * colour_img[..., 2]
        )
        palette_scale = np.clip(mixed_luma / np.maximum(cur_palette_luma, 1e-3), 0.76, 1.42)
        palette_img = colour_img * palette_scale[..., None]

        colour_img = (1.0 - recolor_alpha[..., None]) * legacy_img + recolor_alpha[..., None] * palette_img

    colour_img = np.clip(colour_img, 0.0, 255.0)

    colour_img = colour_img.astype(np.uint8)
    return colour_img, density_img


def _create_xray_outputs(
    composite_img: np.ndarray,
    color_map_mode: str,
    color_map_palette: Tuple[str, ...],
    color_map_jitter: float,
    color_map_seed: Optional[int],
    sample_id: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, Image.Image, Image.Image, np.ndarray]:
    if color_map_mode != "legacy":
        raise ValueError("Only the fixed legacy High/Low LUT is permitted.")
    composite_img = composite_img[..., :2]

    if color_map_mode == "legacy":
        xray_colour = highlow_to_colour(composite_img)
        xray_density = highlow_to_density(composite_img)
    elif color_map_mode in {"density_palette", "density_palette_2d"}:
        base_seed = 0 if color_map_seed is None else int(color_map_seed)
        sample_offset = 0 if sample_id is None else int(sample_id)
        rng = np.random.default_rng(base_seed + sample_offset)
        palette_key = color_map_palette[int(rng.integers(0, len(color_map_palette)))]
        if color_map_mode == "density_palette_2d":
            xray_colour, xray_density = highlow_to_colour_density_palette_2d(
                composite_img,
                palette_key=palette_key,
                jitter=color_map_jitter,
                rng=rng,
            )
        else:
            xray_colour, xray_density = highlow_to_colour_density_palette(
                composite_img,
                palette_key=palette_key,
                jitter=color_map_jitter,
                rng=rng,
            )
    else:
        raise ValueError(f"Unknown color_map_mode '{color_map_mode}'")

    xray_grey = highlow_to_grey(composite_img)
    xray_high = Image.fromarray(composite_img[..., 0], mode="L")
    xray_low = Image.fromarray(composite_img[..., 1], mode="L")

    return xray_colour, xray_grey, xray_high, xray_low, xray_density


def _load_det_compass_dataset_from_cache() -> DatasetDict:
    cache_root = os.environ.get(
        "HF_DATASETS_CACHE", "/path/to/laox_repro/hf_cache/datasets"
    )
    candidates = []
    for cache_dir in glob.glob(
        os.path.join(cache_root, "*___det-compass", "*", "*", "*")
    ):
        if not os.path.isfile(os.path.join(cache_dir, "dataset_info.json")):
            continue
        if glob.glob(os.path.join(cache_dir, "*-test-*.arrow")):
            candidates.append(cache_dir)

    if not candidates:
        raise FileNotFoundError(
            "No complete cached PAGF/DET-COMPASS dataset was found under "
            f"{cache_root}. Run once with network access before using --offline."
        )

    cache_dir = max(candidates, key=os.path.getmtime)
    splits = {}
    for split in ("test", "invisible"):
        shard_paths = sorted(
            glob.glob(os.path.join(cache_dir, f"*-{split}-*.arrow"))
        )
        if not shard_paths:
            continue
        shards = [Dataset.from_file(path) for path in shard_paths]
        splits[split] = (
            shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        )

    if "test" not in splits:
        raise FileNotFoundError(
            f"Cached PAGF/DET-COMPASS test split is missing from {cache_dir}."
        )
    print(f"Loading PAGF/DET-COMPASS directly from cache: {cache_dir}")
    return DatasetDict(splits)


def _load_det_compass_dataset(offline: bool = False):
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        return _load_det_compass_dataset_from_cache()
    with _dataset_lock():
        return load_dataset("PAGF/DET-COMPASS", download_mode="reuse_cache_if_exists")


@contextmanager
def _dataset_lock(timeout_s: int = 1800):
    """
    Serialize HuggingFace dataset preparation to avoid concurrent .incomplete writes.
    """
    lock_path = os.environ.get(
        "DET_COMPASS_DATASET_LOCK",
        "/path/to/laox_repro/hf_cache/datasets/.det_compass_lock",
    )
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    start = time.time()
    with open(lock_path, "w") as f:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start > timeout_s:
                    raise TimeoutError(
                        f"Timeout waiting for dataset lock: {lock_path}"
                    )
                time.sleep(1.0)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_range_float(s: str) -> Tuple[float, float]:
    """
    Parse 'min,max' or 'val' into a float tuple.
    """
    try:
        parts = list(map(float, s.split(",")))
        if len(parts) == 1:
            return parts[0], parts[0]
        if len(parts) == 2:
            return min(parts), max(parts)
        raise ValueError
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid float range '{s}'. Use 'val' or 'min,max'."
        ) from exc


def parse_range_int(s: str) -> Tuple[int, int]:
    """
    Parse 'min,max' or 'val' into an int tuple.
    """
    try:
        parts = list(map(int, s.split(",")))
        if len(parts) == 1:
            return parts[0], parts[0]
        if len(parts) == 2:
            return min(parts), max(parts)
        raise ValueError
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid int range '{s}'. Use 'val' or 'min,max'."
        ) from exc


def parse_container_count_probs(s: str) -> Tuple[float, float, float]:
    parts = tuple(float(v.strip()) for v in s.split(","))
    if parts != (1.0, 0.0, 0.0):
        raise argparse.ArgumentTypeError("This reproduction permits exactly one container: 1,0,0.")
    return parts


def _ensure_mask_size(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask is None:
        return None
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    h = min(mask.shape[0], height)
    w = min(mask.shape[1], width)
    out = np.zeros((height, width), dtype=mask.dtype)
    out[:h, :w] = mask[:h, :w]
    return out


def _crop_arrays_to_min_size(*arrays: np.ndarray) -> List[np.ndarray]:
    heights = [arr.shape[0] for arr in arrays]
    widths = [arr.shape[1] for arr in arrays]
    height = min(heights)
    width = min(widths)
    return [arr[:height, :width] for arr in arrays]


def _decode_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        mask = mask_utils.decode(mask_utils.merge(rles)).astype(bool)
        return _ensure_mask_size(mask, height, width)
    if isinstance(segmentation, dict):
        seg = dict(segmentation)
        if "counts" in seg and isinstance(seg.get("counts"), list):
            rle = mask_utils.frPyObjects(seg, height, width)
            mask = mask_utils.decode(rle).astype(bool)
            return _ensure_mask_size(mask, height, width)
        seg["size"] = [int(height), int(width)]
        mask = mask_utils.decode(seg).astype(bool)
        return _ensure_mask_size(mask, height, width)
    return None


def _bbox_iou_xyxy(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _hash_image_array(arr: np.ndarray) -> str:
    """
    Hash image content with shape to make collisions very unlikely.
    """
    h = hashlib.md5(arr.tobytes()).hexdigest()
    return f"{arr.shape}-{h}"


def _save_png(path: str, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _load_image_as_rgb_array(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _strip_other_images(sample: Dict) -> Image.Image:
    """
    Keep only xray_images[0] (colour) and drop rgb_image and other xray images.
    Returns the remaining xray PIL image.
    """
    if "rgb_image" in sample:
        sample.pop("rgb_image", None)
    if "xray_images" in sample:
        xray_images = sample.get("xray_images") or []
        if not xray_images:
            return None
        keep = xray_images[0]
        sample["xray_images"] = [keep]
        return keep
    return None


def build_local_hash_map(xray_dir: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Returns:
        hash_to_files: hash -> [filenames]
        all_files: list of all filenames considered
    """
    hash_to_files: Dict[str, List[str]] = {}
    all_files: List[str] = []
    for fname in tqdm(sorted(os.listdir(xray_dir)), desc="Indexing local xray files"):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS:
            continue
        all_files.append(fname)
        arr = _load_image_as_rgb_array(os.path.join(xray_dir, fname))
        h = _hash_image_array(arr)
        hash_to_files.setdefault(h, []).append(fname)
    return hash_to_files, all_files


def _prepare_foreground_cache(foreground_samples: List[Dict]) -> None:
    print("Caching foreground samples (highlow + mask arrays)...")
    for sample in tqdm(foreground_samples, desc="Cache foreground"):
        cache = sample.setdefault("_cached_arrays", {})
        if "highlow" not in cache:
            cache["highlow"] = sample_to_highlow(sample, as_image=True)
        if "xray_mask" not in cache:
            cache["xray_mask"] = np.array(sample["xray_image_mask"], dtype=np.uint8)
        if "_mask_area" not in sample:
            sample["_mask_area"] = int((cache["xray_mask"] > 0).sum())


def _prepare_background_cache(
    background_samples: List[Dict],
    image_size: Tuple[int, int],
) -> None:
    print("Caching background samples (padded highlow arrays)...")
    for sample in tqdm(background_samples, desc="Cache background"):
        cache = sample.setdefault("_cached_arrays", {})
        if (
            "highlow_padded" not in cache
            or cache.get("highlow_padded_size") != image_size
        ):
            bg_img_original = sample_to_highlow(sample, as_image=True)
            bg_img, _ = center_pad_unpad(
                bg_img_original, target_size=image_size, constant_values=255
            )
            cache["highlow_padded"] = bg_img
            cache["highlow_padded_size"] = image_size


def match_dataset_to_local_xray(
    xray_dir: str,
    output_path: str,
    splits: Tuple[str, ...] = ("test", "invisible"),
    limit: int = None,
    offline: bool = False,
) -> Dict:
    print("Loading dataset PAGF/DET-COMPASS...")
    dataset_xp = _load_det_compass_dataset(offline=offline)

    print(f"Indexing local xray images in {xray_dir}...")
    hash_to_files, all_files = build_local_hash_map(xray_dir)

    matches: Dict[str, List[Dict]] = {}
    unmatched_dataset: List[Dict] = []
    matched_files = set()

    for split in splits:
        if split not in dataset_xp:
            continue
        ds = dataset_xp[split]
        total = len(ds) if limit is None else min(len(ds), limit)
        print(f"Matching split '{split}' ({total} samples)...")
        for i in tqdm(range(total), desc=f"Matching {split}"):
            sample = ds[i]
            xray_img = _strip_other_images(sample)
            if xray_img is None:
                unmatched_dataset.append({"split": split, "index": i, "reason": "missing_xray"})
                continue

            arr = np.asarray(xray_img.convert("RGB"))
            h = _hash_image_array(arr)
            file_matches = hash_to_files.get(h)
            if not file_matches:
                unmatched_dataset.append({"split": split, "index": i, "reason": "no_local_match"})
                continue

            for fname in file_matches:
                matches.setdefault(fname, []).append({"split": split, "index": i})
                matched_files.add(fname)

    unmatched_files = [f for f in all_files if f not in matched_files]

    result = {
        "xray_dir": xray_dir,
        "splits": list(splits),
        "matches": matches,
        "unmatched_dataset": unmatched_dataset,
        "unmatched_files": unmatched_files,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def _load_matches_split_map(matches_path: str) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Returns:
        file_to_split: filename -> split ("test" or "invisible")
        ambiguous: list of files with multiple splits
    """
    with open(matches_path, "r") as f:
        data = json.load(f)

    file_to_split: Dict[str, str] = {}
    ambiguous: List[Dict] = []
    matches = data.get("matches", {})
    for fname, entries in matches.items():
        splits = sorted({e.get("split") for e in entries if e.get("split")})
        if not splits:
            continue
        if len(splits) > 1:
            ambiguous.append({"file": fname, "splits": splits})
            # pick a stable split to proceed
            file_to_split[fname] = splits[0]
        else:
            file_to_split[fname] = splits[0]
    return file_to_split, ambiguous


def _load_matches_index_map(matches_path: str, split: str) -> Dict[str, int]:
    with open(matches_path, "r") as f:
        data = json.load(f)
    file_to_index: Dict[str, int] = {}
    matches = data.get("matches", {})
    for fname, entries in matches.items():
        for entry in entries:
            if entry.get("split") == split:
                file_to_index[fname] = int(entry.get("index"))
                break
    return file_to_index


def filter_coco_for_test_split(
    coco_path: str,
    matches_path: str,
    xray_test_anno_dir: str,
    max_area_ratio: float = 0.5,
    output_path: str = None,
) -> Tuple[Dict, Dict]:
    print(f"Loading COCO annotations from {coco_path}...")
    with open(coco_path, "r") as f:
        coco = json.load(f)

    original_image_count = len(coco.get("images", []))
    if xray_test_anno_dir:
        print(f"Loading test annotations from {xray_test_anno_dir}...")
        with open(xray_test_anno_dir, "r") as f:
            test_coco = json.load(f)
        test_files = {
            img.get("file_name")
            for img in test_coco.get("images", [])
            if img.get("file_name")
        }
        if test_files:
            keep_images = [
                img for img in coco.get("images", [])
                if img.get("file_name") not in test_files
            ]
            keep_image_ids = {img["id"] for img in keep_images}
            keep_annotations = [
                ann for ann in coco.get("annotations", [])
                if ann.get("image_id") in keep_image_ids
            ]
            coco["images"] = keep_images
            coco["annotations"] = keep_annotations
        print(f"Original images: {original_image_count}")
        print(f"After removing test images: {len(coco.get('images', []))}")

    print(f"Loading matches from {matches_path}...")
    file_to_split, ambiguous = _load_matches_split_map(matches_path)
    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in tqdm(coco.get("annotations", []), desc="Indexing annotations"):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    kept_images = []
    kept_annotations = []
    dropped_images = 0
    dropped_annotations = 0

    for image_id, img in tqdm(images_by_id.items(), desc="Filtering images"):
        if file_to_split.get(img["file_name"]) != "test":
            continue
        img_area = float(img.get("width", 0) * img.get("height", 0))
        if img_area <= 0:
            continue
        ann_list = anns_by_image.get(image_id, [])
        has_large = any(
            (ann.get("area", 0) / img_area) > max_area_ratio for ann in ann_list
        )
        if has_large:
            dropped_images += 1
            dropped_annotations += len(ann_list)
            continue
        kept_images.append(img)
        kept_annotations.extend(ann_list)

    print(f"After removing large masks: {len(kept_images)}")

    filtered_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": coco.get("categories", []),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(filtered_coco, f)

    stats = {
        "num_images": len(coco.get("images", [])),
        "num_annotations": len(coco.get("annotations", [])),
        "kept_images": len(kept_images),
        "kept_annotations": len(kept_annotations),
        "dropped_images": dropped_images,
        "dropped_annotations": dropped_annotations,
        "num_ambiguous_files": len(ambiguous),
    }
    return filtered_coco, stats


def _visualize_filtered_coco_xray(
    filtered_coco: Dict,
    dataset_test,
    matches_path: str,
    output_dir: str,
    limit: int = 20,
) -> Dict:
    print(f"Saving xray color/high/low visualizations to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    file_to_index = _load_matches_index_map(matches_path, split="test")
    saved = 0
    skipped = 0

    for img in filtered_coco.get("images", []):
        if saved >= limit:
            break
        fname = img.get("file_name")
        if not fname:
            skipped += 1
            continue
        idx = file_to_index.get(fname)
        if idx is None:
            skipped += 1
            continue
        sample = dataset_test[idx]
        xray_images = sample.get("xray_images") or []
        if len(xray_images) < 4:
            skipped += 1
            continue

        colour = np.asarray(xray_images[0])
        high = np.asarray(xray_images[2])
        low = np.asarray(xray_images[3])
        if colour.ndim == 2:
            colour = np.repeat(colour[:, :, None], 3, axis=2)
        elif colour.ndim == 3 and colour.shape[2] > 3:
            colour = colour[:, :, :3]
        if high.ndim == 3:
            high = high[:, :, 0]
        if low.ndim == 3:
            low = low[:, :, 0]
        colour, high, low = _crop_arrays_to_min_size(colour, high, low)

        high_rgb = np.repeat(high[:, :, None], 3, axis=2)
        low_rgb = np.repeat(low[:, :, None], 3, axis=2)
        combined = np.concatenate([colour, high_rgb, low_rgb], axis=1)
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        stem = os.path.splitext(os.path.basename(fname))[0] or f"xray_{saved:05d}"
        out_path = os.path.join(output_dir, f"{stem}_xray_color_high_low.png")
        Image.fromarray(combined).save(out_path)
        saved += 1

    summary = {
        "output_dir": output_dir,
        "saved": saved,
        "skipped": skipped,
        "limit": limit,
    }
    print(json.dumps(summary, indent=2))
    return summary


def _visualize_foreground_object_bank(
    foreground_samples: List[Dict],
    output_dir: str,
    limit: int = 0,
) -> Dict:
    print(f"Saving foreground object bank visualizations to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    skipped = 0
    name_counts: Dict[str, int] = {}

    for idx, sample in enumerate(foreground_samples):
        if limit and saved >= limit:
            break
        high = sample.get("raw_xray_image_high")
        low = sample.get("raw_xray_image_low")
        if high is None or low is None:
            skipped += 1
            continue
        high = np.asarray(high)
        low = np.asarray(low)
        if high.ndim == 3:
            high = high[:, :, 0]
        if low.ndim == 3:
            low = low[:, :, 0]
        high, low = _crop_arrays_to_min_size(high, low)
        highlow_img = np.stack([high, low], axis=-1)
        if highlow_img.dtype != np.uint8:
            highlow_img = highlow_img.astype(np.float64)
            max_val = float(highlow_img.max()) if highlow_img.size else 0.0
            if max_val <= 1.0:
                highlow_img = highlow_img * 255.0
            highlow_img = np.clip(highlow_img, 0, 255).astype(np.uint8)
        colour = highlow_to_colour(highlow_img)
        colour = np.clip(colour, 0, 255).astype(np.uint8)
        high = np.clip(high, 0, 255).astype(np.uint8)
        low = np.clip(low, 0, 255).astype(np.uint8)

        source_name = sample.get("source_file_name") or ""
        stem_base = os.path.splitext(os.path.basename(source_name))[0]
        if not stem_base:
            stem_base = f"fg_{idx:06d}"
        name_counts[stem_base] = name_counts.get(stem_base, 0) + 1
        ann_id = sample.get("annotation_id")
        if ann_id is not None:
            stem = f"{stem_base}_ann{ann_id}"
        elif name_counts[stem_base] > 1:
            stem = f"{stem_base}_obj{name_counts[stem_base]:02d}"
        else:
            stem = stem_base
        Image.fromarray(colour).save(os.path.join(output_dir, f"{stem}_color.png"))
        Image.fromarray(high).save(os.path.join(output_dir, f"{stem}_high.png"))
        Image.fromarray(low).save(os.path.join(output_dir, f"{stem}_low.png"))
        mask_img = sample.get("xray_image_mask")
        if mask_img is not None:
            mask = np.asarray(mask_img)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8) * 255
            colour_mask, mask = _crop_arrays_to_min_size(colour, mask)
            Image.fromarray(mask).save(os.path.join(output_dir, f"{stem}_mask.png"))
            mask_on = mask > 0
            overlay = np.array([255, 0, 0], dtype=np.uint8)
            alpha = 0.5
            colour_masked = colour_mask.copy().astype(np.float32)
            colour_masked[mask_on] = (
                (1.0 - alpha) * colour_masked[mask_on] + alpha * overlay
            )
            colour_masked = np.clip(colour_masked, 0, 255).astype(np.uint8)
            Image.fromarray(colour_masked).save(
                os.path.join(output_dir, f"{stem}_color_mask.png")
            )
        saved += 1

    summary = {
        "output_dir": output_dir,
        "saved": saved,
        "skipped": skipped,
        "limit": limit,
        "total": len(foreground_samples),
    }
    print(json.dumps(summary, indent=2))
    return summary


def _build_foreground_samples(
    filtered_coco: Dict,
    dataset_test,
    matches_path: str,
    output_dir: str = None,
) -> List[Dict]:
    print("Building foreground samples from filtered COCO annotations...")
    file_to_index = _load_matches_index_map(matches_path, split="test")
    image_meta = {img["id"]: img for img in filtered_coco.get("images", [])}
    image_id_to_index: Dict[int, int] = {}
    for image_id, img in image_meta.items():
        idx = file_to_index.get(img["file_name"])
        if idx is not None:
            image_id_to_index[image_id] = idx

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        coco_train = {
            "info": filtered_coco.get("info", {}),
            "licenses": filtered_coco.get("licenses", []),
            "images": [],
            "annotations": [],
            "categories": filtered_coco.get("categories", []),
        }
        ann_id = 0
        for image_id, img in image_meta.items():
            idx = image_id_to_index.get(image_id)
            if idx is None:
                continue
            sample = dataset_test[idx]
            width = sample.get("xray_width")
            height = sample.get("xray_height")
            if width is None or height is None:
                xray_images = sample.get("xray_images") or []
                if xray_images:
                    arr = np.asarray(xray_images[0])
                    height, width = arr.shape[0], arr.shape[1]
            if width is None or height is None:
                continue
            coco_train["images"].append(
                {
                    "id": image_id,
                    "file_name": img.get("file_name"),
                    "width": int(width),
                    "height": int(height),
                }
            )
            objects = sample.get("objects", {})
            bboxes = objects.get("xray_bbox") or []
            categories = objects.get("categories") or []
            for bbox, category_id in zip(bboxes, categories):
                if not bbox:
                    continue
                try:
                    category_id = int(category_id)
                except (TypeError, ValueError):
                    category_id = 0
                area = float(bbox[2] * bbox[3]) if bbox else 0.0
                coco_train["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1
        output_path = os.path.join(output_dir, "coco_train.json")
        with open(output_path, "w") as f:
            json.dump(coco_train, f)
        print(f"Saved dataset_xp train annotations to {output_path}")

    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    foreground_samples: List[Dict] = []
    for ann in tqdm(filtered_coco.get("annotations", []), desc="Cropping foregrounds"):
        image_id = ann.get("image_id")
        idx = image_id_to_index.get(image_id)
        if idx is None:
            continue
        img = image_meta.get(image_id)
        if img is None:
            continue
        height = int(img.get("height", 0))
        width = int(img.get("width", 0))
        if height <= 0 or width <= 0:
            continue

        if idx not in cache:
            sample = dataset_test[idx]
            xray_images = sample.get("xray_images") or []
            if len(xray_images) < 4:
                continue
            high = np.asarray(xray_images[2])
            low = np.asarray(xray_images[3])
            cache[idx] = (high, low)
        high, low = cache[idx]

        segmentation = ann.get("segmentation")
        mask = _decode_mask(segmentation, height, width)
        if mask is None and ann.get("bbox"):
            x, y, w, h = [int(v) for v in ann["bbox"]]
            mask = np.zeros((height, width), dtype=bool)
            mask[y : y + h, x : x + w] = True
        if mask is None:
            continue
        mask = _ensure_mask_size(mask, height, width)
        if mask.shape[0] != high.shape[0] or mask.shape[1] != high.shape[1]:
            mask = _ensure_mask_size(mask, high.shape[0], high.shape[1])
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        high_crop = high[y_min : y_max + 1, x_min : x_max + 1]
        low_crop = low[y_min : y_max + 1, x_min : x_max + 1]
        mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1].astype(np.uint8) * 255

        foreground_samples.append(
            {
                "raw_xray_image_high": high_crop.astype(np.uint8),
                "raw_xray_image_low": low_crop.astype(np.uint8),
                "xray_image_mask": Image.fromarray(mask_crop, mode="L"),
                "category": int(ann.get("category_id", 0)),
                "source_file_name": img.get("file_name"),
                "annotation_id": ann.get("id"),
                "xray_visible": True,
            }
        )

    foreground_samples.sort(key=lambda item: (item["source_file_name"], item["annotation_id"]))
    return foreground_samples


def split_coco_by_matches_and_plot(
    coco_path: str,
    matches_path: str,
    output_plot_path: str,
    title: str = "Annotation Area Distribution",
):
    """
    Use HuggingFace matches to split COCO annotations into test/invisible groups,
    then plot area distributions (all/test/invisible) on one figure.
    """
    print(f"Loading COCO annotations from {coco_path}...")
    with open(coco_path, "r") as f:
        coco = json.load(f)

    print(f"Loading matches from {matches_path}...")
    file_to_split, ambiguous = _load_matches_split_map(matches_path)

    image_id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    image_id_to_area = {
        img["id"]: (img.get("width", 0) * img.get("height", 0))
        for img in coco.get("images", [])
    }
    image_id_to_split: Dict[int, str] = {}
    for image_id, fname in image_id_to_file.items():
        split = file_to_split.get(fname)
        if split in ("test", "invisible"):
            image_id_to_split[image_id] = split

    all_areas: List[float] = []
    test_areas: List[float] = []
    invisible_areas: List[float] = []
    all_ratios: List[float] = []
    test_ratios: List[float] = []
    invisible_ratios: List[float] = []
    skipped = 0

    for ann in tqdm(coco.get("annotations", []), desc="Collecting annotation areas"):
        area = ann.get("area", 0)
        all_areas.append(area)
        img_area = image_id_to_area.get(ann.get("image_id"), 0)
        if img_area and img_area > 0:
            ratio = area / float(img_area)
            all_ratios.append(ratio)
        image_id = ann.get("image_id")
        split = image_id_to_split.get(image_id)
        if split == "test":
            test_areas.append(area)
            if img_area and img_area > 0:
                test_ratios.append(ratio)
        elif split == "invisible":
            invisible_areas.append(area)
            if img_area and img_area > 0:
                invisible_ratios.append(ratio)
        else:
            skipped += 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    bins = 50

    axes[0].hist(all_areas, bins=bins, alpha=0.5, label="all")
    axes[0].hist(test_areas, bins=bins, alpha=0.5, label="test")
    axes[0].hist(invisible_areas, bins=bins, alpha=0.5, label="invisible")
    axes[0].set_title(title)
    axes[0].set_xlabel("annotation area")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].hist(all_ratios, bins=bins, alpha=0.5, label="all")
    axes[1].hist(test_ratios, bins=bins, alpha=0.5, label="test")
    axes[1].hist(invisible_ratios, bins=bins, alpha=0.5, label="invisible")
    axes[1].set_title("Annotation Area / Image Area Ratio")
    axes[1].set_xlabel("area ratio")
    axes[1].set_ylabel("count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)

    summary = {
        "coco_path": coco_path,
        "matches_path": matches_path,
        "output_plot_path": output_plot_path,
        "num_annotations": len(all_areas),
        "num_test_annotations": len(test_areas),
        "num_invisible_annotations": len(invisible_areas),
        "num_skipped_annotations": skipped,
        "num_ambiguous_files": len(ambiguous),
        "num_ratio_entries": len(all_ratios),
    }
    return summary


def _sample_transformed_object(
    fg_sample: Dict,
    image_size: Tuple[int, int],
    max_angle: float,
    max_scale: float,
    max_t: float,
    max_perspective: float,
    existing_boxes: List[Tuple[float, float, float, float]],
    gamma: float,
    max_tries: int = 50,
):
    for _ in range(max_tries):
        objs = transform_xray_crops(
            fg_samples=[fg_sample],
            image_size=image_size,
            max_angle=max_angle,
            max_scale=max_scale,
            max_t=max_t,
            max_perspective=max_perspective,
        )
        if not objs:
            continue
        obj = objs[0]
        bbox = get_bbox_from_mask(obj["fg_mask_t"])
        if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        x1 = float(bbox[0])
        y1 = float(bbox[1])
        x2 = float(bbox[0] + bbox[2])
        y2 = float(bbox[1] + bbox[3])
        candidate = (x1, y1, x2, y2)
        if any(_bbox_iou_xyxy(candidate, b) > gamma for b in existing_boxes):
            continue
        return obj, candidate
    return None, None


def create_synthetic_sample_musc(
    foreground_samples: List[Dict],
    background_samples: List[Dict],
    num_objs: int,
    gamma: float,
    composition_mode: str = "xray",
    color_map_mode: str = "legacy",
    color_map_palette: Tuple[str, ...] = ALL_DENSITY_PALETTES,
    color_map_jitter: float = 0.0,
    color_map_seed: Optional[int] = None,
    sample_id: Optional[int] = None,
    image_size: Tuple[int, int] = (580, 880),
    max_angle: float = 180.0,
    max_scale: float = 1.2,
    max_t: float = 0.75,
    max_perspective: float = 0.05,
):
    sample = {
        "xray_images": [],
        "xray_width": None,
        "xray_height": None,
        "objects": {
            "xray_bbox": [],
            "xray_mask": [],
            "categories": [],
        },
    }

    if not background_samples:
        return sample

    bg_sample = random.choice(background_samples)
    cache = bg_sample.setdefault("_cached_arrays", {})
    if (
        "highlow_padded" in cache
        and cache.get("highlow_padded_size") == image_size
    ):
        bg_img = cache["highlow_padded"]
    else:
        bg_img_original = sample_to_highlow(bg_sample, as_image=True)
        bg_img, _ = center_pad_unpad(
            bg_img_original, target_size=image_size, constant_values=255
        )
        cache["highlow_padded"] = bg_img
        cache["highlow_padded_size"] = image_size

    if not foreground_samples or num_objs <= 0:
        selected = []
    else:
        selected = np.random.choice(foreground_samples, size=num_objs, replace=True).tolist()
        selected.sort(key=lambda x: x.get("_mask_area", 0), reverse=True)

    transformed_objs = []
    placed_boxes: List[Tuple[float, float, float, float]] = []
    for fg_sample in selected:
        obj, bbox_xyxy = _sample_transformed_object(
            fg_sample=fg_sample,
            image_size=bg_img.shape[0:2],
            max_angle=max_angle,
            max_scale=max_scale,
            max_t=max_t,
            max_perspective=max_perspective,
            existing_boxes=placed_boxes,
            gamma=gamma,
        )
        if obj is None:
            continue
        placed_boxes.append(bbox_xyxy)
        transformed_objs.append(obj)

    composite_img = bg_img
    for obj in transformed_objs:
        if composition_mode == "xray":
            composite_img = combine_highlow_images(
                composite_img, obj["fg_img_t"], substract=False, maskB=obj["fg_mask_t"]
            )
        elif composition_mode == "rgb":
            mask = obj["fg_mask_t"] > 0
            composite_img = composite_img.copy()
            composite_img[mask] = obj["fg_img_t"][mask]
        else:
            raise ValueError(f"Unknown composition_mode '{composition_mode}'")
        bbox = get_bbox_from_mask(obj["fg_mask_t"])
        sample["objects"]["xray_bbox"].append(bbox)
        sample["objects"]["xray_mask"].append(
            Image.fromarray(obj["fg_mask_t"] > 0).convert("1")
        )
        sample["objects"]["categories"].append(obj["fg_sample"].get("category", "unknown"))

    xray_colour, xray_grey, xray_high, xray_low, xray_density = _create_xray_outputs(
        composite_img=composite_img,
        color_map_mode=color_map_mode,
        color_map_palette=color_map_palette,
        color_map_jitter=color_map_jitter,
        color_map_seed=color_map_seed,
        sample_id=sample_id,
    )

    sample["xray_images"] = [xray_colour, xray_grey, xray_high, xray_low, xray_density]
    sample["xray_height"] = xray_colour.shape[0]
    sample["xray_width"] = xray_colour.shape[1]

    return sample


def create_multi_container_sample_musc(
    foreground_samples, background_samples, num_objs, gamma,
    container_count_probs=(1.0, 0.0, 0.0), **kwargs,
):
    """Compatibility entry point: never sample or concatenate containers."""
    if tuple(container_count_probs) != (1.0, 0.0, 0.0):
        raise ValueError("Only one container is permitted in this reproduction.")
    return create_synthetic_sample_musc(
        foreground_samples, background_samples, num_objs, gamma, **kwargs
    )


def seed_image(base_seed: int, stage: str, image_id: int) -> int:
    """Independent per-image stream; independent of worker count and chunking."""
    payload = f"laox-single-legacy-v1:{base_seed}:{stage}:{image_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed & 0x7fffffff)
    import torch
    torch.manual_seed(seed)
    return seed


def generate_synthetic_dataset_musc(
    foreground_samples: List[Dict],
    background_samples: List[Dict],
    output_dir: str,
    num_generated: int,
    num_objects_range: Tuple[int, int],
    gamma_range: Tuple[float, float],
    max_perspective: float,
    composition_mode: str,
    color_map_mode: str,
    color_map_palette: Tuple[str, ...],
    color_map_jitter: float,
    color_map_seed: Optional[int],
    container_count_probs: Tuple[float, float, float],
    build_segmentation: bool,
    io_workers: int,
    start_id: int,
    info: Dict,
    licenses: List[Dict],
    categories: List[Dict],
    base_seed: int,
    stage: str,
):
    if not background_samples:
        raise ValueError("No background samples available from the 'invisible' split.")
    print(f"Generating {num_generated} synthetic images into {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    xray_dir = os.path.join(output_dir, "xray")
    os.makedirs(xray_dir, exist_ok=True)
    coco_output = {
        "info": info,
        "licenses": licenses,
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    ann_id = 0
    io_executor = ThreadPoolExecutor(max_workers=io_workers if io_workers > 0 else 1)
    io_futures: deque = deque()
    max_pending = max(1, (io_workers if io_workers > 0 else 1) * 4)

    num_min, num_max = num_objects_range
    gamma_min, gamma_max = gamma_range

    for i in tqdm(range(num_generated), desc="musc_augmentation"):
        global_id = start_id + i
        seed_image(base_seed, stage, global_id)
        current_num_objs = np.random.randint(num_min, num_max + 1)
        current_gamma = np.random.uniform(gamma_min, gamma_max)
        sample = create_multi_container_sample_musc(
            foreground_samples=foreground_samples,
            background_samples=background_samples,
            num_objs=current_num_objs,
            gamma=current_gamma,
            container_count_probs=container_count_probs,
            max_perspective=max_perspective,
            composition_mode=composition_mode,
            color_map_mode=color_map_mode,
            color_map_palette=color_map_palette,
            color_map_jitter=color_map_jitter,
            color_map_seed=color_map_seed,
            sample_id=global_id,
        )

        png_filename = f"{global_id:06d}.png"
        png_path = os.path.join(xray_dir, png_filename)
        xray_colour = np.clip(sample["xray_images"][0], 0, 255).astype(np.uint8)
        io_futures.append(io_executor.submit(_save_png, png_path, xray_colour))
        if len(io_futures) >= max_pending:
            io_futures.popleft().result()

        coco_output["images"].append(
            {
                "id": global_id,
                "file_name": png_filename,
                "width": sample["xray_width"],
                "height": sample["xray_height"],
            }
        )

        for bbox, mask, category_id in zip(
            sample["objects"]["xray_bbox"],
            sample["objects"]["xray_mask"],
            sample["objects"]["categories"],
        ):
            try:
                category_id = int(category_id)
            except (TypeError, ValueError):
                category_id = 0
            segmentation = []
            mask_area = bbox[2] * bbox[3] if bbox else 0

            if build_segmentation:
                mask_arr = np.array(mask, dtype=np.uint8)
                mask_arr = (mask_arr > 0).astype(np.uint8)
                contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if contour.shape[0] < 3:
                        continue
                    flattened = contour.reshape(-1, 2).ravel().tolist()
                    if len(flattened) >= 6:
                        segmentation.append(flattened)
                mask_area = int(mask_arr.sum())
            else:
                mask_arr = np.array(mask, dtype=np.uint8)
                if mask_arr.max() > 0:
                    mask_area = int(mask_arr.sum())

            coco_output["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": global_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": mask_area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )
            ann_id += 1

    end_id = start_id + num_generated
    coco_json_path = os.path.join(output_dir, f"coco_{start_id}_{end_id}.json")
    with open(coco_json_path, "w") as f:
        json.dump(coco_output, f)

    while io_futures:
        io_futures.popleft().result()
    io_executor.shutdown(wait=True)

    print(f"Saved synthetic COCO annotations to {coco_json_path}")
    return coco_json_path


def _draw_bboxes_on_rgb_image(
    image_rgb: np.ndarray,
    bboxes: List[List[float]],
    categories: List,
) -> np.ndarray:
    vis = np.asarray(image_rgb, dtype=np.uint8).copy()
    for bbox, category in zip(bboxes, categories):
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = bbox
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w - 1))
        y2 = int(round(y + h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(vis, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.putText(
            vis,
            str(category),
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def generate_preview_color_map_examples(
    foreground_samples: List[Dict],
    background_samples: List[Dict],
    output_dir: str,
    preview_num_samples: int,
    num_objects_range: Tuple[int, int],
    gamma_range: Tuple[float, float],
    max_perspective: float,
    composition_mode: str,
    color_map_mode: str,
    color_map_palette: Tuple[str, ...],
    color_map_jitter: float,
    color_map_seed: Optional[int],
    container_count_probs: Tuple[float, float, float],
):
    if not background_samples:
        raise ValueError("No background samples available from the 'invisible' split.")

    os.makedirs(output_dir, exist_ok=True)

    num_min, num_max = num_objects_range
    gamma_min, gamma_max = gamma_range

    for i in tqdm(range(preview_num_samples), desc="preview_colormap"):
        sample_id = i
        current_num_objs = np.random.randint(num_min, num_max + 1)
        current_gamma = np.random.uniform(gamma_min, gamma_max)

        # Generate once with legacy mapping to freeze object placement/container layout.
        sample = create_multi_container_sample_musc(
            foreground_samples=foreground_samples,
            background_samples=background_samples,
            num_objs=current_num_objs,
            gamma=current_gamma,
            container_count_probs=container_count_probs,
            max_perspective=max_perspective,
            composition_mode=composition_mode,
            color_map_mode="legacy",
            color_map_palette=color_map_palette,
            color_map_jitter=color_map_jitter,
            color_map_seed=color_map_seed,
            sample_id=sample_id,
        )

        legacy_color = np.asarray(sample["xray_images"][0], dtype=np.uint8)
        legacy_bbox_vis = _draw_bboxes_on_rgb_image(
            legacy_color,
            sample["objects"]["xray_bbox"],
            sample["objects"]["categories"],
        )

        stem = f"preview_{sample_id:04d}"
        Image.fromarray(legacy_color).save(os.path.join(output_dir, f"{stem}_legacy.png"))
        Image.fromarray(legacy_bbox_vis).save(os.path.join(output_dir, f"{stem}_legacy_bbox.png"))

        high = np.asarray(sample["xray_images"][2], dtype=np.uint8)
        low = np.asarray(sample["xray_images"][3], dtype=np.uint8)
        highlow = np.stack([high, low], axis=-1)

        for j, palette_key in enumerate(color_map_palette):
            base_seed = 0 if color_map_seed is None else int(color_map_seed)
            rng = np.random.default_rng(base_seed + sample_id * 1000 + j)
            if color_map_mode == "density_palette_2d":
                colour_img, _ = highlow_to_colour_density_palette_2d(
                    highlow,
                    palette_key=palette_key,
                    jitter=color_map_jitter,
                    rng=rng,
                )
            else:
                colour_img, _ = highlow_to_colour_density_palette(
                    highlow,
                    palette_key=palette_key,
                    jitter=color_map_jitter,
                    rng=rng,
                )
            Image.fromarray(colour_img).save(
                os.path.join(output_dir, f"{stem}_palette_{palette_key}.png")
            )

    print(
        json.dumps(
            {
                "preview_dir": output_dir,
                "preview_num_samples": preview_num_samples,
                "num_palettes": len(color_map_palette),
                "color_map_mode": color_map_mode,
                "files_per_sample": 2 + len(color_map_palette),
            },
            indent=2,
        )
    )



def merge_coco_jsons(
    output_dir: str,
    output_path: str = None,
    pattern: str = "coco_*.json",
):
    json_paths = sorted(glob.glob(os.path.join(output_dir, pattern)))
    if not json_paths:
        raise FileNotFoundError(f"No COCO json files found in {output_dir} matching {pattern}")

    merged = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    images_seen = set()
    ann_id = 0
    for idx, path in enumerate(json_paths):
        with open(path, "r") as f:
            data = json.load(f)
        if idx == 0:
            merged["info"] = data.get("info", {})
            merged["licenses"] = data.get("licenses", [])
            merged["categories"] = data.get("categories", [])

        for img in data.get("images", []):
            img_id = img.get("id")
            if img_id in images_seen:
                continue
            merged["images"].append(img)
            images_seen.add(img_id)

        for ann in data.get("annotations", []):
            ann_copy = dict(ann)
            merged["annotations"].append(ann_copy)

    merged["images"].sort(key=lambda item: item["id"])
    merged["annotations"].sort(key=lambda item: (item["image_id"], item["id"]))
    for ann_id, ann in enumerate(merged["annotations"]):
        ann["id"] = ann_id

    if output_path is None:
        output_path = os.path.join(output_dir, "coco.json")
    with open(output_path, "w") as f:
        json.dump(merged, f)

    print(f"Merged {len(json_paths)} files into {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Match DET-COMPASS xray images, split COCO, or generate MUSC-style synthetic data."
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="Root directory for datasets (used with --dataset_gen and --stage).",
    )
    parser.add_argument(
        "--dataset_gen",
        default="",
        help="Generated dataset name under dataset_root.",
    )
    parser.add_argument(
        "--stage",
        default="",
        help="Stage name under dataset path.",
    )
    parser.add_argument(
        "--xray_dir",
        default="/path/to/datasets/det_compass_xp/xray",
        help="Directory containing local xray images to match.",
    )
    parser.add_argument(
        "--output_path",
        default="det_compass_xp_xray_matches.json",
        help="Path to write the filename->(split,index) mapping JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit per split for debugging.",
    )
    parser.add_argument(
        "--mode",
        choices=["match", "split_coco", "augment", "preview", "merge"],
        default="augment",
        help="Run matching, split COCO, or generate MUSC-style augmentation.",
    )
    parser.add_argument(
        "--coco_dir",
        default="/path/to/datasets/det_compass_xp/coco_cutler",
        help="COCO base path; final coco_path is coco_dir_{crop_ablation}.",
    )
    parser.add_argument(
        "--crop_ablation",
        default="musc",
        help="Suffix appended to coco_dir to form coco_path (without .json).",
    )
    parser.add_argument(
        "--xray_test_anno_dir",
        default="/path/to/datasets/det_compass_xp/coco_agnostic.json",
        help="COCO json containing test images to exclude.",
    )
    parser.add_argument(
        "--matches_path",
        default="/path/to/datasets/det_compass_xp/huggingface_matches.json",
        help="Match JSON generated by this script.",
    )
    parser.add_argument(
        "--plot_path",
        default="coco_musc_area_distribution.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--output_dir",
        default="/path/to/datasets/det_compass_xp/musc_aug",
        help="Output directory for synthetic samples.",
    )
    parser.add_argument(
        "--xray_vis_dir",
        default="",
        help="Optional directory to save xray color/high/low visualizations.",
    )
    parser.add_argument(
        "--xray_vis_limit",
        type=int,
        default=20,
        help="Maximum number of filtered xray images to visualize.",
    )
    parser.add_argument(
        "--fg_vis_dir",
        default="",
        help="Optional directory to save cropped foreground object bank visualizations.",
    )
    parser.add_argument(
        "--fg_vis_limit",
        type=int,
        default=0,
        help="Maximum number of foreground objects to visualize (0 for all).",
    )
    parser.add_argument(
        "--filtered_coco_path",
        default="",
        help="Optional path to save filtered COCO (test-only, large areas removed).",
    )
    parser.add_argument(
        "--max_area_ratio",
        type=float,
        default=0.5,
        help="Drop images if any annotation area ratio exceeds this value.",
    )
    parser.add_argument(
        "--num_generated",
        type=int,
        default=0,
        help="Number of synthetic images to generate.",
    )
    parser.add_argument(
        "--preview_num_samples",
        type=int,
        default=6,
        help="Number of preview base scenes to generate for color-map comparison.",
    )
    parser.add_argument(
        "--preview_dir",
        default="preview_colormap",
        help="Output directory (relative to output_dir if not absolute) for preview images.",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="Start id offset for generated images (used for sharded generation).",
    )
    parser.add_argument(
        "--num_objects",
        type=parse_range_int,
        default="3,5",
        help="Object count range per image, e.g. '3,8' or '5'.",
    )
    parser.add_argument(
        "--composition_mode",
        choices=["xray", "rgb"],
        default="xray",
        help="Composition mode: xray uses Beer-Lambert blending, rgb uses copy-paste overwrite.",
    )
    parser.add_argument(
        "--color_map_mode",
        choices=["legacy"],
        default="legacy",
        help="Color mapping mode for generated xray color images.",
    )
    parser.add_argument(
        "--color_map_palette",
        type=parse_palette_list,
        default=ALL_DENSITY_PALETTES,
        help=(
            "Comma-separated palette keys/profile names for density_palette mode. "
            "Examples: 'all', 'kept', 'legacyplus', 'whitezero', 'twod_new', or explicit key list."
        ),
    )
    parser.add_argument(
        "--color_map_jitter",
        type=float,
        default=0.0,
        help="Max per-channel RGB jitter (0-255 scale) for density_palette mode.",
    )
    parser.add_argument(
        "--color_map_seed",
        type=int,
        default=None,
        help="Optional seed for deterministic color-map sampling across images.",
    )
    parser.add_argument(
        "--gamma",
        type=parse_range_float,
        default="0.0,0.3",
        help="Max IoU overlap range, e.g. '0.0,0.4' or '0.2'.",
    )
    parser.add_argument(
        "--container_count_probs",
        type=parse_container_count_probs,
        default=(1.0, 0.0, 0.0),
        help=(
            "Probabilities for sampling 1/2/3 containers in one image. "
            "Format: 'p1,p2,p3' with p1>p2>p3 and p2+p3>=0.30."
        ),
    )
    parser.add_argument(
        "--max_perspective",
        type=float,
        default=0.1,
        help="Max perspective jitter ratio per corner for homography.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=40,
        help="Fixed base seed, inherited from the existing sample demo; per-image streams are worker-independent.",
    )
    parser.add_argument(
        "--build_segmentation",
        action="store_true",
        default=True,
        help="Build COCO polygon segmentation from masks.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Force HuggingFace datasets to load from local cache only.",
    )
    parser.add_argument(
        "--online",
        dest="offline",
        action="store_false",
        help="Disable offline mode and allow downloads.",
    )
    parser.add_argument(
        "--io_workers",
        type=int,
        default=4,
        help="Thread workers for async PNG saves.",
    )
    parser.add_argument(
        "--merge_pattern",
        default="coco_*.json",
        help="Glob pattern for COCO json shards in merge mode.",
    )
    parser.add_argument(
        "--merge_output",
        default="coco.json",
        help="Output filename for merged COCO json (relative to output_dir).",
    )
    args = parser.parse_args()
    if args.mode == "preview":
        parser.error("Color-palette preview is disabled in the fixed-LUT reproduction.")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be an unsigned 32-bit integer")
    if args.color_map_jitter != 0 or args.container_count_probs != (1.0, 0.0, 0.0):
        parser.error("This profile requires one container and zero color jitter.")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    if args.dataset_root and args.dataset_gen:
        base_out = os.path.join(args.dataset_root, args.dataset_gen)
        if args.stage:
            base_out = os.path.join(base_out, args.stage)
        args.output_dir = base_out
        if args.filtered_coco_path:
            args.filtered_coco_path = os.path.join(args.output_dir, args.filtered_coco_path)

    crop_suffix = args.crop_ablation
    if not crop_suffix.endswith(".json"):
        crop_suffix = f"{crop_suffix}.json"
    args.coco_path = f"{args.coco_dir}_{crop_suffix}"

    if args.offline:
        print("Offline mode enabled: using local HuggingFace cache only.")

    if args.mode == "match":
        print("Mode: match")
        match_dataset_to_local_xray(
            xray_dir=args.xray_dir,
            output_path=args.output_path,
            splits=("test", "invisible"),
            limit=args.limit,
            offline=args.offline,
        )
    elif args.mode == "split_coco":
        print("Mode: split_coco")
        summary = split_coco_by_matches_and_plot(
            coco_path=args.coco_path,
            matches_path=args.matches_path,
            output_plot_path=args.plot_path,
            title="COCO Cutler MUSC Annotation Area Distribution",
        )
        print(json.dumps(summary, indent=2))
    elif args.mode == "merge":
        print("Mode: merge")
        output_path = os.path.join(args.output_dir, args.merge_output)
        merge_coco_jsons(
            output_dir=args.output_dir,
            output_path=output_path,
            pattern=args.merge_pattern,
        )
    else:
        print(f"Mode: {args.mode}")
        filtered_coco, stats = filter_coco_for_test_split(
            coco_path=args.coco_path,
            matches_path=args.matches_path,
            xray_test_anno_dir=args.xray_test_anno_dir,
            max_area_ratio=args.max_area_ratio,
            output_path=args.filtered_coco_path or None,
        )
        print(json.dumps(stats, indent=2))

        print("Loading dataset PAGF/DET-COMPASS...,offline={}".format(args.offline))
        dataset_xp = _load_det_compass_dataset(offline=args.offline)
        print("Dropping rgb_image columns to speed up...")
        for split in ("test", "invisible"):
            if split in dataset_xp and "rgb_image" in dataset_xp[split].column_names:
                dataset_xp[split] = dataset_xp[split].remove_columns("rgb_image")

        if args.xray_vis_dir:
            vis_dir = args.xray_vis_dir
            if not os.path.isabs(vis_dir):
                vis_dir = os.path.join(args.output_dir, vis_dir)
            _visualize_filtered_coco_xray(
                filtered_coco=filtered_coco,
                dataset_test=dataset_xp["test"],
                matches_path=args.matches_path,
                output_dir=vis_dir,
                limit=args.xray_vis_limit,
            )

        if args.seed is not None:
            print(f"Setting random seed to {args.seed}...")
            set_seed(args.seed)

        foreground_samples = _build_foreground_samples(
            filtered_coco=filtered_coco,
            dataset_test=dataset_xp["test"],
            matches_path=args.matches_path,
            output_dir=args.output_dir if args.start_id == 0 else None,
        )
        background_samples = list(dataset_xp["invisible"])

        if args.fg_vis_dir:
            fg_vis_dir = args.fg_vis_dir
            if not os.path.isabs(fg_vis_dir):
                fg_vis_dir = os.path.join(args.output_dir, fg_vis_dir)
            _visualize_foreground_object_bank(
                foreground_samples=foreground_samples,
                output_dir=fg_vis_dir,
                limit=args.fg_vis_limit,
            )

        _prepare_foreground_cache(foreground_samples)
        _prepare_background_cache(background_samples, image_size=(580, 880))

        print("Starting synthetic generation...")
        print(
            json.dumps(
                {
                    "foreground_samples": len(foreground_samples),
                    "background_samples": len(background_samples),
                },
                indent=2,
            )
        )

        if args.mode == "preview":
            preview_dir = args.preview_dir
            if not os.path.isabs(preview_dir):
                preview_dir = os.path.join(args.output_dir, preview_dir)
            generate_preview_color_map_examples(
                foreground_samples=foreground_samples,
                background_samples=background_samples,
                output_dir=preview_dir,
                preview_num_samples=args.preview_num_samples,
                num_objects_range=args.num_objects,
                gamma_range=args.gamma,
                max_perspective=args.max_perspective,
                composition_mode=args.composition_mode,
                color_map_mode=args.color_map_mode,
                color_map_palette=args.color_map_palette,
                color_map_jitter=args.color_map_jitter,
                color_map_seed=args.color_map_seed,
                container_count_probs=args.container_count_probs,
            )
        else:
            generate_synthetic_dataset_musc(
                foreground_samples=foreground_samples,
                background_samples=background_samples,
                output_dir=args.output_dir,
                num_generated=args.num_generated,
                num_objects_range=args.num_objects,
                gamma_range=args.gamma,
                max_perspective=args.max_perspective,
                composition_mode=args.composition_mode,
                color_map_mode=args.color_map_mode,
                color_map_palette=args.color_map_palette,
                color_map_jitter=args.color_map_jitter,
                color_map_seed=args.color_map_seed,
                container_count_probs=args.container_count_probs,
                build_segmentation=args.build_segmentation,
                io_workers=args.io_workers,
                start_id=args.start_id,
                info=filtered_coco.get("info", {}),
                licenses=filtered_coco.get("licenses", []),
                categories=filtered_coco.get("categories", []),
                base_seed=args.seed,
                stage=args.stage,
            )


if __name__ == "__main__":
    main()
