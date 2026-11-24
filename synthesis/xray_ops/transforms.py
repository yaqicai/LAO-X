# Derived from DET-COMPASS-CROPS by Lorenzo Vaquero (used with permission);
# modified for LAO-X.
"""Seeding and random geometric transformation of object crops."""
import random

import cv2
import numpy as np

from xray_ops.colour import has_mask_on_border, sample_to_highlow

try:
    import torch
except ImportError:
    torch = None


def _get_cached_highlow_and_mask(fg_sample):
    """
    Cache heavy conversions for foreground samples to avoid repeated PIL/array work.
    """
    cache = fg_sample.setdefault("_cached_arrays", {})
    if "highlow" not in cache or "xray_mask" not in cache:
        cache["highlow"] = sample_to_highlow(fg_sample, as_image=True)
        cache["xray_mask"] = np.array(fg_sample["xray_image_mask"], dtype=np.uint8)
    return cache["highlow"], cache["xray_mask"]


def set_seed(seed: int):
    """
    Set seed for reproducibility across random, numpy, and torch (if available).

    Parameters:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Seed set to: {seed}")


def transform_xray_crops(fg_samples, image_size, max_angle=180, max_scale=1.2, max_t=0.75, max_perspective=0.05):
    """Place each crop at the canvas centre and apply a random rotation, scale,
    translation and small perspective jitter, re-drawing until the mask stays
    clear of the image border (identity after 100 failed tries).

    Returns a list of dicts with keys 'fg_sample', 'fg_img_t' (HxWx3) and
    'fg_mask_t' (HxW); objects whose transformed mask is empty are dropped.
    """
    h, w = image_size

    transformed_objs = []
    for idx, fg_sample in enumerate(fg_samples, 1):
        fg_img, fg_mask = _get_cached_highlow_and_mask(fg_sample)  # HxWxC, HxW

        if fg_img.shape[:2] != fg_mask.shape:
            raise ValueError(
                f"Foreground image/mask shape mismatch: {fg_img.shape[:2]} vs "
                f"{fg_mask.shape}"
            )

        # Tight object crops can be wider or taller than a single container.
        # Fit only oversized crops, leaving a one-pixel border for identity fallback.
        Hf, Wf = fg_mask.shape
        fit_scale = min((h - 2) / float(Hf), (w - 2) / float(Wf), 1.0)
        if fit_scale < 1.0:
            fitted_h = max(1, int(np.floor(Hf * fit_scale)))
            fitted_w = max(1, int(np.floor(Wf * fit_scale)))
            fg_img = cv2.resize(
                fg_img, (fitted_w, fitted_h), interpolation=cv2.INTER_LINEAR
            )
            fg_mask = cv2.resize(
                fg_mask, (fitted_w, fitted_h), interpolation=cv2.INTER_NEAREST
            )
            Hf, Wf = fg_mask.shape

        found_affine_transform = False
        num_tries = 0
        while not found_affine_transform:
            # Random affine transformation
            angle = np.random.uniform(-max_angle, max_angle)
            scale = np.random.uniform(1-(max_scale-1), max_scale)
            tx = np.random.randint(int(-(w//2)*max_t), int((w//2)*max_t))
            ty = np.random.randint(int(-(h//2)*max_t), int((h//2)*max_t))
            center = (w//2, h//2)

            num_tries += 1
            fallback_identity = num_tries > 100
            if fallback_identity:
                print(f'Failed to find affine transform for object {idx} ({fg_sample["category"]}) after {num_tries} tries. Skipping transform.')
                angle = 0
                scale = 1.0
                tx = 0
                ty = 0
                center = (w//2, h//2)
                found_affine_transform = True

            M = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
            M[:,2] += [tx, ty]
            H_affine = np.vstack([M, [0, 0, 1]]).astype(np.float32)

            # Sample a small random homography (perspective jitter)
            src_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            jitter_x = w * max_perspective
            jitter_y = h * max_perspective
            if fallback_identity:
                perspective_offsets = np.zeros((4, 2), dtype=np.float32)
            else:
                perspective_offsets = np.float32([
                    [np.random.uniform(-jitter_x, jitter_x), np.random.uniform(-jitter_y, jitter_y)],
                    [np.random.uniform(-jitter_x, jitter_x), np.random.uniform(-jitter_y, jitter_y)],
                    [np.random.uniform(-jitter_x, jitter_x), np.random.uniform(-jitter_y, jitter_y)],
                    [np.random.uniform(-jitter_x, jitter_x), np.random.uniform(-jitter_y, jitter_y)],
                ])
            dst_corners = src_corners + perspective_offsets
            H_perspective = cv2.getPerspectiveTransform(src_corners, dst_corners)

            H = H_perspective @ H_affine

            # fg_img: Hf x Wf x C, fg_mask: Hf x Wf
            canvas_img = np.ones((h, w, 3), dtype=fg_img.dtype) * 255
            canvas_mask = np.zeros((h, w), dtype=fg_mask.dtype)

            # Compute top-left corner to center the object
            y0 = (h - Hf) // 2
            x0 = (w - Wf) // 2

            # Place the object in the center
            canvas_img[y0:y0+Hf, x0:x0+Wf, :] = fg_img
            canvas_mask[y0:y0+Hf, x0:x0+Wf] = fg_mask

            # Apply transformation to mask
            fg_mask_t = cv2.warpPerspective(canvas_mask, H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

            if not has_mask_on_border(fg_mask_t):
                found_affine_transform = True
                break

        # Apply transformation to image
        fg_img_t = cv2.warpPerspective(canvas_img, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

        area = np.sum(fg_mask_t > 0)
        if area > 0:
            transformed_objs.append({
                'fg_sample': fg_sample,
                'fg_img_t': fg_img_t,
                'fg_mask_t': fg_mask_t,
            })

    return transformed_objs
