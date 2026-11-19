from compass_editor.colour import sample_to_highlow, combine_highlow_images, highlow_to_colour, highlow_to_grey, highlow_to_density
from compass_editor.colour import has_mask_on_border, center_pad_unpad, get_bbox_from_mask
import random
import numpy as np
import cv2
from PIL import Image
import math

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


def _get_cached_rgb_and_mask(fg_sample):
    cache = fg_sample.setdefault("_cached_arrays", {})
    if "rgb_img" not in cache or "rgb_mask" not in cache:
        rgb_img = np.asarray(fg_sample["rgb_image"])[..., 0:3]  # drop alpha
        rgb_mask = np.asarray(fg_sample["rgb_image_mask"]).astype(np.uint8)
        cache["rgb_img"] = rgb_img
        cache["rgb_mask"] = rgb_mask
    return cache["rgb_img"], cache["rgb_mask"]

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

def split_dataset(dataset, only_xray_visible=True, only_rgb_visible=True):
    foreground_samples = []
    background_samples = []

    for sample in dataset:
        if sample['is_background']:
            background_samples.append(sample)
        else:
            is_usable = True
            if only_xray_visible and not sample['xray_visible']:
                is_usable = False
            if only_rgb_visible and not sample['rgb_visible']:
                is_usable = False
            
            if is_usable:
                foreground_samples.append(sample)
    
    return foreground_samples, background_samples


def create_synthetic_sample(
    foreground_samples,
    background_samples,
    min_objs=5,
    max_objs=9,
    generate_rgb=True,
    image_size=(580,880),
    max_angle=180,
    max_scale=1.2,
    max_t=0.75,
    # allowed_categories=None,
    max_perspective=0.05
):
    """
    Create a synthetic sample by compositing transformed foreground objects onto a background.
    Returns a sample dict in the required format.
    Args:
        allowed_categories: Optional iterable of category names to restrict foreground sampling.
        max_perspective: Max ratio (0-1) for random corner offsets when sampling a homography.
    """
    sample = {
        "rgb_image": None,
        "rgb_width": None,
        "rgb_height": None,
        "xray_images": [],
        "xray_width": None,
        "xray_height": None,
        "objects": {
            "rgb_bbox": [] if generate_rgb else None,
            "rgb_mask": [] if generate_rgb else None,
            "xray_bbox": [],
            "xray_mask": [],
            "categories": [],
            "rgb_visible": [],
            "xray_visible": [],
            "dangerous": [],
            "IN_ids": [],
            "WN_ids": []
        }
    }

    # 1. Randomly choose one background_sample
    bg_sample = np.random.choice(background_samples)
    bg_img_original = sample_to_highlow(bg_sample, as_image=True)  # WxHx3
    bg_img, xray_pad = center_pad_unpad(bg_img_original, target_size=image_size, constant_values=255)

    bg_img_rgb = None
    if generate_rgb:
        bg_img_rgb_original = np.asarray(bg_sample['rgb_image'])[..., 0:3]  # We remove the alpha channel
        xray_h, xray_w, _ = bg_img_original.shape
        rgb_h, rgb_w, _ = bg_img_rgb_original.shape

        scale_y = float(rgb_h) / float(xray_h)
        scale_x = float(rgb_w) / float(xray_w)

        rgb_pad = xray_pad

        rgb_pad = (math.ceil((xray_pad[0] + xray_pad[1]) * scale_y), math.ceil((xray_pad[2] + xray_pad[3]) * scale_x))
        image_size_rgb = (rgb_h + rgb_pad[0], rgb_w + rgb_pad[1])

        bg_img_rgb, _ = center_pad_unpad(bg_img_rgb_original, target_size=image_size_rgb, constant_values=255)


    # 2. Randomly choose number of objects and select foreground samples
    num_objs = np.random.randint(min_objs, max_objs+1)

    # if allowed_categories:
    #     allowed_set = set(allowed_categories)
    #     filtered_foreground = [s for s in foreground_samples if s.get('category', 'unknown') in allowed_set]
    #     if not filtered_foreground:
    #         raise ValueError("No foreground samples match the provided allowed_categories.")
    # else:
    filtered_foreground = foreground_samples

    fg_samples = np.random.choice(filtered_foreground, size=num_objs, replace=True).tolist()

    # Sort objects by area (descending, from smallest to biggest)
    fg_samples.sort(key=lambda x: np.sum(np.array(x['xray_image_mask']).astype(np.int32)), reverse=True)

    # Store transformed objects for sorting
    transformed_objs = transform_xray_crops(
        fg_samples=fg_samples,
        image_size=bg_img.shape[0:2],
        max_angle=max_angle,
        max_scale=max_scale,
        max_t=max_t,
        max_perspective=max_perspective
    )

    if generate_rgb:
        apply_transform_to_rgb_crops(
            transformed_xray_objs=transformed_objs,
            image_size=image_size_rgb,
            scale=(scale_y, scale_x)
        )

    composite_img = bg_img
    composite_img_rgb = bg_img_rgb
    for i, obj in enumerate(transformed_objs):
        composite_img = combine_highlow_images(composite_img, obj['fg_img_t'], substract=False, maskB=obj['fg_mask_t'])

        if generate_rgb:
            mask_rgb_obj = np.stack([obj['rgb_mask_t']]*composite_img_rgb.shape[2], axis=-1)
            composite_img_rgb = np.where(mask_rgb_obj, obj['rgb_img_t'], composite_img_rgb)
            bbox_rgb = get_bbox_from_mask(obj['rgb_mask_t'])

            sample['objects']['rgb_bbox'].append(bbox_rgb)
            sample['objects']['rgb_mask'].append(Image.fromarray(obj['rgb_mask_t'] > 0).convert('1'))

        bbox = get_bbox_from_mask(obj['fg_mask_t'])

        sample['objects']['xray_bbox'].append(bbox)
        sample['objects']['xray_mask'].append(Image.fromarray(obj['fg_mask_t'] > 0).convert('1'))
        
        sample['objects']['categories'].append(obj['fg_sample'].get('category', 'unknown'))
        sample['objects']['rgb_visible'].append(obj['fg_sample'].get('rgb_visible', False))
        sample['objects']['xray_visible'].append(obj['fg_sample'].get('xray_visible', False))
        sample['objects']['dangerous'].append(obj['fg_sample'].get('dangerous', False))
        sample['objects']['IN_ids'].append(obj['fg_sample'].get('IN_id', ''))
        sample['objects']['WN_ids'].append(obj['fg_sample'].get('WN_id', ''))
    
    composite_img = composite_img[...,0:2]
    
    # 7. Create xray_images (Colour, Grey, High, Low, Density)
    xray_colour = highlow_to_colour(composite_img)
    xray_grey = highlow_to_grey(composite_img)
    xray_high = Image.fromarray(composite_img[...,0], mode="L")
    xray_low = Image.fromarray(composite_img[...,1], mode="L")
    xray_density = highlow_to_density(composite_img)

    sample['xray_images'] = [xray_colour, xray_grey, xray_high, xray_low, xray_density]
    sample['xray_height'] = xray_colour.shape[0]
    sample['xray_width'] = xray_colour.shape[1]
    if generate_rgb:
        sample['rgb_image'] = composite_img_rgb
        sample['rgb_height'] = composite_img_rgb.shape[0]
        sample['rgb_width'] = composite_img_rgb.shape[1]

    return sample


def transform_xray_crops(fg_samples, image_size, max_angle=180, max_scale=1.2, max_t=0.75, max_perspective=0.05):
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
            # 3. Random affine transformation
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

            # 4. Apply transformation to mask
            fg_mask_t = cv2.warpPerspective(canvas_mask, H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

            if not has_mask_on_border(fg_mask_t):
                found_affine_transform = True
                break

        # 4.b. Apply transformation to image
        fg_img_t = cv2.warpPerspective(canvas_img, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

        area = np.sum(fg_mask_t > 0)
        if area > 0:
            transformed_objs.append({
                'idx': idx,
                'fg_sample': fg_sample,
                'fg_img_t': fg_img_t,
                'fg_mask_t': fg_mask_t,
                'transform': H,
                'transform_details': {
                    'center': center,
                    'angle': angle,
                    'scale': scale,
                    't_norm': (tx / float(w), ty / float(h)),
                    'perspective_offsets_norm': (perspective_offsets / np.array([[w, h]])).tolist()
                }
            })

    return transformed_objs


def apply_transform_to_rgb_crops(transformed_xray_objs, image_size, scale):
    """
    Applies the affine transformation from the x-ray domain to the corresponding RGB image and mask.

    Parameters:
        transformed_xray_objs: list of dicts containing the x-ray transformed foreground samples and affine matrices.
        image_size: (h, w) of the target padded RGB background.
        scale: (scale_y, scale_x) ratio between RGB and X-ray image sizes.

    Returns:
        transformed_objs: list of updated transformed_xray_objs with rgb_img_t and rgb_mask_t added.
    """
    scale_y, scale_x = scale
    h, w = image_size
    updated_objs = []

    for obj in transformed_xray_objs:
        fg_sample = obj['fg_sample']
        rgb_img, rgb_mask = _get_cached_rgb_and_mask(fg_sample)
        T_xray = obj['transform_details']

        H_orig, W_orig = rgb_mask.shape

        # Center the RGB object on a canvas of size (h, w)
        canvas_img = np.full((h, w, 3), 255, dtype=np.uint8)
        canvas_mask = np.zeros((h, w), dtype=np.uint8)

        y0 = (h - H_orig) // 2
        x0 = (w - W_orig) // 2
        canvas_img[y0:y0+H_orig, x0:x0+W_orig, :] = rgb_img
        canvas_mask[y0:y0+H_orig, x0:x0+W_orig] = rgb_mask

        center = (w//2, h//2)
        M_rgb = cv2.getRotationMatrix2D(center, T_xray['angle'], T_xray['scale']).astype(np.float32)

        if 't_norm' in T_xray:
            tx_rgb = T_xray['t_norm'][0] * w
            ty_rgb = T_xray['t_norm'][1] * h
        else:
            tx_rgb = T_xray.get('t', (0, 0))[0] * scale_x
            ty_rgb = T_xray.get('t', (0, 0))[1] * scale_y

        M_rgb[:,2] += [tx_rgb, ty_rgb]
        H_affine_rgb = np.vstack([M_rgb, [0, 0, 1]]).astype(np.float32)

        src_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        offsets_norm = np.array(T_xray.get('perspective_offsets_norm', np.zeros((4, 2))), dtype=np.float32)
        if offsets_norm.shape != (4, 2):
            offsets_norm = np.zeros((4, 2), dtype=np.float32)
        dst_corners = src_corners + offsets_norm * np.array([w, h], dtype=np.float32)
        H_perspective_rgb = cv2.getPerspectiveTransform(src_corners, dst_corners)

        H_rgb = H_perspective_rgb @ H_affine_rgb

        # Apply the transformed homography matrix to the RGB image and mask
        rgb_img_t = cv2.warpPerspective(canvas_img, H_rgb, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        rgb_mask_t = cv2.warpPerspective(canvas_mask, H_rgb, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        obj['rgb_img_t'] = rgb_img_t
        obj['rgb_mask_t'] = rgb_mask_t
        obj['rgb_transform'] = H_rgb
        updated_objs.append(obj)

    return updated_objs
