import numpy as np
import torch
import cv2
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from compass_editor import colour

def xywh_to_xyxy(bbox):
    x_min, y_min, w, h = bbox
    return [x_min, y_min, x_min+w, y_min+h]

def estimate_affine_transform(imgA, imgB, show_matches=False):
    """
    Detect ORB features in imgA & imgB, match them,
    then estimate full 2×3 affine transform M (rotation, scale, shear, translation).
    """
    # gray
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # ORB detect & compute
    orb = cv2.ORB_create(5000)
    kpsA, desA = orb.detectAndCompute(grayA, None)
    kpsB, desB = orb.detectAndCompute(grayB, None)

    # BFMatcher + crossCheck
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desA, desB)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 3:
        raise RuntimeError("Too few matches to estimate affine.")

    # pts arrays
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # full affine under RANSAC
    M, inliers = cv2.estimateAffine2D(ptsA, ptsB, method=cv2.RANSAC)
    if M is None:
        raise RuntimeError("Affine estimation failed.")
    
    if show_matches:
        vis = cv2.drawMatches(imgA, kpsA, imgB, kpsB, matches[:50], None, flags=2)
        cv2.imshow("Matches", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return M

def bbox_to_mask(mask: np.ndarray) -> np.ndarray:
    """
    Creates a binary mask where the axis-aligned bounding box (AABB) of the input mask is filled.
    
    Args:
        mask: 2D binary NumPy array (bool or 0/1).
    
    Returns:
        2D binary NumPy array with AABB filled (dtype: uint8).
    """
    # Get coordinates of foreground pixels
    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        # Empty input mask
        return np.zeros_like(mask, dtype=np.uint8)
    
    # Bounding box coordinates
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Create new mask
    box_mask = np.zeros_like(mask, dtype=np.uint8)
    box_mask[y_min:y_max + 1, x_min:x_max + 1] = 1

    return box_mask


def decompose_full_affine(M):
    """
    Decompose 2×3 M = [a b tx; c d ty] into rotation (deg), scale_x, scale_y, shear, translation.
    """
    a,b,tx = M[0]
    c,d,ty = M[1]
    # scales
    sx = np.hypot(a,c)
    sy = np.hypot(b,d)
    # shear
    shear = (a*b + c*d) / (sy*sy) if sy!=0 else 0
    # rotation from a,c
    rot = np.degrees(np.arctan2(c, a))
    return {"rotation_deg": rot, "scale_x": sx, "scale_y": sy, "shear": shear, "tx": tx, "ty": ty}


def get_sam_predictor():
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")  # 'facebook/sam2-hiera-large'

    return predictor


def image_similarity_ncc(imgA, imgB, mask=None):
    if mask is not None:
        imgA = imgA[mask]
        imgB = imgB[mask]

    imgA = (imgA - np.mean(imgA)) / (np.std(imgA) + 1e-5)
    imgB = (imgB - np.mean(imgB)) / (np.std(imgB) + 1e-5)
    return np.mean(imgA * imgB)  # Higher is better


def suppress_background(sam_predictor, sample, background_samples):
    """
    Given a sample and a list of background_samples, find the best aligned one and suppress background.
    Returns:
        - foreground image (color-mapped)
        - best transformed background (warped)
        - transform parameters (dict)
        - best index in background_samples
    """
    sample_img = np.array(sample["xray_images"][0].convert("RGB"))
    sample_bbox = xywh_to_xyxy([0, 0, sample['xray_width'], sample['xray_height']])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(sample_img)
        sample_mask = sam_predictor.predict(None, None, box=sample_bbox, multimask_output=False)[0][0].astype(np.uint8)

    best_iou = -1
    best_result = None
    best_index = -1

    # Loop over all background samples
    for i, bg in enumerate(background_samples):
        background_img = np.array(bg["xray_images"][0].convert("RGB"))
        background_bbox = xywh_to_xyxy([0, 0, bg['xray_width'], bg['xray_height']])

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam_predictor.set_image(background_img)
            background_mask = sam_predictor.predict(None, None, box=background_bbox, multimask_output=False)[0][0].astype(np.uint8)

        # Compute affine transform from background to sample
        M = estimate_affine_transform(background_img, sample_img)
        transformed_bg_mask = cv2.warpAffine(background_mask, M, (sample_mask.shape[1], sample_mask.shape[0]), flags=cv2.INTER_NEAREST)

        warped_bg = cv2.warpAffine(background_img, M, (sample_img.shape[1], sample_img.shape[0]), flags=cv2.INTER_LINEAR)
        score = image_similarity_ncc(sample_img, warped_bg)

        if score > best_iou:
            best_iou = score
            best_result = {
                "background_img": background_img,
                "background_mask": background_mask,
                "transform": M,
                "bg_index": i
            }

    if best_result is None:
        raise RuntimeError("No suitable background image found.")

    # Decompose transformation
    M = best_result["transform"]
    params = decompose_full_affine(M)

    # Build density images
    sample_density = np.stack([sample["xray_images"][2], sample["xray_images"][3]], axis=-1)
    background_density = np.stack([
        background_samples[best_result["bg_index"]]["xray_images"][2],
        background_samples[best_result["bg_index"]]["xray_images"][3]
    ], axis=-1)

    # Zero out background
    background_density[best_result["background_mask"] == 1] = 0

    # Warp background_density into sample frame
    H_s, W_s = sample_density.shape[:2]
    warped_density = cv2.warpAffine(background_density, M, (W_s, H_s), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    # Subtract and colorize
    fg = colour.combine_highlow_images(sample_density, warped_density, substract=True)
    fg_col = colour.highlow_to_colour(fg)

    return fg_col, fg, params

def apply_mask_and_crop(img: np.ndarray, mask: np.ndarray, fill_val=0) -> np.ndarray:
    """
    Applies a binary mask to an image:
    - Pixels outside the mask are set to `fill_val`.
    - Adds a transparency channel.
    - Crops to the minimal bounding box of the mask.

    Args:
        img (np.ndarray): Input image (H, W, 3) or (H, W, 4).
        mask (np.ndarray): Binary mask (H, W), values 0 or 255 (or 0/1).
        fill_val (int or tuple): Fill value for masked-out areas.

    Returns:
        np.ndarray: Processed image with transparency and cropping.
    """
    assert img.ndim == 3 and img.shape[2] in (3, 4), "Image must have 3 or 4 channels"
    assert mask.ndim == 2, "Mask must be 2D"

    img = np.copy(img)
    mask = np.copy(mask)

    # Normalize mask to boolean
    mask_bool = mask > 0

    # Create alpha channel
    alpha = (mask_bool * 255).astype(np.uint8)

    # Apply fill_val to RGB
    img_rgb = img[..., :3]
    if isinstance(fill_val, int):
        fill_val = (fill_val, fill_val, fill_val)

    masked_rgb = np.where(mask_bool[..., None], img_rgb, fill_val)

    # Stack RGB and alpha
    result = np.dstack([masked_rgb, alpha])

    # Find crop bounds
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)

    if not np.any(rows) or not np.any(cols):
        raise ValueError
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    cropped = result[y_min:y_max+1, x_min:x_max+1]

    cropped = cropped.astype(np.uint8)

    return cropped