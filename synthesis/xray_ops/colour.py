# Derived from DET-COMPASS-CROPS by Lorenzo Vaquero (used with permission);
# modified for LAO-X.
"""High/low-energy X-ray utilities and the fixed DET-COMPASS lookup tables.

The three lookup tables map a (high, low) attenuation pair to the colour,
grey and density renderings of DET-COMPASS. They are read from
``$LAOX_LUT_DIR`` (default: ``synthesis/luts/``) the first time they are
needed; see ``build_luts.py`` for how they are built.
"""
import math
import os
import pickle

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


LUT_DIR_ENV = "LAOX_LUT_DIR"
DEFAULT_LUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "luts"
)
LUT_NAMES = ("colour", "grey", "density")

_LUTS = {}


def lut_dir():
    """Directory holding the raw LUT pickles: $LAOX_LUT_DIR or synthesis/luts/."""
    return os.environ.get(LUT_DIR_ENV) or DEFAULT_LUT_DIR


def lut_path(name):
    """Path of the raw pickle for one table ('colour', 'grey' or 'density')."""
    if name not in LUT_NAMES:
        raise ValueError(f"Unknown LUT '{name}'; expected one of {LUT_NAMES}.")
    return os.path.join(lut_dir(), f"xray_highlow_to_{name}_dict_RAW.pkl")


def _denoise_dict(raw_colour_dict, min_total_occurrences=10000, average_quantile=0.01):
    final_dict = {}
    for grey_key, colour_counter in raw_colour_dict.items():
        total_occurrences = sum(colour_counter.values())
        if total_occurrences < min_total_occurrences:
            continue

        num_unique_colours = len(colour_counter)
        if num_unique_colours == 0:
            continue

        # Determine the number of top colors to average
        num_to_average = math.ceil(average_quantile * num_unique_colours)
        num_to_average = max(1, num_to_average)

        most_common_items = colour_counter.most_common(num_to_average)
        colours_to_average = [item[0] for item in most_common_items]

        if colours_to_average:
            # Determine if grayscale (int) or multi-channel (tuple)
            if isinstance(colours_to_average[0], int):
                # Grayscale averaging
                avg = sum(colours_to_average) / len(colours_to_average)
                final_dict[grey_key] = round(avg)
            else:
                # Multi-channel averaging (e.g., RGB, RGBA)
                num_channels = len(colours_to_average[0])
                channel_totals = [0] * num_channels

                for colour in colours_to_average:
                    for i in range(num_channels):
                        channel_totals[i] += colour[i]

                avg_channels = [round(total / len(colours_to_average)) for total in channel_totals]
                final_dict[grey_key] = tuple(avg_channels)

    return final_dict


def _get_lut(name, min_total_occurrences=10000, average_quantile=0.01):
    """Load, denoise and densify one lookup table (cached per process)."""
    if name not in _LUTS:
        dict_file = lut_path(name)
        if not os.path.isfile(dict_file):
            raise FileNotFoundError(
                f"Lookup table not found: {dict_file}. Build it with "
                f"'python build_luts.py {name}' or set {LUT_DIR_ENV}."
            )
        print(f"Loading {name} LUT {dict_file}")
        with open(dict_file, "rb") as f:
            raw_dict = pickle.load(f)
        final_dict = _denoise_dict(
            raw_dict,
            min_total_occurrences=min_total_occurrences,
            average_quantile=average_quantile,
        )
        _LUTS[name] = interpolate_xray_palette(final_dict)
    return _LUTS[name]


def interpolate_xray_palette(anchor_points, k=5):
    """
    Fills a sparse X-ray color palette using k-NN interpolation in log-attenuation space.
    Supports both grayscale (single value) and RGB (tuple of three values).

    Args:
        anchor_points (dict): A dictionary mapping sparse (high, low) pixel tuples
                              to either grayscale (int/float) or RGB (r, g, b) tuples.
        k (int): The number of nearest neighbors for interpolation. Defaults to 5.

    Returns:
        dict: A complete dictionary mapping every possible (high, low) tuple to an
              interpolated grayscale value or (r, g, b) color.

    Raises:
        ValueError: If the anchor_points dictionary is empty or malformed.
    """
    if not anchor_points:
        raise ValueError("The anchor_points dictionary cannot be empty.")

    known_coords_raw = np.array(list(anchor_points.keys()))
    known_values_raw = list(anchor_points.values())

    # Determine if values are RGB or grayscale
    sample_val = known_values_raw[0]
    is_rgb = isinstance(sample_val, (tuple, list)) and len(sample_val) == 3

    # Convert to 2D array
    known_colors = np.array(known_values_raw, dtype=np.float32)
    if not is_rgb:
        known_colors = known_colors.reshape(-1, 1)

    # 1. Transform anchor coordinates to log-attenuation space
    with np.errstate(divide='ignore'):
        known_coords_log = np.log(255.0 / np.maximum(1, known_coords_raw))

    # 2. Set up and fit k-NN regressors (1 per channel)
    regressors = []
    for i in range(known_colors.shape[1]):
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn.fit(known_coords_log, known_colors[:, i])
        regressors.append(knn)

    # 3. Create a grid for all possible pixel values and transform to log space
    h_vals, l_vals = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
    full_coords_raw = np.vstack([h_vals.ravel(), l_vals.ravel()]).T

    with np.errstate(divide='ignore'):
        full_coords_log = np.log(255.0 / np.maximum(1, full_coords_raw))

    # 4. Predict color values
    preds = [reg.predict(full_coords_log) for reg in regressors]
    predicted_colors = np.stack(preds, axis=-1)
    predicted_colors = np.clip(predicted_colors, 0, 255).astype(np.uint8)

    if not is_rgb:
        # Flatten to single grayscale values
        predicted_colors = predicted_colors[:, 0]

    # 5. Assemble the final lookup dictionary
    filled_dict = {
        tuple(coord): (
            tuple(color) if is_rgb else int(color)
        )
        for coord, color in zip(full_coords_raw, predicted_colors)
    }

    return filled_dict


def quantize_color(rgb):
    return tuple((np.array(rgb)).astype(int))


def sample_to_highlow(sample, as_image=False):
    """Stack the high/low-energy channels of a sample into an HxWx2 uint8 array.

    Accepts a DET-COMPASS sample (``xray_images`` list) or a cropped object
    (``raw_xray_image_high``/``raw_xray_image_low``). With ``as_image=True`` a
    third all-zero channel is appended (HxWx3).
    """
    if 'xray_images' in sample:
        # DET-COMPASS sample
        sample_highlow = np.stack([sample["xray_images"][2], sample["xray_images"][3]], axis=-1)
        if as_image:
            sample_highlow = np.pad(sample_highlow, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)

    elif 'raw_xray_image_high' in sample and 'raw_xray_image_low' in sample:
        # Cropped object-bank sample
        sample_highlow = np.stack([sample["raw_xray_image_high"], sample["raw_xray_image_low"]], axis=-1)
        if as_image:
            sample_highlow = np.pad(sample_highlow, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)

    else:
        raise ValueError('Unrecognized sample format!')

    if sample_highlow.dtype != np.uint8:
        sample_highlow = sample_highlow.astype(np.float64)
        sample_min = np.min(sample_highlow)
        sample_max = np.max(sample_highlow)

        assert sample_min >= 0.0 and sample_max <= 1.0, "Sample highlow data was encoded in an incorrect format!"
        sample_highlow = sample_highlow * 255.0
        sample_highlow = np.clip(sample_highlow, 0, 255).astype(np.uint8)

    return sample_highlow


def highlow_to_colour(high_low_img):
    mapping_dict = _get_lut("colour")

    grey_img = np.array(high_low_img)

    h, w, _ = grey_img.shape
    colour_img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            grey_rgb = quantize_color(grey_img[y, x])
            colour_img[y, x] = mapping_dict.get(grey_rgb, (0, 0, 0))  # Fallback to black
    return colour_img


def combine_highlow_images(imgA: np.ndarray, imgB: np.ndarray, substract=False, maskB=None, epsilon: float = 1e-9) -> np.ndarray:
    """
    Combines (substract=False) or separates (substract=True) the X-ray attenuation
    of two dual-energy images, following the Beer-Lambert law (I = I_0 * exp(-mu*x)).

    The raw pixel values are proportional to the transmitted intensity (I). To work with
    attenuation (mu*x), we need to operate in log space:
    1. Convert raw intensity values to a floating-point representation.
    2. Normalize intensities to get transmittance (T = I / I_0). We approximate I_0
       with the maximum possible pixel value (255).
    3. Calculate absorbance (A = -log(T)), which is proportional to attenuation.
    4. Add (or subtract) the absorbance of imgB to (from) the absorbance of imgA.
    5. Convert the resulting absorbance back to an intensity image.

    Args:
        imgA (np.ndarray): Dual-energy X-ray scan, e.g. the container.
            Expected shape is (height, width, C) with 8-bit pixel values (0-255).
        imgB (np.ndarray): Dual-energy X-ray scan of e.g. a transformed object.
            Must have the same shape and data type as imgA.
        substract (bool): Subtract imgB's absorbance instead of adding it.
        maskB (np.ndarray): Optional HxW mask; outside it the result is imgA.
        epsilon (float): A small constant to prevent division by zero or log(0) errors
            for pixels with an intensity of 0.

    Returns:
        np.ndarray: A NumPy array of the same shape and original data type.
    """
    # --- Input Validation ---
    if imgB.shape != imgA.shape:
        raise ValueError("Input tensors must have the same shape.")
    if imgB.ndim != 3:
        raise ValueError("Input tensors must have a shape of (height, width, C).")

    # Store the original data type to return the final tensor in the same format.
    original_dtype = imgB.dtype

    # --- Conversion to Absorbance (Proportional to Attenuation) ---

    # Convert integer tensors to float for calculations.
    imgB_float = imgB.astype(np.float64)
    imgA_float = imgA.astype(np.float64)

    # Normalize the intensity values to get transmittance (T).
    # We add epsilon to avoid log(0) issues for black pixels (intensity = 0).
    # I_0 is approximated by the maximum value of the data type (255 for 8-bit).
    transmittance_imgB = np.clip((imgB_float + epsilon) / 255.0, 0.0, 1.0)
    transmittance_imgA = np.clip((imgA_float + epsilon) / 255.0, 0.0, 1.0)

    # Calculate absorbance (A = -log(T)). Absorbance is additive.
    absorbance_imgB = -np.log(transmittance_imgB)
    absorbance_imgA = -np.log(transmittance_imgA)

    # --- Combine the attenuations ---
    if substract:
        absorbance_imgC = absorbance_imgA - absorbance_imgB
    else:
        absorbance_imgC = absorbance_imgA + absorbance_imgB

    # Ensure the resulting absorbance is non-negative.
    absorbance_imgC[absorbance_imgC < 0] = 0

    # --- Conversion back to Intensity Image ---

    # T = exp(-A)
    transmittance_imgC = np.exp(-absorbance_imgC)

    # Convert transmittance back to pixel intensity values.
    intensity_imgC = transmittance_imgC * 255.0

    # Clip the values to the valid 8-bit range [0, 255] and convert back
    # to the original data type (e.g., uint8).
    imgC = np.clip(intensity_imgC, 0, 255).astype(original_dtype)

    if maskB is not None:
        maskB = np.stack([maskB]*imgB.shape[2], axis=-1)
        imgC = np.where(maskB, imgC, imgA)

    return imgC


def highlow_to_grey(high_low_img):
    mapping_dict = _get_lut("grey")

    grey_img = np.array(high_low_img)

    h, w, _ = grey_img.shape
    colour_img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            grey_rgb = quantize_color(grey_img[y, x])
            colour_img[y, x] = mapping_dict.get(grey_rgb, (0, 0, 0))  # Fallback to black
    return colour_img


def highlow_to_density(high_low_img):
    mapping_dict = _get_lut("density")

    grey_img = np.array(high_low_img)

    h, w, _ = grey_img.shape
    colour_img = np.zeros((h, w, 1), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            grey_rgb = quantize_color(grey_img[y, x])
            colour_img[y, x] = mapping_dict.get(grey_rgb, (255))  # Fallback to white

    colour_img = colour_img[..., 0]

    return colour_img


def has_mask_on_border(mask):
    # Check last 5 top and bottom rows
    top = mask[0:5, :].flatten()
    bottom = mask[-1:-6:-1, :].flatten()

    # Check left and right columns
    left = mask[:, 0:5].flatten()
    right = mask[:, -1:-6:-1].flatten()

    # Combine all borders
    border = np.concatenate([top, bottom, left, right])

    return np.any(border)


def center_pad_unpad(tensor, target_size, constant_values=0):
    """
    Pads or crops (centered) a HxWxC tensor to target size (target_height, target_width).

    Args:
        tensor (np.ndarray): Input array of shape (H, W, C)
        target_size (tuple): Desired output (height, width)
        constant_values: Padding value

    Returns:
        np.ndarray: Tensor of shape (target_height, target_width, C)
        tuple: Final (top, bottom, left, right) padding (negative = cropped)
    """
    target_height, target_width = target_size

    H, W, C = tensor.shape

    # Determine crop or pad amounts
    pad_h = max(target_height - H, 0)
    pad_w = max(target_width - W, 0)

    crop_h = max(H - target_height, 0)
    crop_w = max(W - target_width, 0)

    # Padding (pad equally on both sides, extra goes to bottom/right)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        tensor = np.pad(tensor, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=constant_values)

    # Cropping (crop equally from both sides, extra from bottom/right)
    crop_top = crop_h // 2
    crop_bottom = crop_h - crop_top
    crop_left = crop_w // 2
    crop_right = crop_w - crop_left

    if crop_h > 0 or crop_w > 0:
        tensor = tensor[crop_top:H - crop_bottom, crop_left:W - crop_right, :]

    final_pad = (pad_top - crop_top, pad_bottom - crop_bottom, pad_left - crop_left, pad_right - crop_right)
    tensor = tensor.copy()

    return tensor, final_pad


def get_bbox_from_mask(mask):  # Bounding box [x_min, y_min, width, height]

    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bbox = [float(x_min), float(y_min), float(x_max-x_min+1), float(y_max-y_min+1)]
    else:
        bbox = [0.0, 0.0, 0.0, 0.0]

    return bbox
