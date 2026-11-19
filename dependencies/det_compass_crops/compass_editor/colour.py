import os
import math
import pickle

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


XRAY_COLOUR_DICT = None
XRAY_DENSITY_DICT = None
XRAY_GREY_DICT = None

def __denoise_dict(raw_colour_dict, min_total_occurrences=10000, average_quantile=0.01):
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

def __compute_xray_colour_dict(min_total_occurrences=10000, average_quantile=0.01):
    global XRAY_COLOUR_DICT

    if XRAY_COLOUR_DICT is not None:
        return
    
    dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xray_highlow_to_colour_dict_RAW.pkl")

    print(f'Loading Colour Dict {dict_file}')
    with open(dict_file, "rb") as f:
        raw_colour_dict = pickle.load(f)

    #print(f'Found {len(raw_colour_dict)} elements in the Colour Dict')

    #print('Denoising Colour Dict')
    final_dict = __denoise_dict(raw_colour_dict, min_total_occurrences=min_total_occurrences, average_quantile=average_quantile)

    #print(f'Reduced Colour Dict to {len(final_dict)} elements')

    #print('Interpolating missing Colour Dict elements')
    XRAY_COLOUR_DICT = interpolate_xray_palette(final_dict)
    #print(f'Interpolated {len(XRAY_COLOUR_DICT)-len(final_dict)} elements (total {len(XRAY_COLOUR_DICT)} elements)')


def __compute_xray_grey_dict(min_total_occurrences=10000, average_quantile=0.01):
    global XRAY_GREY_DICT

    if XRAY_GREY_DICT is not None:
        return
    
    dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xray_highlow_to_grey_dict_RAW.pkl")

    print(f'Loading Grey Dict {dict_file}')
    with open(dict_file, "rb") as f:
        raw_colour_dict = pickle.load(f)

    #print(f'Found {len(raw_colour_dict)} elements in the Grey Dict')

    #print('Denoising Grey Dict')
    final_dict = __denoise_dict(raw_colour_dict, min_total_occurrences=min_total_occurrences, average_quantile=average_quantile)

    #print(f'Reduced Grey Dict to {len(final_dict)} elements')

    #print('Interpolating missing Grey Dict elements')
    XRAY_GREY_DICT = interpolate_xray_palette(final_dict)
    #print(f'Interpolated {len(XRAY_GREY_DICT)-len(final_dict)} elements (total {len(XRAY_GREY_DICT)} elements)')


def __compute_xray_density_dict(min_total_occurrences=10000, average_quantile=0.01):
    global XRAY_DENSITY_DICT

    if XRAY_DENSITY_DICT is not None:
        return
    
    dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xray_highlow_to_density_dict_RAW.pkl")

    print(f'Loading Density Dict {dict_file}')
    with open(dict_file, "rb") as f:
        raw_colour_dict = pickle.load(f)

    # print(f'Found {len(raw_colour_dict)} elements in the Density Dict')

    # print('Denoising Density Dict')
    final_dict = __denoise_dict(raw_colour_dict, min_total_occurrences=min_total_occurrences, average_quantile=average_quantile)

    # print(f'Reduced Density Dict to {len(final_dict)} elements')

    # print('Interpolating missing Density Dict elements')
    XRAY_DENSITY_DICT = interpolate_xray_palette(final_dict)
    # print(f'Interpolated {len(XRAY_DENSITY_DICT)-len(final_dict)} elements (total {len(XRAY_DENSITY_DICT)} elements)')


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
    if 'xray_images' in sample:
        # We are dealing with an original DET-COMPASS sample
        sample_highlow = np.stack([sample["xray_images"][2], sample["xray_images"][3]], axis=-1)
        if as_image:
            sample_highlow = np.pad(sample_highlow, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)


    elif 'xray_image_highlow' in sample:
        # We are dealing with a DET-COMPASS-CROPS sample
        # Originally, I thought that we could directly use "xray_image_highlow"
        # However, there are some images that have too much background substracted (and the object is no longer visible)
        # Therefore, I make a 2-to-1 average with the sample that still contains the background
        sample_highlow_substracted = sample["xray_image_highlow"]
        if as_image:
            sample_highlow_substracted = np.asarray(sample_highlow_substracted)[:, :, 0:3]
        else:
            sample_highlow_substracted = np.asarray(sample_highlow_substracted)[:, :, 0:2]  # Due to the way it is stored in PIL, it has an extra channel of zeros and an extra transparency channel
        
        sample_highlow_raw = np.stack([sample["raw_xray_image_high"], sample["raw_xray_image_low"]], axis=-1)
        if as_image:
            sample_highlow_raw = np.pad(sample_highlow_raw, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)

        sample_highlow = ((2*sample_highlow_substracted.astype(np.float32) + sample_highlow_raw.astype(np.float32)) / 3.0).astype(np.uint8)

    elif 'raw_xray_image_high' in sample and 'raw_xray_image_low' in sample:
        # We are dealing with a DET-COMPASS-CROPS sample without computed highlow
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
    


def highlow_to_colour(high_low_img, mapping_dict=XRAY_COLOUR_DICT):
    if mapping_dict is None:
        __compute_xray_colour_dict()
        mapping_dict = XRAY_COLOUR_DICT

    grey_img = np.array(high_low_img)  # np.stack([xray_high_img, xray_low_img], axis=-1)
    
    h, w, _ = grey_img.shape
    colour_img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            grey_rgb = quantize_color(grey_img[y, x])
            colour_img[y, x] = mapping_dict.get(grey_rgb, (0, 0, 0))  # Fallback to black
    return colour_img


def combine_highlow_images(imgA: np.ndarray, imgB: np.ndarray, substract=False, maskB=None, epsilon: float = 1e-9) -> np.ndarray:
    """
    Isolates the X-ray attenuation of an object in imgA by cancelling the background attenuation imgB (when substract=False).

    This function operates based on the Beer-Lambert law (I = I_0 * exp(-mu*x)).
    The raw pixel values are proportional to the transmitted intensity (I). To work with
    attenuation (mu*x), we need to operate in log space.

    The process is as follows:
    1. Convert raw intensity values to a floating-point representation.
    2. Normalize intensities to get transmittance (T = I / I_0). We approximate I_0
       with the maximum possible pixel value (255).
    3. Calculate absorbance (A = -log(T)), which is proportional to attenuation.
    4. Subtract the absorbance of imgB from the absorbance of imgA.
       This leaves only the absorbance of the isolated object itself.
    5. Convert the resulting object absorbance back to an intensity image for visualization.

    Args:
        imgB (np.ndarray): A NumPy array representing the dual-energy X-ray scan
            of e.g., an empty bag. Expected shape is (height, width, 2) for low and high
            energy channels, with pixel values in 8-bit greyscale (0-255).
        imgA (np.ndarray): A NumPy array representing the dual-energy
            X-ray scan of e.g. the bag containing the object. Must have
            the same shape and data type as bag_tensor.
        epsilon (float): A small constant to prevent division by zero or log(0) errors
            for pixels with an intensity of 0.

    Returns:
        np.ndarray: A NumPy array of the same shape and original data type, representing
            the dual-energy X-ray data for the isolated object.
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

    # --- Isolate the Object's Attenuation ---

    # Subtract imgA's absorbance from the combined absorbance.
    if substract:
        absorbance_imgC = absorbance_imgA - absorbance_imgB
    else:
        absorbance_imgC = absorbance_imgA + absorbance_imgB
    
    # Ensure the resulting absorbance is non-negative.
    absorbance_imgC[absorbance_imgC < 0] = 0


    # --- Conversion back to Intensity Image ---

    # Convert the object's absorbance back to transmittance.
    # T_object = exp(-A_object)
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


def highlow_to_grey(high_low_img, mapping_dict=XRAY_GREY_DICT):
    if mapping_dict is None:
        __compute_xray_grey_dict()
        mapping_dict = XRAY_GREY_DICT

    grey_img = np.array(high_low_img)  # np.stack([xray_high_img, xray_low_img], axis=-1)
    
    h, w, _ = grey_img.shape
    colour_img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            grey_rgb = quantize_color(grey_img[y, x])
            colour_img[y, x] = mapping_dict.get(grey_rgb, (0, 0, 0))  # Fallback to black
    return colour_img

def highlow_to_density(high_low_img, mapping_dict=XRAY_DENSITY_DICT):
    if mapping_dict is None:
        __compute_xray_density_dict()
        mapping_dict = XRAY_DENSITY_DICT

    grey_img = np.array(high_low_img)  # np.stack([xray_high_img, xray_low_img], axis=-1)
    
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
        target_height (int): Desired output height
        target_width (int): Desired output width
    
    Returns:
        np.ndarray: Tensor of shape (target_height, target_width, C)
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
