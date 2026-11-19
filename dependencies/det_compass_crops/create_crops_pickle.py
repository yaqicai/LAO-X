from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from compass_editor import background
from compass_editor import viewer
from PIL import Image
from typing import TypedDict, List
from compass_editor import colour
import gc


VISUALIZE = False

# Load the dataset (test split as example; consider train+val too)
ds = load_dataset("PAGF/DET-COMPASS", name="default", split="all")


predictor = background.get_sam_predictor()  # 'facebook/sam2-hiera-large'

background_samples = []
for sample in ds:
    if len(sample["objects"]["rgb_bbox"]) == 0 and len(sample["objects"]["xray_bbox"]) == 0:
        background_samples.append(sample)


# I create a parallel structure
class CropData(TypedDict):
    category: str
    is_background: bool
    rgb_visible: bool
    xray_visible: bool
    dangerous: bool
    IN_id: str
    WN_id: str
    rgb_width: int
    rgb_height: int
    xray_width: int
    xray_height: int

    rgb_image: Image.Image
    rgb_image_mask: Image.Image

    xray_image_colour: Image.Image
    xray_image_highlow: Image.Image
    xray_image_mask: Image.Image
    
    raw_xray_image_colour: Image.Image
    raw_xray_image_grey: Image.Image
    raw_xray_image_high: Image.Image
    raw_xray_image_low: Image.Image
    raw_xray_image_density: Image.Image



new_ds: List[CropData] = []

# Iterate over samples
for sample_id, sample in tqdm(enumerate(ds), desc="Cropping objects"):

    for bbox_id in range(len(sample["objects"]["xray_bbox"])):
        # Bounding-boxes have format ([x_min, y_min, width, height]), and images have size (sample['rgb_width'], sample['rgb_height']) and  (sample['xray_width'], sample['xray_height'])
        rgb_img = np.array(sample["rgb_image"].convert("RGB"))  # shape: (H, W, 3)
        xray_img = np.array(sample["xray_images"][0].convert("RGB"))  # shape: (H, W, 3)

        rgb_bbox = sample["objects"]["rgb_bbox"][bbox_id]  # [x_min, y_min, width, height]
        xray_bbox = sample["objects"]["xray_bbox"][bbox_id]

        sample_density = np.stack([sample["xray_images"][2], sample["xray_images"][3]], axis=-1)

        foreground_sample_img, foreground_sample_highlow, result = background.suppress_background(sam_predictor=predictor, sample=sample, background_samples=background_samples)
        xray_img_original = xray_img
        xray_img = foreground_sample_img


        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 1. We run SAMv2 on image sample["rgb_image"] using as initialization bbox sample["objects"]["rgb_bbox"][bbox_id], generating an rgb_mask
            predictor.set_image(rgb_img)
            rgb_masks, _, _ = predictor.predict(point_coords=None,
                                                point_labels=None,
                                                box=background.xywh_to_xyxy(np.array(rgb_bbox)),
                                                multimask_output=False)

            # Take the highest scoring mask
            rgb_mask = rgb_masks[0].astype(np.uint8)  # shape: (H, W)

            # 2. The bboxes sample["objects"]["rgb_bbox"][bbox_id] and sample["objects"]["xray_bbox"][bbox_id] are correlated, but have different size and position.
            # We warp the generated rgb_mask from the rgb_bbox to the xray_bbox, generating a pseudo_xray_mask
            x1_rgb, y1_rgb, w_rgb, h_rgb = map(int, rgb_bbox)
            x2_rgb, y2_rgb = x1_rgb + w_rgb, y1_rgb + h_rgb
            rgb_crop = rgb_mask[y1_rgb:y2_rgb, x1_rgb:x2_rgb]

            x1_xray, y1_xray, w_xray, h_xray = map(int, xray_bbox)
            x2_xray, y2_xray = x1_xray + w_xray, y1_xray + h_xray

            pseudo_xray_mask = np.zeros_like(xray_img[:,:,0], dtype=np.uint8)
            resized_mask = cv2.resize(rgb_crop, (w_xray, h_xray), interpolation=cv2.INTER_NEAREST)
            pseudo_xray_mask[y1_xray:y2_xray, x1_xray:x2_xray] = resized_mask
            pseudo_xray_mask = cv2.resize(pseudo_xray_mask, (xray_img.shape[1], xray_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 3. We run SAMv2 on image sample["xray_image"] using as initialization mask pseudo_xray_mask, generating an xray_mask
            predictor.set_image(xray_img)
            # Resize the mask to match SAM2's internal image size (e.g., 256x256)
            resized_mask = cv2.resize(pseudo_xray_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            pseudo_xray_tensor = torch.tensor(resized_mask, dtype=torch.float32, device="cuda").unsqueeze(0)
            xray_masks, _, _ = predictor.predict(point_coords=None,
                                                 point_labels=None,
                                                 box=background.xywh_to_xyxy(np.array(xray_bbox)),
                                                 mask_input=pseudo_xray_tensor,
                                                 multimask_output=False)

            xray_mask = xray_masks[0].astype(np.uint8)

            padded_highlow = np.pad(foreground_sample_highlow, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=255)
            padded_highlow_raw = np.pad(sample_density, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=255)

            padded_high_raw = np.stack([sample["xray_images"][2]] * 3, axis=-1)
            padded_low_raw = np.stack([sample["xray_images"][3]] * 3, axis=-1)
            padded_density_raw = np.stack([sample["xray_images"][4]] * 3, axis=-1)
            
            padded_rgb_mask = np.stack([rgb_mask*255] * 3, axis=-1)
            padded_xray_mask = np.stack([xray_mask*255] * 3, axis=-1)

            # Now I have all the elements to append them to the dict
            sample_rgb_image = Image.fromarray(background.apply_mask_and_crop(rgb_img, rgb_mask), mode="RGBA")
            sample_rgb_image_mask = Image.fromarray(background.apply_mask_and_crop(padded_rgb_mask, rgb_mask, fill_val=0)[:, :, -1] > 0).convert('1')

            sample_xray_image_colour = Image.fromarray(background.apply_mask_and_crop(xray_img, xray_mask))
            sample_xray_image_highlow = Image.fromarray(background.apply_mask_and_crop(padded_highlow, xray_mask, fill_val=255), mode="RGBA")
            sample_xray_image_mask = Image.fromarray(background.apply_mask_and_crop(padded_xray_mask, xray_mask, fill_val=0)[:, :, 0] > 0).convert('1')

            sample_raw_xray_image_colour = Image.fromarray(background.apply_mask_and_crop(xray_img_original, xray_mask), mode="RGBA")
            sample_raw_xray_image_grey = Image.fromarray(background.apply_mask_and_crop(np.array(sample["xray_images"][1].convert("RGB")), xray_mask), mode="RGBA")
            sample_raw_xray_image_highlow = Image.fromarray(background.apply_mask_and_crop(padded_highlow_raw, xray_mask, fill_val=255), mode="RGBA")

            sample_raw_xray_image_high = Image.fromarray(background.apply_mask_and_crop(padded_high_raw, xray_mask, fill_val=255)[:, :, 0], mode="L")
            sample_raw_xray_image_low = Image.fromarray(background.apply_mask_and_crop(padded_low_raw, xray_mask, fill_val=255)[:, :, 0], mode="L")
            sample_raw_xray_image_density = Image.fromarray(background.apply_mask_and_crop(padded_density_raw, xray_mask, fill_val=255)[:, :, 0], mode="L")

            
            crop_sample = CropData(
                category=sample["objects"]["categories"][bbox_id],
                is_background=False,
                rgb_visible=sample["objects"]["rgb_visible"][bbox_id],
                xray_visible=sample["objects"]["xray_visible"][bbox_id],
                dangerous=sample["objects"]["dangerous"][bbox_id],
                IN_id=sample["objects"]["IN_ids"][bbox_id],
                WN_id=sample["objects"]["WN_ids"][bbox_id],
                rgb_width=sample_rgb_image.size[0],
                rgb_height=sample_rgb_image.size[1],
                xray_width=sample_xray_image_colour.size[0],
                xray_height=sample_xray_image_colour.size[1],

                rgb_image=sample_rgb_image,
                rgb_image_mask=sample_rgb_image_mask,

                xray_image_colour=sample_xray_image_colour,
                xray_image_highlow=sample_xray_image_highlow,
                xray_image_mask=sample_xray_image_mask,
                
                raw_xray_image_colour=sample_raw_xray_image_colour,
                raw_xray_image_grey=sample_raw_xray_image_grey,
                raw_xray_image_high=sample_raw_xray_image_high,
                raw_xray_image_low=sample_raw_xray_image_low,
                raw_xray_image_density=sample_raw_xray_image_density
            )

            if VISUALIZE:
                viewer.visualize_crop(crop_sample)

            new_ds.append(crop_sample)


# Iterate over samples
for sample_id, sample in tqdm(enumerate(ds), desc="Retrieving backgrounds"):
    if len(sample["objects"]["xray_bbox"]) > 0:
        continue

    rgb_img = np.array(sample["rgb_image"].convert("RGB"))  # shape: (H, W, 3)
    xray_img = np.array(sample["xray_images"][0].convert("RGB"))  # shape: (H, W, 3)

    rgb_bbox = [0.0, 0.0, sample['rgb_width'], sample['rgb_height']]  # [x_min, y_min, width, height]
    xray_bbox = [0.0, 0.0, sample['xray_width'], sample['xray_height']]  # [x_min, y_min, width, height]

    sample_density = np.stack([sample["xray_images"][2], sample["xray_images"][3]], axis=-1)


    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # 1. We run SAMv2 on image sample["rgb_image"] using as initialization bbox sample["objects"]["rgb_bbox"][bbox_id], generating an rgb_mask
        predictor.set_image(rgb_img)
        rgb_masks, _, _ = predictor.predict(point_coords=None,
                                            point_labels=None,
                                            box=background.xywh_to_xyxy(np.array(rgb_bbox)),
                                            multimask_output=False)

        # Take the highest scoring mask
        rgb_mask = rgb_masks[0].astype(np.uint8)  # shape: (H, W)

        # 2. The bboxes sample["objects"]["rgb_bbox"][bbox_id] and sample["objects"]["xray_bbox"][bbox_id] are correlated, but have different size and position.
        # We warp the generated rgb_mask from the rgb_bbox to the xray_bbox, generating a pseudo_xray_mask
        x1_rgb, y1_rgb, w_rgb, h_rgb = map(int, rgb_bbox)
        x2_rgb, y2_rgb = x1_rgb + w_rgb, y1_rgb + h_rgb
        rgb_crop = rgb_mask[y1_rgb:y2_rgb, x1_rgb:x2_rgb]

        x1_xray, y1_xray, w_xray, h_xray = map(int, xray_bbox)
        x2_xray, y2_xray = x1_xray + w_xray, y1_xray + h_xray

        pseudo_xray_mask = np.zeros_like(xray_img[:,:,0], dtype=np.uint8)
        resized_mask = cv2.resize(rgb_crop, (w_xray, h_xray), interpolation=cv2.INTER_NEAREST)
        pseudo_xray_mask[y1_xray:y2_xray, x1_xray:x2_xray] = resized_mask
        pseudo_xray_mask = cv2.resize(pseudo_xray_mask, (xray_img.shape[1], xray_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 3. We run SAMv2 on image sample["xray_image"] using as initialization mask pseudo_xray_mask, generating an xray_mask
        predictor.set_image(xray_img)
        # Resize the mask to match SAM2's internal image size (e.g., 256x256)
        resized_mask = cv2.resize(pseudo_xray_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        pseudo_xray_tensor = torch.tensor(resized_mask, dtype=torch.float32, device="cuda").unsqueeze(0)
        xray_masks, _, _ = predictor.predict(point_coords=None,
                                                point_labels=None,
                                                box=background.xywh_to_xyxy(np.array(xray_bbox)),
                                                mask_input=pseudo_xray_tensor,
                                                multimask_output=False)

        xray_mask = xray_masks[0].astype(np.uint8)

        # For the tray, for some reason the mask is inverted:
        xray_mask = 1 - xray_mask

        padded_highlow_raw = np.pad(sample_density, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)

        padded_high_raw = np.stack([sample["xray_images"][2]] * 3, axis=-1)
        padded_low_raw = np.stack([sample["xray_images"][3]] * 3, axis=-1)
        padded_density_raw = np.stack([sample["xray_images"][4]] * 3, axis=-1)

        # rgb_mask presents holes for the background trays so we will fill them
        rgb_mask = background.bbox_to_mask(rgb_mask)
        
        padded_rgb_mask = np.stack([rgb_mask*255] * 3, axis=-1)
        padded_xray_mask = np.stack([xray_mask*255] * 3, axis=-1)

        # Now I have all the elements to append them to the dict
        sample_rgb_image = Image.fromarray(background.apply_mask_and_crop(rgb_img, rgb_mask), mode="RGBA")
        sample_rgb_image_mask = Image.fromarray(background.apply_mask_and_crop(padded_rgb_mask, rgb_mask, fill_val=0)[:, :, -1] > 0).convert('1')

        sample_xray_image_colour = Image.fromarray(background.apply_mask_and_crop(xray_img, xray_mask))
        sample_xray_image_highlow = Image.fromarray(background.apply_mask_and_crop(padded_highlow_raw, xray_mask, fill_val=255), mode="RGBA")
        sample_xray_image_mask = Image.fromarray(background.apply_mask_and_crop(padded_xray_mask, xray_mask, fill_val=0)[:, :, 0] > 0).convert('1')

        sample_raw_xray_image_colour = Image.fromarray(background.apply_mask_and_crop(xray_img, xray_mask), mode="RGBA")
        sample_raw_xray_image_grey = Image.fromarray(background.apply_mask_and_crop(np.array(sample["xray_images"][1].convert("RGB")), xray_mask), mode="RGBA")
        sample_raw_xray_image_highlow = Image.fromarray(background.apply_mask_and_crop(padded_highlow_raw, xray_mask, fill_val=255), mode="RGBA")

        sample_raw_xray_image_high = Image.fromarray(background.apply_mask_and_crop(padded_high_raw, xray_mask, fill_val=255)[:, :, 0], mode="L")
        sample_raw_xray_image_low = Image.fromarray(background.apply_mask_and_crop(padded_low_raw, xray_mask, fill_val=255)[:, :, 0], mode="L")
        sample_raw_xray_image_density = Image.fromarray(background.apply_mask_and_crop(padded_density_raw, xray_mask, fill_val=255)[:, :, 0], mode="L")

        
        crop_sample = CropData(
            category='background',
            is_background=True,
            rgb_visible=False,
            xray_visible=False,
            dangerous=False,
            IN_id='',
            WN_id='',
            rgb_width=sample_rgb_image.size[0],
            rgb_height=sample_rgb_image.size[1],
            xray_width=sample_xray_image_colour.size[0],
            xray_height=sample_xray_image_colour.size[1],

            rgb_image=sample_rgb_image,
            rgb_image_mask=sample_rgb_image_mask,

            xray_image_colour=sample_xray_image_colour,
            xray_image_highlow=sample_xray_image_highlow,
            xray_image_mask=sample_xray_image_mask,
            
            raw_xray_image_colour=sample_raw_xray_image_colour,
            raw_xray_image_grey=sample_raw_xray_image_grey,
            raw_xray_image_high=sample_raw_xray_image_high,
            raw_xray_image_low=sample_raw_xray_image_low,
            raw_xray_image_density=sample_raw_xray_image_density
        )

        if VISUALIZE:
            viewer.visualize_crop(crop_sample)

        new_ds.append(crop_sample)


# We free some space:

print("Cleaning RAM")
del predictor
del background_samples
del ds
del colour.XRAY_COLOUR_DICT_ORIGINAL
del colour.XRAY_COLOUR_DICT
torch.cuda.empty_cache()
gc.collect()

print("Saving dataset list to Pickle")
import pickle
with open("DET-COMPASS-CROPS.pkl", "wb") as f:
    pickle.dump(new_ds, f)

print("Done!")