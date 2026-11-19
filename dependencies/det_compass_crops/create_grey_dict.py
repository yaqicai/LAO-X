from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
from collections import Counter
import pickle

from compass_editor.colour import quantize_color

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compass_editor", "xray_highlow_to_grey_dict_RAW.pkl")


def main():
    # Ask before overwriting the file
    if os.path.exists(OUTPUT_FILE):
        response = input(f"File {OUTPUT_FILE} already exists. Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborting to avoid overwriting the file.")
            return

    # Load the dataset (test split as example; consider train+val too)
    ds = load_dataset("PAGF/DET-COMPASS", name="default", split="test")

    # Dictionary: grey_key -> count of each RGB color
    conversion_accumulator = defaultdict(Counter)

    # Iterate over samples
    for sample in tqdm(ds, desc="Building conversion dictionary"):
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

    print(f'Saving RAW Grey dict to {OUTPUT_FILE}')
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(conversion_accumulator, f)


if __name__ == "__main__":
    main()
