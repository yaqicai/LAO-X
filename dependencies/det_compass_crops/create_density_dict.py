from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
from collections import Counter
import pickle

from compass_editor.colour import quantize_color

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compass_editor", "xray_highlow_to_density_dict_RAW.pkl")


def main():
    # Ask before overwriting the file
    if os.path.exists(OUTPUT_FILE):
        response = input(f"File {OUTPUT_FILE} already exists. Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborting to avoid overwriting the file.")
            return

    # Load the dataset (test split as example; consider train+val too)
    ds = load_dataset("PAGF/DET-COMPASS", name="default", split="all")

    # Dictionary: grey_key -> count of each RGB color
    conversion_accumulator = defaultdict(Counter)

    # Iterate over samples
    for sample in tqdm(ds, desc="Building conversion dictionary"):
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

    print(f'Saving RAW Density dict to {OUTPUT_FILE}')
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(conversion_accumulator, f)


if __name__ == "__main__":
    main()
