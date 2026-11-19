from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
from collections import Counter
import pickle

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compass_editor", "xray_highlow_to_colour_dict_RAW.pkl")


def accumulate_colour(conversion_accumulator, highlow, colour):
    # highlow: HxWx2 uint8, colour: HxWx3 uint8
    grey_code = (highlow[...,0].astype(np.uint32) << 8) | highlow[...,1].astype(np.uint32)  # 0..65535
    colour_code = (
        (colour[...,0].astype(np.uint32) << 16) |
        (colour[...,1].astype(np.uint32) << 8) |
        colour[...,2].astype(np.uint32)
    )
    combo = (grey_code.astype(np.uint64) << 24) | colour_code  # 唯一键
    flat, counts = np.unique(combo.reshape(-1), return_counts=True)
    grey_flat = (flat >> 24).astype(np.uint32)
    colour_flat = (flat & 0xFFFFFF).astype(np.uint32)

    for g, c, cnt in zip(grey_flat, colour_flat, counts):
        c_tuple = ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
        conversion_accumulator[(int(g >> 8), int(g & 0xFF))][c_tuple] += int(cnt)


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
        xray_colour_img = np.array(sample["xray_images"][0])  # shape: (H, W, 3)
        xray_high_img   = np.array(sample["xray_images"][2])  # shape: (H, W)
        xray_low_img    = np.array(sample["xray_images"][3])  # shape: (H, W)
        xray_grey_img   = np.stack([xray_high_img, xray_low_img], axis=-1)

        accumulate_colour(conversion_accumulator, xray_grey_img, xray_colour_img)

    print(f'Saving RAW Colour dict to {OUTPUT_FILE}')
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(conversion_accumulator, f)


if __name__ == "__main__":
    main()
