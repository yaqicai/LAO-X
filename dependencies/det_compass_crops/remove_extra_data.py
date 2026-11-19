import os
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, concatenate_datasets
import PIL


SPLIT = "easy"

def remove_extra_data(input_dir, output_dir, batch_size=300, preserve_masks=False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Load classes.csv to create class name to ID mapping
    classes_df = pd.read_csv('classes.csv')
    class_name_to_id = dict(zip(classes_df['class'], classes_df['ID']))
    print(f"Loaded {len(class_name_to_id)} class mappings from classes.csv")

    # Get all pickle files
    pickle_files = list(input_path.glob('*_sample.pkl'))
    print(f"Found {len(pickle_files)} sample files")
    
    sample_list = []
    batch_count = 0
    
    # Process each file
    for pickle_file in tqdm(pickle_files, desc="Processing samples"):
        try:
            # Load sample
            with open(pickle_file, 'rb') as f:
                sample = pickle.load(f)
            
            # We remove the extra data
            sample["xray_images"] = sample["xray_images"][:1]
            
            # Delete masks if not preserving them
            if not preserve_masks:
                del sample['objects']['rgb_mask']
                del sample['objects']['xray_mask']

            # Create class_IDs list by mapping category names to IDs
            sample["objects"]["class_IDs"] = []
            for category in sample["objects"]["categories"]:
                if category in class_name_to_id:
                    sample["objects"]["class_IDs"].append(class_name_to_id[category])
                else:
                    print(f"Warning: Category '{category}' not found in classes.csv")
                    sample["objects"]["class_IDs"].append(-1)  # Use -1 for unknown categories
            
            del sample['objects']['categories']
            del sample['objects']['dangerous']
            del sample['objects']['IN_ids']
            del sample['objects']['WN_ids']

            # Remove the alpha channel from the PIL 'rgb_image' (if it has 4 channels) (but first check if it is a PIL image)
            if isinstance(sample['rgb_image'], PIL.Image.Image):
                if sample['rgb_image'].mode == 'RGBA':
                    sample['rgb_image'] = sample['rgb_image'].convert('RGB')
            else:  # It is a numpy array
                if sample['rgb_image'].shape[2] == 4:
                    sample['rgb_image'] = sample['rgb_image'][:, :, :3]
                
                sample['rgb_image'] = PIL.Image.fromarray(sample['rgb_image'])

            if isinstance(sample['xray_images'][0], PIL.Image.Image) and sample['xray_images'][0].mode == 'RGBA':
                sample['xray_images'][0] = sample['xray_images'][0].convert('RGB')
            else:  # It is a numpy array
                if sample['xray_images'][0].shape[2] == 4:
                    sample['xray_images'][0] = sample['xray_images'][0][:, :, :3]
                
                sample['xray_images'][0] = PIL.Image.fromarray(sample['xray_images'][0])

            # Handle masks based on preserve_masks flag (after xray_images[0] is converted to PIL)
            if preserve_masks:
                # Preserve masks - convert to binary PIL images and resize to match xray_images[0]
                target_w, target_h = sample['xray_images'][0].size
                
                # Helper function to convert a mask to binary PIL Image and resize
                def process_mask(mask):
                    """Convert mask to binary PIL Image and resize to target size."""
                    # Ensure it's a PIL Image in binary mode
                    if not isinstance(mask, PIL.Image.Image):
                        # Convert numpy array to PIL Image
                        if isinstance(mask, np.ndarray):
                            # Ensure binary (0/255 or 0/1)
                            if mask.max() <= 1:
                                mask = (mask * 255).astype(np.uint8)
                            mask = PIL.Image.fromarray(mask, mode='L').convert('1')
                        else:
                            mask = PIL.Image.fromarray(np.array(mask)).convert('1')
                    elif mask.mode != '1':
                        mask = mask.convert('1')
                    # Resize to match xray_images[0] size
                    return mask.resize((target_w, target_h), resample=PIL.Image.NEAREST)
                
                # Process all rgb_mask entries
                if 'rgb_mask' in sample['objects'] and len(sample['objects']['rgb_mask']) > 0:
                    processed_rgb_masks = []
                    for rgb_mask in sample['objects']['rgb_mask']:
                        processed_rgb_masks.append(process_mask(rgb_mask))
                    sample['objects']['rgb_mask'] = processed_rgb_masks
                else:
                    # Create empty mask list if not present
                    sample['objects']['rgb_mask'] = [PIL.Image.new('1', (target_w, target_h), 0)]
                
                # Process all xray_mask entries
                if 'xray_mask' in sample['objects'] and len(sample['objects']['xray_mask']) > 0:
                    processed_xray_masks = []
                    for xray_mask in sample['objects']['xray_mask']:
                        processed_xray_masks.append(process_mask(xray_mask))
                    sample['objects']['xray_mask'] = processed_xray_masks
                else:
                    # Create empty mask list if not present
                    sample['objects']['xray_mask'] = [PIL.Image.new('1', (target_w, target_h), 0)]

            # Resize rgb_image to match xray_images[0] and rescale rgb_bbox
            if isinstance(sample['rgb_image'], PIL.Image.Image):
                rgb_img = sample['rgb_image']
            else:
                rgb_img = PIL.Image.fromarray(sample['rgb_image'])
            xray_img = sample['xray_images'][0]
            if not isinstance(xray_img, PIL.Image.Image):
                xray_img = PIL.Image.fromarray(xray_img)
            orig_w, orig_h = rgb_img.size
            target_w, target_h = xray_img.size
            scale_w = target_w / orig_w
            scale_h = target_h / orig_h
            # Resize with high quality
            rgb_img_resized = rgb_img.resize((target_w, target_h), resample=PIL.Image.LANCZOS)
            sample['rgb_image'] = rgb_img_resized
            # Rescale all rgb_bbox
            if 'rgb_bbox' in sample['objects']:
                new_bboxes = []
                for bbox in sample['objects']['rgb_bbox']:
                    x, y, w, h = bbox
                    x_new = x * scale_w
                    y_new = y * scale_h
                    w_new = w * scale_w
                    h_new = h * scale_h
                    new_bboxes.append([x_new, y_new, w_new, h_new])
                sample['objects']['rgb_bbox'] = new_bboxes

            sample_list.append(sample)
            
            # Save batch when we reach batch_size
            if len(sample_list) >= batch_size:
                batch_dataset = Dataset.from_list(sample_list)
                batch_path = output_path / f"batch_{batch_count:04d}"
                batch_dataset.save_to_disk(str(batch_path))
                print(f"Saved batch {batch_count} with {len(sample_list)} samples to {batch_path}")
                
                sample_list = []
                batch_count += 1

        except Exception as e:
            print(f"Error processing {pickle_file.name}: {e}")
    
    # Save remaining samples as the last batch
    if sample_list:
        batch_dataset = Dataset.from_list(sample_list)
        batch_path = output_path / f"batch_{batch_count:04d}"
        batch_dataset.save_to_disk(str(batch_path))
        print(f"Saved final batch {batch_count} with {len(sample_list)} samples to {batch_path}")
        batch_count += 1
    
    print(f"Saved {batch_count} batches to disk")
    
    # Load and concatenate all batches
    print("Loading and concatenating batches...")
    datasets = []
    for i in tqdm(range(batch_count), desc="Loading batches"):
        batch_path = output_path / f"batch_{i:04d}"
        dataset = Dataset.load_from_disk(str(batch_path))
        datasets.append(dataset)
    
    # Concatenate all datasets
    print("Concatenating datasets...")
    final_dataset = concatenate_datasets(datasets)
    print(f"Final dataset has {len(final_dataset)} samples")
    
    # Save the dataset to hub
    print("Pushing to hub...")
    final_dataset.push_to_hub("lorenzovaquero/DET-COMPASS-SYN", private=True, split=SPLIT)
    print("Successfully pushed to hub!")


if __name__ == "__main__":
    # Configuration
    INPUT_DIR =  "./DET-COMPASS-SYN/" + SPLIT
    OUTPUT_DIR = "./DET-COMPASS-SYN-HF-FULL/" + SPLIT
    BATCH_SIZE = 150
    PRESERVE_MASKS = False
    
    print("Removing extra data...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Preserve masks: {PRESERVE_MASKS}")
    
    # Run the processing
    remove_extra_data(INPUT_DIR, OUTPUT_DIR, BATCH_SIZE, preserve_masks=PRESERVE_MASKS)
    
    print("\nCompleted!") 