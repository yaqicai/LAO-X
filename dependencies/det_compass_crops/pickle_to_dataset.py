import pickle
import gc
import os
import shutil
from datasets import Dataset, concatenate_datasets


SAVE_TO_HUB = False

BATCH_SIZE = 300  # Adjust based on your memory constraints
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "DET-COMPASS-CROPS_tmp_batches")
FINAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "DET-COMPASS-CROPS")

# Ensure temp batch directory exists
os.makedirs(TMP_DIR, exist_ok=True)

# Load full data from pickle
with open("DET-COMPASS-CROPS.pkl", "rb") as f:
    data = pickle.load(f)

# Split into batches and save each batch to disk
batch_files = []
print("Creating and saving batches...")
for i in range(0, len(data), BATCH_SIZE):
    batch_data = data[i:i+BATCH_SIZE]
    ds_batch = Dataset.from_list(batch_data)
    batch_path = os.path.join(TMP_DIR, f"batch_{i//BATCH_SIZE}")
    ds_batch.save_to_disk(batch_path)
    batch_files.append(batch_path)

    # Cleanup to save memory
    del ds_batch
    del batch_data
    gc.collect()

# Reload batches and concatenate
print("Reloading and concatenating batches...")
datasets = [Dataset.load_from_disk(path) for path in batch_files]
final_dataset = concatenate_datasets(datasets)

# Save final dataset to disk
print("Saving final dataset...")
final_dataset.save_to_disk(FINAL_DIR)

# Optionally, delete temporary batches
shutil.rmtree(TMP_DIR)

# Optionally, save to hub
if SAVE_TO_HUB:
    final_dataset.push_to_hub("lorenzovaquero/DET-COMPASS-CROPS", private=True)

print("Done!")
