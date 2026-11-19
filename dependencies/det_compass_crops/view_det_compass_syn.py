from datasets import load_dataset
from compass_editor.viewer import launch_dataset_viewer_synthetic
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import os

LOCAL = True

SPLITS = ['easy', 'medium', 'hard', 'extreme']
SPLIT = SPLITS[0]
LOCAL_PATH = os.path.join("DET-COMPASS-SYN-HF-FULL", SPLIT)

# Load dataset
if LOCAL:
    datasets = []
    batch_count = len(os.listdir(LOCAL_PATH))
    for i in tqdm(range(batch_count), desc="Loading batches"):
        batch_path = os.path.join(LOCAL_PATH, f"batch_{i:04d}")
        dataset = Dataset.load_from_disk(str(batch_path))
        datasets.append(dataset)
    dataset = concatenate_datasets(datasets)

else:
    dataset = load_dataset("lorenzovaquero/DET-COMPASS-SYN", name="default", split=SPLIT)

launch_dataset_viewer_synthetic(dataset)