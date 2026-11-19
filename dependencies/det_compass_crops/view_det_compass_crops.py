from datasets import load_dataset
from compass_editor.viewer import launch_dataset_viewer

# Load dataset
dataset = load_dataset("lorenzovaquero/DET-COMPASS-CROPS", split="train")
launch_dataset_viewer(dataset)