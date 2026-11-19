from datasets import load_dataset
from compass_editor.dataset import split_dataset, create_synthetic_sample, set_seed
from compass_editor.viewer import visualize_image

# Load dataset
print('Loading "lorenzovaquero/DET-COMPASS-CROPS" dataset...')
dataset = load_dataset("lorenzovaquero/DET-COMPASS-CROPS", split="train")

print('Splitting dataset...')
foreground_samples, background_samples = split_dataset(dataset, only_xray_visible=True, only_rgb_visible=True)

seeds = [40, 35, 7, 15, 36]
for i in seeds:
    set_seed(i)
    print('Creating sample...')
    sample = create_synthetic_sample(foreground_samples=foreground_samples, background_samples=background_samples, generate_rgb=True)

    visualize_image(sample)