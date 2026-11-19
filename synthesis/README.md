# DET-COMPASS-CROPS


## Crops Dataset:

<div align="center">
  <img src="./figs/qualitative_crops.png" alt="Crop Sample">
</div>

## Synthetic Dataset Creator:

<div align="center">
  <img src="./figs/qualitative_synthetic.png" alt="Synthetic Sample">
</div>


## Huggingface Datasets (obtained using this repo):

 - [Original DET-COMPASS (RAXO) Dataset](https://huggingface.co/datasets/PAGF/DET-COMPASS)
 - [X-Ray Crops Dataset](https://huggingface.co/datasets/lorenzovaquero/DET-COMPASS-CROPS)
 - [X-Ray Synthetic Dataset](https://huggingface.co/datasets/lorenzovaquero/DET-COMPASS-SYN)


## Usage:

### 1. Compute correspondende dicts from high-low attenuation values to colour/grayscale/density colors (using "PAGF/DET-COMPASS")
```console
python create_colour_dict.py
python create_grey_dict.py
python create_density_dict.py
```


### 2. Create a Crops Dataset (pickle format) with the masked crops of the objects in "PAGF/DET-COMPASS"
```console
python create_crops_pickle.py
```


### 3. Convert the pickle Crops Dataset to huggingface datasets format (and optionally upload it to "lorenzovaquero/DET-COMPASS-CROPS")
```console
python pickle_to_dataset.py

python view_det_compass_crops.py  # Optional, if uploaded to HF
```


### 4. Create your Synthetic Dataset (using "lorenzovaquero/DET-COMPASS-CROPS")
```console
python create_synthetic_dataset.py
```

### 5. Convert the pickle Crops Dataset to huggingface datasets format (and optionally upload it to "lorenzovaquero/DET-COMPASS-SYN")
```console
python remove_extra_data.py

python view_det_compass_syn.py  # Optional, if uploaded to HF
```