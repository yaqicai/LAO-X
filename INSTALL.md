# Installation

Tested with Python 3.10, PyTorch 2.5.1 + CUDA 12.1 (training/oversegmentation
also with PyTorch 2.7.1 + CUDA 12.8 on RTX 5090).

## Option A: exact conda environment

```bash
bash environment/rebuild_environment.sh standard   # or: 5090
```

This creates the environment from `environment/*.rebuild.yml`, installs the
pinned Detectron2 fork, installs the bundled `sam2/` package in editable mode,
clones DINOv3 at the pinned commit, and builds the MultiScaleDeformableAttention
CUDA op.

## Option B: manual

```bash
conda create -n laox python=3.10 -y
conda activate laox
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Detectron2 (only needed for oversegmentation and the CutLER object bank)
FORCE_CUDA=1 python -m pip install --no-build-isolation \
  'git+https://github.com/MaureenZOU/detectron2-xyz.git@42121d75e10d9f858f3a91b6a39f5722c02868f0'
(cd granularity_divide_and_conquer/mask2former/modeling/pixel_decoder/ops && \
  FORCE_CUDA=1 python -m pip install . --no-build-isolation)

# DINOv3 (only needed for oversegmentation)
git clone https://github.com/facebookresearch/dinov3.git granularity_divide_and_conquer/dinov3
git -C granularity_divide_and_conquer/dinov3 checkout 54694f7627fd815f62a5dcc82944ffa6153bbb76

# SAM 2 / UnSAMv2
pip install -e sam2
```

Ignore the pip warning `detectron2 0.6 requires iopath<0.1.10`.

## Synthesis only (CPU)

The synthesis generator needs neither a GPU nor Detectron2:

```bash
pip install numpy==1.26.4 opencv-python==4.10.0.84 Pillow==10.4.0 scikit-learn==1.6.1 \
            datasets==3.6.0 pyarrow==20.0.0 pycocotools==2.0.10 matplotlib tqdm
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

The generator also needs the three high/low-energy lookup dictionaries
(`xray_highlow_to_{colour,grey,density}_dict_RAW.pkl`) in
`dependencies/det_compass_crops/compass_editor/`. Build them from DET-COMPASS
with `create_colour_dict_fast.py`, `create_grey_dict.py` and
`create_density_dict.py` in `dependencies/det_compass_crops/`.

## Checkpoints

See the weights table in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
