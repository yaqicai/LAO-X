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
            datasets==3.6.0 pyarrow==20.0.0 pycocotools==2.0.10 tqdm
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

The generator needs the three high/low-energy lookup tables
(`xray_highlow_to_{colour,grey,density}_dict_RAW.pkl`) in `$LAOX_LUT_DIR`
(default `synthesis/luts/`). Build them once from DET-COMPASS:

```bash
python synthesis/build_luts.py all
```

## Checkpoints

Place the weights in `$LAOX_CKPT_ROOT` (default `checkpoints/`):

| File | Used by | Source |
| --- | --- | --- |
| `sam2.1_hiera_small.pt` | training initialization | `bash sam2/checkpoints/download_ckpts.sh` |
| `cutler_cascade_final.pth` | object bank, oversegmentation | [CutLER model zoo](https://github.com/facebookresearch/CutLER#model-zoo) |
| `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | oversegmentation | [DINOv3](https://github.com/facebookresearch/dinov3) (ViT-B/16, LVD-1689M) |

OpenCLIP ViT-L-14-336 (MuSc) and CascadePSP (oversegmentation refinement) are
downloaded automatically on first use.

## Interactive-evaluation benchmarks

`evaluation/interactive` expects the GraCo benchmarks (GrabCut, Berkeley,
DAVIS, SBD, PascalPart, PartImageNet) under `$LAOX_DATA_ROOT/interactive/`; see
`evaluation/interactive/config.yml`.
