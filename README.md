# LAO-X

Official implementation of "Localize Any Object in X-Ray Security Scans without
Human Annotation" (NeurIPS 2026).

[Project website](https://yaqicai.github.io/LAO-X/) · [Paper](https://arxiv.org/abs/2610.07326)

LAO-X learns class-agnostic object segmentation for X-ray security images
without manual masks. It

1. builds an **object bank** from unlabeled X-ray scans by fusing MuSc anomaly
   heatmaps with CutLER proposals,
2. **synthesizes** multi-object X-ray images by Beer–Lambert composition of the
   banked objects' high/low-energy channels onto one empty container per
   image, rendered to colour with lookup tables built from DET-COMPASS
   (easy / hard / extreme stages),
3. produces **oversegmentation** pseudo-labels with granularity
   divide-and-conquer and **fuses** them with the synthetic object masks, and
4. trains a granularity-controllable SAM 2 model (UnSAMv2) with an
   **easy-to-hard curriculum**.

This repository is built on [UnSAMv2](https://github.com/yujunwei04/UnSAMv2)
and includes modified copies of [MuSc](https://github.com/xrli-U/MuSc),
[CutLER](https://github.com/facebookresearch/CutLER) and the DET-COMPASS-CROPS
synthetic-data generator. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Repository layout

```text
.
├── pipeline/                         # one script per pipeline step (start here)
├── third_party/MuSc/                 # step 1: anomaly heatmaps (modified MuSc)
├── third_party/CutLER/               # step 1: heatmap-guided object masks (modified CutLER)
├── synthesis/                        # step 2: single-container X-ray synthesis
├── granularity_divide_and_conquer/   # step 3: oversegmentation pseudo-labels
├── pseudo_label_generate.py          # step 4: class-agnostic labels + fusion
├── util_xray.py, gra_fusion.py       #         COCO / fusion utilities
├── sam2/                             # step 5: UnSAMv2 model, configs, curriculum training
├── test_det_v2.py                    # step 6: whole-image evaluation
├── scripts/                          #         evaluation, cross-dataset and analysis scripts
├── evaluation/                       #         interactive (click) evaluation
├── ablation/                         # ablation experiments
├── config.py                         # dataset locations
└── environment/                      # conda environments
```

## Installation

See [INSTALL.md](INSTALL.md).

## Data and paths

The source dataset is [DET-COMPASS](https://huggingface.co/datasets/PAGF/DET-COMPASS)
(`default` configuration, `test` and `invisible` splits, all five energy
channels). Cross-dataset evaluation additionally uses CLCXray, DvXray, HiXray,
PIDray and PIXray. Datasets and checkpoints are not distributed with this
repository.

All scripts read their locations from environment variables:

| Variable | Default | Contents |
| --- | --- | --- |
| `LAOX_DATA_ROOT` | `<repo>/datasets` | datasets (layout in [config.py](config.py)) and generated synthetic data |
| `LAOX_CKPT_ROOT` | `<repo>/checkpoints` | `sam2.1_hiera_small.pt`, `cutler_cascade_final.pth`, `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| `LAOX_RESULTS_ROOT` | `<repo>/results` | object-bank intermediates, training runs, analyses |
| `LAOX_LUT_DIR` | `<repo>/synthesis/luts` | the three high/low-energy lookup tables |
| `PYTHON`, `GPUS` | `python`, `0,...,7` | interpreter, visible GPUs |

The DET-COMPASS source images are expected in
`$LAOX_DATA_ROOT/det_compass_xp/xray/`, and the files used by synthesis in
`$LAOX_DATA_ROOT/det_compass_xp/annotations/`:

| File | Description |
| --- | --- |
| `coco_cutler_musc.json` | object bank (output of step 1) |
| `coco_agnostic.json` | held-out evaluation images (class-agnostic COCO); excluded from the object bank |
| `huggingface_matches.json` | local file name → DET-COMPASS split/index (`synthesis/generate.py match`) |

These annotation files are not distributed with this repository at the moment.

## Pipeline

```bash
bash pipeline/01_object_bank.sh     # MuSc heatmaps + CutLER -> coco_cutler_musc.json   (1 GPU)
bash pipeline/02_synthesize.sh      # easy / hard / extreme synthetic images            (CPU)
bash pipeline/03_oversegment.sh     # oversegmentation pseudo-labels                    (GPUs)
bash pipeline/04_pseudo_labels.sh   # class-agnostic labels + fusion                    (CPU)
bash pipeline/05_train.sh           # curriculum training                               (8 GPUs)
bash pipeline/06_evaluate.sh        # held-out DET-COMPASS; DATASET=CLCXray ... for cross-dataset
```

Synthesis stages (16 generation workers; output is identical for any worker count):

| Stage | Images | Objects / image | Overlap `gamma` |
| --- | ---: | --- | --- |
| easy | 8,000 | 2–4 | 0.0–0.05 |
| hard | 8,000 | 4–7 | 0.0–0.3 |
| extreme | 8,000 | 6–9 | 0.0–0.5 |

Every synthetic image uses exactly one container from the DET-COMPASS
`invisible` split. Objects are composed in the high/low-energy domain and the
result is rendered with three fixed lookup tables
(`xray_highlow_to_{colour,grey,density}_dict_RAW.pkl`, built by
`python synthesis/build_luts.py all`). Each image draws from its own random
stream derived from the base seed (default 40), the stage and the image id, so
the output is byte-identical across runs and worker counts. See
[synthesis/README.md](synthesis/README.md).

Curriculum training (8 GPUs × batch size 2, initialized from SAM 2.1 Hiera-S;
each stage starts from the previous stage's checkpoint):

| Stage | Samples | Base LR | Vision LR argument | Epochs |
| --- | --- | --- | --- | ---: |
| easy | 0–4000 | 1.6e-4 | 4.8e-6 | 80 |
| hard | 0–6000 | 1.2e-4 | 3.2e-6 | 64 |
| extreme | 0–8000 | 1e-4 | 3e-6 | 48 |

Evaluation uses the extreme-stage `checkpoint_48.pt`, granularities
0.1–0.8 and mask threshold 0.5.

## Ablations

| Script | Ablation |
| --- | --- |
| `ablation/object_bank_cutler_only.sh` | CutLER-only object bank without MuSc; `VARIANT=n1` / `n3` keeps CutLER's top 1 / 3 predictions per image |
| `ablation/composition_rgb.sh` | RGB copy-paste instead of high/low-energy composition |
| `ablation/extreme_difficulty.sh {00_07,00_09,9_12,12_15}` | harder final curriculum stage |
| `ablation/label_source.sh {holistic,part}` | train on object masks only / oversegmentation only |

## License

LAO-X is released under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
license ([LICENSE](LICENSE)); it may be used for non-commercial purposes only.
Third-party code included in this repository keeps its original license and
copyright notices: the UnSAMv2 and SAM 2 code is under the Apache License 2.0
([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0), [sam2/LICENSE](sam2/LICENSE)) and
MuSc and TokenCut are under the MIT license. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

If you use LAO-X in your research, please cite:

```bibtex
@misc{cai2026localizeobjectxraysecurity,
      title={Localize Any Object in X-Ray Security Scans without Human Annotation},
      author={Yaqi Cai and Mingxuan Liu and Lorenzo Vaquero and Ning Wang and Nan Pu and Feng Xue and Elisa Ricci and Nicu Sebe},
      year={2026},
      eprint={2610.07326},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.07326},
}
```
