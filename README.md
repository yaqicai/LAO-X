# LAO-X

LAO-X learns class-agnostic object segmentation for X-ray security images
without manual masks. It

1. builds an **object bank** from unlabeled X-ray scans by fusing MuSc anomaly
   heatmaps with CutLER proposals,
2. **synthesizes** multi-object X-ray images by Beer–Lambert composition of the
   banked objects' high/low-energy channels onto one empty container per
   image, rendered to colour, grey and density images with lookup tables built
   from DET-COMPASS (easy / hard / extreme stages),
3. produces **oversegmentation** pseudo-labels with granularity
   divide-and-conquer and **fuses** them with the synthetic ground truth, and
4. trains a granularity-controllable SAM 2 model (UnSAMv2) with an
   **easy-to-hard curriculum**.

This repository is built on [UnSAMv2](https://github.com/yujunwei04/UnSAMv2)
and includes modified copies of [MuSc](https://github.com/xrli-U/MuSc),
[CutLER](https://github.com/facebookresearch/CutLER) and the DET-COMPASS-CROPS
synthetic-data generator. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Repository layout

```text
.
├── dependencies/det_compass_crops/   # X-ray synthesis (single container, LUT rendering)
├── third_party/MuSc/                 # object bank, step 1a: anomaly heatmaps
├── third_party/CutLER/               # object bank, step 1b: heatmap-guided masks
├── granularity_divide_and_conquer/   # oversegmentation pseudo-labels
├── pseudo_label_generate.py          # class-agnostic labels + fusion
├── util_xray.py, gra_fusion.py       # COCO / fusion utilities
├── sam2/                             # UnSAMv2 model, configs and training code
├── test_det_v2.py, test_instances.py # whole-image evaluation
├── evaluation/                       # interactive + whole-image evaluation
├── scripts/                          # task scripts (synthesis, evaluation, analysis)
├── ablation/                         # ablation experiment scripts
└── environment/                      # conda environments
```

## Installation

See [INSTALL.md](INSTALL.md).

## Data

The source dataset is [DET-COMPASS](https://huggingface.co/datasets/PAGF/DET-COMPASS)
(`default` configuration, `test` and `invisible` splits, all five energy
channels). Cross-dataset evaluation additionally uses CLCXray, DvXray, HiXray,
PIDray and PIXray. Datasets and checkpoints are not distributed with this
repository. All dataset paths in the scripts are written as
`/path/to/datasets/...`; edit [config.py](config.py) and the script headers to
point at your copies.

## Pipeline

| Step | Entry point |
| --- | --- |
| 1a. MuSc heatmaps | `third_party/MuSc/examples/musc_main.py` (`--dataset_name mvtec_xp`) |
| 1b. CutLER × MuSc object bank | `third_party/CutLER/cutler/demo/demo.py --musc_dir ... --fusion_mode mul_blend_mul_v2 --coco_output ...` |
| 2. Synthesis | `scripts/task_create_synthetic_dataset_object_bank.sh` |
| 3. Oversegmentation | `granularity_divide_and_conquer/run_gra_dico_aug.sh <dataset> <stage>` |
| 4. Label fusion | `pseudo_label_generate.py --fusion --fusion-overlap-thresh 0.95 ...` |
| 5. Curriculum training | `sam2/training/train_curriculum_augmentation_restart_8.sh` |
| 6. Evaluation | `scripts/task_test_augmentation_restart.sh` |

Synthesis stages:

| Stage | Images | Objects / image | Overlap `gamma` |
| --- | ---: | --- | --- |
| easy | 8,000 | 2–4 | 0.0–0.05 |
| hard | 8,000 | 4–7 | 0.0–0.3 |
| extreme | 8,000 | 6–9 | 0.0–0.5 |

Curriculum training (8 GPUs × batch size 2):

| Stage | Samples | Base LR | Epochs |
| --- | --- | --- | ---: |
| easy | 0–4000 | 1.6e-4 | 80 |
| hard | 0–6000 | 1.2e-4 | 64 |
| extreme | 0–8000 | 1e-4 | 48 |

Every synthetic image uses exactly one container from the DET-COMPASS
`invisible` split. Objects are composed in the high/low-energy domain and the
result is rendered with three fixed lookup tables
(`xray_highlow_to_{colour,grey,density}_dict_RAW.pkl`) built from DET-COMPASS
by `dependencies/det_compass_crops/create_{colour_dict_fast,grey_dict,density_dict}.py`.
Each image draws from its own random stream derived from the base seed
(default 40), the stage and the image id, so the output is byte-identical
across runs and worker counts.

## License

Code written for LAO-X and the UnSAMv2/SAM 2 code it builds on are released
under the Apache License 2.0 ([LICENSE](LICENSE)). Third-party components keep
their own licenses; in particular the modified CutLER code in
`third_party/CutLER` is under CC BY-NC-SA 4.0. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
