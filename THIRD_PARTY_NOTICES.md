# Third-party components

| Component | Location | Upstream | Base commit | License | Modified for LAO-X |
| --- | --- | --- | --- | --- | --- |
| UnSAMv2 | repository root, `evaluation/`, `granularity_divide_and_conquer/` | [yujunwei04/UnSAMv2](https://github.com/yujunwei04/UnSAMv2) | — | Apache-2.0 ([LICENSE](LICENSE)) | yes |
| SAM 2 | `sam2/` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | — | Apache-2.0 ([sam2/LICENSE](sam2/LICENSE)); cc_torch BSD-3-Clause ([sam2/LICENSE_cctorch](sam2/LICENSE_cctorch)) | yes |
| Mask2Former ops / CutLER configs used by divide-and-conquer | `granularity_divide_and_conquer/` | via UnSAMv2 | — | see upstream | no |
| MuSc | `third_party/MuSc/` | [xrli-U/MuSc](https://github.com/xrli-U/MuSc) | `72d58ad` | MIT ([LICENSE](third_party/MuSc/LICENSE)) | yes |
| CutLER | `third_party/CutLER/` | [facebookresearch/CutLER](https://github.com/facebookresearch/CutLER) | `cca0a27` | CC BY-NC-SA 4.0 ([LICENSE](third_party/CutLER/LICENSE)) | yes |
| TokenCut | `third_party/CutLER/third_party/TokenCut/` | [YangtaoWANG95/TokenCut](https://github.com/YangtaoWANG95/TokenCut) | — | MIT | no |
| DET-COMPASS-CROPS | `dependencies/det_compass_crops/` | Lorenzo Vaquero, DET-COMPASS-CROPS | `3932fc1` | used with the author's permission | yes |
| Detectron2 (xyz fork) | not vendored; installed by `environment/rebuild_environment.sh` | [MaureenZOU/detectron2-xyz](https://github.com/MaureenZOU/detectron2-xyz) | `42121d7` | Apache-2.0 | no |
| DINOv3 | not vendored; cloned into `granularity_divide_and_conquer/dinov3` | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | `54694f7` | DINOv3 License | no |

Modified files keep their original copyright headers. Modifications to
CutLER files are distributed under CC BY-NC-SA 4.0, the license of the
original files.

## Pretrained weights (not distributed)

| Weights | Used by | Source |
| --- | --- | --- |
| SAM 2.1 Hiera-S `sam2.1_hiera_small.pt` | training initialization | [SAM 2](https://github.com/facebookresearch/sam2#download-checkpoints) |
| CutLER `cutler_cascade_final.pth` | object bank, oversegmentation | [CutLER](https://github.com/facebookresearch/CutLER#model-zoo) |
| DINOv3 ViT-B/16 (`dinov3_vitb16_pretrain_lvd1689m`) | oversegmentation | [DINOv3](https://github.com/facebookresearch/dinov3) |
| OpenCLIP ViT-L-14-336 (OpenAI) | MuSc | downloaded automatically by MuSc |
| CascadePSP | oversegmentation refinement | downloaded automatically by `segmentation-refinement` |
