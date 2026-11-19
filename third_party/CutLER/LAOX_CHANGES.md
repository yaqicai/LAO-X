# LAO-X changes to CutLER

- Upstream: https://github.com/facebookresearch/CutLER, commit `cca0a27`
  ("Update README.md").
- `third_party/TokenCut/` holds the content of the upstream git submodule
  (https://github.com/YangtaoWANG95/TokenCut, pinned at `5eba77f`), vendored as
  plain files; MIT license (`third_party/TokenCut/LICENSE`). LAO-X does not
  modify it intentionally (not re-diffed against a TokenCut checkout).
  `.gitmodules` is kept as upstream but is not used by the LAO-X repository.
- License: CutLER is CC BY-NC-SA 4.0 (`LICENSE`); TokenCut, the bilateral solver
  and the CRF code are MIT (see `README.md`, "License"). **The LAO-X
  modifications and additions listed below are released under CC BY-NC-SA 4.0**
  and are marked here as required by that license.
- Use in LAO-X: object-bank step 2, CutLER masks on X-ray images guided by MuSc
  heatmaps (`cutler/demo/laox_object_bank.sh`), and CutLER-only / MaskCut
  ablation banks (`maskcut.sh`).
- `.gitignore` is unchanged from upstream. Its patterns `*/*.sh` and
  `*/datasets/` match some upstream-tracked files (`videocutler/*.sh`,
  `maskcut/run_maskcut_with_submitit.sh`, `videocutler/datasets/`); they are
  force-tracked in the LAO-X repository. The LAO-X files added here are not ignored.

All files not listed here are identical to upstream, except that the upstream
demo/README images and videos (`docs/`, `cutler/demo/imgs/`, `maskcut/imgs/`,
`videocutler/docs/`) are not included.

## Modified files

| File | Change |
| --- | --- |
| `cutler/demo/demo.py` | New options `--musc_dir`, `--fusion_mode` (`mul`, `add`, `blend`, `mul_mul`, `mul_mul_sigmoid`, `mul_blend_mul`, `mul_blend_mul_v2`, `ori`) and `--musc_{sigmoid_coeff,sharp_temp,soft_temp,power,norm_eps}`: each input is fused with its MuSc heatmap (JET colours inverted to [0, 1]) before inference. `--coco_output` writes all predicted masks as a COCO json (image id = numeric file stem when possible, category `fg`, score kept). `--vis_demo` gates saving visualizations (plus the fused input under `no_anno/`); the output dir is created if missing. |
| `cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml` | `TEST.DETECTIONS_PER_IMAGE: 100 -> 1` (at most one mask per image in the object bank). |
| `cutler/config/cutler_config.py` | Adds `MODEL.ROI_BOX_HEAD.{USE_FED_LOSS, USE_SIGMOID_CE, FED_LOSS_FREQ_WEIGHT_POWER, FED_LOSS_NUM_CLASSES}` defaults for older Detectron2 versions. |
| `cutler/modeling/meta_arch/rcnn.py` | Fallback `move_device_like` when Detectron2 lacks it. |
| `cutler/modeling/roi_heads/fast_rcnn.py` | Fallback import of `get_fed_loss_cls_weights` from the bundled `cutler/data`. |
| `cutler/modeling/roi_heads/roi_heads.py` | `colored` is optional. |
| `maskcut/maskcut.py` | GPU affinity + eigenvector solver `fast_affinity_and_eigen` replaces the SciPy `eigh` path in `maskcut_forward` (unused helpers `build_knn_affinity`, `build_sparse_laplacian` added); masks are built at the actual resized resolution; if no mask is found the image is retried with tau x0.5 and x0.25; returns the resized image; new flags `--pretrain-type` (accepted but unused; DINO is always used) and `--num_images` (if > 0, processes sorted images `[12, 12 + num_images)`); a flat image folder is accepted as `--dataset-path`; per-image errors are no longer skipped. |
| `maskcut/dino.py` | Comment only. |

## Added files

| File | Purpose |
| --- | --- |
| `cutler/demo/laox_object_bank.sh` | Object-bank step 2: `demo.py` with `--musc_dir ... --fusion_mode mul_blend_mul_v2 --coco_output .../coco_cutler_musc.json`; `MODE=ori` gives the CutLER-only bank `coco_cutler_ori.json`. |
| `cutler/demo/run_musc_fusion_ablation.py` | Sweep of the `mul_blend_mul_v2` parameters on a DET-COMPASS subset with COCOeval bbox metrics. |
| `cutler/demo/export_musc_heat_to_coco.py` | Thresholds MuSc heatmaps directly into a COCO mask json (heatmap-only baseline). |
| `export_hf_det_compass_to_coco_agnostic.py` | Exports PAGF/DET-COMPASS boxes (HuggingFace `datasets` cache) for the local X-ray images to a class-agnostic COCO json. |
| `maskcut.sh` | MaskCut N=1 (960 px) / N=3 (480 px) runs, visualization, and the CutLER-only bank. |
| `maskcut/maskcut_v1.py` | Experimental MaskCut variant (centre-bias partition flip, patch size 14, backbone type in the json name). Its dinov2/dinov3/clip branches need `dino.ViTFeat_new`, which is not included; only `--pretrain-type dino` runs. |
| `visualize_maskcut.py` | Draws MaskCut masks and boxes from a COCO json onto the images. |
| `third_party/TokenCut/**` | Vendored submodule content (see above). |
| `LAOX_CHANGES.md` | This file. |
