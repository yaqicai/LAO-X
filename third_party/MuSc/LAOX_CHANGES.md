# LAO-X changes to MuSc

- Upstream: https://github.com/xrli-U/MuSc, commit `72d58ad` ("Add files via upload").
- License: MIT (see `LICENSE`, unchanged). The LAO-X modifications below are
  distributed under the same license.
- Use in LAO-X: object-bank step 1, zero-shot anomaly heatmaps for unlabeled
  X-ray images (`scripts/laox_heatmaps.sh`). Only the forward heatmaps
  `*_musc_vanilla.png` are used downstream; the inverse maps are not.

All files not listed here are identical to upstream, except that the upstream
`assets/*.png` README figures are not included.

## Modified files

| File | Change |
| --- | --- |
| `configs/musc.yaml` | New keys `datasets.{max_images, annotation_path, image_root, crop_mode, annotation_mode}` and `models.{msm_topmin_min, msm_topmin_max}` (defaults keep upstream behaviour: MSM interval 0-0.3). |
| `examples/musc_main.py` | CLI flags for the new config keys. Any `--img_resize` value is mapped to the fixed non-square input `[560, 840]` (H, W) used for all LAO-X runs. |
| `datasets/mvtec.py`, `datasets/visa.py`, `datasets/btad.py` | Optional `max_images` cap on the number of test images. |
| `models/musc.py` | Imports dataset modules via `sys.path` (works from any cwd); adds the `mvtec_xp` dataset and the COCO bbox-crop datasets (`pixray`, `CLCXray`, `DvXray`, `pidray`, `HiXray`, `xray_coco`, `coco_bbox`); `class_name ALL` for other datasets means one category `ALL`; passes `max_images`; supports non-square inputs (patch grid from `clip_model.visual.grid_size` for LNAMD reshaping and map interpolation); configurable MSM interval; visualization writes `<stem>_musc_<crop_mode>.png` and `<stem>_musc_inverse_<crop_mode>.png` (plus original/crop images for bbox datasets) instead of a single `<name>`; unknown `dataset_name` raises an error. |
| `models/backbone/open_clip/transform.py` | A tuple/list `image_size` resizes to (H, W) without centre crop. |
| `models/backbone/open_clip/transformer.py` | Attention-map accumulation handles a non-square patch grid. |
| `models/modules/_LNAMD.py` | `_embed(features, shape_info=None)` reshapes patch tokens to a given (non-square) grid. |
| `models/modules/_MSM.py` | `resolve_topmin_interval` clamps the interval to `[0, N-1]`/`[1, N]` (upstream swapped the bounds); `MSM` calls `compute_scores_slow` (same scores, lower GPU memory) instead of `compute_scores_fast`. |
| `models/modules/_RsCIN.py` | Skips k values not in `(0, N)` and returns scores unchanged for fewer than 2 images or constant scores. |
| `scripts/musc.sh` | Replaced the MVTec/VisA/BTAD examples with the LAO-X X-ray bbox-crop runs (pidray, HiXray by default; pixray, CLCXray, DvXray selectable), paths from `LAOX_DATA_ROOT`/`LAOX_RESULTS_ROOT`. |

## Added files

| File | Purpose |
| --- | --- |
| `datasets/mvtec_xp.py` | Loader for a flat folder of unlabeled images (`--dataset_name mvtec_xp --data_path <dir>`); used for the object bank. |
| `datasets/coco_bbox.py` | Loader that crops COCO bbox/mask annotations (`vanilla` or aspect-preserving `ratio_expand` crops) for per-object heatmaps. |
| `scripts/musc_strict.sh` | Small sweep of MSM interval / `r_list` / feature layers on CLCXray crops, with a metrics summary. |
| `scripts/laox_heatmaps.sh` | Object-bank step 1: runs MuSc on an image folder and collects the forward heatmaps into one flat folder named like the source images. |
| `LAOX_CHANGES.md` | This file. |
