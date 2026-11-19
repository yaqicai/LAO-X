# LAO-X X-ray synthesis

This directory generates the synthetic training stages of LAO-X. Each
synthetic image is made from **exactly one empty container** taken from the
DET-COMPASS `invisible` split. Objects from the object bank are pasted onto it
in the high/low-energy domain (Beer–Lambert composition), and the result is
rendered with three fixed lookup tables built from DET-COMPASS:
high/low → colour, grey and density. Each image comes with COCO boxes and
polygon masks for its objects.

| File | Purpose |
| --- | --- |
| `run_stage.sh` | Generates one stage: runs parallel workers, then merges their annotations |
| `generate.py` | Generator with three subcommands: `match`, `augment`, `merge` |
| `build_luts.py` | Builds the three lookup tables from DET-COMPASS |
| `compass_editor/` | High/low-energy utilities, LUT rendering and object transforms |
| `compare_outputs.py` | Compares two output directories file by file (SHA-256) |
| `eval_coco_cutler_against_gt.py` | Scores object-bank COCO files against labelled COCO (bbox AP/AR) |

The CPU-only Python requirements are listed in [INSTALL.md](../INSTALL.md).
The generator also imports `torch`, but only to seed it.

## Inputs

| Input | Default location (`$LAOX_DATA_ROOT` defaults to `<repo>/datasets`) |
| --- | --- |
| DET-COMPASS: HuggingFace [`PAGF/DET-COMPASS`](https://huggingface.co/datasets/PAGF/DET-COMPASS), `default` configuration, `test` and `invisible` splits | HuggingFace cache |
| Object bank: COCO mask JSON on DET-COMPASS `test` images, from the MuSc × CutLER step | passed with `--object-bank-json` (e.g. `.../annotations/coco_cutler_musc.json`) |
| Held-out labelled COCO. Its images are removed from the object bank | `$LAOX_DATA_ROOT/det_compass_xp/annotations/coco_agnostic.json` |
| Matches JSON that maps local file names to HuggingFace `(split, index)` | `$LAOX_DATA_ROOT/det_compass_xp/annotations/huggingface_matches.json` |
| Lookup tables `xray_highlow_to_{colour,grey,density}_dict_RAW.pkl` | `$LAOX_LUT_DIR`, default `synthesis/luts/` |

**DET-COMPASS cache.** The generator uses the standard HuggingFace environment
variables (`HF_HOME`, `HF_DATASETS_CACHE`). By default (`--offline`) it reads
the cached Arrow shards directly from
`$HF_DATASETS_CACHE/*___det-compass/<config>/<version>/<revision>/`. On the
first run, pass `--online` so that the dataset can be downloaded.

**Matches JSON.** Build it once from the local DET-COMPASS colour images:

```bash
python synthesis/generate.py match --online \
    --xray-dir "$LAOX_DATA_ROOT/det_compass_xp/xray" \
    --output "$LAOX_DATA_ROOT/det_compass_xp/annotations/huggingface_matches.json"
```

**Lookup tables.** Build them from DET-COMPASS. The colour and grey tables use
the `test` split; the density table uses all splits.

```bash
python synthesis/build_luts.py all              # writes to $LAOX_LUT_DIR or synthesis/luts/
python synthesis/build_luts.py density --force  # rebuild a single table
```

`colour` is the vectorised builder. `colour-slow` is the original per-pixel
builder. Both count the same values, but they insert keys in a different
order, so ties in the later denoising and k-NN densification can resolve
differently. To reproduce published results, keep the tables you trained
with. The generator reads the pickles from `$LAOX_LUT_DIR` if that variable
is set, and from `synthesis/luts/` otherwise.

## Stages

| Stage | Images | Objects / image | `gamma` (max box IoU) |
| --- | ---: | --- | --- |
| easy | 8,000 | 2–4 | 0.0–0.05 |
| hard | 8,000 | 4–7 | 0.0–0.3 |
| extreme | 8,000 | 6–9 | 0.0–0.5 |

All stages use 16 worker processes with 2 PNG-writer threads each (the
`run_stage.sh` defaults) and base seed 40.

```bash
export LAOX_DATA_ROOT=/path/to/datasets
ANN="$LAOX_DATA_ROOT/det_compass_xp/annotations"
bash synthesis/run_stage.sh --object-bank-json "$ANN/coco_cutler_musc.json" \
    --dataset-gen crop_aug --stage easy    --num-objects 2,4 --gamma 0.0,0.05
bash synthesis/run_stage.sh --object-bank-json "$ANN/coco_cutler_musc.json" \
    --dataset-gen crop_aug --stage hard    --num-objects 4,7 --gamma 0.0,0.3
bash synthesis/run_stage.sh --object-bank-json "$ANN/coco_cutler_musc.json" \
    --dataset-gen crop_aug --stage extreme --num-objects 6,9 --gamma 0.0,0.5
```

`run_stage.sh` options are `--dataset-root`, `--workers`, `--num-generated`,
`--io-workers`, `--seed`, `--online`, `--test-coco`, `--matches-json` and
`--composition-mode xray|rgb`; run it with `--help` for details. `rgb`
(copy-paste instead of Beer–Lambert) is the composition ablation. The
interpreter is taken from the `PYTHON` environment variable. The script will
not write into a stage directory that is not empty.

## Output

```text
<dataset-root>/<dataset-gen>/<stage>/
├── xray/000000.png ...          # colour renderings, one per image id
├── coco_<start>_<end>.json      # annotations of one worker shard
├── coco.json                    # all shards merged (annotation ids renumbered)
└── coco_train.json              # DET-COMPASS boxes of the object-bank source images
```

Each annotation contains a box, a polygon segmentation and the mask area. The
category of each object is copied from the object-bank annotation.

## Determinism

Image `i` of stage `s` is generated from its own random stream. The stream is
seeded with `sha256("laox-single-legacy-v1:<seed>:<s>:<i>")` and applied to
Python `random`, NumPy, OpenCV and torch. OpenCV and BLAS run single-threaded,
and `PYTHONHASHSEED` is fixed. Because of this, `xray/*.png`, every
`coco_*.json` and `coco_train.json` are byte-identical across runs. This holds
for the same arguments, the same inputs, LUTs and library versions. The
image files and `coco.json` also do not depend on `--workers` or
`--io-workers`; only the names of the shard files do. To check two runs:

```bash
python synthesis/compare_outputs.py RUN_A/easy RUN_B/easy   # exit 1 on any difference
```

## Attribution

This code is derived from **DET-COMPASS-CROPS** by **Lorenzo Vaquero**, and is
used and modified with his permission. LAO-X's changes are the object-bank
input, single-container composition, the shardable deterministic seeding and
the COCO export. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
DET-COMPASS is by its respective authors; see its
[dataset card](https://huggingface.co/datasets/PAGF/DET-COMPASS).
