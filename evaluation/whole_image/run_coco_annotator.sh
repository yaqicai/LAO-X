#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python coco_annotator.py \
    --image-dir /path/to/datasets/pixray/PIXray \
    --model-config configs/unsamv2_small.yaml \
    --checkpoint /path/to/LAO-X/results/train/det_compass_xp_divide_and_conquer_1000/log/checkpoints/checkpoint.pt \
    --output-dir /path/to/LAO-X/Official_eval/ \
    --granularities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --start-index 0 \
    --end-index 100
    # --granularities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \ 
    
    "$@"