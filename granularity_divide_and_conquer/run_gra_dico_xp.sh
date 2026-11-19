#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python gra_divide_conquer.py \
--input-dir /path/to/datasets/det_compass_xp \
--output-dir /path/to/LAO-X/results/cutler/det_compass_xp/granularity_divide_and_conquer/vis \
--confidence-threshold 0.3 \
--thetas 0.9 0.8 0.7 0.6 0.5 \
--NMS-iou 0.8 \
--patch-size 16 \
--local-size 768 \
--postprocess True \
--preprocess True \
--granularity True \
--start-id 0 \
--end-id 10000 \
--detection-per-image 1000 \
--opts MODEL.WEIGHTS /path/to/LAO-X/granularity_divide_and_conquer/ckpts/cutler_cascade_final.pth
# --input-dir /path/to/input/image \
# --output-dir /path/to/save/output \
# --opts MODEL.WEIGHTS /path/to/cutler_checkpoint
