eval "$(conda shell.bash hook)" && conda activate UnSAMv2

export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=/path/to/hf_cache/hub


cd /path/to/LAO-X/dependencies/det_compass_crops/
bash task_create_synthetic_dataset_musc.sh --workers 8 --stage extreme_9_12 --num-generated 8000 --num-objects 9,12 --gamma 0.0,0.5 --dataset-root /path/to/datasets/ --dataset-gen crop_aug_xp --io-workers 2

cd /path/to/LAO-X/granularity_divide_and_conquer
GPUS=0,1,2,3,4,5,6,7 bash run_gra_dico_aug.sh crop_aug_xp extreme_9_12


cd /path/to/LAO-X
python -c "from util_xray import *; convert_to_class_agnostic('/path/to/datasets/crop_aug_xp/extreme_9_12/coco.json')"
python pseudo_label_generate.py --dataset crop_aug_xp --coco-name coco_agnostic.json --stage extreme_9_12
python pseudo_label_generate.py --fusion --fusion-name anno_fusion --dataset crop_aug_xp --stage extreme_9_12 --fusion-overlap-thresh 0.95

cd /path/to/LAO-X/sam2/training
bash train_curriculum_augmentation_restart_4.sh crop_musc_aug_fusion_bs16_xp  \
    --data-root-name crop_aug_xp \
    --anno-comment fusion \
    --stages easy,hard,extreme_9_12 \
    --stage-ranges easy:0-4000,hard:0-6000,extreme_9_12:0-8000 \
    --stage-base-lrs easy:1.6e-4,hard:1.2e-4,extreme_9_12:1e-4 \
    --stage-vision-lrs easy:4.8e-6,hard:3.2e-6,extreme_9_12:3e-6 \
    --stage-epochs easy:80,hard:64,extreme_9_12:48 \
    --batch-size 4
