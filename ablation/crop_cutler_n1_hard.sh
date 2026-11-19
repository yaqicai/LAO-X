eval "$(conda shell.bash hook)" && conda activate UnSAMv2

export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=/path/to/hf_cache/hub


cd /path/to/LAO-X/dependencies/det_compass_crops/
bash task_create_synthetic_dataset_musc.sh --workers 8 --stage hard --num-generated 8000 --num-objects 4,7 --gamma 0.0,0.3 --dataset-root /path/to/datasets/ --dataset-gen crop_aug_cutler_n1_xp --io-workers 2 --crop_ablation ori


cd /path/to/LAO-X/granularity_divide_and_conquer
GPUS=0,1,2,3,4,5,6,7 bash run_gra_dico_aug.sh crop_aug_cutler_n1_xp hard 


cd /path/to/LAO-X
python -c "from util_xray import *; convert_to_class_agnostic('/path/to/datasets/crop_aug_cutler_n1_xp/hard/coco.json')"
python pseudo_label_generate.py --dataset crop_aug_cutler_n1_xp --coco-name coco_agnostic.json --stage hard
python pseudo_label_generate.py --fusion --fusion-name anno_fusion --dataset crop_aug_cutler_n1_xp --stage hard --fusion-overlap-thresh 0.95

bash train_curriculum_augmentation_restart_8.sh crop_aug_cutler_n1_fusion_bs16 \
    --data-root-name crop_aug_cutler_n1_xp \
    --anno-comment fusion \
    --stages easy,hard,extreme \
    --stage-ranges easy:0-4000,hard:0-6000,extreme:0-8000 \
    --stage-base-lrs easy:1.6e-4,hard:1.2e-4,extreme:1e-4 \
    --stage-vision-lrs easy:4.8e-6,hard:3.2e-6,extreme:3e-6 \
    --stage-epochs easy:80,hard:64,extreme:48 \
    --batch-size 2