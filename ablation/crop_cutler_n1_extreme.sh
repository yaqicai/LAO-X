eval "$(conda shell.bash hook)" && conda activate UnSAMv2

export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=/path/to/hf_cache/hub


cd /path/to/LAO-X/dependencies/det_compass_crops/
bash task_create_synthetic_dataset_musc.sh --workers 8 --stage extreme --num-generated 8000 --num-objects 6,9 --gamma 0.0,0.5 --dataset-root /path/to/datasets/ --dataset-gen crop_aug_cutler_n1_xp --io-workers 2 --crop_ablation ori


cd /path/to/LAO-X/granularity_divide_and_conquer
GPUS=0,1,2,3,4,5,6,7 bash run_gra_dico_aug.sh crop_aug_cutler_n1_xp extreme


cd /path/to/LAO-X
python -c "from util_xray import *; convert_to_class_agnostic('/path/to/datasets/crop_aug_cutler_n1_xp/extreme/coco.json')"
python pseudo_label_generate.py --dataset crop_aug_cutler_n1_xp --coco-name coco_agnostic.json --stage extreme
python pseudo_label_generate.py --fusion --fusion-name anno_fusion --dataset crop_aug_cutler_n1_xp --stage extreme --fusion-overlap-thresh 0.95