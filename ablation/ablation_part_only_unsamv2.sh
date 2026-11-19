eval "$(conda shell.bash hook)" && conda activate UnSAMv2

cd /path/to/LAO-X/sam2/training

bash train_curriculum_augmentation_restart_8.sh crop_musc_aug_part_only_unsamv2_bs16  \
    --data-root-name crop_aug_xp \
    --anno-comment cutler_clean \
    --stages easy,hard,extreme \
    --stage-ranges easy:0-4000,hard:0-6000,extreme:0-8000 \
    --stage-base-lrs easy:1.6e-4,hard:1.2e-4,extreme:1e-4 \
    --stage-vision-lrs easy:4.8e-6,hard:3.2e-6,extreme:3e-6 \
    --stage-epochs easy:80,hard:64,extreme:48 \
    --batch-size 2