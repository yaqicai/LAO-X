# # Test_det_v2
# # -----------------
# # python test_det_v2.py --dataset det_compass_syn_100 --weights "/path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt"
# # python test_det_v2.py --dataset det_compass_syn_100 --weights "/path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt" --visualize-image /path/to/datasets/det_compass_syn_100/xray/000003.png --visualize-out /path/to/LAO-X/results/UnSAMv2/000003_vis.png
# # python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --gra 0.3

# # det_compass_syn_100 1
# # python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --gra 0.3 --eval-mode bbox,seg --start_id 0 --end_id 1

# # det_compass_syn_100 100
# # python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --gra 0.15 --eval-mode bbox,seg --start_id 0 --end_id 100
# # python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --gra 0.3 --eval-mode bbox,seg --start_id 0 --end_id 100
# # python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --gra 0.5 --eval-mode bbox,seg --start_id 0 --end_id 100
# # python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --gra 0.7 --eval-mode bbox,seg --start_id 0 --end_id 100
# # python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --gra 1.0 --eval-mode bbox,seg --start_id 0 --end_id 100

# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5 --start_id 0 --end_id 100 --comment _augmentation_restart_fusion_easy
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5 --eval-mode bbox --output-dir ./output_eval --val_json temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100  --comment _augmentation_restart_fusion_easy 
# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5,0.7 --start_id 0 --end_id 100 --comment _augmentation_restart_fusion_easy 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5,0.7 --eval-mode bbox --output-dir ./output_eval --val_json temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100  --comment _augmentation_restart_fusion_easy
# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5,0.7,1.0 --start_id 0 --end_id 100 --comment _augmentation_restart_fusion_easy
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5,0.7,1.0 --eval-mode bbox --output-dir ./output_eval --val_json temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100  --comment _augmentation_restart_fusion_easy

# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5 --start_id 0 --end_id 100 --comment _augmentation_restart_fusion_hard
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5 --eval-mode bbox --output-dir ./output_eval --val_json temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100  --comment _augmentation_restart_fusion_hard 
# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5,0.7 --start_id 0 --end_id 100 --comment _augmentation_restart_fusion_hard 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5,0.7 --eval-mode bbox --output-dir ./output_eval --val_json temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100  --comment _augmentation_restart_fusion_hard
# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5,0.7,1.0 --start_id 0 --end_id 100 --comment _augmentation_restart_fusion_hard 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5,0.7,1.0 --eval-mode bbox --output-dir ./output_eval --val_json temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100  --comment _augmentation_restart_fusion_hard

# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5 --start_id 0 --end_id 100 --comment _augmentation_restart_fusion_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5 --eval-mode bbox --output-dir ./output_eval --val_json temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100  --comment _augmentation_restart_fusion_extreme 
# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5,0.7 --start_id 0 --end_id 100 --comment _augmentation_restart_fusion_extreme 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5,0.7 --eval-mode bbox --output-dir ./output_eval --val_json temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100  --comment _augmentation_restart_fusion_extreme
# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5,0.7,1.0 --start_id 0 --end_id 100 --comment _augmentation_restart_fusion_extreme 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5,0.7,1.0 --eval-mode bbox --output-dir ./output_eval --val_json temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100  --comment _augmentation_restart_fusion_extreme

# # det_compass_syn_100 5000
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --gra 0.15 --eval-mode bbox,seg --start_id 0 --end_id 5000
# # python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --gra 0.3 --eval-mode bbox,seg --start_id 0 --end_id 5000
# # python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --gra 0.5 --eval-mode bbox,seg --start_id 0 --end_id 5000

# # Pixray 5047
# python test_det_v2.py --method unsamv2 --dataset PIXray --sam2-ckpt /path/to/LAO-X/results/train/pixray/log/checkpoints/checkpoint_5.pt --gra 0.15 --eval-mode bbox,seg --start_id 0 --end_id 5047
# python test_det_v2.py --method unsamv2 --dataset PIXray --sam2-ckpt results/train/augmentation_restart_fusion_095/easy/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 100 --comment _augmentation_restart_fusion_095_easy
# python test_det_v2.py --method unsamv2 --dataset PIXray --sam2-ckpt results/train/det_compass_xp_divide_and_conquer_1000/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 100 --comment _divide_and_conquer_1000 
# # python test_det_v2.py --method unsamv2 --dataset PIXray --sam2-ckpt /path/to/LAO-X/results/train/pixray/log/checkpoints/checkpoint_5.pt --gra 0.5 --eval-mode bbox,seg --start_id 0 --end_id 5047
# # python test_det_v2.py --method unsamv2 --dataset PIXray --merge-gra 0.3,0.5 --eval-mode bbox,seg --output-dir ./output_eval --val_json /path/to/datasets/PIXray/coco_agnostic.json

# # Test_det_v3
# # -----------------
# # python test_det_v3.py -c configs/unsamv2_small_training.yaml --use-cluster 0 --num-gpus 1 --resume-from /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt
# # python test_det_v3.py -c configs/unsamv2_small_training.yaml --use-cluster 0 --num-gpus 1 --resume-from /path/to/LAO-X/results/train/det_compass_syn_100/dir/checkpoints/checkpoint_5.pt --img-folder /path/to/datasets/det_compass_syn_100/xray --gt-folder /path/to/datasets/det_compass_syn_100 --granularity 0.3

# # Gra fusion
# # python gra_fusion.py --gras 0.15,0.3,0.5
# # python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.3,0.5 --eval-mode bbox --output-dir ./output_eval --val_json /path/to/LAO-X/temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100

# # det_compass_syn_100 100
# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5,0.7 --start_id 0 --end_id 100
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5,0.7 --eval-mode bbox --output-dir ./output_eval --val_json /path/to/LAO-X/temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100
# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5,0.7,1.0 --start_id 0 --end_id 100
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5,0.7,1.0 --eval-mode bbox --output-dir ./output_eval --val_json /path/to/LAO-X/temp_det_compass_syn_100_val_0_100.json --start_id 0 --end_id 100

# # det_compass_syn_100 5000
# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5 --start_id 0 --end_id 5000
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5 --eval-mode bbox --output-dir ./output_eval --val_json /path/to/LAO-X/temp_det_compass_syn_100_val_0_5000.json --start_id 0 --end_id 5000
# python gra_fusion.py --method unsamv2 --dataset det_compass_syn_100 --gras 0.15,0.3,0.5,0.7 --start_id 0 --end_id 5000
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --merge-gra 0.15,0.3,0.5,0.7 --eval-mode bbox --output-dir ./output_eval --val_json /path/to/LAO-X/temp_det_compass_syn_100_val_0_5000.json --start_id 0 --end_id 5000 --merge_results_files /path/to/LAO-X/output_eval/gra_fusion_PIXray_img0_5047_unsamv2_0.15_0.3_0.5_0.7/coco_instances_results.json

# # Pixray 5047
# python gra_fusion.py --method unsamv2 --dataset PIXray --gras 0.15,0.3,0.5 --start_id 0 --end_id 5047
# python test_det_v2.py --method unsamv2 --dataset PIXray --merge-gra 0.15,0.3,0.5 --eval-mode bbox --output-dir ./output_eval --val_json /path/to/LAO-X/temp_PIXray_val_0_5047.json --start_id 0 --end_id 5047
# python gra_fusion.py --method unsamv2 --dataset PIXray --gras 0.15,0.3,0.5,0.7 --start_id 0 --end_id 5047
# python test_det_v2.py --method unsamv2 --dataset PIXray --merge-gra 0.15,0.3,0.5,0.7 --eval-mode bbox --output-dir ./output_eval --val_json /path/to/LAO-X/temp_PIXray_val_0_5047.json --start_id 0 --end_id 5047

# python gra_fusion.py --method unsamv2 --dataset PIXray --gras 0.3,0.5 --start_id 0 --end_id 5047
# python test_det_v2.py --method unsamv2 --dataset PIXray --merge-gra 0.3,0.5 --eval-mode bbox --output-dir ./output_eval --val_json /path/to/LAO-X/temp_PIXray_val_0_5047.json --start_id 0 --end_id 5047
# python gra_fusion.py --method unsamv2 --dataset PIXray --gras 0.3,0.5,0.7 --start_id 0 --end_id 5047
# python test_det_v2.py --method unsamv2 --dataset PIXray --merge-gra 0.3,0.5,0.7 --eval-mode bbox --output-dir ./output_eval --val_json /path/to/LAO-X/temp_PIXray_val_0_5047.json --start_id 0 --end_id 5047


# # Test instances
# python test_instances.py --dataset det_compass_syn_100 --image_id 000030 --start_id 0 --end_id 5000
# python test_instances.py --dataset PIXray --image_id 0 --start_id 0 --end_id 5047
# python test_instances.py --dataset PIXray --image_id 1 --start_id 0 --end_id 5047

# python test_instances.py --dataset det_compass_xp --image_id 0013 --start_id 0 --end_id 1927

# python test_instances.py --dataset PIXray --image_id 16 --start_id 0 --end_id 5047 --gras 0.3 --comment _divide_and_conquer_1000
# python test_instances.py --dataset PIXray --image_id 16 --start_id 0 --end_id 5047 --gras 0.3 --comment _cutler_ori_N3_gra10
# python test_instances.py --dataset PIXray --image_id 16 --start_id 0 --end_id 100 --gras 0.7 --comment _augmentation_restart_fusion_095_easy

MASK_THRESHOLD=0.5 DATASET=det_compass_xp CKPT_SUFFIX=48 START_ID=0 END_ID=100 GPUS=0 GRAS=1.0 STAGES=extreme bash task_test_augmentation_restart.sh crop_musc_aug_fusion_bs16 
MASK_THRESHOLD=0.5 DATASET=CLCXray CKPT_SUFFIX=48 START_ID=0 END_ID=100 GPUS=0 GRAS=1.0 STAGES=extreme bash task_test_augmentation_restart.sh crop_musc_aug_fusion_bs16
MASK_THRESHOLD=0.5 DATASET=DvXray CKPT_SUFFIX=48 START_ID=0 END_ID=100 GPUS=0 GRAS=1.0 STAGES=extreme bash task_test_augmentation_restart.sh crop_musc_aug_fusion_bs16
MASK_THRESHOLD=0.5 DATASET=HiXray CKPT_SUFFIX=48 START_ID=0 END_ID=100 GPUS=0 GRAS=1.0 STAGES=extreme bash task_test_augmentation_restart.sh crop_musc_aug_fusion_bs16
MASK_THRESHOLD=0.5 DATASET=pidray CKPT_SUFFIX=48 START_ID=0 END_ID=100 GPUS=0 GRAS=1.0 STAGES=extreme bash task_test_augmentation_restart.sh crop_musc_aug_fusion_bs16