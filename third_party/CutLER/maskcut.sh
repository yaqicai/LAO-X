# DINO v1
python maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir /path/to/datasets/maskcut/det_compass_xp --vit-arch base --patch-size 8 --tau 0.15 --num_images 150 --N 1 --pretrain-type dino --fixed_size 960

python maskcut.py --dataset-path /path/to/datasets/det_compass_syn_100/xray/ --out-dir /path/to/datasets/maskcut/det_compass_syn_100/ --vit-arch base --patch-size 16 --pretrain-type dinov3 --pretrain_path /path/to/models/dinov3_vitb16_pretrain.pth

python maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir /path/to/datasets/maskcut/det_compass_xp --vit-arch base --patch-size 14 --tau 0.15 --num_images 50 --N 3 --pretrain-type dinov2 --pretrain_path /path/to/models/dinov2_vitb14_pretrain.pth

python maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir /path/to/datasets/maskcut/det_compass_xp --vit-arch base --patch-size 16 --tau 0.15 --num_images 50 --N 1 --pretrain-type dinov3 --pretrain_path /path/to/models/dinov3_vitb16_pretrain.pth --fixed_size 960   --dinov3-repo /path/to/LAO-X/granularity_divide_and_conquer/dinov3
python maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir /path/to/datasets/maskcut/det_compass_xp --vit-arch base --patch-size 14 --tau 0.15 --num_images 50 --N 1 --pretrain-type dinov2 --pretrain_path /path/to/models/dinov2_vitb14_pretrain.pth --fixed_size 960

python demo.py --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml --input /path/to/datasets/det_compass_xp/*.png --opts MODEL.WEIGHTS /path/to/LAO-X/granularity_divide_and_conquer/ckpts/cutler_cascade_final.pth

python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize480_tau0.15_N3_0_1.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize480_tau0.15_N3_0_1.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp_dinov3_N1 --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize960_tau0.15_N1_0_1_dinov3.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp_dinov3_N1_v2 --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize960_tau0.15_N1_0_1_dinov3.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp_dinov2_N1_v2 --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize960_tau0.15_N1_0_1_dinov2.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp_dinov2_N1_v3 --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize960_tau0.15_N1_0_1_dinov2.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp_dinov2_N1_v5 --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize960_tau0.15_N1_0_1_dinov2.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp_dinov2_N1_anyup_v5 --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize480_tau0.15_N1_0_1_dino.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp_dinov2_N1_v6 --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize960_tau0.15_N1_0_1_dino.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp_dinov2_N1_v6_960 --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize960_tau0.15_N1_0_1.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp_dinov2_N1_v7_480 --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize480_tau0.15_N1_0_1_dino.json
python visualize_maskcut.py --dataset-path /path/to/datasets/det_compass_xp/ --out-dir ./maskcut/det_compass_xp_dinov1_N1_960_kqv --ann-path /path/to/datasets/maskcut/det_compass_xp/imagenet_train_fixsize960_tau0.15_N1_0_1.json