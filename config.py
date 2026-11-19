import os

# Xray Detection Datasets Roots
# TODO change datasets root
Root = "/path/to/datasets"
# SoyAgeingRoot = os.path.join(Root, "SoyAgeing")
XrayRoot = {
    
    # "det_compass_syn_312": os.path.join(Root, "det_compass_syn_312"),
    # "det_compass_syn_100": os.path.join(Root, "det_compass_syn_100"),
    "PIXray": os.path.join(Root, "pixray"), # /path/to/datasets/PIXray/PIXray/default_coco_xray_v1.json
    "CLCXray": os.path.join(Root, "CLCXray"),
    "DvXray": os.path.join(Root, "DvXray"),
    "pidray": os.path.join(Root, "pidray"),
    "HiXray": os.path.join(Root, "HiXray"),
    "det_compass_xp": os.path.join(Root, "det_compass_xp"),
    "crop_aug_easy": os.path.join(Root, "crop_aug/easy"),
    "crop_aug_hard": os.path.join(Root, "crop_aug/hard"),
    "crop_aug_extreme": os.path.join(Root, "crop_aug/extreme"),
    "WIXray": os.path.join(Root, "WIXray-master"),
}
XrayImgRoot = {
    # "det_compass_syn_312": os.path.join(Root, "det_compass_syn_312/xray"),
    # "det_compass_syn_100": os.path.join(Root, "det_compass_syn_100/xray"),
    "PIXray": os.path.join(Root, "pixray/PIXray"), # /path/to/datasets/PIXray/PIXray/default_coco_xray_v1.json
    "CLCXray": os.path.join(Root, "CLCXray/images/test2017"),
    "DvXray": os.path.join(Root, "DvXray/DvXray_Positive_Samples"),
    "pidray": os.path.join(Root, "pidray/images/test2017"),
    "HiXray": os.path.join(Root, "HiXray/images/test2017"),
    "DET-COMPASS": os.path.join(Root, "det_compass_xp/xray"),
    "det_compass_xp": os.path.join(Root, "det_compass_xp/xray"),
    "crop_aug_easy": os.path.join(Root, "crop_aug/easy/xray"),
    "crop_aug_hard": os.path.join(Root, "crop_aug/hard/xray"),
    "crop_aug_extreme": os.path.join(Root, "crop_aug/extreme/xray"),
    "WIXray": os.path.join(Root, "WIXray-master/images/val2017"),
}
XrayImgRootVal = {
     # /path/to/datasets/det_compass_xp/coco_agnostic.json
    # "det_compass_syn_312": os.path.join(Root, "det_compass_syn_312/xray"),
    # "det_compass_syn_100": os.path.join(Root, "det_compass_syn_100/xray"),
    "PIXray": os.path.join(Root, "pixray/PIXray"), # /path/to/datasets/pixray/coco_agnostic.json
    "CLCXray": os.path.join(Root, "CLCXray/val2017"), # /path/to/datasets/CLCXray/coco_agnostic_val.json
    "DvXray": os.path.join(Root, "DvXray/DvXray_Positive_Samples"), # not in /path/to/datasets/DvXray/coco_agnostic.json
    "pidray": os.path.join(Root, "pidray/pidray/train"),  # /path/to/datasets/pidray/pidray/annotations/xray_train.json
    "HiXray": os.path.join(Root, "HiXray/train/train_image"), # HiXray/train/train_annotation/.txt
    "det_compass_xp": os.path.join(Root, "det_compass_xp/xray"),
}
XrayImgRootValAnno = {
    # "det_compass_syn_312": os.path.join(Root, "det_compass_syn_312/xray"),
    # "det_compass_syn_100": os.path.join(Root, "det_compass_syn_100/xray"),
    "PIXray": os.path.join(Root, "pixray/coco_agnostic.json"), # /path/to/datasets/pixray/coco_agnostic.json
    "CLCXray": os.path.join(Root, "CLCXray/coco_agnostic_val.json"), # /path/to/datasets/CLCXray/coco_agnostic_val.json
    "DvXray": os.path.join(Root, "DvXray/coco_agnostic.json"), # not in /path/to/datasets/DvXray/coco_agnostic.json
    "pidray": os.path.join(Root, "pidray/pidray/annotations/coco_agnostic.json"),  # /path/to/datasets/pidray/pidray/annotations/xray_train.json
    "HiXray": os.path.join(Root, "HiXray/train/train_annotation/"), # HiXray/train/train_annotation/.txt
    "det_compass_xp": "/path/to/datasets/crop_aug_temp/temp/coco_train.json", # /path/to/datasets/det_compass_xp/coco_agnostic.json
}


json_syn_path_v1 = "/path/to/datasets/det_compass_syn/default_coco_xray_v1.json"
json_origin_path_v1 = "/path/to/datasets/det_compass/default_coco_xray_v1.json"
json_syn_path_v2 = "/path/to/datasets/det_compass_syn/default_coco_xray_v2.json"
json_origin_path_v2 = "/path/to/datasets/det_compass/default_coco_xray_v2.json"
json_syn_path = "/path/to/datasets/det_compass_syn/default_coco_xray.json"
json_origin_path = "/path/to/datasets/det_compass/default_coco_xray.json"
categories_all_path = "/path/to/datasets/det_compass/categories_all.json"
categories_visible_path = "/path/to/datasets/det_compass/categories_visible.json"
categories_visible_in_all_path = "/path/to/datasets/det_compass/categories_visible_in_all.json"


# Data augment configs
image_size = 448
crop_pct = 0.875
interpolation = 3
trans_type = "imagenet"
n_views = 2

# Model configs
num_proxy_base = 1  # number of base proxies per classes
num_proxy_hard = None  # number of hard proxies per classes
num_proxy_local = 0
mlp_out_dim = 10240  # MLP head project dimension
sk_num_iter = 3  # number of Sinkhorn-knopp iters
sk_epsilon = 0.05  # epsilon of Sinkhorn-knopp

# Dataset config
prop_train_labels = 0.5

# Experiment config
temperature = 0.1

# DINO
dino_pretrain_path = "./your_dino_pretrained_path"
