"""Dataset locations used by LAO-X.

All datasets live under one root, taken from the ``LAOX_DATA_ROOT`` environment
variable (default: ``<repo>/datasets``). Each entry below is relative to it.
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
Root = os.environ.get("LAOX_DATA_ROOT", os.path.join(REPO_ROOT, "datasets"))

# Dataset directory (annotations are looked up inside it).
XrayRoot = {
    "PIXray": os.path.join(Root, "pixray"),
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

# Image directory of the evaluation split.
XrayImgRoot = {
    "PIXray": os.path.join(Root, "pixray/PIXray"),
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

# Validation images and class-agnostic annotations, used by
# scripts/sweep_pred_iou_thresh_sam2.py to tune pred_iou_thresh.
XrayImgRootVal = {
    "PIXray": os.path.join(Root, "pixray/PIXray"),
    "CLCXray": os.path.join(Root, "CLCXray/val2017"),
    "DvXray": os.path.join(Root, "DvXray/DvXray_Positive_Samples"),
    "pidray": os.path.join(Root, "pidray/pidray/train"),
    "HiXray": os.path.join(Root, "HiXray/train/train_image"),
    "det_compass_xp": os.path.join(Root, "det_compass_xp/xray"),
}
XrayImgRootValAnno = {
    "PIXray": os.path.join(Root, "pixray/coco_agnostic.json"),
    "CLCXray": os.path.join(Root, "CLCXray/coco_agnostic_val.json"),
    "DvXray": os.path.join(Root, "DvXray/coco_agnostic.json"),
    "pidray": os.path.join(Root, "pidray/pidray/annotations/coco_agnostic.json"),
    "HiXray": os.path.join(Root, "HiXray/train/train_annotation/"),
    "det_compass_xp": os.path.join(Root, "det_compass_xp/coco_agnostic.json"),
}
