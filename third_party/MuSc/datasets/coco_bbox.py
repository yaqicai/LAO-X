import json
import math
import os
import random
from collections import defaultdict
from enum import Enum

import PIL.Image
import PIL.ImageDraw
import torch
from torchvision import transforms


_CLASSNAMES = ["ALL"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NEAREST = getattr(getattr(transforms, "InterpolationMode", None), "NEAREST", PIL.Image.NEAREST)


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CocoBBoxDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        clip_transformer=None,
        random_seed=42,
        divide_num=1,
        divide_iter=0,
        annotation_path=None,
        image_root=None,
        max_images=None,
        crop_mode="vanilla",
        annotation_mode="auto",
        target_mask_ratio=0.2,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classname = classname
        self.annotation_path = annotation_path or self._default_annotation_path(source)
        self.image_root = image_root or self._default_image_root(source)
        self.max_images = max_images
        self.crop_mode = crop_mode
        self.annotation_mode = annotation_mode
        self.target_mask_ratio = target_mask_ratio
        if self.crop_mode not in ("vanilla", "ratio_expand"):
            raise ValueError(f"Unsupported crop_mode: {self.crop_mode}")
        if self.annotation_mode not in ("auto", "mask", "bbox"):
            raise ValueError(f"Unsupported annotation_mode: {self.annotation_mode}")
        resize_size = self._to_2tuple(resize)
        image_size = self._to_2tuple(imagesize)
        self.target_crop_aspect = image_size[1] / image_size[0]

        self.data_to_iterate = self.get_image_data()
        if divide_num > 1:
            self.data_to_iterate = self.sub_datasets(
                self.data_to_iterate, divide_num, divide_iter, random_seed
            )

        if clip_transformer is None:
            self.transform_img = transforms.Compose(
                [
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        else:
            self.transform_img = clip_transformer

        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=NEAREST),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        self.imagesize = (3, *image_size)

    def _to_2tuple(self, value):
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value, value)

    def _default_annotation_path(self, source):
        for filename in ("coco_agnostic.json", "coco.json", "full_test.json"):
            path = os.path.join(source, filename)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"No COCO annotation file found under {source}. "
            "Pass --annotation_path explicitly."
        )

    def _default_image_root(self, source):
        for dirname in ("PIXray", "xray", "images/val2017", "val2017", "test2017"):
            path = os.path.join(source, dirname)
            if os.path.isdir(path):
                return path
        return source

    def sub_datasets(self, full_datasets, divide_num, divide_iter, random_seed=42):
        if divide_num == 0:
            return full_datasets
        random.seed(random_seed)

        id_dict = defaultdict(list)
        for i, item in enumerate(full_datasets):
            id_dict[item["category_name"]].append(i)

        sub_id_list = []
        for key in sorted(id_dict.keys()):
            type_id_list = id_dict[key]
            random.shuffle(type_id_list)
            divide_list = [
                type_id_list[i : i + divide_num]
                for i in range(0, len(type_id_list), divide_num)
            ]
            sub_list = [
                divide_list[i][divide_iter]
                for i in range(len(divide_list))
                if len(divide_list[i]) > divide_iter
            ]
            sub_id_list.extend(sub_list)
        return [full_datasets[idx] for idx in sub_id_list]

    def __getitem__(self, idx):
        item = self.data_to_iterate[idx]
        image = PIL.Image.open(item["image_path"]).convert("RGB")
        width, height = image.size
        gt_crop_box = self._crop_box(item["bbox"], width, height)
        full_mask = self._make_mask(item, (width, height))
        crop_box = self._select_crop_box(item, full_mask, width, height)
        image_crop = image.crop(crop_box)
        mask_crop = full_mask.crop(crop_box)

        image_tensor = self.transform_img(image_crop)
        mask_tensor = self.transform_mask(mask_crop) > 0

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "is_anomaly": 1,
            "image_path": item["display_path"],
            "original_image_path": item["image_path"],
            "crop_box": torch.tensor(crop_box, dtype=torch.int64),
            "gt_crop_box": torch.tensor(gt_crop_box, dtype=torch.int64),
            "crop_mode": self.crop_mode,
            "annotation_mode": self._annotation_mode(item),
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        with open(self.annotation_path, "r") as f:
            coco = json.load(f)

        categories = {
            category["id"]: category.get("name", str(category["id"]))
            for category in coco.get("categories", [])
        }
        selected_class = None
        if self.classname is not None and str(self.classname).lower() != "all":
            selected_class = str(self.classname)

        anns_by_image = defaultdict(list)
        for ann in coco.get("annotations", []):
            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            category_name = categories.get(ann.get("category_id"), str(ann.get("category_id")))
            if selected_class is not None and category_name != selected_class:
                continue
            anns_by_image[ann["image_id"]].append(ann)

        images = []
        for image_info in coco.get("images", []):
            image_anns = anns_by_image.get(image_info["id"], [])
            if image_anns:
                images.append((image_info, image_anns))
        images.sort(key=lambda x: x[0].get("file_name", ""))

        data_to_iterate = []
        for image_info, image_anns in images:
            image_path = os.path.join(self.image_root, image_info["file_name"])
            for ann in sorted(image_anns, key=lambda x: x.get("id", 0)):
                category_name = categories.get(ann.get("category_id"), str(ann.get("category_id")))
                data_to_iterate.append(
                    {
                        "image_path": image_path,
                        "display_path": self._display_path(image_info["file_name"], ann, category_name),
                        "bbox": ann["bbox"],
                        "segmentation": ann.get("segmentation"),
                        "category_name": category_name,
                    }
                )
                if self.max_images is not None and len(data_to_iterate) >= self.max_images:
                    break
            if self.max_images is not None and len(data_to_iterate) >= self.max_images:
                break

        if not data_to_iterate:
            raise RuntimeError(
                f"No bbox samples found in {self.annotation_path} for class {self.classname}."
            )
        return data_to_iterate

    def _display_path(self, file_name, ann, category_name):
        stem, ext = os.path.splitext(os.path.basename(file_name))
        category_dir = category_name.replace("/", "_")
        ann_id = ann.get("id", len(stem))
        return os.path.join(
            self.source,
            "_musc_bbox_input",
            category_dir,
            f"{stem}_ann{ann_id}{ext or '.png'}",
        )

    def _crop_box(self, bbox, width, height):
        x, y, w, h = bbox[:4]
        left = max(0, int(math.floor(x)))
        top = max(0, int(math.floor(y)))
        right = min(width, int(math.ceil(x + w)))
        bottom = min(height, int(math.ceil(y + h)))
        if right <= left:
            right = min(width, left + 1)
        if bottom <= top:
            bottom = min(height, top + 1)
        return left, top, right, bottom

    def _select_crop_box(self, item, mask, width, height):
        gt_box = self._crop_box(item["bbox"], width, height)
        if self.crop_mode == "vanilla":
            return gt_box
        return self._ratio_expand_crop_box(item, mask, width, height)

    def _ratio_expand_crop_box(self, item, mask, width, height):
        left, top, right, bottom = self._crop_box(item["bbox"], width, height)
        bbox_w = max(1, right - left)
        bbox_h = max(1, bottom - top)
        aspect = self.target_crop_aspect
        object_area = self._object_area(item, mask, width, height)
        if object_area <= 0:
            return left, top, right, bottom

        target_area = object_area / self.target_mask_ratio
        target_w = math.sqrt(target_area * aspect)
        target_h = math.sqrt(target_area / aspect)
        if target_w < bbox_w:
            target_w = bbox_w
            target_h = target_w / aspect
        if target_h < bbox_h:
            target_h = bbox_h
            target_w = target_h * aspect

        if target_w > width:
            target_w = width
            target_h = target_w / aspect
        if target_h > height:
            target_h = height
            target_w = target_h * aspect

        target_w = max(1, min(width, int(math.ceil(target_w))))
        target_h = max(1, min(height, int(math.ceil(target_h))))
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        return self._centered_box(center_x, center_y, target_w, target_h, width, height)

    def _annotation_mode(self, item):
        if self.annotation_mode == "bbox":
            return "bbox"
        if self.annotation_mode == "mask":
            return "mask" if self._has_polygon_segmentation(item) else "bbox"
        return "mask" if self._has_polygon_segmentation(item) else "bbox"

    def _has_polygon_segmentation(self, item):
        segmentation = item.get("segmentation")
        if not isinstance(segmentation, list):
            return False
        for polygon in segmentation:
            if isinstance(polygon, list) and len(polygon) >= 6:
                return True
        return False

    def _object_area(self, item, mask, width, height):
        if self._annotation_mode(item) == "mask":
            return sum(mask.histogram()[1:])

        left, top, right, bottom = self._crop_box(item["bbox"], width, height)
        return max(1, (right - left) * (bottom - top))

    def _centered_box(self, center_x, center_y, crop_w, crop_h, image_w, image_h):
        left = int(round(center_x - crop_w / 2))
        top = int(round(center_y - crop_h / 2))
        right = left + crop_w
        bottom = top + crop_h

        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > image_w:
            left -= right - image_w
            right = image_w
        if bottom > image_h:
            top -= bottom - image_h
            bottom = image_h

        left = max(0, left)
        top = max(0, top)
        return left, top, right, bottom

    def _make_mask(self, item, image_size):
        mask = PIL.Image.new("L", image_size, 0)
        draw = PIL.ImageDraw.Draw(mask)
        drew_polygon = False
        if self._annotation_mode(item) == "mask":
            for polygon in item["segmentation"]:
                if isinstance(polygon, list) and len(polygon) >= 6:
                    points = list(zip(polygon[0::2], polygon[1::2]))
                    draw.polygon(points, fill=255)
                    drew_polygon = True

        if not drew_polygon:
            draw.rectangle(self._crop_box(item["bbox"], *image_size), fill=255)
        return mask
