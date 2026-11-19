# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
from typing import Any, Dict, List, Optional, Tuple
import os
import time

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.amg import (
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    MaskData,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SAM2AutomaticMaskGenerator:
    def __init__(
        self,
        model: SAM2Base,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        use_m2m: bool = False,
        multimask_output: bool = True,
        granularity: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Using a SAM 2 model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM 2 with a HieraL backbone.

        Arguments:
          model (Sam): The SAM 2 model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          mask_threshold (float): Threshold for binarizing the mask logits
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          use_m2m (bool): Whether to add a one step refinement using previous mask predictions.
          multimask_output (bool): Whether to output multimask at each point of the grid.
          granularity (float or None): Granularity parameter to control mask generation detail level.
            If None, uses model default behavior.
        """

        t0 = time.perf_counter()
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils  # type: ignore  # noqa: F401
            except ImportError as e:
                print("Please install pycocotools")
                raise e

        t_pred_start = time.perf_counter()
        self.predictor = SAM2ImagePredictor(
            model,
            max_hole_area=min_mask_region_area,
            max_sprinkle_area=min_mask_region_area,
        )
        t_pred_end = time.perf_counter()
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.use_m2m = use_m2m
        self.multimask_output = multimask_output
        self.granularity = granularity
        t1 = time.perf_counter()
        if os.getenv("SAM2_PROFILE", "") == "1":
            print(
                "[SAM2AutomaticMaskGenerator] init predictor={:.3f}s total={:.3f}s".format(
                    t_pred_end - t_pred_start,
                    t1 - t0,
                ),
                flush=True,
            )

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2AutomaticMaskGenerator":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2AutomaticMaskGenerator): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def generate(self, image: np.ndarray, gra: float = 1.0) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        t0 = time.perf_counter()
        # Generate masks
        t_gen_start = time.perf_counter()
        mask_data = self._generate_masks(image, gra)
        t_gen_end = time.perf_counter()

        # Encode masks
        t_encode_start = time.perf_counter()
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]
        t_encode_end = time.perf_counter()

        # Write mask records
        t_pack_start = time.perf_counter()
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)
        t_pack_end = time.perf_counter()
        t1 = time.perf_counter()
        if os.getenv("SAM2_PROFILE", "") == "1":
            print(
                "[SAM2AutomaticMaskGenerator] generate masks={:.3f}s encode={:.3f}s pack={:.3f}s total={:.3f}s".format(
                    t_gen_end - t_gen_start,
                    t_encode_end - t_encode_start,
                    t_pack_end - t_pack_start,
                    t1 - t0,
                ),
                flush=True,
            )

        return curr_anns

    def _generate_masks(self, image: np.ndarray, gra: float = 1.0) -> MaskData:
        t0 = time.perf_counter()
        orig_size = image.shape[:2] 
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        t_crop_total = 0.0
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            t_crop_start = time.perf_counter()
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size, gra)
            data.cat(crop_data)
            t_crop_total += time.perf_counter() - t_crop_start

        # Remove duplicate masks between crops
        t_nms_start = time.perf_counter()
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        t_nms_end = time.perf_counter()
        data.to_numpy()
        t1 = time.perf_counter()
        if os.getenv("SAM2_PROFILE", "") == "1":
            print(
                "[SAM2AutomaticMaskGenerator] _generate_masks crops={:.3f}s nms={:.3f}s total={:.3f}s".format(
                    t_crop_total,
                    t_nms_end - t_nms_start,
                    t1 - t0,
                ),
                flush=True,
            )
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
        gra: float = 1.0,
    ) -> MaskData:
        # Crop the image and calculate embeddings
        t0 = time.perf_counter()
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        t_set_start = time.perf_counter()
        self.predictor.set_image(cropped_im)
        t_set_end = time.perf_counter()

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        t_batch_total = 0.0
        batch_count = 0
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            t_batch_start = time.perf_counter()
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size, normalize=True, gra=gra
            )
            data.cat(batch_data)
            del batch_data
            t_batch_total += time.perf_counter() - t_batch_start
            batch_count += 1
        self.predictor.reset_predictor()

        # Remove duplicates within this crop.
        t_nms_start = time.perf_counter()
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)
        t_nms_end = time.perf_counter()

        # Return to the original image frame
        t_uncrop_start = time.perf_counter()
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
        t_uncrop_end = time.perf_counter()

        t1 = time.perf_counter()
        if os.getenv("SAM2_PROFILE", "") == "1":
            print(
                "[SAM2AutomaticMaskGenerator] _process_crop set_image={:.3f}s batches={:.3f}s(n={}) nms={:.3f}s uncrop={:.3f}s total={:.3f}s".format(
                    t_set_end - t_set_start,
                    t_batch_total,
                    batch_count,
                    t_nms_end - t_nms_start,
                    t_uncrop_end - t_uncrop_start,
                    t1 - t0,
                ),
                flush=True,
            )
        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        normalize=False,
        gra: float = 1.0,
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        t0 = time.perf_counter()
        points = torch.as_tensor(
            points, dtype=torch.float32, device=self.predictor.device
        )
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        
        # Use granularity as a simple float or None
        gra_value = gra if gra is not None else 1.0
                
        t_pred_start = time.perf_counter()
        masks, iou_preds, low_res_masks = self.predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=self.multimask_output,
            return_logits=True,
            gra=gra_value,
            granularity=None,  # Explicitly set to None to avoid issues
        )
        t_pred_end = time.perf_counter()
        # Serialize predictions and store in MaskData
        t_pack_start = time.perf_counter()
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=points.repeat_interleave(masks.shape[1], dim=0),
            low_res_masks=low_res_masks.flatten(0, 1),
        )
        del masks
        t_pack_end = time.perf_counter()

        if not self.use_m2m:
            t_filter_start = time.perf_counter()
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)
            # Calculate and filter by stability score
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
            t_filter_end = time.perf_counter()
        else:
            t_filter_start = time.perf_counter()
            # One step refinement using previous mask predictions
            in_points = self.predictor._transforms.transform_coords(
                data["points"], normalize=normalize, orig_hw=im_size
            )
            labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, ious = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch, gra=gra
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)

            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
            t_filter_end = time.perf_counter()

        # Threshold masks and calculate boxes
        t_rle_start = time.perf_counter()
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]
        t_rle_end = time.perf_counter()
        t1 = time.perf_counter()
        if os.getenv("SAM2_PROFILE", "") == "1":
            print(
                "[SAM2AutomaticMaskGenerator] _process_batch pred={:.3f}s pack={:.3f}s filter={:.3f}s rle={:.3f}s total={:.3f}s".format(
                    t_pred_end - t_pred_start,
                    t_pack_end - t_pack_start,
                    t_filter_end - t_filter_start,
                    t_rle_end - t_rle_start,
                    t1 - t0,
                ),
                flush=True,
            )

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def refine_with_m2m(self, points, point_labels, low_res_masks, points_per_batch, gra=1.0):
        new_masks = []
        new_iou_preds = []

        for cur_points, cur_point_labels, low_res_mask in batch_iterator(
            points_per_batch, points, point_labels, low_res_masks
        ):
            # Use granularity as a simple float for M2M
            gra_value = gra if gra is not None else 1.0
            
            best_masks, best_iou_preds, _ = self.predictor._predict(
                cur_points[:, None, :],
                cur_point_labels[:, None],
                mask_input=low_res_mask[:, None, :],
                multimask_output=False,
                return_logits=True,
                gra=gra_value,
                granularity=None,  # Explicitly set to None to avoid issues
            )
            new_masks.append(best_masks)
            new_iou_preds.append(best_iou_preds)
        masks = torch.cat(new_masks, dim=0)
        return masks, torch.cat(new_iou_preds, dim=0)
