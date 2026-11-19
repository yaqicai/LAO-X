# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/demo/demo.py

import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json
import datetime

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from pycocotools import mask as mask_util
import sys
sys.path.append('./')
sys.path.append('../')
from config import add_cutler_config

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "CutLER detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Disable the use of SyncBN normalization when running on a CPU
    # SyncBN is not supported on CPU and can cause errors, so we switch to BN instead
    if cfg.MODEL.DEVICE == 'cpu' and cfg.MODEL.RESNETS.NORM == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--musc_dir",
        help="LAO-X: directory of JET-colorized MuSc heatmaps named <input stem>.png; "
        "if given, each input image is fused with its heatmap (see --fusion_mode).",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--vis_demo",
        action="store_true",
        help="Save visualization images when --output is set.",
    )
    parser.add_argument(
        "--coco_output",
        default=None,
        type=str,
        help="Path to save COCO-format annotations (json).",
    )
    parser.add_argument(
        "--fusion_mode",
        help="Fusion mode for heatmap and original image.",
        default="mul",
        choices=["mul", "add", "blend", "mul_mul", "mul_mul_sigmoid", "mul_blend_mul", "mul_blend_mul_v2", "ori"],
        type=str,
    )
    parser.add_argument(
        "--musc_sigmoid_coeff",
        type=float,
        default=-5.0,
        help="Coefficient inside the sigmoid exponent for mul_blend_mul_v2.",
    )
    parser.add_argument(
        "--musc_sharp_temp",
        type=float,
        default=0.8,
        help="Relative max-response threshold for the sharp MuSC gate in mul_blend_mul_v2.",
    )
    parser.add_argument(
        "--musc_soft_temp",
        type=float,
        default=0.6,
        help="Relative max-response threshold for the soft MuSC gate in mul_blend_mul_v2.",
    )
    parser.add_argument(
        "--musc_power",
        type=float,
        default=3.0,
        help="Nonlinear enhancement power applied to the MuSC heatmap in mul_blend_mul_v2.",
    )
    parser.add_argument(
        "--musc_norm_eps",
        type=float,
        default=1e-6,
        help="Small value used to avoid division by zero when normalizing MuSC weights.",
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def recover_heatmap_from_image(img):
    """LAO-X: invert cv2.COLORMAP_JET, mapping a BGR heatmap image back to [0, 1]."""
    from scipy.spatial.distance import cdist
    # cv2 images are BGR, matching the output of cv2.applyColorMap.
    image = img
    if image is None:
        raise ValueError("Failed to read heatmap image")

    h, w, c = image.shape

    # JET lookup table: the colour of every grey level 0..255, shape (256, 3).
    gradient_values = np.arange(256, dtype=np.uint8).reshape(1, 256)
    colormap_lut = cv2.applyColorMap(gradient_values, cv2.COLORMAP_JET).squeeze()

    # For every pixel, find the nearest LUT colour (Euclidean distance);
    # its index is the original 0..255 grey value. dists has shape (N, 256).
    image_flat = image.reshape(-1, 3).astype(np.float32)
    lut_flat = colormap_lut.astype(np.float32)
    dists = cdist(image_flat, lut_flat, 'euclidean')
    original_int_values = np.argmin(dists, axis=1)

    # Reshape and rescale to [0, 1].
    recovered_map_int = original_int_values.reshape(h, w)
    recovered_map_float = recovered_map_int / 255.0

    return recovered_map_float

def apply_adaptive_enhancement(mask):
    """LAO-X (unused, kept for reference): keep only responses within 80% of the
    maximum and stretch them back to [0, 1]."""
    max_val = mask.max()
    dynamic_threshold = max_val * 0.8
    mask_clipped = np.clip(mask, dynamic_threshold, max_val)
    if max_val - dynamic_threshold > 1e-5:
        mask_final = (mask_clipped - dynamic_threshold) / (max_val - dynamic_threshold)
    else:
        mask_final = mask_clipped * 0 # avoid division by zero: all-black map

    return mask_final


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    coco_images = []
    coco_annotations = []
    image_id = 0
    ann_id = 0
    if args.output and not os.path.exists(args.output):
        os.mkdir(args.output)
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            if args.musc_dir is not None:
                img_name = os.path.basename(path).split('.')[0]
                musc_path = os.path.join(args.musc_dir, img_name + '.png')
                musc_data = read_image(musc_path, format="BGR")
                # musc_heat = recover_heatmap_from_image(musc_path)
                if musc_data.shape[0] != img.shape[0] or musc_data.shape[1] != img.shape[1]:
                    musc_data = cv2.resize(musc_data, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                musc_heat = recover_heatmap_from_image(musc_data)
                musc_f = np.expand_dims(musc_heat, axis=-1)
                
                img_f = img.astype(np.float32) / 255.0
                musc_data = musc_data.astype(np.float32) / 255.0
                # musc_f = musc_data.astype(np.float32) / 255.0
                if args.fusion_mode == "mul":
                    fused = img_f * musc_f
                elif args.fusion_mode == "mul_mul":
                    fused = img_f * (musc_f ** 3)
                elif args.fusion_mode == "mul_mul_sigmoid":
                    musc_f = 1 / (1 + np.exp(-10 * (musc_f - 0.5)))
                    fused = img_f * (musc_f ** 3)
                elif args.fusion_mode == "mul_blend_mul":
                    musc_f = 1 / (1 + np.exp(-5 * (musc_f - 0.5)))
                    fused = (img_f * 0.5 + musc_f * 0.5)
                elif args.fusion_mode == "mul_blend_mul_v2":
                    # musc_f_enhance = apply_adaptive_enhancement(musc_f)
                    musc_max = musc_f.max()
                    musc_f_dyn_sharp = 1 / (
                        1 + np.exp(args.musc_sigmoid_coeff * (musc_f - musc_max * args.musc_sharp_temp))
                    )
                    musc_f_dyn_soft = 1 / (
                        1 + np.exp(args.musc_sigmoid_coeff * (musc_f - musc_max * args.musc_soft_temp))
                    )
                    musc_enhanced = (musc_f ** args.musc_power) * musc_f_dyn_soft
                    fused = (img_f + musc_data) * (
                        musc_enhanced / max(float(musc_enhanced.max()), args.musc_norm_eps)
                        + musc_f_dyn_sharp
                    ) / 4
                elif args.fusion_mode == "add":
                    fused = np.clip(img_f + musc_f, 0.0, 1.0)
                elif args.fusion_mode == "blend":
                    fused = np.clip(img_f + musc_f, 0.0, 1.0)
                else:  
                    fused = img_f
                img = (fused * 255.0).astype(np.uint8)
            fused_img = img.copy()
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if "instances" in predictions:
                instances = predictions["instances"].to("cpu")
                if instances.has("pred_masks"):
                    masks = instances.pred_masks.numpy()
                    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
                    scores = instances.scores.numpy() if instances.has("scores") else None
                    source_file_name = os.path.basename(path)
                    source_stem = os.path.splitext(source_file_name)[0]
                    try:
                        source_image_id = int(source_stem)
                    except ValueError:
                        source_image_id = image_id
                    coco_images.append(
                        {
                            "id": source_image_id,
                            "file_name": source_file_name,
                            "width": img.shape[1],
                            "height": img.shape[0],
                            "date_captured": datetime.datetime.utcnow().isoformat(" "),
                            "license": 1,
                            "coco_url": "",
                            "flickr_url": "",
                        }
                    )
                    for idx, m in enumerate(masks):
                        rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
                        rle["counts"] = rle["counts"].decode("ascii")
                        area = float(mask_util.area(rle))
                        if area < 1:
                            continue
                        bbox = mask_util.toBbox(rle).tolist()
                        coco_annotations.append(
                            {
                                "id": ann_id,
                                "image_id": source_image_id,
                                "category_id": int(classes[idx]) + 1 if classes is not None else 1,
                                "iscrowd": 0,
                                "area": area,
                                "bbox": bbox,
                                "segmentation": rle,
                                "score": float(scores[idx]) if scores is not None else 1.0,
                                "width": img.shape[1],
                                "height": img.shape[0],
                            }
                        )
                        ann_id += 1
                    image_id += 1

            if args.output:
                if args.vis_demo:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                    if os.path.isdir(args.output):
                        no_anno_dir = os.path.join(args.output, "no_anno")
                        os.makedirs(no_anno_dir, exist_ok=True)
                        raw_filename = os.path.join(no_anno_dir, os.path.basename(path))
                    else:
                        root, ext = os.path.splitext(args.output)
                        raw_filename = f"{root}_no_anno{ext or '.png'}"
                    cv2.imwrite(raw_filename, fused_img)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
        if coco_images:
            coco_output = args.coco_output
            if coco_output is None and args.output:
                if os.path.isdir(args.output):
                    coco_output = os.path.join(args.output, "annotations.json")
                else:
                    coco_output = os.path.splitext(args.output)[0] + ".json"
            if coco_output is not None:
                coco = {
                    "info": {
                        "description": "CutLER demo annotations",
                        "url": "",
                        "version": "1.0",
                        "year": datetime.datetime.utcnow().year,
                        "contributor": "",
                        "date_created": datetime.datetime.utcnow().isoformat(" "),
                    },
                    "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
                    "categories": [{"id": 1, "name": "fg", "supercategory": "fg"}],
                    "images": coco_images,
                    "annotations": coco_annotations,
                }
                with open(coco_output, "w") as f:
                    json.dump(coco, f)
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
