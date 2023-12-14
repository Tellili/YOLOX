#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
import cv2
import numpy as np
from tqdm import tqdm

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    return parser

def measure_inference_time(model_path, image_path, input_shape):
    start_time = time.time()
    origin_img = cv2.imread(image_path)
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(model_path)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    
    
    output = session.run(None, ort_inputs)
    

    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        # print(len(final_boxes))
    inference_time = time.time() - start_time
    return inference_time

if __name__ == '__main__':
    args = make_parser().parse_args()
    input_shape = tuple(map(int, args.input_shape.split(',')))
    path = "/data/haythem/openml/YOLOX/datasets/3k_dataset/data/"
    
    total_time = 0
    filenames = os.listdir(path)[:10]
    for filename in tqdm(filenames):
        image_path = path + filename
        inference_time = measure_inference_time(args.model, image_path, input_shape)
        total_time += inference_time

    print(f"Total time for inference on {len(filenames)} images: {total_time} seconds")