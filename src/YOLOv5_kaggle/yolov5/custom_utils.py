# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from PIL import Image
import ast
import matplotlib.pyplot as plt
import torch
from WBF.ensemble_boxes.ensemble_boxes_nms import nms
from WBF.ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion as wbf

def plot_bboxes(img_path, img, df, bboxes, tracked_points=None):
    if img is None:
        img = np.array(Image.open(img_path))
    img_w, img_h = img.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.imshow(img)
    if df is not None:
        gt_bboxes = df.query("img_path == @img_path")["annotations"].values[0]
        for i in range(len(gt_bboxes)):
            x_min, y_min, width, height = (
                gt_bboxes[i]["x"],
                gt_bboxes[i]["y"],
                gt_bboxes[i]["width"],
                gt_bboxes[i]["height"],
            )
            rect = plt.Rectangle(
                [x_min, y_min], width, height, ec="b", fc="none", lw=3.0
            )
            ax.add_patch(rect)
    for i in range(len(bboxes)):
        if len(bboxes[i].split()) == 5:
            bbox = bboxes[i].split()[1:]
            conf = np.round(float(bboxes[i].split()[0]), 2)
        else:
            bbox = bboxes[i].split()
            conf = 1
        x_min, y_min, width, height = list(map(int, bbox))
        rect = plt.Rectangle([x_min, y_min], width, height, ec="r", fc="none", lw=1.5)
        ax.add_patch(rect)
        ax.annotate(
            f"{conf}",
            xy=(x_min, y_min - 5),
            size=15,
            color="r",
        )
    if tracked_points is not None:
        for i in range(len(tracked_points)):
            points = tracked_points[i][:2]
            point_id = tracked_points[i][2]
            ax.plot([points[0]], [points[1]], color="purple", marker="o")
            ax.annotate(
                f"{point_id}",
                xy=(points[0] - 3, points[1] - 5),
                size=15,
                color="purple",
            )
    plt.show();

def get_ensemble(MODELS, iou):
    MODEL_PATHS = [x[0] for x in MODELS]
    IMG_SIZES = [x[1] for x in MODELS]
    CONF_THRESHS = [x[2] for x in MODELS]
    models = []
    for i in range(len(MODEL_PATHS)):
        model = torch.hub.load(
            "../input/yolov5reef/yolov5",
            "custom",
            path=MODEL_PATHS[i],
            source="local",
            force_reload=True,
        )
        model.conf = CONF_THRESHS[i]
        model.iou = iou
        models.append(model)
    return models, MODEL_PATHS, IMG_SIZES, CONF_THRESHS

def ensemble_predict(
    models,
    img_sizes,
    img,
    augment,
    skip_box_thr,
    iou_thres,
    final_thres,
    conf_type,
    add_prediction = None,
    mod_weights=None,
    WBF=True,
):
    all_scores = []
    all_bboxes = []
    all_classes = []
    final_bboxes = []
    xy_for_track = []
    weights = []
    for i in range(len(models)):
        img_size = img_sizes[i]
        prediction = (
            models[i](img, size=img_sizes[i], augment=augment[i]).xyxyn[0].cpu().numpy()
        )
        if prediction.size != 0:
            conf = prediction[:, 4]
            bboxes_array = prediction[:, :4]
            bboxes_array = np.clip(bboxes_array, 0, 1)
            all_bboxes.append(bboxes_array)
            all_scores.append(conf)
            all_classes.append(["0"] * bboxes_array.shape[0])
            if mod_weights is not None:
                weights.append(mod_weights[i])
    if add_prediction is not None:
        all_bboxes.append(add_prediction[0])
        all_scores.append(add_prediction[1])
        all_classes.append(["0"] * add_prediction[1].shape[0])
    if len(all_bboxes):
        if len(weights) == 0:
            weights = None
        if WBF:
            boxes, scores, _ = wbf(
                boxes_list=all_bboxes,
                scores_list=all_scores,
                labels_list=all_classes,
                weights=weights,
                iou_thr=iou_thres,
                skip_box_thr=skip_box_thr,
                conf_type=conf_type,
            )
        else:
            boxes, scores, _ = nms(
                boxes=all_bboxes,
                scores=all_scores,
                labels=all_classes,
                iou_thr=iou_thres,
                weights=weights,
            )
        boxes = boxes[scores > final_thres]
        scores = scores[scores > final_thres]
        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box
            width = (xmax - xmin) * 1280
            height = (ymax - ymin) * 720
            xmin *= 1280
            ymin *= 720
            xmin, ymin, width, height = map(int, [xmin, ymin, width, height])
            xy_for_track.append([int(xmin+width/2), int(ymin + height/2), width, height, score])
            final_bboxes.append(
                " ".join(list(map(str, [score, xmin, ymin, width, height])))
            )
    return final_bboxes, xy_for_track

