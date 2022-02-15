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

# +

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression
from WBF.ensemble_boxes.ensemble_boxes_nms import nms, soft_nms
from WBF.ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion


# -

def check_xy(xmin, ymin, width, height, img_w, img_h):
    xmin = xmin if xmin >= 0 else 0
    ymin = ymin if ymin >= 0 else 0
    width = width if (xmin + width) <= img_w else img_w - xmin
    height = height if (ymin + height) <= img_h else img_h - ymin
    return xmin, ymin, width, height


def get_model(model_path, device):
    device = torch.device(device)
    half = device.type != "cpu"
    model = attempt_load(
        model_path,
        map_location=device,
    )
    if half:
        model.half()
    model.eval()
    with torch.no_grad():
        zero_img = torch.zeros((1, 3, 1280, 1280), device=device)  # init img
        _ = model(zero_img.half() if half else zero_img)[0].cpu().detach().numpy()
    return model, half, device


def bottom_letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, 128), np.mod(dh, 128)  # wh padding
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = 0, int(round(dh + 0.1))
    left, right = 0, 0
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def load_image(path, device, img=None, img_size=2560, half=True):
    if img is None:
        img = cv2.imread(path)  # BGR
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h0, w0 = img.shape[:2]  # orig hw

    img, r, (dw, dh) = bottom_letterbox(img, new_shape=img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device, non_blocking=True)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, h0, w0, int(dh)

def y4predict(img, h0, w0, dh, model, conf_thres, iou_thr, augment, method="nms"):
    h1 = img.shape[2] - dh
    w1 = img.shape[3]
    final_bboxes = []
    list_bboxes = []
    all_scores = []
    all_bboxes = []
    with torch.no_grad():
        prediction = model(img, augment=augment)[0].cpu().detach().numpy()[0]
    prediction = prediction[prediction[:, 4] > conf_thres]
    if prediction.size:
        conf = prediction[:, 4] * prediction[:, 5]
        bboxes_array = np.zeros(prediction[:, :4].shape)
        bboxes_array[:, 0] = (prediction[:, 0] - prediction[:, 2] / 2) / w1
        bboxes_array[:, 1] = (prediction[:, 1] - prediction[:, 3] / 2) / h1
        bboxes_array[:, 2] = (prediction[:, 0] + prediction[:, 2] / 2) / w1
        bboxes_array[:, 3] = (prediction[:, 1] + prediction[:, 3] / 2) / h1
        bboxes_array = np.clip(bboxes_array, 0, 1)
        all_bboxes.append(bboxes_array)
        all_scores.append(conf)
        classes = [["0"] * bboxes_array.shape[0]]
        if method == "nms":
            boxes, scores, _ = nms(
                boxes=all_bboxes,
                scores=all_scores,
                labels=classes,
                iou_thr=iou_thr,
                weights=None,
            )
        elif method == "soft_nms":
            boxes, scores, _ = soft_nms(
                boxes=all_bboxes,
                scores=all_scores,
                labels=classes,
                method=2,
                iou_thr=iou_thr,
                sigma=0.5,
                thresh=conf_thres,
                weights=None,
            )
        for b, s in zip(boxes, scores):
            xmin, ymin, xmax, ymax = b
            width = (xmax - xmin) * w0
            height = (ymax - ymin) * h0
            xmin *= w0
            ymin *= h0
            xmin, ymin, width, height = map(int, [xmin, ymin, width, height])
            list_bboxes.append([s, xmin, ymin, width, height])
            final_bboxes.append(
                " ".join(list(map(str, [s, xmin, ymin, width, height])))
            )
    return final_bboxes, list_bboxes, (boxes, scores)



def get_bbox_from_resized(
    img, model, conf_thres, iou_thres, skip_box_thr, h0, w0, conf_type, method="nms_yolo"
):
    h1, w1 = img.shape[2:]
    final_bboxes = []
    with torch.no_grad():
        prediction = model(img)[0].cpu().detach()
    if method == "nms_yolo":
        pred_yolo = non_max_suppression(
            prediction=prediction,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            merge=False,
            classes=None,
            agnostic=False,
        )[0].numpy()
        for xmin, ymin, xmax, ymax, score, _ in pred_yolo:
            width = ((xmax - xmin) / w1) * w0
            height = ((ymax - ymin) / h1) * w0
            xmin = (xmin / w1) * w0
            ymin = (ymin / h1) * h0
            xmin, ymin, width, height = list(map(int, [xmin, ymin, width, height]))
            xmin, ymin, width, height = check_xy(xmin, ymin, width, height, w0, h0)
            final_bboxes.append(" ".join(map(str, [score, xmin, ymin, width, height])))
    else:
        scores = []
        bboxes = []
        prediction = prediction[0].numpy()
        prediction = prediction[prediction[:, 4] > conf_thres]
        conf = prediction[:, 4] * prediction[:, 5]
        bbox = prediction[:, :4].copy()
        bboxes_array = np.zeros(bbox.shape)
        bboxes_array[:, 0] = (bbox[:, 0] - bbox[:, 2] / 2) / w1
        bboxes_array[:, 1] = (bbox[:, 1] - bbox[:, 3] / 2) / h1
        bboxes_array[:, 2] = (bbox[:, 0] + bbox[:, 2] / 2) / w1
        bboxes_array[:, 3] = (bbox[:, 1] + bbox[:, 3] / 2) / h1
        bboxes_array = np.clip(bboxes_array, 0, 1)
        bboxes.append(bboxes_array)
        scores.append(conf)
        classes = [["0"] * bboxes_array.shape[0]]

        if method == "wbf":
            boxes, scores, _ = weighted_boxes_fusion(
                boxes_list=bboxes,
                scores_list=scores,
                labels_list=classes,
                weights=None,
                iou_thr=iou_thres,
                skip_box_thr=skip_box_thr,
                conf_type=conf_type,
            )
        elif method == "nms":
            boxes, scores, _ = nms(
                boxes=bboxes,
                scores=scores,
                labels=classes,
                iou_thr=iou_thr,
                weights=None,
            )
        for b, s in zip(boxes, scores):
            xmin, ymin, xmax, ymax = b
            width = (xmax - xmin) * w0
            height = (ymax - ymin) * h0
            xmin *= w0
            ymin *= h0
            xmin, ymin, width, height = map(int, [xmin, ymin, width, height])
            final_bboxes.append(
                " ".join(list(map(str, [s, xmin, ymin, width, height])))
            )
    return final_bboxes


def get_bbox_pad(
    img,
    models,
    conf_thres,
    iou_thres,
    skip_box_thr,
    h0,
    w0,
    conf_type,
    method="nms_yolo",
):
    h1, w1 = img.shape[2:]
    final_bboxes = []
    predictions = []
    for model in models:
        with torch.no_grad():
            single_prediction = model(img)[0].cpu().detach()
            predictions.append(single_prediction)
    if (
        torch.sum(
            torch.tensor([torch.sum(x[0][:, 4] > conf_thres) for x in predictions]) > 0
        )
        > len(models) // 2
    ):
        prediction = torch.cat([x for x in predictions], 1)
        if method == "nms_yolo":
            pred_yolo = non_max_suppression(
                prediction=prediction,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                merge=False,
                classes=None,
                agnostic=False,
            )[0].numpy()
            for xmin, ymin, xmax, ymax, score, _ in pred_yolo:
                width = xmax - xmin
                height = ymax - ymin
                xmin, ymin, width, height = list(map(int, [xmin, ymin, width, height]))
                xmin, ymin, width, height = check_xy(xmin, ymin, width, height, w0, h0)
                final_bboxes.append(
                    " ".join(map(str, [score, xmin, ymin, width, height]))
                )
        else:
            scores = []
            bboxes = []
            prediction = prediction[0].numpy()
            prediction = prediction[prediction[:, 4] > conf_thres]
            conf = prediction[:, 4] * prediction[:, 5]
            bbox = prediction[:, :4].copy()
            bboxes_array = np.zeros(bbox.shape)
            bboxes_array[:, 0] = (bbox[:, 0] - bbox[:, 2] / 2) / w0
            bboxes_array[:, 1] = (bbox[:, 1] - bbox[:, 3] / 2) / h0
            bboxes_array[:, 2] = (bbox[:, 0] + bbox[:, 2] / 2) / w0
            bboxes_array[:, 3] = (bbox[:, 1] + bbox[:, 3] / 2) / h0
            bboxes_array = np.clip(bboxes_array, 0, 1)
            bboxes.append(bboxes_array)
            scores.append(conf)
            classes = [["0"] * bboxes_array.shape[0]]

            if method == "wbf":
                boxes, scores, _ = weighted_boxes_fusion(
                    boxes_list=bboxes,
                    scores_list=scores,
                    labels_list=classes,
                    weights=None,
                    iou_thr=iou_thres,
                    skip_box_thr=skip_box_thr,
                    conf_type=conf_type,
                )
            elif method == "nms":
                boxes, scores, _ = nms(
                    boxes=bboxes,
                    scores=scores,
                    labels=classes,
                    iou_thr=iou_thres,
                    weights=None,
                )
            for b, s in zip(boxes, scores):
                xmin, ymin, xmax, ymax = b
                width = (xmax - xmin) * w0
                height = (ymax - ymin) * h0
                xmin *= w0
                ymin *= h0
                xmin, ymin, width, height = map(int, [xmin, ymin, width, height])
                final_bboxes.append(
                    " ".join(list(map(str, [s, xmin, ymin, width, height])))
                )
    return final_bboxes

def plot_bboxes(img_path, img, df, bboxes, bbox_format="coco"):
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
            rect = plt.Rectangle([x_min, y_min], width, height, ec="b", fc="none", lw=3.0)
            ax.add_patch(rect)
    if bbox_format == "coco":
        for i in range(len(bboxes)):
            if len(bboxes[i].split()) == 5:
                bbox = bboxes[i].split()[1:]
                conf = np.round(float(bboxes[i].split()[0]), 2)
            else:
                bbox = bboxes[i].split()
                conf = 1
            x_center, y_center, width, height = list(map(float, bbox))
            x_min = int((x_center - width / 2) * img_w)
            y_min = int((y_center - height / 2) * img_h)
            width = int(width * img_w)
            height = int(height * img_h)
            rect = plt.Rectangle(
                [x_min, y_min], width, height, ec="r", fc="none", lw=2.0
            )
            ax.add_patch(rect)
            ax.annotate(
                f"{conf}",
                xy=(x_min, y_min - 5),
                size=15,
                color="r",
            )
    elif bbox_format == "from_yolo":
        for i in range(len(bboxes)):
            if len(bboxes[i].split()) == 5:
                bbox = bboxes[i].split()[1:]
                conf = np.round(float(bboxes[i].split()[0]), 2)
            else:
                bbox = bboxes[i].split()
                conf = 1
            x_min, y_min, width, height = list(map(int, bbox))
            rect = plt.Rectangle(
                [x_min, y_min], width, height, ec="r", fc="none", lw=1.5
            )
            ax.add_patch(rect)
            ax.annotate(
                f"{conf}",
                xy=(x_min, y_min - 5),
                size=15,
                color="r",
            )
    plt.show();


