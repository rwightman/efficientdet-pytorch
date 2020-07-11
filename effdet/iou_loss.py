'''
Based on:
 https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/86a370aa2cadea6ba7e5dffb2efc4bacc4c863ea/utils/box/box_utils.py#L47

 Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression
 https://arxiv.org/pdf/1911.08287.pdf
 Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression
 https://giou.stanford.edu/GIoU.pdf
 UnitBox: An Advanced Object Detection Network
 https://arxiv.org/pdf/1608.01471.pdf

 Important!!! (in case of c_iou_loss)
 targets -> bboxes1, preds -> bboxes2
 '''

import torch
from torch import nn
import numpy as np

eps = 10e-16


def compute_iou(bboxes1, bboxes2):
    "bboxes1 of shape [N, 4] and bboxes2 of shape [N, 4]"
    assert bboxes1.size(0) == bboxes2.size(0)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    min_x2 = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    max_x1 = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    min_y2 = torch.min(bboxes1[:, 3], bboxes2[:, 3])
    max_y1 = torch.max(bboxes1[:, 1], bboxes2[:, 1])

    inter = torch.where(min_x2 - max_x1 > 0, min_x2 - max_x1, torch.tensor(0.)) * \
            torch.where(min_y2 - max_y1 > 0, min_y2 - max_y1, torch.tensor(0.))
    union = area1 + area2 - inter
    iou = inter / union
    iou = torch.clamp(iou, min=0, max=1.0)
    return iou


def compute_g_iou(bboxes1, bboxes2):
    "box1 of shape [N, 4] and box2 of shape [N, 4]"
    #assert bboxes1.size(0) == bboxes2.size(0)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    min_x2 = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    max_x1 = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    min_y2 = torch.min(bboxes1[:, 3], bboxes2[:, 3])
    max_y1 = torch.max(bboxes1[:, 1], bboxes2[:, 1])
    inter = torch.clamp(min_x2 - max_x1, min=0) * torch.clamp(min_y2 - max_y1, min=0)
    union = area1 + area2 - inter
    C = (torch.max(bboxes1[:, 2], bboxes2[:, 2]) - torch.min(bboxes1[:, 0], bboxes2[:, 0])) * \
        (torch.max(bboxes1[:, 3], bboxes2[:, 3]) - torch.min(bboxes1[:, 1], bboxes2[:, 1]))
    g_iou = inter / union - (C - union) / C
    g_iou = torch.clamp(g_iou, min=0, max=1.0)
    return g_iou


def compute_d_iou(bboxes1, bboxes2):
    "bboxes1 of shape [N, 4] and bboxes2 of shape [N, 4]"
    #assert bboxes1.size(0) == bboxes2.size(0)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    min_x2 = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    max_x1 = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    min_y2 = torch.min(bboxes1[:, 3], bboxes2[:, 3])
    max_y1 = torch.max(bboxes1[:, 1], bboxes2[:, 1])
    inter = torch.clamp(min_x2 - max_x1, min=0) * torch.clamp(min_y2 - max_y1, min=0)
    union = area1 + area2 - inter
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    # squared euclidian distance between the target and predicted bboxes
    d_2 = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
    # squared length of the diagonal of the minimum bbox that encloses both bboxes
    c_2 = (torch.max(bboxes1[:, 2], bboxes2[:, 2]) - torch.min(bboxes1[:, 0], bboxes2[:, 0])) ** 2 + (
            torch.max(bboxes1[:, 3], bboxes2[:, 3]) - torch.min(bboxes1[:, 1], bboxes2[:, 1])) ** 2
    d_iou = inter / union - d_2 / c_2
    d_iou = torch.clamp(d_iou, min=-1.0, max=1.0)

    return d_iou


def compute_c_iou(bboxes1, bboxes2):
    "bboxes1 of shape [N, 4] and bboxes2 of shape [N, 4]"
    #assert bboxes1.size(0) == bboxes2.size(0)
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    area1 = w1 * h1
    area2 = w2 * h2
    min_x2 = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    max_x1 = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    min_y2 = torch.min(bboxes1[:, 3], bboxes2[:, 3])
    max_y1 = torch.max(bboxes1[:, 1], bboxes2[:, 1])

    inter = torch.clamp(min_x2 - max_x1, min=0) * torch.clamp(min_y2 - max_y1, min=0)
    union = area1 + area2 - inter

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
    # squared euclidian distance between the target and predicted bboxes
    d_2 = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
    # squared length of the diagonal of the minimum bbox that encloses both bboxes
    c_2 = (torch.max(bboxes1[:, 2], bboxes2[:, 2]) - torch.min(bboxes1[:, 0], bboxes2[:, 0])) ** 2 + (
            torch.max(bboxes1[:, 3], bboxes2[:, 3]) - torch.min(bboxes1[:, 1], bboxes2[:, 1])) ** 2
    iou = inter / union
    v = 4 / np.pi ** 2 * (np.arctan(w1 / h1) - np.arctan(w2 / h2)) ** 2
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v + eps)
    c_iou = iou - (d_2 / c_2 + alpha * v)
    c_iou = torch.clamp(c_iou, min=-1.0, max=1.0)
    return c_iou



