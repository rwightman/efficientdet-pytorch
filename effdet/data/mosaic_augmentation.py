import random
import math
import torch
import numpy as np
import cv2
from PIL import Image
import os


class MosaicAugmentation:
    """Randomly selects 3 other training images to form a 4 tiled image, with
        different aspect ratios. Original implementation from
        https://github.com/ultralytics/yolov5
        with slight changes to fit this codebase.
    Args:
        index: Current image to be transformed
        parser: Parser to obtain annotations from images
        degrees: image rotation (deg)
        translate: image translation (deg)
        scale: image scale
        shear: image shear
        perspective: image perspective(+/- fraction)
    """

    def __init__(
        self,
        index,
        parser,
        data_dir,
        img_size=640,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
    ):
        self.index = index
        self._parser = parser
        self.data_dir = data_dir
        self.img_size = img_size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        self.indices = range(0, len(self._parser.img_ids))
        self.augment = True

    def __call__(self):
        # loads images in a 4-mosaic
        annotations4 = []
        s = self.img_size
        yc, xc = [
            int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border
        ]  # mosaic center x, y
        indices = [self.index] + random.choices(
            self.indices, k=3
        )  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = load_image(self, index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full(
                    (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
                )  # base image with 4 tiles
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = (
                    xc,
                    yc,
                    min(xc + w, s * 2),
                    min(s * 2, yc + h),
                )
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[
                y1b:y2b, x1b:x2b
            ]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            ann = self._parser.get_ann_info(index)
            annotations = np.zeros((len(ann["bbox"]), 5))
            # print("Ann Bbox", ann['bbox'])
            # print("ann cls", ann['cls'])
            for idx, bbox in enumerate(ann["bbox"]):
                bb = list(bbox.copy())
                cls = [ann["cls"][idx]]
                bb = cls + bb
                annotations[idx] = bb

            if self._parser.yxyx:
                annotations[:, 0:5] = annotations[:, [0, 2, 1, 4, 3]]  # to xyxy

            if annotations.size:
                annotations[:, 1:] = add_padding(annotations[:, 1:], padw, padh)
            annotations4.append(annotations)

        # Concat/clip annotations
        annotations4 = np.concatenate(annotations4, 0)
        for x in annotations4[:, 1:]:
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, annotations4 = replicate(img4, annotations4)  # replicate

        # Augment
        img4, annotations4 = random_perspective(
            img4,
            annotations4,
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            border=self.mosaic_border,
        )  # border to remove

        annotations4[:, 0:5] = (
            annotations4[:, [0, 2, 1, 4, 3]]
            if self._parser.yxyx
            else annotations4
        )
        classes = annotations4[:, 0]
        bboxes = annotations4[:, 1:5]
        img = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
        img_pil4 = Image.fromarray(img)
        return img_pil4, bboxes, classes, self.img_size


def load_image(self, index):
    img_info = self._parser.img_infos[index]
    img_path = self.data_dir / img_info["file_name"]
    img = cv2.imread(os.fsdecode(img_path))  # BGR
    assert img is not None, "Image Not Found " + img_path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        img = cv2.resize(
            img,
            (int(w0 * r), int(h0 * r)),
            interpolation=cv2.INTER_AREA
            if r < 1 and not self.augment
            else cv2.INTER_LINEAR,
        )
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized


def box_candidates(
    box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16
):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment,
    # wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + eps) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def add_padding(x, padw, padh):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] + padw
    y[:, 1] = x[:, 1] + padh
    y[:, 2] = x[:, 2] + padw
    y[:, 3] = x[:, 3] + padh
    return y


def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1)
    # , scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(
        -perspective, perspective
    )  # x perspective (about y)
    P[2, 1] = random.uniform(
        -perspective, perspective
    )  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(
        random.uniform(-shear, shear) * math.pi / 180
    )  # x shear (deg)
    S[1, 0] = math.tan(
        random.uniform(-shear, shear) * math.pi / 180
    )  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (
        (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any()
    ):  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(
            n, 8
        )  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = (
            np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)))
            .reshape(4, n)
            .T
        )

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(
            box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.10
        )
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets
