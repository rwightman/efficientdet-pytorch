""" COCO transforms (quick and dirty)

Hacked together by Ross Wightman
"""
import torch
from PIL import Image
import numpy as np
import random
import math

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


class ImageToNumpy:

    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img, annotations


class ImageToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype), annotations


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


def clip_boxes(boxes, img_size):
    height, width = img_size
    boxes = np.where(boxes < 0, np.zeros_like(boxes), boxes)
    boxes[:, 2] = np.where(boxes[:, 2] > height, (height - 1) * np.ones_like(boxes[:, 2]), boxes[:, 2])
    boxes[:, 3] = np.where(boxes[:, 3] > width, (width - 1) * np.ones_like(boxes[:, 3]), boxes[:, 3])
    return boxes


def clip_boxes_remove_empty(boxes, classes, img_size):
    boxes = clip_boxes(boxes, img_size)
    indices = np.where(np.sum(boxes, axis=1) != 0)[0]
    if len(indices) < len(boxes):
        boxes = boxes[indices, :]
        classes = classes[indices]
    return boxes, classes


def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size


class ResizePad:

    def __init__(self, target_size: int, interpolation: str = 'bilinear'):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation

    def __call__(self, img, anno: dict):
        width, height = img.size

        img_scale_y = self.target_size[0] / height
        img_scale_x = self.target_size[1] / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]))
        interp_method = _pil_interp(self.interpolation)
        img = img.resize((scaled_w, scaled_h), interp_method)
        new_img.paste(img)

        if 'bbox' in anno:
            # FIXME haven't tested this path since not currently using dataset annotations for train/eval
            bbox = anno['bbox']
            bbox[:, :4] *= img_scale
            anno['bbox'], anno['cls'] = clip_boxes_remove_empty(bbox, anno['cls'], (scaled_h, scaled_w))

        anno['scale'] = 1. / img_scale  # back to original

        return new_img, anno


class RandomResizePad:

    def __init__(self, target_size: int, scale: tuple = (0.1, 2.0), interpolation: str = 'bilinear',
                 fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.scale = scale
        self.interpolation = interpolation
        self.fill_color = fill_color

    def _get_params(self, img):
        # Select a random scale factor.
        scale_factor = random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]

        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.size
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * random.uniform(0, 1))
        return scaled_h, scaled_w, offset_y, offset_x, img_scale

    def __call__(self, img, anno: dict):
        scaled_h, scaled_w, offset_y, offset_x, img_scale = self._get_params(img)

        interp_method = _pil_interp(self.interpolation)
        img = img.resize((scaled_w, scaled_h), interp_method)
        right, lower = min(scaled_w, offset_x + self.target_size[1]), min(scaled_h, offset_y + self.target_size[0])
        img = img.crop((offset_x, offset_y, right, lower))
        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
        new_img.paste(img)

        if 'bbox' in anno:
            # FIXME haven't tested this path since not currently using dataset annotations for train/eval
            bbox = anno['bbox']
            bbox[:, :4] *= img_scale
            box_offset = np.stack([offset_y, offset_x] * 2)
            bbox -= box_offset
            anno['bbox'], anno['cls'] = clip_boxes_remove_empty(bbox, anno['cls'], (scaled_h, scaled_w))

        anno['scale'] = 1. / img_scale  # back to original

        return new_img, anno


class RandomFlip:

    def __init__(self, horizontal=True, vertical=False, prob=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.prob = prob

    def _get_params(self):
        do_horizontal = random.random() < self.prob if self.horizontal else False
        do_vertical = random.random() < self.prob if self.vertical else False
        return do_horizontal, do_vertical

    def __call__(self, img, annotations: dict):
        do_horizontal, do_vertical = self._get_params()
        width, height = img.size

        if do_horizontal and do_vertical:
            img = img.transpose(Image.ROTATE_180)
            if 'bbox' in annotations:
                bbox = annotations['bbox']
                bbox[:, 0] = height - bbox[:, 0]
                bbox[:, 2] = height - bbox[:, 2]
                bbox[:, 1] = width - bbox[:, 1]
                bbox[:, 3] = width - bbox[:, 3]
                annotations['bbox'] = bbox
        elif do_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if 'bbox' in annotations:
                bbox = annotations['bbox']
                bbox[:, 1] = width - bbox[:, 1]
                bbox[:, 3] = width - bbox[:, 3]
                annotations['bbox'] = bbox
        elif do_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if 'bbox' in annotations:
                bbox = annotations['bbox']
                bbox[:, 0] = height - bbox[:, 0]
                bbox[:, 2] = height - bbox[:, 2]
                annotations['bbox'] = bbox

        return img, annotations


class Compose:

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, annotations: dict):
        for t in self.transforms:
            img, annotations = t(img, annotations)
        return img, annotations


def transforms_coco_eval(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

    image_tfl = [
        ResizePad(target_size=img_size, interpolation=interpolation),
        ImageToNumpy(),
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf


def transforms_coco_train(
        img_size=224,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

    image_tfl = [
        RandomFlip(horizontal=True, prob=0.5),
        RandomResizePad(
            target_size=img_size, interpolation=interpolation, fill_color=tuple([int(round(255 * x)) for x in mean])),
        ImageToNumpy(),
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf
