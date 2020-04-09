import torch
from PIL import Image
import numpy as np

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


class ResizePad:

    def __init__(self, target_size: int, interpolation: str = 'bilinear'):
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, img, annotations: dict):
        width, height = img.size
        if height > width:
            scale = self.target_size / height
            scaled_height = self.target_size
            scaled_width = int(width * scale)
        else:
            scale = self.target_size / width
            scaled_height = int(height * scale)
            scaled_width = self.target_size

        new_img = Image.new("RGB", (self.target_size, self.target_size))
        interp_method = _pil_interp(self.interpolation)
        img = img.resize((scaled_width, scaled_height), interp_method)
        new_img.paste(img)

        if 'bbox' in annotations:
            # FIXME haven't tested this path since not currently using dataset annotations for train/eval
            bbox = annotations['bbox']
            bbox[:, :4] *= scale
            bbox = clip_boxes(bbox, (scaled_height, scaled_width))
            indices = np.where(np.sum(bbox, axis=1) != 0)[0]
            if len(indices) < len(bbox):
                bbox = np.take(bbox, indices)
                annotations['cls'] = np.take(annotations['cls'], indices)
            annotations['bbox'] = bbox

        annotations['scale'] = 1. / scale  # back to original

        return new_img, annotations


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
