""" Pascal VOC dataset parser

Copyright 2020 Ross Wightman
"""
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

from .parser_config import VocParserCfg


class VocParser:

    DEFAULT_CLASSES = (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, cfg: VocParserCfg):
        self.yxyx = cfg.bbox_yxyx
        self.has_labels = cfg.has_labels
        self.keep_difficult = cfg.keep_difficult
        self.include_bboxes_ignore = False
        self.min_size = cfg.min_size
        self.correct_bbox = 1

        self.ann_template = cfg.ann_template  # absolute file path template
        self.img_template = cfg.img_template  # relative to img_dir

        classes = cfg.classes or self.DEFAULT_CLASSES
        self.cat_ids = []
        self.cat_to_label = {cat: i + 1 for i, cat in enumerate(classes)}
        self.img_ids = []
        self.img_ids_invalid = []
        self.img_infos = []

        self._anns = None
        self._img_to_ann = None
        self._load_annotations(cfg.split_filename)

    def _load_annotations(self, split_filename):

        with open(split_filename) as f:
            ids = f.readlines()
        self.img_ids = [x.strip("\n") for x in ids]
        self.img_id_to_idx = {k: v for k, v in enumerate(self.img_ids)}
        self._img_to_ann = defaultdict(list)
        self._anns = []

        for img_idx, img_id in enumerate(self.img_ids):
            filename = self.img_template % img_id
            xml_path = self.ann_template % img_id
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            self.img_infos.append(dict(id=img_id, file_name=filename, width=width, height=height))

            for obj_idx, obj in enumerate(root.findall('object')):
                name = obj.find('name').text
                label = self.cat_to_label[name]
                difficult = int(obj.find('difficult').text)
                bnd_box = obj.find('bndbox')
                bbox = [
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)
                ]
                self._anns.append(dict(img_idx=img_idx, label=label, bbox=bbox, difficult=difficult))
                self._img_to_ann[img_idx].append(obj_idx)

    def get_ann_info(self, idx):
        ann_indices = self._img_to_ann[idx]
        ann_info = [self._anns[i] for i in ann_indices]
        return self._parse_ann_info(ann_info)

    def _parse_ann_info(self, ann_info):
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for ann in ann_info:
            ignore = False
            x1, y1, x2, y2 = ann['bbox']
            label = ann['label']
            if self.min_size:
                w = x2 - x1
                h = y2 - y1
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if self.yxyx:
                bbox = [y1, x1, y2, x2]
            else:
                bbox = ann['bbox']
            if ann['difficult'] or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        if not bboxes:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0, ), dtype=np.float32)
        else:
            bboxes = (np.array(bboxes, ndmin=2, dtype=np.float32) - 1) #.clip(min=0)
            labels = np.array(labels, dtype=np.float32)

        if self.include_bboxes_ignore:
            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
                labels_ignore = np.zeros((0, ), dtype=np.float32)
            else:
                bboxes_ignore = (np.array(bboxes_ignore, ndmin=2, dtype=np.float32) - 1) #.clip(min=0)
                labels_ignore = np.array(labels_ignore, dtype=np.float32)

        ann = dict(
            bbox=bboxes.astype(np.float32),
            cls=labels.astype(np.int64))

        if self.include_bboxes_ignore:
            ann.update(dict(
                bbox_ignore=bboxes_ignore.astype(np.float32),
                cls_ignore=labels_ignore.astype(np.int64)))
        return ann

