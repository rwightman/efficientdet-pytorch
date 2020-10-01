""" COCO dataset parser

Copyright 2020 Ross Wightman
"""
import numpy as np
from pycocotools.coco import COCO
from .parser import Parser
from .parser_config import CocoParserCfg


class CocoParser(Parser):

    def __init__(self, cfg: CocoParserCfg):
        super().__init__(
            bbox_yxyx=cfg.bbox_yxyx,
            has_labels=cfg.has_labels,
            include_masks=cfg.include_masks,
            include_bboxes_ignore=cfg.include_bboxes_ignore,
            ignore_empty_gt=cfg.has_labels and cfg.ignore_empty_gt,
            min_img_size=cfg.min_img_size
        )
        self.cat_ids_as_labels = True  # this is the default for original TF EfficientDet models
        self.coco = None
        self._load_annotations(cfg.ann_filename)

    def get_ann_info(self, idx):
        img_id = self.img_ids[idx]
        return self._parse_img_ann(img_id)

    def _load_annotations(self, ann_file):
        assert self.coco is None
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat_names = [c['name'] for c in self.coco.loadCats(ids=self.cat_ids)]
        if not self.cat_ids_as_labels:
            self.cat_id_to_label = {cat_id: i + self.label_offset for i, cat_id in enumerate(self.cat_ids)}
        img_ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for img_id in sorted(self.coco.imgs.keys()):
            info = self.coco.loadImgs([img_id])[0]
            if (min(info['width'], info['height']) < self.min_img_size or
                    (self.ignore_empty_gt and img_id not in img_ids_with_ann)):
                self.img_ids_invalid.append(img_id)
                continue
            self.img_ids.append(img_id)
            self.img_infos.append(info)

    def _parse_img_ann(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        bboxes = []
        bboxes_ignore = []
        cls = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if self.include_masks and ann['area'] <= 0:
                continue
            if w < 1 or h < 1:
                continue

            if self.yxyx:
                bbox = [y1, x1, y1 + h, x1 + w]
            else:
                bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                if self.include_bboxes_ignore:
                    bboxes_ignore.append(bbox)
            else:
                bboxes.append(bbox)
                cls.append(self.cat_id_to_label[ann['category_id']] if self.cat_id_to_label else ann['category_id'])

        if bboxes:
            bboxes = np.array(bboxes, ndmin=2, dtype=np.float32)
            cls = np.array(cls, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.array([], dtype=np.int64)

        if self.include_bboxes_ignore:
            if bboxes_ignore:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2, dtype=np.float32)
            else:
                bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(bbox=bboxes, cls=cls)

        if self.include_bboxes_ignore:
            ann['bbox_ignore'] = bboxes_ignore

        return ann
