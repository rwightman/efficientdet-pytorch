""" OpenImages dataset parser

Copyright 2020 Ross Wightman
"""
import numpy as np
import os
import logging

from .parser import Parser
from .parser_config import OpenImagesParserCfg

_logger = logging.getLogger(__name__)


class OpenImagesParser(Parser):

    def __init__(self, cfg: OpenImagesParserCfg):
        super().__init__(
            bbox_yxyx=cfg.bbox_yxyx,
            has_labels=cfg.has_labels,
            include_masks=False,  # FIXME to support someday
            include_bboxes_ignore=False,
            ignore_empty_gt=cfg.has_labels and cfg.ignore_empty_gt,
            min_img_size=cfg.min_img_size
        )
        self.img_prefix_levels = cfg.prefix_levels
        self.mask_prefix_levels = 1
        self._anns = None  # access via get_ann_info()
        self._img_to_ann = None
        self._load_annotations(
            categories_filename=cfg.categories_filename,
            img_info_filename=cfg.img_info_filename,
            img_filename=cfg.img_filename,
            masks_filename=cfg.masks_filename,
            bbox_filename=cfg.bbox_filename
        )

    def _load_annotations(
            self,
            categories_filename: str,
            img_info_filename: str,
            img_filename: str,
            masks_filename: str,
            bbox_filename: str,
    ):
        import pandas as pd  # For now, blow up on pandas req only when trying to load open images anno

        _logger.info('Loading categories...')
        classes_df = pd.read_csv(categories_filename, header=None)
        self.cat_ids = classes_df[0].tolist()
        self.cat_names = classes_df[1].tolist()
        self.cat_id_to_label = {c: i + self.label_offset for i, c in enumerate(self.cat_ids)}

        def _img_filename(img_id):
            # build image filenames that are relative to img_dir
            filename = img_filename % img_id
            if self.img_prefix_levels:
                levels = [c for c in img_id[:self.img_prefix_levels]]
                filename = os.path.join(*levels, filename)
            return filename

        def _mask_filename(mask_path):
            # FIXME finish
            if self.mask_prefix_levels:
                levels = [c for c in mask_path[:self.mask_prefix_levels]]
                mask_path = os.path.join(*levels, mask_path)
            return mask_path

        def _load_img_info(csv_file, select_img_ids=None):
            _logger.info('Read img_info csv...')
            img_info_df = pd.read_csv(csv_file, index_col='id')

            _logger.info('Filter images...')
            if select_img_ids is not None:
                img_info_df = img_info_df.loc[select_img_ids]
            img_info_df = img_info_df[
                (img_info_df['width'] >= self.min_img_size) & (img_info_df['height'] >= self.min_img_size)]

            _logger.info('Mapping ids...')
            img_info_df['img_id'] = img_info_df.index
            img_info_df['file_name'] = img_info_df.index.map(lambda x: _img_filename(x))
            img_info_df = img_info_df[['img_id', 'file_name', 'width', 'height']]
            img_sizes = img_info_df[['width', 'height']].values
            self.img_infos = img_info_df.to_dict('records')
            self.img_ids = img_info_df.index.values.tolist()
            img_id_to_idx = {img_id: idx for idx, img_id in enumerate(self.img_ids)}
            return img_sizes, img_id_to_idx

        if self.include_masks and self.has_labels:
            masks_df = pd.read_csv(masks_filename)

            # NOTE currently using dataset masks anno ImageIDs to form valid img_ids from the dataset
            anno_img_ids = sorted(masks_df['ImageID'].unique())
            img_sizes, img_id_to_idx = _load_img_info(img_info_filename, select_img_ids=anno_img_ids)

            masks_df['ImageIdx'] = masks_df['ImageID'].map(img_id_to_idx)
            if np.issubdtype(masks_df.ImageIdx.dtype, np.floating):
                masks_df = masks_df.dropna(axis='rows')
                masks_df['ImageIdx'] = masks_df.ImageIdx.astype(np.int32)
            masks_df.sort_values('ImageIdx', inplace=True)
            ann_img_idx = masks_df['ImageIdx'].values
            img_sizes = img_sizes[ann_img_idx]
            masks_df['BoxXMin'] = masks_df['BoxXMin'] * img_sizes[:, 0]
            masks_df['BoxXMax'] = masks_df['BoxXMax'] * img_sizes[:, 0]
            masks_df['BoxYMin'] = masks_df['BoxYMin'] * img_sizes[:, 1]
            masks_df['BoxYMax'] = masks_df['BoxYMax'] * img_sizes[:, 1]
            masks_df['LabelIdx'] = masks_df['LabelName'].map(self.cat_id_to_label)
            # FIXME remap mask filename with _mask_filename

            self._anns = dict(
                bbox=masks_df[['BoxXMin', 'BoxYMin', 'BoxXMax', 'BoxYMax']].values.astype(np.float32),
                label=masks_df[['LabelIdx']].values.astype(np.int32),
                mask_path=masks_df[['MaskPath']].values
            )
            _, ri, rc = np.unique(ann_img_idx, return_index=True, return_counts=True)
            self._img_to_ann = list(zip(ri, rc))  # index, count tuples
        elif self.has_labels:
            _logger.info('Loading bbox...')
            bbox_df = pd.read_csv(bbox_filename)

            # NOTE currently using dataset box anno ImageIDs to form valid img_ids from the larger dataset.
            # FIXME use *imagelabels.csv or imagelabels-boxable.csv for negative examples (without box?)
            anno_img_ids = sorted(bbox_df['ImageID'].unique())
            img_sizes, img_id_to_idx = _load_img_info(img_info_filename, select_img_ids=anno_img_ids)

            _logger.info('Process bbox...')
            bbox_df['ImageIdx'] = bbox_df['ImageID'].map(img_id_to_idx)
            if np.issubdtype(bbox_df.ImageIdx.dtype, np.floating):
                bbox_df = bbox_df.dropna(axis='rows')
                bbox_df['ImageIdx'] = bbox_df.ImageIdx.astype(np.int32)
            bbox_df.sort_values('ImageIdx', inplace=True)
            ann_img_idx = bbox_df['ImageIdx'].values
            img_sizes = img_sizes[ann_img_idx]
            bbox_df['XMin'] = bbox_df['XMin'] * img_sizes[:, 0]
            bbox_df['XMax'] = bbox_df['XMax'] * img_sizes[:, 0]
            bbox_df['YMin'] = bbox_df['YMin'] * img_sizes[:, 1]
            bbox_df['YMax'] = bbox_df['YMax'] * img_sizes[:, 1]
            bbox_df['LabelIdx'] = bbox_df['LabelName'].map(self.cat_id_to_label).astype(np.int32)

            self._anns = dict(
                bbox=bbox_df[['XMin', 'YMin', 'XMax', 'YMax']].values.astype(np.float32),
                label=bbox_df[['LabelIdx', 'IsGroupOf']].values.astype(np.int32),
            )
            _, ri, rc = np.unique(ann_img_idx, return_index=True, return_counts=True)
            self._img_to_ann = list(zip(ri, rc))  # index, count tuples
        else:
            _load_img_info(img_info_filename)

        _logger.info('Annotations loaded!')

    def get_ann_info(self, idx):
        if not self.has_labels:
            return dict()
        start_idx, num_ann = self._img_to_ann[idx]
        ann_keys = tuple(self._anns.keys())
        ann_values = tuple(self._anns[k][start_idx:start_idx + num_ann] for k in ann_keys)
        return self._parse_ann_info(idx, ann_keys, ann_values)

    def _parse_ann_info(self, img_idx, ann_keys, ann_values):
        """
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        if self.include_masks:
            assert 'mask_path' in ann_keys
            gt_masks = []

        for ann in zip(*ann_values):
            ann = dict(zip(ann_keys, ann))
            x1, y1, x2, y2 = ann['bbox']
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            label = ann['label'][0]
            iscrowd = False
            if len(ann['label']) > 1:
                iscrowd = ann['label'][1]
            if self.yxyx:
                bbox = np.array([y1, x1, y2, x2], dtype=np.float32)
            else:
                bbox = ann['bbox']
            if iscrowd:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(label)
            # if self.include_masks:
            #     img_info = self.img_infos[img_idx]
            #     mask_img = SegmentationMask(ann['mask_filename'], img_info['width'], img_info['height'])
            #     gt_masks.append(mask_img)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, ndmin=2, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if self.include_bboxes_ignore:
            if gt_bboxes_ignore:
                gt_bboxes_ignore = np.array(gt_bboxes_ignore, ndmin=2, dtype=np.float32)
            else:
                gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(bbox=gt_bboxes, cls=gt_labels)

        if self.include_bboxes_ignore:
            ann.update(dict(bbox_ignore=gt_bboxes_ignore, cls_ignore=np.array([], dtype=np.int64)))
        if self.include_masks:
            ann['masks'] = gt_masks
        return ann
