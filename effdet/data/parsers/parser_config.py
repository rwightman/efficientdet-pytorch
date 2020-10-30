""" Dataset parser configs

Copyright 2020 Ross Wightman
"""
from dataclasses import dataclass

__all__ = ['CocoParserCfg', 'OpenImagesParserCfg', 'VocParserCfg']


@dataclass
class CocoParserCfg:
    ann_filename: str  # absolute path
    include_masks: bool = False
    include_bboxes_ignore: bool = False
    has_labels: bool = True
    bbox_yxyx: bool = True
    min_img_size: int = 32
    ignore_empty_gt: bool = False


@dataclass
class VocParserCfg:
    split_filename: str
    ann_filename: str
    img_filename: str = '%.jpg'
    keep_difficult: bool = True
    classes: list = None
    add_background: bool = True
    has_labels: bool = True
    bbox_yxyx: bool = True
    min_img_size: int = 32
    ignore_empty_gt: bool = False


@dataclass
class OpenImagesParserCfg:
    categories_filename: str
    img_info_filename: str
    bbox_filename: str
    img_label_filename: str = ''
    masks_filename: str = ''
    img_filename: str = '%s.jpg'  # relative to dataset img_dir
    task: str = 'obj'
    prefix_levels: int = 1
    add_background: bool = True
    has_labels: bool = True
    bbox_yxyx: bool = True
    min_img_size: int = 32
    ignore_empty_gt: bool = False
