""" Dataset factory

Copyright 2020 Ross Wightman
"""
import os
from collections import OrderedDict
from pathlib import Path

from .dataset_config import *
from .parsers import *
from .dataset import DetectionDatset
from .parsers import create_parser


def create_dataset(name, root, splits=('train', 'val')):
    if isinstance(splits, str):
        splits = (splits,)
    name = name.lower()
    root = Path(root)
    dataset_cls = DetectionDatset
    datasets = OrderedDict()
    if name.startswith('coco'):
        if 'coco2014' in name:
            dataset_cfg = Coco2014Cfg()
        else:
            dataset_cfg = Coco2017Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                data_dir=root / Path(split_cfg['img_dir']),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )
    elif name.startswith('voc'):
        if 'voc0712' in name:
            dataset_cfg = Voc0712Cfg()
        elif 'voc2007' in name:
            dataset_cfg = Voc2007Cfg()
        else:
            dataset_cfg = Voc2012Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            if isinstance(split_cfg['split_filename'], (tuple, list)):
                assert len(split_cfg['split_filename']) == len(split_cfg['ann_filename'])
                parser = None
                for sf, af, id in zip(
                        split_cfg['split_filename'], split_cfg['ann_filename'], split_cfg['img_dir']):
                    parser_cfg = VocParserCfg(
                        split_filename=root / sf,
                        ann_filename=os.path.join(root, af),
                        img_filename=os.path.join(id, dataset_cfg.img_filename))
                    if parser is None:
                        parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
                    else:
                        other_parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
                        parser.merge(other=other_parser)
            else:
                parser_cfg = VocParserCfg(
                    split_filename=root / split_cfg['split_filename'],
                    ann_filename=os.path.join(root, split_cfg['ann_filename']),
                    img_filename=os.path.join(split_cfg['img_dir'], dataset_cfg.img_filename),
                )
                parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
            datasets[s] = dataset_cls(data_dir=root, parser=parser)
    elif name.startswith('openimages'):
        if 'challenge2019' in name:
            dataset_cfg = OpenImagesObjChallenge2019Cfg()
        else:
            dataset_cfg = OpenImagesObjV5Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            parser_cfg = OpenImagesParserCfg(
                categories_filename=root / dataset_cfg.categories_map,
                img_info_filename=root / split_cfg['img_info'],
                bbox_filename=root / split_cfg['ann_bbox'],
                img_label_filename=root / split_cfg['ann_img_label'],
                img_filename=dataset_cfg.img_filename,
                prefix_levels=split_cfg['prefix_levels'],
                has_labels=split_cfg['has_labels'],
            )
            datasets[s] = dataset_cls(
                data_dir=root / Path(split_cfg['img_dir']),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg)
            )
    else:
        assert False, f'Unknown dataset parser ({name})'

    datasets = list(datasets.values())
    return datasets if len(datasets) > 1 else datasets[0]
