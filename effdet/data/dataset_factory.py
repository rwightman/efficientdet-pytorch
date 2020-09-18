import os
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
    img_dirs = []
    parsers = []
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
            img_dirs.append(root / Path(split_cfg['img_dir']))
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
            )
            parsers.append(create_parser(dataset_cfg.parser, cfg=parser_cfg))
    elif name.startswith('voc'):
        if 'VOC2007' in name:
            dataset_cfg = Voc2007Cfg()
        else:
            dataset_cfg = Voc2012Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            img_dirs.append(root / Path(split_cfg['img_dir']))
            parser_cfg = VocParserCfg(
                split_filename=root / split_cfg['split_filename'],
                ann_template=os.path.join(root, dataset_cfg.ann_template),
            )
            parsers.append(create_parser(dataset_cfg.parser, cfg=parser_cfg))
    elif name.startswith('openimages'):
        if 'challenge2019' in name:
            dataset_cfg = OpenImagesObjChallenge2019Cfg()
        else:
            dataset_cfg = OpenImagesObjV5Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            img_dirs.append(root / Path(split_cfg['img_dir']))
            parser_cfg = OpenImagesParserCfg(
                categories_filename=root / dataset_cfg.categories_map,
                img_info_filename=root / split_cfg['img_info'],
                bbox_filename=root / split_cfg['ann_bbox'],
                img_label_filename=root / split_cfg['ann_img_label'],
                img_template=dataset_cfg.img_template,
                prefix_levels=split_cfg['prefix_levels']

            )
            parsers.append(create_parser(dataset_cfg.parser, cfg=parser_cfg))
    else:
        assert False, f'Unknown dataset parser ({name})'

    datasets = [DetectionDatset(img_dir, parser=parser) for img_dir, parser in zip(img_dirs, parsers)]
    return datasets if len(datasets) > 1 else datasets[0]
