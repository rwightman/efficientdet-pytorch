""" Detection dataset

Hacked together by Ross Wightman
"""
import torch.utils.data as data

from PIL import Image
from .parsers import create_parser


class DetectionDatset(data.Dataset):
    """`Object Detection Dataset. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, data_dir, parser=None, parser_kwargs=None, transform=None):
        super(DetectionDatset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.data_dir = data_dir
        if isinstance(parser, str):
            self.parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self.parser = parser
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_id = self.parser.img_ids[index]
        img_info = self.parser.img_infos[index]
        target = dict(img_id=img_id, img_size=(img_info['width'], img_info['height']))
        if self.parser.has_labels:
            ann = self.parser.get_ann_info(index)
            target.update(ann)

        img_path = self.data_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.parser.img_ids)
