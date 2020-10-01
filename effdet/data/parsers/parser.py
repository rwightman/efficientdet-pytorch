from numbers import Integral
from typing import List, Union, Dict, Any


class Parser:
    """ Parser base class.

    The attributes listed below make up a public interface common to all parsers. They can be accessed directly
    once the dataset is constructed and annotations are populated.

    Attributes:

        cat_names (list[str]):
            list of category (class) names, with background class at position 0.
        cat_ids (list[union[str, int]):
            list of dataset specific, unique integer or string category ids, does not include background
        cat_id_to_label (dict):
            map from category id to integer 1-indexed class label

        img_ids (list):
            list of dataset specific, unique image ids corresponding to valid samples in dataset
        img_ids_invalid (list):
            list of image ids corresponding to invalid images, not used as samples
        img_infos (list[dict]):
            image info, list of info dicts with filename, width, height for each image sample
    """
    def __init__(
            self,
            bbox_yxyx: bool = False,
            has_labels: bool = True,
            include_masks: bool = False,
            include_bboxes_ignore: bool = False,
            ignore_empty_gt: bool = False,
            min_img_size: int = 32,
    ):
        """
        Args:
            yxyx (bool): output coords in yxyx format, otherwise xyxy
            has_labels (bool): dataset has labels (for training validation, False usually for test sets)
            include_masks (bool): include segmentation masks in target output (not supported yet for any dataset)
            include_bboxes_ignore (bool): include ignored bbox in target output
            ignore_empty_gt (bool): ignore images with no ground truth (no negative images)
            min_img_size (bool): ignore images with width or height smaller than this number
            sub_sample (int): sample every N images from the dataset
        """
        # parser config, determines how dataset parsed and validated
        self.yxyx = bbox_yxyx
        self.has_labels = has_labels
        self.include_masks = include_masks
        self.include_bboxes_ignore = include_bboxes_ignore
        self.ignore_empty_gt = ignore_empty_gt
        self.min_img_size = min_img_size
        self.label_offset = 1

        # Category (class) metadata. Populated by _load_annotations()
        self.cat_names: List[str] = []
        self.cat_ids: List[Union[str, Integral]] = []
        self.cat_id_to_label: Dict[Union[str, Integral], Integral] = dict()

        # Image metadata. Populated by _load_annotations()
        self.img_ids: List[Union[str, Integral]] = []
        self.img_ids_invalid: List[Union[str, Integral]] = []
        self.img_infos: List[Dict[str, Any]] = []

    @property
    def cat_dicts(self):
        """return category names and labels in format compatible with TF Models Evaluator
        list[dict(name=<class name>, id=<class label>)]
        """
        return [
            dict(
                name=name,
                id=cat_id if not self.cat_id_to_label else self.cat_id_to_label[cat_id]
            ) for name, cat_id in zip(self.cat_names, self.cat_ids)]

    @property
    def max_label(self):
        if self.cat_id_to_label:
            return max(self.cat_id_to_label.values())
        else:
            assert len(self.cat_ids) and isinstance(self.cat_ids[0], Integral)
            return max(self.cat_ids)
