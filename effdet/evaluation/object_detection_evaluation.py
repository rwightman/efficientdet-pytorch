import logging

import numpy as np

from effdet.evaluation.metrics import compute_precision_recall, compute_average_precision, compute_cor_loc
from effdet.evaluation.per_image_evaluation import PerImageEvaluation


class ObjectDetectionEvaluation:
    """Internal implementation of Pascal object detection metrics."""

    def __init__(self,
                 num_gt_classes,
                 matching_iou_threshold=0.5,
                 nms_iou_threshold=1.0,
                 nms_max_output_boxes=10000,
                 recall_lower_bound=0.0,
                 recall_upper_bound=1.0,
                 use_weighted_mean_ap=False,
                 label_id_offset=0,
                 group_of_weight=0.0,
                 per_image_eval_class=PerImageEvaluation):
        """Constructor.
        Args:
            num_gt_classes: Number of ground-truth classes.
            matching_iou_threshold: IOU threshold used for matching detected boxes to ground-truth boxes.
            nms_iou_threshold: IOU threshold used for non-maximum suppression.
            nms_max_output_boxes: Maximum number of boxes returned by non-maximum suppression.
            recall_lower_bound: lower bound of recall operating area
            recall_upper_bound: upper bound of recall operating area
            use_weighted_mean_ap: (optional) boolean which determines if the mean
                average precision is computed directly from the scores and tp_fp_labels of all classes.
            label_id_offset: The label id offset.
            group_of_weight: Weight of group-of boxes.If set to 0, detections of the
                correct class within a group-of box are ignored. If weight is > 0, then
                if at least one detection falls within a group-of box with
                matching_iou_threshold, weight group_of_weight is added to true
                positives. Consequently, if no detection falls within a group-of box,
                weight group_of_weight is added to false negatives.
            per_image_eval_class: The class that contains functions for computing per image metrics.
        Raises:
            ValueError: if num_gt_classes is smaller than 1.
        """
        if num_gt_classes < 1:
            raise ValueError('Need at least 1 groundtruth class for evaluation.')

        self.per_image_eval = per_image_eval_class(
            num_gt_classes=num_gt_classes,
            matching_iou_threshold=matching_iou_threshold,
            nms_iou_threshold=nms_iou_threshold,
            nms_max_output_boxes=nms_max_output_boxes,
            group_of_weight=group_of_weight)
        self.recall_lower_bound = recall_lower_bound
        self.recall_upper_bound = recall_upper_bound
        self.group_of_weight = group_of_weight
        self.num_class = num_gt_classes
        self.use_weighted_mean_ap = use_weighted_mean_ap
        self.label_id_offset = label_id_offset

        self.gt_boxes = {}
        self.gt_class_labels = {}
        self.gt_masks = {}
        self.gt_is_difficult_list = {}
        self.gt_is_group_of_list = {}
        self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=float)
        self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

        self._initialize_detections()

    def _initialize_detections(self):
        """Initializes internal data structures."""
        self.detection_keys = set()
        self.scores_per_class = [[] for _ in range(self.num_class)]
        self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
        self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
        self.average_precision_per_class = np.empty(self.num_class, dtype=float)
        self.average_precision_per_class.fill(np.nan)
        self.precisions_per_class = [np.nan] * self.num_class
        self.recalls_per_class = [np.nan] * self.num_class
        self.sum_tp_class = [np.nan] * self.num_class

        self.corloc_per_class = np.ones(self.num_class, dtype=float)

    def clear_detections(self):
        self._initialize_detections()

    def add_single_ground_truth_image_info(
            self, image_key, gt_boxes, gt_class_labels,
            gt_is_difficult_list=None, gt_is_group_of_list=None, gt_masks=None):
        """Adds groundtruth for a single image to be used for evaluation.
        Args:
            image_key: A unique string/integer identifier for the image.
            gt_boxes: float32 numpy array of shape [num_boxes, 4] containing
                `num_boxes` groundtruth boxes of the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
            gt_class_labels: integer numpy array of shape [num_boxes]
                containing 0-indexed groundtruth classes for the boxes.
            gt_is_difficult_list: A length M numpy boolean array denoting
                whether a ground truth box is a difficult instance or not. To support
                the case that no boxes are difficult, it is by default set as None.
            gt_is_group_of_list: A length M numpy boolean array denoting
                whether a ground truth box is a group-of box or not. To support the case
                that no boxes are groups-of, it is by default set as None.
            gt_masks: uint8 numpy array of shape [num_boxes, height, width]
                containing `num_boxes` groundtruth masks. The mask values range from 0 to 1.
        """
        if image_key in self.gt_boxes:
            logging.warning('image %s has already been added to the ground truth database.', image_key)
            return

        self.gt_boxes[image_key] = gt_boxes
        self.gt_class_labels[image_key] = gt_class_labels
        self.gt_masks[image_key] = gt_masks
        if gt_is_difficult_list is None:
            num_boxes = gt_boxes.shape[0]
            gt_is_difficult_list = np.zeros(num_boxes, dtype=bool)
        gt_is_difficult_list = gt_is_difficult_list.astype(dtype=bool)
        self.gt_is_difficult_list[image_key] = gt_is_difficult_list
        if gt_is_group_of_list is None:
            num_boxes = gt_boxes.shape[0]
            gt_is_group_of_list = np.zeros(num_boxes, dtype=bool)
        if gt_masks is None:
            num_boxes = gt_boxes.shape[0]
            mask_presence_indicator = np.zeros(num_boxes, dtype=bool)
        else:
            mask_presence_indicator = (np.sum(gt_masks, axis=(1, 2)) == 0).astype(dtype=bool)

        gt_is_group_of_list = gt_is_group_of_list.astype(dtype=bool)
        self.gt_is_group_of_list[image_key] = gt_is_group_of_list

        # ignore boxes without masks
        masked_gt_is_difficult_list = gt_is_difficult_list | mask_presence_indicator
        for class_index in range(self.num_class):
            num_gt_instances = np.sum(
                gt_class_labels[~masked_gt_is_difficult_list & ~gt_is_group_of_list] == class_index)
            num_groupof_gt_instances = self.group_of_weight * np.sum(
                gt_class_labels[gt_is_group_of_list & ~masked_gt_is_difficult_list] == class_index)
            self.num_gt_instances_per_class[class_index] += num_gt_instances + num_groupof_gt_instances
            if np.any(gt_class_labels == class_index):
                self.num_gt_imgs_per_class[class_index] += 1

    def add_single_detected_image_info(
            self, image_key, detected_boxes, detected_scores, detected_class_labels, detected_masks=None):
        """Adds detections for a single image to be used for evaluation.
        Args:
            image_key: A unique string/integer identifier for the image.
            detected_boxes: float32 numpy array of shape [num_boxes, 4] containing
                `num_boxes` detection boxes of the format [ymin, xmin, ymax, xmax] in
                absolute image coordinates.
            detected_scores: float32 numpy array of shape [num_boxes] containing
                detection scores for the boxes.
            detected_class_labels: integer numpy array of shape [num_boxes] containing
                0-indexed detection classes for the boxes.
            detected_masks: np.uint8 numpy array of shape [num_boxes, height, width]
                containing `num_boxes` detection masks with values ranging between 0 and 1.
        Raises:
            ValueError: if the number of boxes, scores and class labels differ in length.
        """
        if len(detected_boxes) != len(detected_scores) or len(detected_boxes) != len(detected_class_labels):
            raise ValueError(
                'detected_boxes, detected_scores and '
                'detected_class_labels should all have same lengths. Got'
                '[%d, %d, %d]' % len(detected_boxes), len(detected_scores),
                len(detected_class_labels))

        if image_key in self.detection_keys:
            logging.warning('image %s has already been added to the detection result database', image_key)
            return

        self.detection_keys.add(image_key)
        if image_key in self.gt_boxes:
            gt_boxes = self.gt_boxes[image_key]
            gt_class_labels = self.gt_class_labels[image_key]
            # Masks are popped instead of look up. The reason is that we do not want
            # to keep all masks in memory which can cause memory overflow.
            gt_masks = self.gt_masks.pop(image_key)
            gt_is_difficult_list = self.gt_is_difficult_list[image_key]
            gt_is_group_of_list = self.gt_is_group_of_list[image_key]
        else:
            gt_boxes = np.empty(shape=[0, 4], dtype=float)
            gt_class_labels = np.array([], dtype=int)
            if detected_masks is None:
                gt_masks = None
            else:
                gt_masks = np.empty(shape=[0, 1, 1], dtype=float)
            gt_is_difficult_list = np.array([], dtype=bool)
            gt_is_group_of_list = np.array([], dtype=bool)
        scores, tp_fp_labels, is_class_correctly_detected_in_image = \
            self.per_image_eval.compute_object_detection_metrics(
                detected_boxes=detected_boxes,
                detected_scores=detected_scores,
                detected_class_labels=detected_class_labels,
                gt_boxes=gt_boxes,
                gt_class_labels=gt_class_labels,
                gt_is_difficult_list=gt_is_difficult_list,
                gt_is_group_of_list=gt_is_group_of_list,
                detected_masks=detected_masks,
                gt_masks=gt_masks)

        for i in range(self.num_class):
            if scores[i].shape[0] > 0:
                self.scores_per_class[i].append(scores[i])
                self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
        self.num_images_correctly_detected_per_class += is_class_correctly_detected_in_image

    def evaluate(self):
        """Compute evaluation result.
        Returns:
            A dict with the following fields -
                average_precision: float numpy array of average precision for each class.
                mean_ap: mean average precision of all classes, float scalar
                precisions: List of precisions, each precision is a float numpy array
                recalls: List of recalls, each recall is a float numpy array
                corloc: numpy float array
                mean_corloc: Mean CorLoc score for each class, float scalar
        """
        if (self.num_gt_instances_per_class == 0).any():
            logging.warning(
                'The following classes have no ground truth examples: %s',
                np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0)) + self.label_id_offset)

        if self.use_weighted_mean_ap:
            all_scores = np.array([], dtype=float)
            all_tp_fp_labels = np.array([], dtype=bool)
        for class_index in range(self.num_class):
            if self.num_gt_instances_per_class[class_index] == 0:
                continue
            if not self.scores_per_class[class_index]:
                scores = np.array([], dtype=float)
                tp_fp_labels = np.array([], dtype=float)
            else:
                scores = np.concatenate(self.scores_per_class[class_index])
                tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
            if self.use_weighted_mean_ap:
                all_scores = np.append(all_scores, scores)
                all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
            precision, recall = compute_precision_recall(
                scores, tp_fp_labels, self.num_gt_instances_per_class[class_index])
            recall_within_bound_indices = [
                index for index, value in enumerate(recall) if
                value >= self.recall_lower_bound and value <= self.recall_upper_bound
            ]
            recall_within_bound = recall[recall_within_bound_indices]
            precision_within_bound = precision[recall_within_bound_indices]

            self.precisions_per_class[class_index] = precision_within_bound
            self.recalls_per_class[class_index] = recall_within_bound
            self.sum_tp_class[class_index] = tp_fp_labels.sum()
            average_precision = compute_average_precision(precision_within_bound, recall_within_bound)
            self.average_precision_per_class[class_index] = average_precision
            logging.debug('average_precision: %f', average_precision)

        self.corloc_per_class = compute_cor_loc(
            self.num_gt_imgs_per_class, self.num_images_correctly_detected_per_class)

        if self.use_weighted_mean_ap:
            num_gt_instances = np.sum(self.num_gt_instances_per_class)
            precision, recall = compute_precision_recall(all_scores, all_tp_fp_labels, num_gt_instances)
            recall_within_bound_indices = [
                index for index, value in enumerate(recall) if
                value >= self.recall_lower_bound and value <= self.recall_upper_bound
            ]
            recall_within_bound = recall[recall_within_bound_indices]
            precision_within_bound = precision[recall_within_bound_indices]
            mean_ap = compute_average_precision(precision_within_bound, recall_within_bound)
        else:
            mean_ap = np.nanmean(self.average_precision_per_class)
        mean_corloc = np.nanmean(self.corloc_per_class)

        return dict(
            per_class_ap=self.average_precision_per_class, mean_ap=mean_ap,
            per_class_precision=self.precisions_per_class,
            per_class_recall=self.recalls_per_class,
            per_class_corlocs=self.corloc_per_class, mean_corloc=mean_corloc)
