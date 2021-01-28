from abc import ABCMeta
from abc import abstractmethod
#import collections
import logging
import unicodedata
import numpy as np

from .fields import InputDataFields, DetectionResultFields
from .object_detection_evaluation import ObjectDetectionEvaluation


def create_category_index(categories):
    """Creates dictionary of COCO compatible categories keyed by category id.
    Args:
        categories: a list of dicts, each of which has the following keys:
            'id': (required) an integer id uniquely identifying this category.
            'name': (required) string representing category name e.g., 'cat', 'dog', 'pizza'.
    Returns:
        category_index: a dict containing the same entries as categories, but keyed
            by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


class DetectionEvaluator(metaclass=ABCMeta):
    """Interface for object detection evalution classes.
    Example usage of the Evaluator:
    ------------------------------
    evaluator = DetectionEvaluator(categories)
    # Detections and groundtruth for image 1.
    evaluator.add_single_gt_image_info(...)
    evaluator.add_single_detected_image_info(...)
    # Detections and groundtruth for image 2.
    evaluator.add_single_gt_image_info(...)
    evaluator.add_single_detected_image_info(...)
    metrics_dict = evaluator.evaluation()
    """

    def __init__(self, categories):
        """Constructor.
        Args:
          categories: A list of dicts, each of which has the following keys -
            'id': (required) an integer id uniquely identifying this category.
            'name': (required) string representing category name e.g., 'cat', 'dog'.
        """
        self._categories = categories

    def observe_result_dict_for_single_example(self, eval_dict):
        """Observes an evaluation result dict for a single example.
        When executing eagerly, once all observations have been observed by this
        method you can use `.evaluation()` to get the final metrics.
        When using `tf.estimator.Estimator` for evaluation this function is used by
        `get_estimator_eval_metric_ops()` to construct the metric update op.
        Args:
            eval_dict: A dictionary that holds tensors for evaluating an object
                detection model, returned from
                eval_util.result_dict_for_single_example().
        Returns:
            None when executing eagerly, or an update_op that can be used to update
            the eval metrics in `tf.estimator.EstimatorSpec`.
        """
        raise NotImplementedError('Not implemented for this evaluator!')

    @abstractmethod
    def add_single_ground_truth_image_info(self, image_id, gt_dict):
        """Adds groundtruth for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            gt_dict: A dictionary of groundtruth numpy arrays required for evaluations.
        """
        pass

    @abstractmethod
    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            detections_dict: A dictionary of detection numpy arrays required for evaluation.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluates detections and returns a dictionary of metrics."""
        pass

    @abstractmethod
    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        pass


class ObjectDetectionEvaluator(DetectionEvaluator):
    """A class to evaluation detections."""

    def __init__(self,
                 categories,
                 matching_iou_threshold=0.5,
                 recall_lower_bound=0.0,
                 recall_upper_bound=1.0,
                 evaluate_corlocs=False,
                 evaluate_precision_recall=False,
                 metric_prefix=None,
                 use_weighted_mean_ap=False,
                 evaluate_masks=False,
                 group_of_weight=0.0):
        """Constructor.
        Args:
            categories: A list of dicts, each of which has the following keys -
                'id': (required) an integer id uniquely identifying this category.
                'name': (required) string representing category name e.g., 'cat', 'dog'.
            matching_iou_threshold: IOU threshold to use for matching groundtruth boxes to detection boxes.
            recall_lower_bound: lower bound of recall operating area.
            recall_upper_bound: upper bound of recall operating area.
            evaluate_corlocs: (optional) boolean which determines if corloc scores are to be returned or not.
            evaluate_precision_recall: (optional) boolean which determines if
                precision and recall values are to be returned or not.
            metric_prefix: (optional) string prefix for metric name; if None, no prefix is used.
            use_weighted_mean_ap: (optional) boolean which determines if the mean
                average precision is computed directly from the scores and tp_fp_labels of all classes.
            evaluate_masks: If False, evaluation will be performed based on boxes. If
                True, mask evaluation will be performed instead.
            group_of_weight: Weight of group-of boxes.If set to 0, detections of the
                correct class within a group-of box are ignored. If weight is > 0, then
                if at least one detection falls within a group-of box with
                matching_iou_threshold, weight group_of_weight is added to true
                positives. Consequently, if no detection falls within a group-of box,
                weight group_of_weight is added to false negatives.
        Raises:
            ValueError: If the category ids are not 1-indexed.
        """
        super(ObjectDetectionEvaluator, self).__init__(categories)
        self._num_classes = max([cat['id'] for cat in categories])
        if min(cat['id'] for cat in categories) < 1:
            raise ValueError('Classes should be 1-indexed.')
        self._matching_iou_threshold = matching_iou_threshold
        self._recall_lower_bound = recall_lower_bound
        self._recall_upper_bound = recall_upper_bound
        self._use_weighted_mean_ap = use_weighted_mean_ap
        self._label_id_offset = 1
        self._evaluate_masks = evaluate_masks
        self._group_of_weight = group_of_weight
        self._evaluation = ObjectDetectionEvaluation(
            num_gt_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            recall_lower_bound=self._recall_lower_bound,
            recall_upper_bound=self._recall_upper_bound,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset,
            group_of_weight=self._group_of_weight)
        self._image_ids = set([])
        self._evaluate_corlocs = evaluate_corlocs
        self._evaluate_precision_recall = evaluate_precision_recall
        self._metric_prefix = (metric_prefix + '_') if metric_prefix else ''
        self._build_metric_names()

    def _build_metric_names(self):
        """Builds a list with metric names."""
        if self._recall_lower_bound > 0.0 or self._recall_upper_bound < 1.0:
            self._metric_names = [
                self._metric_prefix + 'Precision/mAP@{}IOU@[{:.1f},{:.1f}]Recall'.format(
                    self._matching_iou_threshold, self._recall_lower_bound, self._recall_upper_bound)
            ]
        else:
            self._metric_names = [
                self._metric_prefix + 'Precision/mAP@{}IOU'.format(self._matching_iou_threshold)
            ]
        if self._evaluate_corlocs:
            self._metric_names.append(
                self._metric_prefix + 'Precision/meanCorLoc@{}IOU'.format(self._matching_iou_threshold))

        category_index = create_category_index(self._categories)
        for idx in range(self._num_classes):
            if idx + self._label_id_offset in category_index:
                category_name = category_index[idx + self._label_id_offset]['name']
                category_name = unicodedata.normalize('NFKD', category_name)
                self._metric_names.append(
                    self._metric_prefix + 'PerformanceByCategory/AP@{}IOU/{}'.format(
                        self._matching_iou_threshold, category_name))
                if self._evaluate_corlocs:
                    self._metric_names.append(
                        self._metric_prefix + 'PerformanceByCategory/CorLoc@{}IOU/{}'.format(
                            self._matching_iou_threshold, category_name))

    def add_single_ground_truth_image_info(self, image_id, gt_dict):
        """Adds groundtruth for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            gt_dict: A dictionary containing -
                InputDataFields.gt_boxes: float32 numpy array
                    of shape [num_boxes, 4] containing `num_boxes` groundtruth boxes of
                    the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
                InputDataFields.gt_classes: integer numpy array
                    of shape [num_boxes] containing 1-indexed groundtruth classes for the boxes.
                InputDataFields.gt_difficult: Optional length M numpy boolean array
                    denoting whether a ground truth box is a difficult instance or not.
                    This field is optional to support the case that no boxes are difficult.
                InputDataFields.gt_instance_masks: Optional numpy array of shape
                    [num_boxes, height, width] with values in {0, 1}.
        Raises:
            ValueError: On adding groundtruth for an image more than once. Will also
                raise error if instance masks are not in groundtruth dictionary.
        """
        if image_id in self._image_ids:
            return

        gt_classes = gt_dict[InputDataFields.gt_classes] - self._label_id_offset
        # If the key is not present in the gt_dict or the array is empty
        # (unless there are no annotations for the groundtruth on this image)
        # use values from the dictionary or insert None otherwise.
        if (InputDataFields.gt_difficult in gt_dict and
                (gt_dict[InputDataFields.gt_difficult].size or not gt_classes.size)):
            gt_difficult = gt_dict[InputDataFields.gt_difficult]
        else:
            gt_difficult = None
            # FIXME disable difficult flag warning, will support flag eventually
            # if not len(self._image_ids) % 1000:
            #     logging.warning('image %s does not have groundtruth difficult flag specified', image_id)
        gt_masks = None
        if self._evaluate_masks:
            if InputDataFields.gt_instance_masks not in gt_dict:
                raise ValueError('Instance masks not in groundtruth dictionary.')
            gt_masks = gt_dict[InputDataFields.gt_instance_masks]
        self._evaluation.add_single_ground_truth_image_info(
            image_key=image_id,
            gt_boxes=gt_dict[InputDataFields.gt_boxes],
            gt_class_labels=gt_classes,
            gt_is_difficult_list=gt_difficult,
            gt_masks=gt_masks)
        self._image_ids.update([image_id])

    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            detections_dict: A dictionary containing -
                DetectionResultFields.detection_boxes: float32 numpy
                    array of shape [num_boxes, 4] containing `num_boxes` detection boxes
                    of the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
                DetectionResultFields.detection_scores: float32 numpy
                    array of shape [num_boxes] containing detection scores for the boxes.
                DetectionResultFields.detection_classes: integer numpy
                    array of shape [num_boxes] containing 1-indexed detection classes for the boxes.
                DetectionResultFields.detection_masks: uint8 numpy array
                    of shape [num_boxes, height, width] containing `num_boxes` masks of
                    values ranging between 0 and 1.
        Raises:
            ValueError: If detection masks are not in detections dictionary.
        """
        detection_classes = detections_dict[DetectionResultFields.detection_classes] - self._label_id_offset
        detection_masks = None
        if self._evaluate_masks:
            if DetectionResultFields.detection_masks not in detections_dict:
                raise ValueError('Detection masks not in detections dictionary.')
            detection_masks = detections_dict[DetectionResultFields.detection_masks]
        self._evaluation.add_single_detected_image_info(
            image_key=image_id,
            detected_boxes=detections_dict[DetectionResultFields.detection_boxes],
            detected_scores=detections_dict[DetectionResultFields.detection_scores],
            detected_class_labels=detection_classes,
            detected_masks=detection_masks)

    def evaluate(self):
        """Compute evaluation result.
        Returns:
          A dictionary of metrics with the following fields -
          1. summary_metrics:
                '<prefix if not empty>_Precision/mAP@<matching_iou_threshold>IOU': mean
                average precision at the specified IOU threshold.
          2. per_category_ap: category specific results with keys of the form
                '<prefix if not empty>_PerformanceByCategory/
                mAP@<matching_iou_threshold>IOU/category'.
        """
        metrics = self._evaluation.evaluate()
        pascal_metrics = {self._metric_names[0]: metrics['mean_ap']}
        if self._evaluate_corlocs:
            pascal_metrics[self._metric_names[1]] = metrics['mean_corloc']
        category_index = create_category_index(self._categories)
        for idx in range(metrics['per_class_ap'].size):
            if idx + self._label_id_offset in category_index:
                category_name = category_index[idx + self._label_id_offset]['name']
                category_name = unicodedata.normalize('NFKD', category_name)
                display_name = self._metric_prefix + 'PerformanceByCategory/AP@{}IOU/{}'.format(
                    self._matching_iou_threshold, category_name)
                pascal_metrics[display_name] = metrics['per_class_ap'][idx]

                # Optionally add precision and recall values
                if self._evaluate_precision_recall:
                    display_name = self._metric_prefix + 'PerformanceByCategory/Precision@{}IOU/{}'.format(
                        self._matching_iou_threshold, category_name)
                    pascal_metrics[display_name] = metrics['per_class_precision'][idx]
                    display_name = self._metric_prefix + 'PerformanceByCategory/Recall@{}IOU/{}'.format(
                        self._matching_iou_threshold, category_name)
                    pascal_metrics[display_name] = metrics['per_class_recall'][idx]

                # Optionally add CorLoc metrics.classes
                if self._evaluate_corlocs:
                    display_name = self._metric_prefix + 'PerformanceByCategory/CorLoc@{}IOU/{}'.format(
                        self._matching_iou_threshold, category_name)
                    pascal_metrics[display_name] = metrics['per_class_corloc'][idx]

        return pascal_metrics

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._evaluation = ObjectDetectionEvaluation(
            num_gt_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset)
        self._image_ids.clear()


class PascalDetectionEvaluator(ObjectDetectionEvaluator):
    """A class to evaluation detections using PASCAL metrics."""

    def __init__(self, categories, matching_iou_threshold=0.5):
        super(PascalDetectionEvaluator, self).__init__(
            categories,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            metric_prefix='PascalBoxes',
            use_weighted_mean_ap=False)


class WeightedPascalDetectionEvaluator(ObjectDetectionEvaluator):
    """A class to evaluation detections using weighted PASCAL metrics.
    Weighted PASCAL metrics computes the mean average precision as the average
    precision given the scores and tp_fp_labels of all classes. In comparison,
    PASCAL metrics computes the mean average precision as the mean of the
    per-class average precisions.
    This definition is very similar to the mean of the per-class average
    precisions weighted by class frequency. However, they are typically not the
    same as the average precision is not a linear function of the scores and
    tp_fp_labels.
    """

    def __init__(self, categories, matching_iou_threshold=0.5):
        super(WeightedPascalDetectionEvaluator, self).__init__(
            categories,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            metric_prefix='WeightedPascalBoxes',
            use_weighted_mean_ap=True)


class PrecisionAtRecallDetectionEvaluator(ObjectDetectionEvaluator):
    """A class to evaluation detections using precision@recall metrics."""

    def __init__(self,
                 categories,
                 matching_iou_threshold=0.5,
                 recall_lower_bound=0.,
                 recall_upper_bound=1.0):
        super(PrecisionAtRecallDetectionEvaluator, self).__init__(
            categories,
            matching_iou_threshold=matching_iou_threshold,
            recall_lower_bound=recall_lower_bound,
            recall_upper_bound=recall_upper_bound,
            evaluate_corlocs=False,
            metric_prefix='PrecisionAtRecallBoxes',
            use_weighted_mean_ap=False)


class OpenImagesDetectionEvaluator(ObjectDetectionEvaluator):
    """A class to evaluation detections using Open Images V2 metrics.
      Open Images V2 introduce group_of type of bounding boxes and this metric
      handles those boxes appropriately.
    """

    def __init__(self,
                 categories,
                 matching_iou_threshold=0.5,
                 evaluate_masks=False,
                 evaluate_corlocs=False,
                 metric_prefix='OpenImagesV5',
                 group_of_weight=0.0):
        """Constructor.
        Args:
            categories: A list of dicts, each of which has the following keys -
                'id': (required) an integer id uniquely identifying this category.
                'name': (required) string representing category name e.g., 'cat', 'dog'.
            matching_iou_threshold: IOU threshold to use for matching groundtruth
                boxes to detection boxes.
            evaluate_masks: if True, evaluator evaluates masks.
            evaluate_corlocs: if True, additionally evaluates and returns CorLoc.
            metric_prefix: Prefix name of the metric.
            group_of_weight: Weight of the group-of bounding box. If set to 0 (default
                for Open Images V2 detection protocol), detections of the correct class
                within a group-of box are ignored. If weight is > 0, then if at least
                one detection falls within a group-of box with matching_iou_threshold,
                weight group_of_weight is added to true positives. Consequently, if no
                detection falls within a group-of box, weight group_of_weight is added
                to false negatives.
        """

        super(OpenImagesDetectionEvaluator, self).__init__(
            categories,
            matching_iou_threshold,
            evaluate_corlocs,
            metric_prefix=metric_prefix,
            group_of_weight=group_of_weight,
            evaluate_masks=evaluate_masks)

    def add_single_ground_truth_image_info(self, image_id, gt_dict):
        """Adds groundtruth for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            gt_dict: A dictionary containing -
                InputDataFields.gt_boxes: float32 numpy array
                    of shape [num_boxes, 4] containing `num_boxes` groundtruth boxes of
                    the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
                InputDataFields.gt_classes: integer numpy array
                    of shape [num_boxes] containing 1-indexed groundtruth classes for the boxes.
                InputDataFields.gt_group_of: Optional length M
                    numpy boolean array denoting whether a groundtruth box contains a group of instances.
        Raises:
            ValueError: On adding groundtruth for an image more than once.
        """
        if image_id in self._image_ids:
            return

        gt_classes = (gt_dict[InputDataFields.gt_classes] - self._label_id_offset)
        # If the key is not present in the gt_dict or the array is empty
        # (unless there are no annotations for the groundtruth on this image)
        # use values from the dictionary or insert None otherwise.
        if (InputDataFields.gt_group_of in gt_dict and
                (gt_dict[InputDataFields.gt_group_of].size or not gt_classes.size)):
            gt_group_of = gt_dict[InputDataFields.gt_group_of]
        else:
            gt_group_of = None
            # FIXME disable warning for now, will add group_of flag eventually
            # if not len(self._image_ids) % 1000:
            #     logging.warning('image %s does not have groundtruth group_of flag specified', image_id)
        if self._evaluate_masks:
            gt_masks = gt_dict[InputDataFields.gt_instance_masks]
        else:
            gt_masks = None

        self._evaluation.add_single_ground_truth_image_info(
            image_id,
            gt_dict[InputDataFields.gt_boxes],
            gt_classes,
            gt_is_difficult_list=None,
            gt_is_group_of_list=gt_group_of,
            gt_masks=gt_masks)
        self._image_ids.update([image_id])


class OpenImagesChallengeEvaluator(OpenImagesDetectionEvaluator):
    """A class implements Open Images Challenge metrics.
      Both Detection and Instance Segmentation evaluation metrics are implemented.
      Open Images Challenge Detection metric has two major changes in comparison
      with Open Images V2 detection metric:
      - a custom weight might be specified for detecting an object contained in a group-of box.
      - verified image-level labels should be explicitly provided for evaluation: in case an
      image has neither positive nor negative image level label of class c, all detections of
      this class on this image will be ignored.

      Open Images Challenge Instance Segmentation metric allows to measure performance
      of models in case of incomplete annotations: some instances are
      annotations only on box level and some - on image-level. In addition,
      image-level labels are taken into account as in detection metric.

      Open Images Challenge Detection metric default parameters:
      evaluate_masks = False
      group_of_weight = 1.0

      Open Images Challenge Instance Segmentation metric default parameters:
      evaluate_masks = True
      (group_of_weight will not matter)
    """

    def __init__(
            self,
            categories,
            evaluate_masks=False,
            matching_iou_threshold=0.5,
            evaluate_corlocs=False,
            group_of_weight=1.0):
        """Constructor.
        Args:
            categories: A list of dicts, each of which has the following keys -
                'id': (required) an integer id uniquely identifying this category.
                'name': (required) string representing category name e.g., 'cat', 'dog'.
            evaluate_masks: set to true for instance segmentation metric and to false
                for detection metric.
            matching_iou_threshold: IOU threshold to use for matching groundtruth
                boxes to detection boxes.
            evaluate_corlocs: if True, additionally evaluates and returns CorLoc.
            group_of_weight: Weight of group-of boxes. If set to 0, detections of the
                correct class within a group-of box are ignored. If weight is > 0, then
                if at least one detection falls within a group-of box with
                matching_iou_threshold, weight group_of_weight is added to true
                positives. Consequently, if no detection falls within a group-of box,
                weight group_of_weight is added to false negatives.
        """
        if not evaluate_masks:
            metrics_prefix = 'OpenImagesDetectionChallenge'
        else:
            metrics_prefix = 'OpenImagesInstanceSegmentationChallenge'

        super(OpenImagesChallengeEvaluator, self).__init__(
            categories,
            matching_iou_threshold,
            evaluate_masks=evaluate_masks,
            evaluate_corlocs=evaluate_corlocs,
            group_of_weight=group_of_weight,
            metric_prefix=metrics_prefix)

        self._evaluatable_labels = {}

    def add_single_ground_truth_image_info(self, image_id, gt_dict):
        """Adds groundtruth for a single image to be used for evaluation.
        Args:
            image_id: A unique string/integer identifier for the image.
            gt_dict: A dictionary containing -
                InputDataFields.gt_boxes: float32 numpy array of shape [num_boxes, 4]
                    containing `num_boxes` groundtruth boxes of the format [ymin, xmin, ymax, xmax]
                    in absolute image coordinates.
                InputDataFields.gt_classes: integer numpy array of shape [num_boxes]
                    containing 1-indexed groundtruth classes for the boxes.
                InputDataFields.gt_image_classes: integer 1D
                    numpy array containing all classes for which labels are verified.
                InputDataFields.gt_group_of: Optional length M
                numpy boolean array denoting whether a groundtruth box contains a group of instances.
        Raises:
            ValueError: On adding groundtruth for an image more than once.
        """
        super(OpenImagesChallengeEvaluator,
              self).add_single_ground_truth_image_info(image_id, gt_dict)
        input_fields = InputDataFields
        gt_classes = gt_dict[input_fields.gt_classes] - self._label_id_offset
        image_classes = np.array([], dtype=int)
        if input_fields.gt_image_classes in gt_dict:
            image_classes = gt_dict[input_fields.gt_image_classes]
        elif input_fields.gt_labeled_classes in gt_dict:
            image_classes = gt_dict[input_fields.gt_labeled_classes]
        image_classes -= self._label_id_offset
        self._evaluatable_labels[image_id] = np.unique(
            np.concatenate((image_classes, gt_classes)))

    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.
        Args:
          image_id: A unique string/integer identifier for the image.
          detections_dict: A dictionary containing -
            DetectionResultFields.detection_boxes: float32 numpy
              array of shape [num_boxes, 4] containing `num_boxes` detection boxes
              of the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
            DetectionResultFields.detection_scores: float32 numpy
              array of shape [num_boxes] containing detection scores for the boxes.
            DetectionResultFields.detection_classes: integer numpy
              array of shape [num_boxes] containing 1-indexed detection classes for
              the boxes.
        Raises:
          ValueError: If detection masks are not in detections dictionary.
        """
        if image_id not in self._image_ids:
            # Since for the correct work of evaluator it is assumed that groundtruth
            # is inserted first we make sure to break the code if is it not the case.
            self._image_ids.update([image_id])
            self._evaluatable_labels[image_id] = np.array([])

        detection_classes = detections_dict[DetectionResultFields.detection_classes] - self._label_id_offset
        allowed_classes = np.where(np.isin(detection_classes, self._evaluatable_labels[image_id]))
        detection_classes = detection_classes[allowed_classes]
        detected_boxes = detections_dict[DetectionResultFields.detection_boxes][allowed_classes]
        detected_scores = detections_dict[DetectionResultFields.detection_scores][allowed_classes]

        if self._evaluate_masks:
            detection_masks = detections_dict[DetectionResultFields.detection_masks][allowed_classes]
        else:
            detection_masks = None
        self._evaluation.add_single_detected_image_info(
            image_key=image_id,
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            detected_class_labels=detection_classes,
            detected_masks=detection_masks)

    def clear(self):
        """Clears stored data."""

        super(OpenImagesChallengeEvaluator, self).clear()
        self._evaluatable_labels.clear()

