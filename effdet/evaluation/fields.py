
class InputDataFields(object):
    """Names for the input tensors.
    Holds the standard data field names to use for identifying input tensors. This
    should be used by the decoder to identify keys for the returned tensor_dict
    containing input tensors. And it should be used by the model to identify the
    tensors it needs.
    Attributes:
        image: image.
        image_additional_channels: additional channels.
        key: unique key corresponding to image.
        filename: original filename of the dataset (without common path).
        gt_image_classes: image-level class labels.
        gt_image_confidences: image-level class confidences.
        gt_labeled_classes: image-level annotation that indicates the
            classes for which an image has been labeled.
        gt_boxes: coordinates of the ground truth boxes in the image.
        gt_classes: box-level class labels.
        gt_confidences: box-level class confidences. The shape should be
            the same as the shape of gt_classes.
        gt_label_types: box-level label types (e.g. explicit negative).
        gt_is_crowd: [DEPRECATED, use gt_group_of instead]
            is the groundtruth a single object or a crowd.
        gt_area: area of a groundtruth segment.
        gt_difficult: is a `difficult` object
        gt_group_of: is a `group_of` objects, e.g. multiple objects of the
            same class, forming a connected group, where instances are heavily
            occluding each other.
        gt_instance_masks: ground truth instance masks.
        gt_instance_boundaries: ground truth instance boundaries.
        gt_instance_classes: instance mask-level class labels.
        gt_label_weights: groundtruth label weights.
        gt_weights: groundtruth weight factor for bounding boxes.
        image_height: height of images, used to decode
        image_width: width of images, used to decode
    """
    image = 'image'
    key = 'image_id'
    filename = 'filename'
    gt_boxes = 'bbox'
    gt_classes = 'cls'
    gt_confidences = 'confidences'
    gt_label_types = 'label_types'
    gt_image_classes = 'img_cls'
    gt_image_confidences = 'img_confidences'
    gt_labeled_classes = 'labeled_cls'
    gt_is_crowd = 'is_crowd'
    gt_area = 'area'
    gt_difficult = 'difficult'
    gt_group_of = 'group_of'
    gt_instance_masks = 'instance_masks'
    gt_instance_boundaries = 'instance_boundaries'
    gt_instance_classes = 'instance_classes'
    image_height = 'img_height'
    image_width = 'img_width'
    image_size = 'img_size'


class DetectionResultFields(object):
    """Naming conventions for storing the output of the detector.
    Attributes:
        source_id: source of the original image.
        key: unique key corresponding to image.
        detection_boxes: coordinates of the detection boxes in the image.
        detection_scores: detection scores for the detection boxes in the image.
        detection_multiclass_scores: class score distribution (including background)
            for detection boxes in the image including background class.
        detection_classes: detection-level class labels.
        detection_masks: contains a segmentation mask for each detection box.
    """

    key = 'image_id'
    detection_boxes = 'bbox'
    detection_scores = 'score'
    detection_classes = 'cls'
    detection_masks = 'masks'


class BoxListFields(object):
    """Naming conventions for BoxLists.
    Attributes:
        boxes: bounding box coordinates.
        classes: classes per bounding box.
        scores: scores per bounding box.
        weights: sample weights per bounding box.
        objectness: objectness score per bounding box.
        masks: masks per bounding box.
        boundaries: boundaries per bounding box.
        keypoints: keypoints per bounding box.
        keypoint_heatmaps: keypoint heatmaps per bounding box.
        is_crowd: is_crowd annotation per bounding box.
    """
    boxes = 'boxes'
    classes = 'classes'
    scores = 'scores'
    weights = 'weights'
    confidences = 'confidences'
    objectness = 'objectness'
    masks = 'masks'
    boundaries = 'boundaries'
    keypoints = 'keypoints'
    keypoint_visibilities = 'keypoint_visibilities'
    keypoint_heatmaps = 'keypoint_heatmaps'
    is_crowd = 'is_crowd'
    group_of = 'group_of'
