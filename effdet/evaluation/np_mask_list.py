import numpy as np
from .np_box_list import *

EPSILON = 1e-7


class MaskList(BoxList):
    """Convenience wrapper for BoxList with masks.
  
    BoxMaskList extends the np_box_list.BoxList to contain masks as well.
    In particular, its constructor receives both boxes and masks. Note that the
    masks correspond to the full image.
    """

    def __init__(self, box_data, mask_data):
        """Constructs box collection.
    
        Args:
            box_data: a numpy array of shape [N, 4] representing box coordinates
            mask_data: a numpy array of shape [N, height, width] representing masks
                with values are in {0,1}. The masks correspond to the full
                image. The height and the width will be equal to image height and width.
    
        Raises:
            ValueError: if bbox data is not a numpy array
            ValueError: if invalid dimensions for bbox data
            ValueError: if mask data is not a numpy array
            ValueError: if invalid dimension for mask data
        """
        super(MaskList, self).__init__(box_data)
        if not isinstance(mask_data, np.ndarray):
            raise ValueError('Mask data must be a numpy array.')
        if len(mask_data.shape) != 3:
            raise ValueError('Invalid dimensions for mask data.')
        if mask_data.dtype != np.uint8:
            raise ValueError('Invalid data type for mask data: uint8 is required.')
        if mask_data.shape[0] != box_data.shape[0]:
            raise ValueError('There should be the same number of boxes and masks.')
        self.data['masks'] = mask_data

    def get_masks(self):
        """Convenience function for accessing masks.
    
        Returns:
            a numpy array of shape [N, height, width] representing masks
        """
        return self.get_field('masks')


def boxlist_to_masklist(boxlist):
    """Converts a BoxList containing 'masks' into a BoxMaskList.
  
    Args:
        boxlist: An np_box_list.BoxList object.
  
    Returns:
        An BoxMaskList object.
  
    Raises:
        ValueError: If boxlist does not contain `masks` as a field.
    """
    if not boxlist.has_field('masks'):
        raise ValueError('boxlist does not contain mask field.')
    masklist = MaskList(box_data=boxlist.get(), mask_data=boxlist.get_field('masks'))
    extra_fields = boxlist.get_extra_fields()
    for key in extra_fields:
        if key != 'masks':
            masklist.data[key] = boxlist.get_field(key)
    return masklist


def area_mask(masks):
    """Computes area of masks.

    Args:
        masks: Numpy array with shape [N, height, width] holding N masks. Masks
        values are of type np.uint8 and values are in {0,1}.

    Returns:
        a numpy array with shape [N*1] representing mask areas.

    Raises:
        ValueError: If masks.dtype is not np.uint8
    """
    if masks.dtype != np.uint8:
        raise ValueError('Masks type should be np.uint8')
    return np.sum(masks, axis=(1, 2), dtype=np.float32)


def intersection_mask(masks1, masks2):
    """Compute pairwise intersection areas between masks.

    Args:
        masks1: a numpy array with shape [N, height, width] holding N masks. Masks
            values are of type np.uint8 and values are in {0,1}.
        masks2: a numpy array with shape [M, height, width] holding M masks. Masks
            values are of type np.uint8 and values are in {0,1}.

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area.

    Raises:
        ValueError: If masks1 and masks2 are not of type np.uint8.
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError('masks1 and masks2 should be of type np.uint8')
    n = masks1.shape[0]
    m = masks2.shape[0]
    answer = np.zeros([n, m], dtype=np.float32)
    for i in np.arange(n):
        for j in np.arange(m):
            answer[i, j] = np.sum(np.minimum(masks1[i], masks2[j]), dtype=np.float32)
    return answer


def iou_mask(masks1, masks2):
    """Computes pairwise intersection-over-union between mask collections.

    Args:
        masks1: a numpy array with shape [N, height, width] holding N masks. Masks
            values are of type np.uint8 and values are in {0,1}.
        masks2: a numpy array with shape [M, height, width] holding N masks. Masks
            values are of type np.uint8 and values are in {0,1}.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.

    Raises:
        ValueError: If masks1 and masks2 are not of type np.uint8.
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError('masks1 and masks2 should be of type np.uint8')
    intersect = intersection(masks1, masks2)
    area1 = area(masks1)
    area2 = area(masks2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / np.maximum(union, EPSILON)


def ioa_mask(masks1, masks2):
    """Computes pairwise intersection-over-area between box collections.

    Intersection-over-area (ioa) between two masks, mask1 and mask2 is defined as
    their intersection area over mask2's area. Note that ioa is not symmetric,
    that is, IOA(mask1, mask2) != IOA(mask2, mask1).

    Args:
        masks1: a numpy array with shape [N, height, width] holding N masks. Masks
            values are of type np.uint8 and values are in {0,1}.
        masks2: a numpy array with shape [M, height, width] holding N masks. Masks
            values are of type np.uint8 and values are in {0,1}.

    Returns:
        a numpy array with shape [N, M] representing pairwise ioa scores.

    Raises:
        ValueError: If masks1 and masks2 are not of type np.uint8.
    """
    if masks1.dtype != np.uint8 or masks2.dtype != np.uint8:
        raise ValueError('masks1 and masks2 should be of type np.uint8')
    intersect = intersection(masks1, masks2)
    areas = np.expand_dims(area(masks2), axis=0)
    return intersect / (areas + EPSILON)


def area_masklist(masklist):
    """Computes area of masks.
  
    Args:
        masklist: BoxMaskList holding N boxes and masks
  
    Returns:
        a numpy array with shape [N*1] representing mask areas
    """
    return area_mask(masklist.get_masks())


def intersection_masklist(masklist1, masklist2):
    """Compute pairwise intersection areas between masks.
  
    Args:
        masklist1: BoxMaskList holding N boxes and masks
        masklist2: BoxMaskList holding M boxes and masks
  
    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    return intersection_mask(masklist1.get_masks(), masklist2.get_masks())


def iou_masklist(masklist1, masklist2):
    """Computes pairwise intersection-over-union between box and mask collections.
  
    Args:
        masklist1: BoxMaskList holding N boxes and masks
        masklist2: BoxMaskList holding M boxes and masks
  
    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    return iou_mask(masklist1.get_masks(), masklist2.get_masks())


def ioa_masklist(masklist1, masklist2):
    """Computes pairwise intersection-over-area between box and mask collections.
  
    Intersection-over-area (ioa) between two masks mask1 and mask2 is defined as
    their intersection area over mask2's area. Note that ioa is not symmetric,
    that is, IOA(mask1, mask2) != IOA(mask2, mask1).
  
    Args:
        masklist1: BoxMaskList holding N boxes and masks
        masklist2: BoxMaskList holding M boxes and masks
  
    Returns:
        a numpy array with shape [N, M] representing pairwise ioa scores.
    """
    return ioa_mask(masklist1.get_masks(), masklist2.get_masks())


def gather_masklist(masklist, indices, fields=None):
    """Gather boxes from BoxMaskList according to indices.
  
    By default, gather returns boxes corresponding to the input index list, as
    well as all additional fields stored in the masklist (indexing into the
    first dimension).  However one can optionally only gather from a
    subset of fields.
  
    Args:
        masklist: BoxMaskList holding N boxes
        indices: a 1-d numpy array of type int_
        fields: (optional) list of fields to also gather from.  If None (default), all fields
            are gathered from.  Pass an empty fields list to only gather the box coordinates.
  
    Returns:
        submasklist: a BoxMaskList corresponding to the subset of the input masklist specified by indices
  
    Raises:
        ValueError: if specified field is not contained in masklist or if the indices are not of type int_
    """
    if fields is not None:
        if 'masks' not in fields:
            fields.append('masks')
    return boxlist_to_masklist(gather_boxlist(boxlist=masklist, indices=indices, fields=fields))


def sort_by_field_masklist(masklist, field, order=SortOrder.DESCEND):
    """Sort boxes and associated fields according to a scalar field.
  
    A common use case is reordering the boxes according to descending scores.
  
    Args:
        masklist: BoxMaskList holding N boxes.
        field: A BoxMaskList field for sorting and reordering the BoxMaskList.
        order: (Optional) 'descend' or 'ascend'. Default is descend.
  
    Returns:
        sorted_masklist: A sorted BoxMaskList with the field in the specified order.
    """
    return boxlist_to_masklist(sort_by_field_boxlist(boxlist=masklist, field=field, order=order))


def non_max_suppression_mask(masklist, max_output_size=10000, iou_threshold=1.0, score_threshold=-10.0):
    """Non maximum suppression.
  
    This op greedily selects a subset of detection bounding boxes, pruning
    away boxes that have high IOU (intersection over union) overlap (> thresh)
    with already selected boxes. In each iteration, the detected bounding box with
    highest score in the available pool is selected.
  
    Args:
        masklist: BoxMaskList holding N boxes.  Must contain a 'scores' field representing
            detection scores. All scores belong to the same class.
        max_output_size: maximum number of retained boxes
        iou_threshold: intersection over union threshold.
        score_threshold: minimum score threshold. Remove the boxes with scores
            less than this value. Default value is set to -10. A very
            low threshold to pass pretty much all the boxes, unless
            the user sets a different score threshold.
  
    Returns:
        an BoxMaskList holding M boxes where M <= max_output_size
  
    Raises:
        ValueError: if 'scores' field does not exist
        ValueError: if threshold is not in [0, 1]
        ValueError: if max_output_size < 0
    """
    if not masklist.has_field('scores'):
        raise ValueError('Field scores does not exist')
    if iou_threshold < 0. or iou_threshold > 1.0:
        raise ValueError('IOU threshold must be in [0, 1]')
    if max_output_size < 0:
        raise ValueError('max_output_size must be bigger than 0.')

    masklist = filter_scores_greater_than(masklist, score_threshold)
    if masklist.num_boxes() == 0:
        return masklist

    masklist = sort_by_field_boxlist(masklist, 'scores')

    # Prevent further computation if NMS is disabled.
    if iou_threshold == 1.0:
        if masklist.num_boxes() > max_output_size:
            selected_indices = np.arange(max_output_size)
            return gather_masklist(masklist, selected_indices)
        else:
            return masklist

    masks = masklist.get_masks()
    num_masks = masklist.num_boxes()

    # is_index_valid is True only for all remaining valid boxes,
    is_index_valid = np.full(num_masks, 1, dtype=bool)
    selected_indices = []
    num_output = 0
    for i in range(num_masks):
        if num_output < max_output_size:
            if is_index_valid[i]:
                num_output += 1
                selected_indices.append(i)
                is_index_valid[i] = False
                valid_indices = np.where(is_index_valid)[0]
                if valid_indices.size == 0:
                    break

                intersect_over_union = iou_mask(np.expand_dims(masks[i], axis=0), masks[valid_indices])
                intersect_over_union = np.squeeze(intersect_over_union, axis=0)
                is_index_valid[valid_indices] = np.logical_and(
                    is_index_valid[valid_indices],
                    intersect_over_union <= iou_threshold)
    return gather_masklist(masklist, np.array(selected_indices))


def multi_class_non_max_suppression_mask(masklist, score_thresh, iou_thresh, max_output_size):
    """Multi-class version of non maximum suppression.
  
    This op greedily selects a subset of detection bounding boxes, pruning away boxes that have
    high IOU (intersection over union) overlap (> thresh) with already selected boxes.  It
    operates independently for each class for which scores are provided (via the scores field
    of the input box_list), pruning boxes with score less than a provided threshold prior to
    applying NMS.
  
    Args:
        masklist: BoxMaskList holding N boxes.  Must contain a 'scores' field representing detection
            scores.  This scores field is a tensor that can be 1 dimensional (in the case of a
            single class) or 2-dimensional, in which case we assume that it takes the shape
            [num_boxes, num_classes]. We further assume that this rank is known statically and
            that scores.shape[1] is also known (i.e., the number of classes is fixed and known
            at graph construction time).
        score_thresh: scalar threshold for score (low scoring boxes are removed).
        iou_thresh: scalar threshold for IOU (boxes that that high IOU overlap with previously
            selected boxes are removed).
        max_output_size: maximum number of retained boxes per class.
  
    Returns:
        a masklist holding M boxes with a rank-1 scores field representing
        corresponding scores for each box with scores sorted in decreasing order
        and a rank-1 classes field representing a class label for each box.
    Raises:
        ValueError: if iou_thresh is not in [0, 1] or if input masklist does not have a valid scores field.
    """
    if not 0 <= iou_thresh <= 1.0:
        raise ValueError('thresh must be between 0 and 1')
    if not isinstance(masklist, MaskList):
        raise ValueError('masklist must be a masklist')
    if not masklist.has_field('scores'):
        raise ValueError('input masklist must have \'scores\' field')
    scores = masklist.get_field('scores')
    if len(scores.shape) == 1:
        scores = np.reshape(scores, [-1, 1])
    elif len(scores.shape) == 2:
        if scores.shape[1] is None:
            raise ValueError('scores field must have statically defined second dimension')
    else:
        raise ValueError('scores field must be of rank 1 or 2')

    num_boxes = masklist.num_boxes()
    num_scores = scores.shape[0]
    num_classes = scores.shape[1]

    if num_boxes != num_scores:
        raise ValueError('Incorrect scores field length: actual vs expected.')

    selected_boxes_list = []
    for class_idx in range(num_classes):
        masklist_and_class_scores = MaskList(box_data=masklist.get(), mask_data=masklist.get_masks())
        class_scores = np.reshape(scores[0:num_scores, class_idx], [-1])
        masklist_and_class_scores.add_field('scores', class_scores)
        masklist_filt = filter_scores_greater_than(masklist_and_class_scores, score_thresh)
        nms_result = non_max_suppression(
            masklist_filt,
            max_output_size=max_output_size,
            iou_threshold=iou_thresh,
            score_threshold=score_thresh)
        nms_result.add_field('classes', np.zeros_like(nms_result.get_field('scores')) + class_idx)
        selected_boxes_list.append(nms_result)
    selected_boxes = concatenate_boxlist(selected_boxes_list)
    sorted_boxes = sort_by_field_boxlist(selected_boxes, 'scores')
    return boxlist_to_masklist(boxlist=sorted_boxes)


def prune_non_overlapping_masklist(masklist1, masklist2, minoverlap=0.0):
    """Prunes the boxes in list1 that overlap less than thresh with list2.
  
    For each mask in masklist1, we want its IOA to be more than minoverlap
    with at least one of the masks in masklist2. If it does not, we remove
    it. If the masks are not full size image, we do the pruning based on boxes.
  
    Args:
        masklist1: BoxMaskList holding N boxes and masks.
        masklist2: BoxMaskList holding M boxes and masks.
        minoverlap: Minimum required overlap between boxes, to count them as overlapping.
  
    Returns:
        A pruned masklist with size [N', 4].
    """
    intersection_over_area = ioa_masklist(masklist2, masklist1)  # [M, N] tensor
    intersection_over_area = np.amax(intersection_over_area, axis=0)  # [N] tensor
    keep_bool = np.greater_equal(intersection_over_area, np.array(minoverlap))
    keep_inds = np.nonzero(keep_bool)[0]
    new_masklist1 = gather_masklist(masklist1, keep_inds)
    return new_masklist1


def concatenate_masklist(masklists, fields=None):
    """Concatenate list of masklists.
  
    This op concatenates a list of input masklists into a larger
    masklist.  It also
    handles concatenation of masklist fields as long as the field tensor
    shapes are equal except for the first dimension.
  
    Args:
        masklists: list of BoxMaskList objects
        fields: optional list of fields to also concatenate.  By default, all
            fields from the first BoxMaskList in the list are included in the concatenation.
  
    Returns:
        a masklist with number of boxes equal to sum([masklist.num_boxes() for masklist in masklist])
    Raises:
        ValueError: if masklists is invalid (i.e., is not a list, is empty, or contains non
            masklist objects), or if requested fields are not contained in all masklists
    """
    if fields is not None:
        if 'masks' not in fields:
            fields.append('masks')
    return boxlist_to_masklist(concatenate_boxlist(boxlists=masklists, fields=fields))


def filter_scores_greater_than_masklist(masklist, thresh):
    """Filter to keep only boxes and masks with score exceeding a given threshold.
  
    This op keeps the collection of boxes and masks whose corresponding scores are
    greater than the input threshold.
  
    Args:
        masklist: BoxMaskList holding N boxes and masks.  Must contain a
            'scores' field representing detection scores.
        thresh: scalar threshold
  
    Returns:
        a BoxMaskList holding M boxes and masks where M <= N
  
    Raises:
        ValueError: if masklist not a BoxMaskList object or if it does not have a scores field
    """
    if not isinstance(masklist, MaskList):
        raise ValueError('masklist must be a BoxMaskList')
    if not masklist.has_field('scores'):
        raise ValueError('input masklist must have \'scores\' field')
    scores = masklist.get_field('scores')
    if len(scores.shape) > 2:
        raise ValueError('Scores should have rank 1 or 2')
    if len(scores.shape) == 2 and scores.shape[1] != 1:
        raise ValueError('Scores should have rank 1 or have shape consistent with [None, 1]')
    high_score_indices = np.reshape(np.where(np.greater(scores, thresh)), [-1]).astype(np.int32)
    return gather_masklist(masklist, high_score_indices)
