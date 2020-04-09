import torch
import torch.nn as nn
from .anchors import Anchors, generate_detections, MAX_DETECTION_POINTS


def _post_process(config, cls_outputs, box_outputs):
    """Selects top-k predictions.

    This code is adapted from offical TensorFlow impl

    Args:
        config: a parameter dictionary that includes `min_level`, `max_level`,  `batch_size`, and `num_classes`.

        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].
    """
    batch_size = cls_outputs[0].shape[0]

    cls_outputs_all = []
    box_outputs_all = []
    # Concatenates class and box of all levels into one tensor.
    for level in range(config.num_levels):
        cls_outputs_all.append(
            cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, config.num_classes]))
        box_outputs_all.append(
            box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4]))
    cls_outputs_all = torch.cat(cls_outputs_all, 1)
    box_outputs_all = torch.cat(box_outputs_all, 1)

    # cls_outputs_all has a shape of [batch_size, N, num_classes] and
    # box_outputs_all has a shape of [batch_size, N, 4].
    cls_outputs_all_after_topk = []
    box_outputs_all_after_topk = []
    indices_all = []
    classes_all = []
    for index in range(batch_size):
        cls_outputs_per_sample = cls_outputs_all[index]
        box_outputs_per_sample = box_outputs_all[index]
        _, cls_topk_indices = torch.topk(cls_outputs_per_sample.flatten(), k=MAX_DETECTION_POINTS)

        # Gets top-k class and box scores.
        indices = cls_topk_indices / config.num_classes
        classes = cls_topk_indices % config.num_classes

        box_outputs_after_topk = torch.index_select(box_outputs_per_sample, 0, indices)
        box_outputs_all_after_topk.append(box_outputs_after_topk)

        indices_gather = torch.index_select(cls_outputs_per_sample, 0, indices)
        cls_outputs_after_topk = torch.gather(indices_gather, 1, classes.unsqueeze(1))
        cls_outputs_all_after_topk.append(cls_outputs_after_topk)

        indices_all.append(indices)
        classes_all.append(classes)

    # Concatenates via the batch dimension.
    cls_outputs_all_after_topk = torch.stack(cls_outputs_all_after_topk, dim=0)
    box_outputs_all_after_topk = torch.stack(box_outputs_all_after_topk, dim=0)
    indices_all = torch.stack(indices_all, dim=0)
    classes_all = torch.stack(classes_all, dim=0)

    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all


## wrap the model with post-proc for a clean DataParallel module boundary
class _Wrapper(nn.Module):

    def __init__(self, model, config):
        super(_Wrapper, self).__init__()
        self.model = model
        self.config = config

    def forward(self, x):
        x = self.model(x)
        return _post_process(self.config, *x)


class DetBenchEval(nn.Module):
    def __init__(self, model, config):
        super(DetBenchEval, self).__init__()
        self.config = config
        self.model = _Wrapper(model, config)
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)
        self._anchor_cache = None

    def forward(self, x, image_ids, image_scales):
        class_out, box_out, indices, classes = self.model(x)

        # FIXME do this in PyTorch
        batch_detections = []
        class_out = class_out.cpu().numpy()
        box_out = box_out.cpu().numpy()
        if self._anchor_cache is None:
            anchor_boxes = self.anchors.boxes.cpu().numpy()
            self._anchor_cache = anchor_boxes
        else:
            anchor_boxes = self._anchor_cache
        indices = indices.cpu().numpy()
        classes = classes.cpu().numpy()
        image_ids = image_ids.cpu().numpy()
        image_scale = image_scales.cpu().numpy()
        for i in range(x.shape[0]):
            detections = generate_detections(
                class_out[i], box_out[i], anchor_boxes, indices[i], classes[i],
                image_ids[i], image_scale[i], self.config.num_classes)
            batch_detections.append(detections)

        return batch_detections

