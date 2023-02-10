""" PyTorch EfficientDet support benches

Hacked together by Ross Wightman
"""
from typing import Optional, Dict, List

import torch
import torch.nn as nn

from .anchors import Anchors, AnchorLabeler, generate_detections
from .loss import DetectionLoss

# try:
#     torch.div(torch.ones(1), torch.ones(1), rounding_mode='floor')
#     has_rounding_mode = True
# except TypeError:
#     has_rounding_mode = False


def _post_process(
        cls_outputs: List[torch.Tensor],
        box_outputs: List[torch.Tensor],
        num_levels: int,
        num_classes: int,
        max_detection_points: int = 5000,
):
    """Selects top-k predictions.

    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.

    Args:
        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].

        num_levels (int): number of feature levels

        num_classes (int): number of output classes
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat([
        cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, num_classes])
        for level in range(num_levels)], 1)

    box_outputs_all = torch.cat([
        box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4])
        for level in range(num_levels)], 1)

    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=max_detection_points)
    # FIXME change someday, will have to live with annoying warnings for a while as testing impl breaks torchscript
    # indices_all = torch.div(cls_topk_indices_all, num_classes, rounding_mode='trunc')
    indices_all = cls_topk_indices_all // num_classes
    classes_all = cls_topk_indices_all % num_classes

    box_outputs_all_after_topk = torch.gather(
        box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))

    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, num_classes))
    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))

    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all


@torch.jit.script
def _batch_detection(
        batch_size: int, class_out, box_out, anchor_boxes, indices, classes,
        img_scale: Optional[torch.Tensor] = None,
        img_size: Optional[torch.Tensor] = None,
        max_det_per_image: int = 100,
        soft_nms: bool = False,
):
    batch_detections = []
    # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
    for i in range(batch_size):
        img_scale_i = None if img_scale is None else img_scale[i]
        img_size_i = None if img_size is None else img_size[i]
        detections = generate_detections(
            class_out[i],
            box_out[i],
            anchor_boxes,
            indices[i],
            classes[i],
            img_scale_i,
            img_size_i,
            max_det_per_image=max_det_per_image,
            soft_nms=soft_nms,
        )
        batch_detections.append(detections)
    return torch.stack(batch_detections, dim=0)


class DetBenchPredict(nn.Module):
    def __init__(self, model):
        super(DetBenchPredict, self).__init__()
        self.model = model
        self.config = model.config  # FIXME remove this when we can use @property (torchscript limitation)
        self.num_levels = model.config.num_levels
        self.num_classes = model.config.num_classes
        self.anchors = Anchors.from_config(model.config)
        self.max_detection_points = model.config.max_detection_points
        self.max_det_per_image = model.config.max_det_per_image
        self.soft_nms = model.config.soft_nms

    def forward(self, x, img_info: Optional[Dict[str, torch.Tensor]] = None):
        class_out, box_out = self.model(x)
        class_out, box_out, indices, classes = _post_process(
            class_out,
            box_out,
            num_levels=self.num_levels,
            num_classes=self.num_classes,
            max_detection_points=self.max_detection_points,
        )
        if img_info is None:
            img_scale, img_size = None, None
        else:
            img_scale, img_size = img_info['img_scale'], img_info['img_size']
        return _batch_detection(
            x.shape[0],
            class_out,
            box_out,
            self.anchors.boxes,
            indices,
            classes,
            img_scale,
            img_size,
            max_det_per_image=self.max_det_per_image,
            soft_nms=self.soft_nms,
        )


class DetBenchTrain(nn.Module):
    def __init__(self, model, create_labeler=True):
        super(DetBenchTrain, self).__init__()
        self.model = model
        self.config = model.config  # FIXME remove this when we can use @property (torchscript limitation)
        self.num_levels = model.config.num_levels
        self.num_classes = model.config.num_classes
        self.anchors = Anchors.from_config(model.config)
        self.max_detection_points = model.config.max_detection_points
        self.max_det_per_image = model.config.max_det_per_image
        self.soft_nms = model.config.soft_nms
        self.anchor_labeler = None
        if create_labeler:
            self.anchor_labeler = AnchorLabeler(
                self.anchors,
                self.num_classes,
                match_threshold=0.5,
            )
        self.loss_fn = DetectionLoss(model.config)

    def forward(self, x, target: Dict[str, torch.Tensor]):
        class_out, box_out = self.model(x)
        if self.anchor_labeler is None:
            # target should contain pre-computed anchor labels if labeler not present in bench
            assert 'label_num_positives' in target
            cls_targets = [target[f'label_cls_{l}'] for l in range(self.num_levels)]
            box_targets = [target[f'label_bbox_{l}'] for l in range(self.num_levels)]
            num_positives = target['label_num_positives']
        else:
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
                target['bbox'],
                target['cls'],
            )

        loss, class_loss, box_loss = self.loss_fn(
            class_out,
            box_out,
            cls_targets,
            box_targets,
            num_positives,
        )
        output = {'loss': loss, 'class_loss': class_loss, 'box_loss': box_loss}
        if not self.training:
            # if eval mode, output detections for evaluation
            class_out_pp, box_out_pp, indices, classes = _post_process(
                class_out,
                box_out,
                num_levels=self.num_levels,
                num_classes=self.num_classes,
                max_detection_points=self.max_detection_points,
            )
            output['detections'] = _batch_detection(
                x.shape[0],
                class_out_pp,
                box_out_pp,
                self.anchors.boxes,
                indices,
                classes,
                target['img_scale'],
                target['img_size'],
                max_det_per_image=self.max_det_per_image,
                soft_nms=self.soft_nms,
            )
        return output


def unwrap_bench(model):
    # Unwrap a model in support bench so that various other fns can access the weights and attribs of the
    # underlying model directly
    if hasattr(model, 'module'):  # unwrap DDP or EMA
        return unwrap_bench(model.module)
    elif hasattr(model, 'model'):  # unwrap Bench -> model
        return unwrap_bench(model.model)
    else:
        return model
