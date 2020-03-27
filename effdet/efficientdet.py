import torch
import torch.nn as nn
import logging
from collections import OrderedDict
from timm import create_model



class EfficientDet(nn.Module):
    """ PyTorch EfficentDet
    """
    def __init__(self, config):
        super(EfficientDet, self).__init__()
        self.backbone = create_model(config.backbone, features_only=True)
        self.fpn = BiFpn(config, self.backbone.feature_info())
        self.box_net = BoxNet(config)
        self.class_net = ClassNet(config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box
