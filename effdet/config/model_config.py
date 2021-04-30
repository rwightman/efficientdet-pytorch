"""EfficientDet Configurations

Adapted from official impl at https://github.com/google/automl/tree/master/efficientdet

TODO use a different config system (OmegaConfig -> Hydra?), separate model from train specific hparams
"""

from omegaconf import OmegaConf
from copy import deepcopy


def default_detection_model_configs():
    """Returns a default detection configs."""
    h = OmegaConf.create()

    # model name.
    h.name = 'tf_efficientdet_d1'

    h.backbone_name = 'tf_efficientnet_b1'
    h.backbone_args = None  # FIXME sort out kwargs vs config for backbone creation
    h.backbone_indices = None

    # model specific, input preprocessing parameters
    h.image_size = (640, 640)

    # dataset specific head parameters
    h.num_classes = 90

    # feature + anchor config
    h.min_level = 3
    h.max_level = 7
    h.num_levels = h.max_level - h.min_level + 1
    h.num_scales = 3
    h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    # ratio w/h: 2.0 means w=1.4, h=0.7. Can be computed with k-mean per dataset.
    # aspect ratios can be specified as below too, pairs will be calc as sqrt(val), 1/sqrt(val)
    #h.aspect_ratios = [1.0, 2.0, 0.5]
    h.anchor_scale = 4.0

    # FPN and head config
    h.pad_type = 'same'  # original TF models require an equivalent of Tensorflow 'SAME' padding
    h.act_type = 'swish'
    h.norm_layer = None  # defaults to batch norm when None
    h.norm_kwargs = dict(eps=.001, momentum=.01)
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_channels = 88
    h.separable_conv = True
    h.apply_resample_bn = True
    h.conv_after_downsample = False
    h.conv_bn_relu_pattern = False
    h.use_native_resize_op = False
    h.downsample_type = 'max'
    h.upsample_type = 'nearest'
    h.redundant_bias = True  # original TF models have back to back bias + BN layers, not necessary!
    h.head_bn_level_first = False  # change order of BN in head repeat list of lists, True for torchscript compat
    h.head_act_type = None  # activation for heads, same as act_type if None

    h.fpn_name = None
    h.fpn_config = None
    h.fpn_drop_path_rate = 0.  # No stochastic depth in default. NOTE not currently used, unstable training

    # classification loss (used by train bench)
    h.alpha = 0.25
    h.gamma = 1.5
    h.label_smoothing = 0.  # only supported if legacy_focal == False, haven't produced great results
    h.legacy_focal = False  # use legacy focal loss (less stable, lower memory use in some cases)
    h.jit_loss = False  # torchscript jit for loss fn speed improvement, can impact stability and/or increase mem usage

    # localization loss (used by train bench)
    h.delta = 0.1
    h.box_loss_weight = 50.0

    # nms
    h.soft_nms = False  # use soft-nms, this is incredibly slow
    h.max_detection_points = 5000  # max detections for post process, input to NMS
    h.max_det_per_image = 100  # max detections per image limit, output of NMS

    return h


efficientdet_model_param_dict = dict(
    # Models with PyTorch friendly padding and my PyTorch pretrained backbones, training TBD
    efficientdet_d0=dict(
        name='efficientdet_d0',
        backbone_name='efficientnet_b0',
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.1),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d0-f3276ba8.pth',
    ),
    efficientdet_d1=dict(
        name='efficientdet_d1',
        backbone_name='efficientnet_b1',
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d1-bb7e98fe.pth',
    ),
    efficientdet_d2=dict(
        name='efficientdet_d2',
        backbone_name='efficientnet_b2',
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url='',  # no pretrained weights yet
    ),
    efficientdet_d3=dict(
        name='efficientdet_d3',
        backbone_name='efficientnet_b3',
        image_size=(896, 896),
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url='',  # no pretrained weights yet
    ),
    efficientdet_d4=dict(
        name='efficientdet_d4',
        backbone_name='efficientnet_b4',
        image_size=(1024, 1024),
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
    ),
    efficientdet_d5=dict(
        name='efficientdet_d5',
        backbone_name='efficientnet_b5',
        image_size=(1280, 1280),
        fpn_channels=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url='',
    ),

    # My own experimental configs with alternate models, training TBD
    # Note: any 'timm' model in the EfficientDet family can be used as a backbone here.
    resdet50=dict(
        name='resdet50',
        backbone_name='resnet50',
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        act_type='relu',
        redundant_bias=False,
        separable_conv=False,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/resdet50_416-08676892.pth',
    ),
    cspresdet50=dict(
        name='cspresdet50',
        backbone_name='cspresnet50',
        image_size=(768, 768),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        act_type='leaky_relu',
        head_act_type='silu',
        downsample_type='bilinear',
        upsample_type='bilinear',
        redundant_bias=False,
        separable_conv=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/cspresdet50b-386da277.pth',
    ),
    cspresdext50=dict(
        name='cspresdext50',
        backbone_name='cspresnext50',
        image_size=(640, 640),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        act_type='leaky_relu',
        redundant_bias=False,
        separable_conv=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url='',
    ),
    cspresdext50pan=dict(
        name='cspresdext50pan',
        backbone_name='cspresnext50',
        image_size=(640, 640),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=88,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        act_type='leaky_relu',
        fpn_name='pan_fa',  # PAN FPN experiment
        redundant_bias=False,
        separable_conv=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/cspresdext50pan-92fdd094.pth',
    ),
    cspdarkdet53=dict(
        name='cspdarkdet53',
        backbone_name='cspdarknet53',
        image_size=(640, 640),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        act_type='leaky_relu',
        redundant_bias=False,
        separable_conv=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        backbone_indices=(3, 4, 5),
        url='',
    ),
    cspdarkdet53m=dict(
        name='cspdarkdet53m',
        backbone_name='cspdarknet53',
        image_size=(768, 768),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=96,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        fpn_name='qufpn_fa',
        act_type='leaky_relu',
        head_act_type='mish',
        downsample_type='bilinear',
        upsample_type='bilinear',
        redundant_bias=False,
        separable_conv=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        backbone_indices=(3, 4, 5),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/cspdarkdet53m-79062b2d.pth',
    ),
    mixdet_m=dict(
        name='mixdet_m',
        backbone_name='mixnet_m',
        image_size=(512, 512),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.1),
        url='',  # no pretrained weights yet
    ),
    mixdet_l=dict(
        name='mixdet_l',
        backbone_name='mixnet_l',
        image_size=(640, 640),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url='',  # no pretrained weights yet
    ),
    mobiledetv2_110d=dict(
        name='mobiledetv2_110d',
        backbone_name='mobilenetv2_110d',
        image_size=(384, 384),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=48,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        act_type='relu6',
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.05),
        url='',  # no pretrained weights yet
    ),
    mobiledetv2_120d=dict(
        name='mobiledetv2_120d',
        backbone_name='mobilenetv2_120d',
        image_size=(512, 512),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=56,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        act_type='relu6',
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.1),
        url='',  # no pretrained weights yet
    ),
    mobiledetv3_large=dict(
        name='mobiledetv3_large',
        backbone_name='mobilenetv3_large_100',
        image_size=(512, 512),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        act_type='hard_swish',
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.1),
        url='',  # no pretrained weights yet
    ),
    efficientdet_q0=dict(
        name='efficientdet_q0',
        backbone_name='efficientnet_b0',
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        fpn_name='qufpn_fa',  # quad-fpn + fast attn experiment
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.1),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_q0-bdf1bdb5.pth',
    ),
    efficientdet_q1=dict(
        name='efficientdet_q1',
        backbone_name='efficientnet_b1',
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        fpn_name='qufpn_fa',  # quad-fpn + fast attn experiment
        downsample_type='bilinear',
        upsample_type='bilinear',
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_q1b-d0612140.pth',
    ),
    efficientdet_q2=dict(
        name='efficientdet_q2',
        backbone_name='efficientnet_b2',
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        fpn_name='qufpn_fa',  # quad-fpn + fast attn experiment
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_q2-0f7564e5.pth',
    ),
    efficientdet_w0=dict(
        name='efficientdet_w0',  # 'wide'
        backbone_name='efficientnet_b0',
        image_size=(512, 512),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=80,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(
            drop_path_rate=0.1,
            feature_location='depthwise'),  # features from after DW/SE in IR block
        url='',  # no pretrained weights yet
    ),
    efficientdet_es=dict(
        name='efficientdet_es',   #EdgeTPU-Small
        backbone_name='efficientnet_es',
        image_size=(512, 512),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=72,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type='',
        act_type='relu',
        redundant_bias=False,
        head_bn_level_first=True,
        separable_conv=False,
        backbone_args=dict(drop_path_rate=0.1),
        url='',
    ),
    efficientdet_em=dict(
        name='efficientdet_em',  # Edge-TPU Medium
        backbone_name='efficientnet_em',
        image_size=(640, 640),
        aspect_ratios=[1.0, 2.0, 0.5],
        fpn_channels=96,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        pad_type='',
        act_type='relu',
        redundant_bias=False,
        head_bn_level_first=True,
        separable_conv=False,
        backbone_args=dict(drop_path_rate=0.2),
        url='',  # no pretrained weights yet
    ),
    efficientdet_lite0=dict(
        name='efficientdet_lite0',
        backbone_name='efficientnet_lite0',
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        act_type='relu',
        redundant_bias=False,
        head_bn_level_first=True,
        backbone_args=dict(drop_path_rate=0.1),
        url='',
    ),

    # Models ported from Tensorflow with pretrained backbones ported from Tensorflow
    tf_efficientdet_d0=dict(
        name='tf_efficientdet_d0',
        backbone_name='tf_efficientnet_b0',
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_34-f153e0cf.pth',
    ),
    tf_efficientdet_d1=dict(
        name='tf_efficientdet_d1',
        backbone_name='tf_efficientnet_b1',
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d1_40-a30f94af.pth'
    ),
    tf_efficientdet_d2=dict(
        name='tf_efficientdet_d2',
        backbone_name='tf_efficientnet_b2',
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d2_43-8107aa99.pth',
    ),
    tf_efficientdet_d3=dict(
        name='tf_efficientdet_d3',
        backbone_name='tf_efficientnet_b3',
        image_size=(896, 896),
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3_47-0b525f35.pth',
    ),
    tf_efficientdet_d4=dict(
        name='tf_efficientdet_d4',
        backbone_name='tf_efficientnet_b4',
        image_size=(1024, 1024),
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4_49-f56376d9.pth',
    ),
    tf_efficientdet_d5=dict(
        name='tf_efficientdet_d5',
        backbone_name='tf_efficientnet_b5',
        image_size=(1280, 1280),
        fpn_channels=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5_51-c79f9be6.pth',
    ),
    tf_efficientdet_d6=dict(
        name='tf_efficientdet_d6',
        backbone_name='tf_efficientnet_b6',
        image_size=(1280, 1280),
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d6_52-4eda3773.pth'
    ),
    tf_efficientdet_d7=dict(
        name='tf_efficientdet_d7',
        backbone_name='tf_efficientnet_b6',
        image_size=(1536, 1536),
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        anchor_scale=5.0,
        fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7_53-6d1d7a95.pth'
    ),
    tf_efficientdet_d7x=dict(
        name='tf_efficientdet_d7x',
        backbone_name='tf_efficientnet_b7',
        image_size=(1536, 1536),
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        anchor_scale=4.0,
        max_level=8,
        fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7x-f390b87c.pth'
    ),

    #  Models ported from Tensorflow AdvProp+AA weights
    #  https://github.com/google/automl/blob/master/efficientdet/Det-AdvProp.md
    tf_efficientdet_d0_ap=dict(
        name='tf_efficientdet_d0_ap',
        backbone_name='tf_efficientnet_b0',
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_ap-d0cdbd0a.pth',
    ),
    tf_efficientdet_d1_ap=dict(
        name='tf_efficientdet_d1_ap',
        backbone_name='tf_efficientnet_b1',
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d1_ap-7721d075.pth'
    ),
    tf_efficientdet_d2_ap=dict(
        name='tf_efficientdet_d2_ap',
        backbone_name='tf_efficientnet_b2',
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d2_ap-a2995c19.pth',
    ),
    tf_efficientdet_d3_ap=dict(
        name='tf_efficientdet_d3_ap',
        backbone_name='tf_efficientnet_b3',
        image_size=(896, 896),
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3_ap-e4a2feab.pth',
    ),
    tf_efficientdet_d4_ap=dict(
        name='tf_efficientdet_d4_ap',
        backbone_name='tf_efficientnet_b4',
        image_size=(1024, 1024),
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4_ap-f601a5fc.pth',
    ),
    tf_efficientdet_d5_ap=dict(
        name='tf_efficientdet_d5_ap',
        backbone_name='tf_efficientnet_b5',
        image_size=(1280, 1280),
        fpn_channels=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill_color=0,
        backbone_args=dict(drop_path_rate=0.2),
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5_ap-3673ae5d.pth',
    ),

    # The lite configs are in TF automl repository but no weights yet and listed as 'not final'
    tf_efficientdet_lite0=dict(
        name='tf_efficientdet_lite0',
        backbone_name='tf_efficientnet_lite0',
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        act_type='relu',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.1),
        # unlike other tf_ models, this was not ported from tf automl impl, but trained from tf pretrained efficient lite
        # weights using this code, will likely replace if/when official det-lite weights are released
        url='https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite0-f5f303a9.pth',
    ),
    tf_efficientdet_lite1=dict(
        name='tf_efficientdet_lite1',
        backbone_name='tf_efficientnet_lite1',
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        act_type='relu',
        backbone_args=dict(drop_path_rate=0.2),
        url='',  # no pretrained weights yet
    ),
    tf_efficientdet_lite2=dict(
        name='tf_efficientdet_lite2',
        backbone_name='tf_efficientnet_lite2',
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        act_type='relu',
        backbone_args=dict(drop_path_rate=0.2),
        url='',
    ),
    tf_efficientdet_lite3=dict(
        name='tf_efficientdet_lite3',
        backbone_name='tf_efficientnet_lite3',
        image_size=(896, 896),
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        act_type='relu',
        backbone_args=dict(drop_path_rate=0.2),
        url='',
    ),
    tf_efficientdet_lite4=dict(
        name='tf_efficientdet_lite4',
        backbone_name='tf_efficientnet_lite4',
        image_size=(1024, 1024),
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        act_type='relu',
        backbone_args=dict(drop_path_rate=0.2),
        url='',
    ),
)


def get_efficientdet_config(model_name='tf_efficientdet_d1'):
    """Get the default config for EfficientDet based on model name."""
    h = default_detection_model_configs()
    h.update(efficientdet_model_param_dict[model_name])
    h.num_levels = h.max_level - h.min_level + 1
    h = deepcopy(h)  # may be unnecessary, ensure no references to param dict values
    # OmegaConf.set_struct(h, True)  # FIXME good idea?
    return h
