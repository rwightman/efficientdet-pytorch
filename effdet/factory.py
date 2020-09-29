from .efficientdet import EfficientDet, HeadNet
from .bench import DetBenchTrain, DetBenchPredict
from .config import get_efficientdet_config
from .helpers import load_pretrained, load_checkpoint


def create_model(
        model_name, bench_task='', num_classes=None, pretrained=False,
        checkpoint_path='', checkpoint_ema=False, **kwargs):

    config = get_efficientdet_config(model_name)
    return create_model_from_config(
        config, bench_task=bench_task, num_classes=num_classes, pretrained=pretrained,
        checkpoint_path=checkpoint_path, checkpoint_ema=checkpoint_ema, **kwargs)


def create_model_from_config(
        config, bench_task='', num_classes=None, pretrained=False,
        checkpoint_path='', checkpoint_ema=False, **kwargs):

    pretrained_backbone = kwargs.pop('pretrained_backbone', True)
    if pretrained or checkpoint_path:
        pretrained_backbone = False  # no point in loading backbone weights

    # Config overrides, override some config value from args. FIXME need a cleaner mechanism or allow
    # config defs via files.
    redundant_bias = kwargs.pop('redundant_bias', None)
    if redundant_bias is not None:
        # override config if set to something
        config.redundant_bias = redundant_bias

    label_smoothing = kwargs.pop('label_smoothing', None)
    if label_smoothing is not None:
        config.label_smoothing = label_smoothing

    legacy_focal = kwargs.pop('legacy_focal', None)
    if legacy_focal is not None:
        config.legacy_focal = legacy_focal

    jit_loss = kwargs.pop('jit_loss', None)
    if jit_loss is not None:
        config.jit_loss = jit_loss

    # create the base model
    model = EfficientDet(config, pretrained_backbone=pretrained_backbone, **kwargs)

    # pretrained weights are always spec'd for original config, load them before we change the model
    if pretrained:
        load_pretrained(model, config.url)

    # reset model head if num_classes doesn't match configs
    if num_classes is not None and num_classes != config.num_classes:
        model.reset_head(num_classes=num_classes)

    # load an argument specified training checkpoint
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, use_ema=checkpoint_ema)

    # wrap model in task specific training/prediction bench if set
    if bench_task == 'train':
        model = DetBenchTrain(model)
    elif bench_task == 'predict':
        model = DetBenchPredict(model)
    return model
