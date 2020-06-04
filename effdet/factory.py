from .efficientdet import EfficientDet
from .bench import DetBenchTrain, DetBenchPredict
from .config import get_efficientdet_config
from .helpers import load_pretrained, load_checkpoint


def create_model(
        model_name, bench_task='', pretrained=False, checkpoint_path='', checkpoint_ema=False, **kwargs):
    config = get_efficientdet_config(model_name)

    pretrained_backbone = kwargs.pop('pretrained_backbone', True)
    if pretrained or checkpoint_path:
        pretrained_backbone = False  # no point in loading backbone weights

    redundant_bias = kwargs.pop('redundant_bias', None)
    if redundant_bias is not None:
        # override config if set to something
        config.redundant_bias = redundant_bias

    model = EfficientDet(config, pretrained_backbone=pretrained_backbone, **kwargs)

    # FIXME handle different head classes / anchors and re-init of necessary layers w/ pretrained load

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, use_ema=checkpoint_ema)
    elif pretrained:
        load_pretrained(model, config.url)

    # wrap model in task specific bench if set
    if bench_task == 'train':
        model = DetBenchTrain(model, config)
    elif bench_task == 'predict':
        model = DetBenchPredict(model, config)
    return model


def create_model_from_config(config, bench_name='', pretrained=False, checkpoint_path='', **kwargs):
    model = EfficientDet(config, **kwargs)

    # FIXME handle different head classes / anchors and re-init of necessary layers w/ pretrained load

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    elif pretrained:
        load_pretrained(model, config.url)

    # wrap model in task specific bench if set
    if bench_name == 'train':
        model = DetBenchTrain(model, config)
    elif bench_name == 'predict':
        model = DetBenchPredict(model, config)
    return model
