from .transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def resolve_input_config(args, model_config=None, model=None):
    if not isinstance(args, dict):
        args = vars(args)
    input_config = {}
    if not model_config and model is not None and hasattr(model, 'config'):
        model_config = model.config

    # Resolve input/image size
    in_chans = 3
    # if 'chans' in args and args['chans'] is not None:
    #     in_chans = args['chans']

    input_size = (in_chans, 512, 512)
    # if 'input_size' in args and args['input_size'] is not None:
    #     assert isinstance(args['input_size'], (tuple, list))
    #     assert len(args['input_size']) == 3
    #     input_size = tuple(args['input_size'])
    #     in_chans = input_size[0]  # input_size overrides in_chans
    # elif 'img_size' in args and args['img_size'] is not None:
    #     assert isinstance(args['img_size'], int)
    #     input_size = (in_chans, args['img_size'], args['img_size'])
    if 'input_size' in model_config:
        input_size = tuple(model_config['input_size'])
    elif 'image_size' in model_config:
        input_size = (in_chans,) + tuple(model_config['image_size'])
    assert isinstance(input_size, tuple) and len(input_size) == 3
    input_config['input_size'] = input_size

    # resolve interpolation method
    input_config['interpolation'] = 'bicubic'
    if 'interpolation' in args and args['interpolation']:
        input_config['interpolation'] = args['interpolation']
    elif 'interpolation' in model_config:
        input_config['interpolation'] = model_config['interpolation']

    # resolve dataset + model mean for normalization
    input_config['mean'] = IMAGENET_DEFAULT_MEAN
    if 'mean' in args and args['mean'] is not None:
        mean = tuple(args['mean'])
        if len(mean) == 1:
            mean = tuple(list(mean) * in_chans)
        else:
            assert len(mean) == in_chans
        input_config['mean'] = mean
    elif 'mean' in model_config:
        input_config['mean'] = model_config['mean']

    # resolve dataset + model std deviation for normalization
    input_config['std'] = IMAGENET_DEFAULT_STD
    if 'std' in args and args['std'] is not None:
        std = tuple(args['std'])
        if len(std) == 1:
            std = tuple(list(std) * in_chans)
        else:
            assert len(std) == in_chans
        input_config['std'] = std
    elif 'std' in model_config:
        input_config['std'] = model_config['std']

    # resolve letterbox fill color
    input_config['fill_color'] = 'mean'
    if 'fill_color' in args and args['fill_color'] is not None:
        input_config['fill_color'] = args['fill_color']
    elif 'fill_color' in model_config:
        input_config['fill_color'] = model_config['fill_color']

    return input_config
