import itertools

from omegaconf import OmegaConf


def bifpn_config(min_level, max_level, weight_method=None):
    """BiFPN config.
    Adapted from https://github.com/google/automl/blob/56815c9986ffd4b508fe1d68508e268d129715c1/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        p.nodes.append({
            'reduction': 1 << i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)],
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        p.nodes.append({
            'reduction': 1 << i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)],
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))
    return p


def panfpn_config(min_level, max_level, weight_method=None):
    """PAN FPN config.

    This defines FPN layout from Path Aggregation Networks as an alternate to
    BiFPN, it does not implement the full PAN spec.

    Paper: https://arxiv.org/abs/1803.01534
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level, min_level - 1, -1):
        # top-down path.
        offsets = [level_last_id(i), level_last_id(i + 1)] if i != max_level else [level_last_id(i)]
        p.nodes.append({
            'reduction': 1 << i,
            'inputs_offsets': offsets,
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level, max_level + 1):
        # bottom-up path.
        offsets = [level_last_id(i), level_last_id(i - 1)] if i != min_level else [level_last_id(i)]
        p.nodes.append({
            'reduction': 1 << i,
            'inputs_offsets': offsets,
            'weight_method': weight_method,
        })
        node_ids[i].append(next(id_cnt))

    return p


def qufpn_config(min_level, max_level, weight_method=None):
    """A dynamic quad fpn config that can adapt to different min/max levels.

    It extends the idea of BiFPN, and has four paths:
        (up_down -> bottom_up) + (bottom_up -> up_down).

    Paper: https://ieeexplore.ieee.org/document/9225379
    Ref code: From contribution to TF EfficientDet
    https://github.com/google/automl/blob/eb74c6739382e9444817d2ad97c4582dbe9a9020/efficientdet/keras/fpn_configs.py
    """
    p = OmegaConf.create()
    weight_method = weight_method or 'fastattn'
    quad_method = 'fastattn'
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    level_first_id = lambda level: node_ids[level][0]
    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path 1.
        p.nodes.append({
            'reduction': 1 << i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    for i in range(min_level + 1, max_level):
        # bottom-up path 2.
        p.nodes.append({
            'reduction': 1 << i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))

    i = max_level
    p.nodes.append({
        'reduction': 1 << i,
        'inputs_offsets': [level_first_id(i)] + [level_last_id(i - 1)],
        'weight_method': weight_method
    })
    node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(min_level + 1, max_level + 1, 1):
        # bottom-up path 3.
        p.nodes.append({
            'reduction': 1 << i,
            'inputs_offsets': [
                level_first_id(i), level_last_id(i - 1) if i != min_level + 1 else level_first_id(i - 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(max_level - 1, min_level, -1):
        # top-down path 4.
        p.nodes.append({
            'reduction': 1 << i,
            'inputs_offsets': [node_ids[i][0]] + [node_ids[i][-1]] + [level_last_id(i + 1)],
            'weight_method': weight_method
        })
        node_ids[i].append(next(id_cnt))
    i = min_level
    p.nodes.append({
        'reduction': 1 << i,
        'inputs_offsets': [node_ids[i][0]] + [level_last_id(i + 1)],
        'weight_method': weight_method
    })
    node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    # NOTE: the order of the quad path is reversed from the original, my code expects the output of
    # each FPN repeat to be same as input from backbone, in order of increasing reductions
    for i in range(min_level, max_level + 1):
        # quad-add path.
        p.nodes.append({
            'reduction': 1 << i,
            'inputs_offsets': [node_ids[i][2], node_ids[i][4]],
            'weight_method': quad_method
        })
        node_ids[i].append(next(id_cnt))

    return p


def get_fpn_config(fpn_name, min_level=3, max_level=7):
    if not fpn_name:
        fpn_name = 'bifpn_fa'
    name_to_config = {
        'bifpn_sum': bifpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
        'bifpn_attn': bifpn_config(min_level=min_level, max_level=max_level, weight_method='attn'),
        'bifpn_fa': bifpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
        'pan_sum': panfpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
        'pan_fa': panfpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
        'qufpn_sum': qufpn_config(min_level=min_level, max_level=max_level, weight_method='sum'),
        'qufpn_fa': qufpn_config(min_level=min_level, max_level=max_level, weight_method='fastattn'),
    }
    return name_to_config[fpn_name]
