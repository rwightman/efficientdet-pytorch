import torch
import os
import logging
from collections import OrderedDict

from timm.models import load_checkpoint

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def load_pretrained(model, url, filter_fn=None, strict=True):
    if not url:
        logging.warning("Pretrained model URL is empty, using random initialization. "
                        "Did you intend to use a `tf_` variant of the model?")
        return
    state_dict = load_state_dict_from_url(url, progress=False, map_location='cpu')
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)
    model.load_state_dict(state_dict, strict=strict)
