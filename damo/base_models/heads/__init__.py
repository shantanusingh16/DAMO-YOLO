# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from .zero_head import ZeroHead
from .fpn_head import FPNHead


def build_head(cfg):

    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'ZeroHead':
        return ZeroHead(**head_cfg)
    elif name == 'FPNHead':
        return FPNHead(**head_cfg)
    else:
        raise NotImplementedError
