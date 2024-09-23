# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from .giraffe_fpn_btn import GiraffeNeckV2
from .fpn import FPN


def build_neck(cfg):
    neck_cfg = copy.deepcopy(cfg)
    name = neck_cfg.pop('name')
    if name == 'GiraffeNeckV2':
        return GiraffeNeckV2(**neck_cfg)
    elif name == 'FPN':
        return FPN(**neck_cfg)
    else:
        raise NotImplementedError
