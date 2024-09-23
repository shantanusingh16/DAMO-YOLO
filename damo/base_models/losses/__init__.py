# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from .dice_loss import DiceLoss
from .cross_entropy import CrossEntropyLoss


def build_loss(cfg):

    loss_cfg = copy.deepcopy(cfg)
    name = loss_cfg.pop('name')
    if name == 'DiceLoss':
        return DiceLoss(**loss_cfg)
    elif name == 'CrossEntropyLoss':
        return CrossEntropyLoss(**loss_cfg)
    else:
        raise NotImplementedError('loss name not supported')
