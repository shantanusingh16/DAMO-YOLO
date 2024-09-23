# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from .tinynas_csp import load_tinynas_net as load_tinynas_net_csp
from .tinynas_res import load_tinynas_net as load_tinynas_net_res
from .tinynas_mob import load_tinynas_net as load_tinynas_net_mob
from .repvit import load_repvit_net


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'TinyNAS_res':
        return load_tinynas_net_res(backbone_cfg)
    elif name == 'TinyNAS_csp':
        return load_tinynas_net_csp(backbone_cfg)
    elif name == 'TinyNAS_mob':
        return load_tinynas_net_mob(backbone_cfg)
    elif name == 'RepViT':
        return load_repvit_net(backbone_cfg)
    else:
        print(f'{name} is not supported yet!')
