#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        self.train.batch_size = 256
        self.train.base_lr_per_img = 0.01 / 64
        self.train.min_lr_ratio = 0.05
        self.train.weight_decay = 5e-4
        self.train.momentum = 0.9
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        # augment
        self.train.augment.transform.image_max_range = (640, 640)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 0.2
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)

        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        # backbone
        RepViT = {
            'name': 'RepViT',
            'model_type': 'repvit_m0_9',
            'init_cfg': {
                'type':'Pretrained',
                'checkpoint':'./ckpts/repvit_m0_9_distill_450e.pth',
                'strict':False
            },
            'out_indices': (3,7,21,24),  # (3,7,21,24)
        }

        self.model.backbone = RepViT

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 1.0,
            'hidden_ratio': 1.0,
            'in_channels': [96, 192, 384],
            'out_channels': [64, 128, 256],
            'act': 'relu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 1,
            'in_channels': [64, 128, 256],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.05,
            'nms_iou_thre': 0.7,
            'legacy': False,
        }
        self.model.head = ZeroHead

        FPNNeck = {
            'name': 'FPN',
            'in_channels': [48, 96, 192, 384],
            'out_channels': 256,
            'num_outs': 4,
        }
        self.model.seg_neck = FPNNeck

        FPNHead = {
            'name':'FPNHead',
            'in_channels':[256, 256, 256, 256],
            'in_index':[0, 1, 2, 3],
            'feature_strides':[4, 8, 16, 32],
            'channels':128,
            'dropout_ratio':0.1,
            'num_classes':6,
            'norm_cfg':None,
            'align_corners':False,
            'loss_decode':{'name': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0},
        }
        self.model.seg_head = FPNHead

        self.dataset.class_names = ['object', ]
