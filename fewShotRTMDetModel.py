import torch
# import RTMDet model
from mmdet.models.detectors import RTMDet

from mmdet.registry import MODELS

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.losses import CrossEntropyLoss
from mmdet.models.losses import Accuracy
from mmdet.models import RTMDet
from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead, MaskFeatModule
from mmdet.registry import MODELS



@MODELS.register_module()
class FewShotRTMDet(RTMDet):

    def __init__(self, num_classes, num_shots=5, num_epoch=30):
        super().__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            neck=neck,
        )
        self.num_classes = num_classes
        self.num_shots = num_shots
        self.loss = CrossEntropyLoss()
        self.epoch = num_epoch

    def fit(self, train_data, labels):
        for epoch in range(self.num_epoch):
            for image, label in train_data:
                # Extract features from the image
                features = self.model(image)
                # Classify the features
                predictions = self.fc(features)
                # Compute the loss
                loss = self.loss(labels, predictions)
                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self, test_data, labels):
        for image, label in test_data:
            # Extract features from the image
            features = self.model(image)
            # Classify the features
            predictions = self.fc(features)
            # Compute the accuracy
            acc = Accuracy(labels, predictions)
            return acc


if __name__ == '__main__':
    # Define the configuration for the model
    backbone = dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.33,
        widen_factor=1.25,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True))
    neck = dict(
        type='CSPNeXtPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True))
    rpn_head = dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor = dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head = dict(
        type='RTMDetInsSepBNHead',
        num_classes=17,
        in_channels=320,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=320,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss', loss_weight=2.0, eps=5e-06, reduction='mean'))

    # Create the model instance
    model = FewShotRTMDet(
        num_classes=17,
        num_shots=5,
        num_epoch=1
    )
    print(model)
    # model.fit(
    #     train_data=,
    #     labels=,
    # )
    #
    # accuracy = model.evaluate(
    #     test_data=,
    #     labels=,
    # )
    #
    # model.save('rtmdet-few-shot-test.pkl')
