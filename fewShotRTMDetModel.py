import torch
import torch.nn as nn

from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
from mmdet.models.dense_heads.rpn_head import RPNHead
# import RTMDet model
from mmdet.models.detectors import RTMDet
from roi_heads.multi_relation_roi_head import MultiRelationRoIHead


class FewShotRTMDet(RTMDet):
    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 num_support=5):
        super(RTMDet, self).__init__(backbone, neck, bbox_roi_extractor, bbox_head, train_cfg, test_cfg, pretrained)

        self.roi_head = StandardRoIHead(self.bbox_roi_extractor, bbox_head)
        self.few_shot_head = MultiRelationRoIHead(rpn_head['out_channels'], num_support)

    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def extract_feat_support(self, support_imgs):
        # Extract feature maps for the support set
        x = self.backbone(support_imgs.reshape((-1,) + support_imgs.shape[2:]))
        x = self.neck(x)
        return x

    def extract_feat_query(self, query_img):
        # Extract feature map for the query image
        x = self.backbone(query_img)
        x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, support_imgs):
        # Extract feature maps for the query image and the support set
        x = self.extract_feat(img)
        supports = self.extract_feat_support(support_imgs)

        # Concatenate the query image and support set feature maps
        num_support = support_imgs.shape[0]
        supports = supports.reshape(num_support, -1, *supports.shape[1:])
        supports = supports.mean(dim=0, keepdim=True).expand(x[0].shape[0], -1, x[0].shape[2], x[0].shape[3])
        x = torch.cat([x, supports], dim=1)

        # Pass the concatenated feature maps through the few-shot head
        x = self.few_shot_head(x)

        # Pass the output of the few-shot head to the ROI head
        roi_feats = self.roi_head(x, None, img_metas)

        # Compute loss
        bbox_targets = self.roi_head.get_target(gt_bboxes, gt_labels)
        loss_bbox = self.roi_head.loss(bbox_targets, roi_feats)
        return dict(loss_bbox=loss_bbox)


if __name__ == '__main__':
    # Define the configuration for the model
    backbone = ResNet(depth=50)
    neck = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)
    rpn_head = {'type': 'RPNHead', 'in_channels': 256, 'feat_channels': 256, 'anchor_scales': [8],
                'anchor_ratios': [0.5, 1.0, 2.0], 'anchor_strides': [4, 8, 16, 32, 64]}
    bbox_roi_extractor = {'type': 'SingleRoIExtractor',
                          'roi_layer': {'type': 'RoIAlign', 'out_size': 7, 'sample_num': 2}, 'out_channels': 256,
                          'featmap_strides': [4, 8, 16, 32]}
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

    # Define the train and test configurations
    train_cfg = {'rpn': {
        'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.7, 'neg_iou_thr': 0.3, 'min_pos_iou': 0.3,
                     'ignore_iof_thr': -1},
        'sampler': {'type': 'RandomSampler', 'num': 256, 'pos_fraction': 0.5, 'neg_pos_ub': -1,
                    'add_gt_as_proposals': True}, 'allowed_border': 0, 'pos_weight': -1, 'debug': False}, 'rcnn': {
        'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.5, 'min_pos_iou': 0.5,
                     'ignore_iof_thr': -1},
        'sampler': {'type': 'RandomSampler', 'num': 512, 'pos_fraction': 0.25, 'neg_pos_ub': -1,
                    'add_gt_as_proposals': True}, 'pos_weight': -1, 'debug': False}}
    test_cfg = {'rpn': {'nms_across_levels': False, 'nms_pre': 1000, 'nms_post': 1000, 'max_num': 1000, 'nms_thr': 0.7,
                        'min_bbox_size': 0},
                'rcnn': {'score_thr': 0.05, 'nms': {'type': 'nms', 'iou_thr': 0.5}, 'max_per_img': 100}}

    # Create the model instance
    model = FewShotRTMDet(backbone=backbone,
                          neck=neck,
                          rpn_head=rpn_head,
                          bbox_roi_extractor=bbox_roi_extractor,
                          bbox_head=bbox_head,
                          train_cfg=train_cfg,
                          test_cfg=test_cfg,
                          pretrained=None,
                          num_support=5)