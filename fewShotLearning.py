import torch
import torch.nn as nn

from mmdet.models.roi_heads.bbox_heads import BBoxHead
# import RTMDet model
from mmdet.models.detectors import RTMDet


class FewShotHead(nn.Module):
    def __init__(self, in_channels, num_support):
        super(FewShotHead, self).__init__()
        self.conv = nn.Conv2d(in_channels * (num_support + 1), in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


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

        self.rpn_head = builder.build_head(rpn_head)
        self.few_shot_head = FewShotHead(rpn_head['out_channels'], num_support)

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

        # Pass the output of the few-shot head to the RPN and ROI head
        rpn_outs = self.rpn_head(x)
        rois = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rpn_outs[0],
            img_metas)
        bbox_feats = self.bbox_head(rois)

        # Compute loss
        losses = self.bbox_head.loss(bbox_feats, rois, gt_bboxes, gt_labels)
        return losses