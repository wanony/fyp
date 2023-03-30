from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from mmdet.models.dense_heads import RTMDetInsSepBNHead
from mmdet.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class FewShotRTMDetInsSepBNHead(RTMDetInsSepBNHead):
    def __init__(self, num_ways: int, num_shots: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_ways = num_ways
        self.num_shots = num_shots

    def forward(self, feats: Tuple[Tensor, ...], support_info: Optional[dict] = None) -> tuple:
        # First, compute the usual outputs
        cls_scores, bbox_preds, kernel_preds, mask_feat = super().forward(feats)

        # If support_info is provided, perform few-shot learning
        if support_info is not None:
            support_feats = support_info["support_feats"]
            support_labels = support_info["support_labels"]

            # Compute support set outputs
            support_cls_scores, support_bbox_preds, support_kernel_preds, support_mask_feats = super().forward(
                support_feats)

            # Calculate prototypes for each class in the support set for each level
            support_cls_scores_level = []
            for level in range(len(support_cls_scores)):
                level_support_cls_scores = []
                for idx in range(self.num_ways):
                    level_support_cls_scores.append(support_cls_scores[level][support_labels[idx]].mean(dim=0))
                support_cls_scores_level.append(torch.stack(level_support_cls_scores, dim=0))

            # Compute the distances between the query set and support set prototypes for each level
            dists = []
            for level_cls_scores, level_support_cls_scores in zip(cls_scores, support_cls_scores_level):
                level_dists = torch.stack(
                    [torch.norm(level_cls_scores - level_support_cls_scores[i], dim=1) for i in range(self.num_ways)],
                    dim=1
                )
                dists.append(level_dists)

            # Calculate the softmax probabilities for the class predictions for each level
            pred_probs = [F.softmax(-level_dists, dim=1) for level_dists in dists]

            return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feat, tuple(pred_probs)

        # Otherwise, return the usual outputs
        return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feat


