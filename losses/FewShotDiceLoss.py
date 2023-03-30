from typing import Optional

from mmdet.models.losses import DiceLoss
from mmdet.registry import MODELS


@MODELS.register_module()
class FewShotDiceLoss(DiceLoss):
    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None,
                support_info: Optional[dict] = None):
        if support_info is not None:
            query_pred = pred
            query_target = target
            support_pred = support_info["support_pred"]
            support_target = support_info["support_target"]

            query_loss = super().forward(
                query_pred,
                query_target,
                weight=weight,
                reduction_override=reduction_override,
                avg_factor=avg_factor
            )

            support_loss = super().forward(
                support_pred,
                support_target,
                weight=weight,
                reduction_override=reduction_override,
                avg_factor=avg_factor
            )

            return query_loss, support_loss

        return super().forward(
            pred,
            target,
            weight=weight,
            reduction_override=reduction_override,
            avg_factor=avg_factor
        )
