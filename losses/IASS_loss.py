import torch
import torch.nn.functional as F

class InstanceAwareSemanticSegmentationLoss(torch.nn.Module):
    def __init__(self):
        super(InstanceAwareSemanticSegmentationLoss, self).__init__()

    def forward(self, mask_preds, mask_feats, gt_masks, gt_labels):
        """
        Args:
            mask_preds (Tensor): Predicted masks of shape (N, H, W), where N is the number of instances.
            mask_feats (Tensor): Mask features of shape (N, F, H, W), where F is the number of feature channels.
            gt_masks (Tensor): Ground truth masks of shape (N, H, W).
            gt_labels (Tensor): Ground truth labels of shape (N,).
        Returns:
            iass_loss (Tensor): The instance-aware semantic segmentation loss.
        """

        # Calculate classification loss
        mask_scores = torch.sigmoid(torch.sum(mask_preds * mask_feats, dim=(1, 2, 3)))
        gt_mask_scores = torch.where(gt_masks.sum(dim=(1, 2)) > 0, 1.0, 0.0)
        classification_loss = F.binary_cross_entropy(mask_scores, gt_mask_scores)

        # Calculate mask loss
        # Resize the predicted masks to match the ground truth mask size
        mask_preds_resized = F.interpolate(mask_preds.unsqueeze(1), size=gt_masks.shape[-2:], mode='bilinear', align_corners=False)
        mask_loss = F.binary_cross_entropy_with_logits(mask_preds_resized, gt_masks.unsqueeze(1))

        # Combine the losses
        iass_loss = classification_loss + mask_loss

        return iass_loss
