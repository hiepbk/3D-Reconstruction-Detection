# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES


def smoothness_loss(refined_points, pseudo_points, reduction='mean'):
    """Smoothness regularization loss for point cloud refinement.
    
    Penalizes large local variations in corrections (residuals) to ensure
    smooth refinement. This prevents overfitting and produces more natural
    point cloud corrections.
    
    Args:
        refined_points (torch.Tensor): Refined point cloud with shape [B, N, C].
        pseudo_points (torch.Tensor): Original pseudo point cloud with shape [B, N, C].
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
    
    Returns:
        torch.Tensor: Smoothness loss value.
    """
    # Compute corrections (residuals)
    corrections = refined_points - pseudo_points  # (B, N, C)
    
    # Compute variance of corrections across points
    # This penalizes large variations in corrections
    correction_var = corrections.var(dim=1, unbiased=False)  # (B, C)
    
    # Average over channels and batch
    if reduction == 'mean':
        return correction_var.mean()
    elif reduction == 'sum':
        return correction_var.sum()
    elif reduction == 'none':
        return correction_var
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


@LOSSES.register_module()
class SmoothnessLoss(nn.Module):
    """Smoothness regularization loss for point cloud refinement.
    
    Args:
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
            Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """
    
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SmoothnessLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, refined_points, pseudo_points, reduction_override=None, **kwargs):
        """Forward function of loss calculation.
        
        Args:
            refined_points (torch.Tensor): Refined point cloud with shape [B, N, C].
            pseudo_points (torch.Tensor): Original pseudo point cloud with shape [B, N, C].
            reduction_override (str, optional): Method to reduce losses.
                Options: 'none', 'sum', 'mean'. Defaults to None.
        
        Returns:
            torch.Tensor: Smoothness loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss = smoothness_loss(
            refined_points, pseudo_points, reduction=reduction
        )
        
        return loss * self.loss_weight

