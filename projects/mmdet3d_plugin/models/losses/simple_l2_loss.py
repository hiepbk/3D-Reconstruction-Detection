# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES


def simple_l2_loss(pred_points, gt_points, reduction='mean'):
    """Simple point-wise L2 loss (O(N) complexity, no distance matrices).
    
    Assumes points are already aligned (same order). This is the simplest
    possible loss that doesn't depend on creating distance matrices.
    
    Args:
        pred_points (torch.Tensor): Predicted point cloud with shape [B, N, C].
        gt_points (torch.Tensor): Ground truth point cloud with shape [B, N, C].
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
    
    Returns:
        torch.Tensor: L2 loss value.
    """
    # Simple point-wise L2 distance (O(N), no distance matrices)
    diff = pred_points - gt_points  # (B, N, C)
    l2_dist = (diff ** 2).sum(dim=2)  # (B, N) - L2 distance per point
    
    if reduction == 'mean':
        return l2_dist.mean()
    elif reduction == 'sum':
        return l2_dist.sum()
    elif reduction == 'none':
        return l2_dist
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


@LOSSES.register_module()
class SimpleL2Loss(nn.Module):
    """Simple point-wise L2 loss for point cloud refinement.
    
    This is a memory-efficient loss that doesn't create distance matrices.
    O(N) complexity instead of O(N*M).
    
    Args:
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
            Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """
    
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SimpleL2Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred_points, gt_points, reduction_override=None, **kwargs):
        """Forward function of loss calculation.
        
        Args:
            pred_points (torch.Tensor): Predicted point cloud with shape [B, N, C].
            gt_points (torch.Tensor): Ground truth point cloud with shape [B, N, C].
            reduction_override (str, optional): Method to reduce losses.
                Options: 'none', 'sum', 'mean'. Defaults to None.
        
        Returns:
            torch.Tensor: L2 loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss = simple_l2_loss(
            pred_points, gt_points, reduction=reduction
        )
        
        return loss * self.loss_weight

