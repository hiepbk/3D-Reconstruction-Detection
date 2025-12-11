# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


def color_loss(pred_colors, gt_colors, mode='l1', reduction='mean'):
    """Color loss for point cloud refinement.
    
    Computes loss between predicted and ground truth colors.
    Uses Chamfer-like matching: for each predicted point, find closest GT point
    and compute color difference.
    
    Args:
        pred_colors (torch.Tensor): Predicted colors with shape [B, M, 3] (RGB).
        gt_colors (torch.Tensor): Ground truth colors with shape [B, N, 3] (RGB).
        mode (str): Loss mode. Options: 'l1', 'l2', 'smooth_l1'. Defaults to 'l1'.
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
    
    Returns:
        torch.Tensor: Color loss value.
    """
    B, M, _ = pred_colors.shape
    _, N, _ = gt_colors.shape
    
    # Compute pairwise color distances: (B, M, N)
    color_dists = torch.cdist(pred_colors, gt_colors, p=2)  # L2 distance in RGB space
    
    # For each predicted color, find closest GT color
    min_dists, _ = color_dists.min(dim=2)  # (B, M)
    
    # Average over points
    if reduction == 'mean':
        return min_dists.mean()
    elif reduction == 'sum':
        return min_dists.sum()
    elif reduction == 'none':
        return min_dists
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


@LOSSES.register_module()
class ColorLoss(nn.Module):
    """Color loss for point cloud refinement.
    
    Args:
        mode (str): Loss mode. Options: 'l1', 'l2', 'smooth_l1'. 
            Defaults to 'l1'.
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
            Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """
    
    def __init__(self, mode='l1', reduction='mean', loss_weight=1.0):
        super(ColorLoss, self).__init__()
        assert mode in ['l1', 'l2', 'smooth_l1']
        assert reduction in ['none', 'sum', 'mean']
        self.mode = mode
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred_colors, gt_colors, reduction_override=None, **kwargs):
        """Forward function of loss calculation.
        
        Args:
            pred_colors (torch.Tensor): Predicted colors with shape [B, M, 3] (RGB).
            gt_colors (torch.Tensor): Ground truth colors with shape [B, N, 3] (RGB).
            reduction_override (str, optional): Method to reduce losses.
                Options: 'none', 'sum', 'mean'. Defaults to None.
        
        Returns:
            torch.Tensor: Color loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss = color_loss(
            pred_colors, gt_colors, 
            mode=self.mode, 
            reduction=reduction
        )
        
        return loss * self.loss_weight

