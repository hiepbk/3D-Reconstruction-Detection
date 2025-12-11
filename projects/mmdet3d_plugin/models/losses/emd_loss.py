# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


def emd_loss(pred_points, gt_points, temperature=0.1, reduction='mean'):
    """Approximate Earth Mover's Distance using soft assignment.
    
    Uses a differentiable approximation via soft assignment (temperature-scaled softmin).
    This is more efficient than exact EMD and works well for point cloud refinement.
    
    Args:
        pred_points (torch.Tensor): Predicted point cloud with shape [B, M, C].
        gt_points (torch.Tensor): Ground truth point cloud with shape [B, N, C].
        temperature (float): Temperature for soft assignment. Lower = sharper assignment.
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
    
    Returns:
        torch.Tensor: EMD loss value.
    """
    B, M, C = pred_points.shape
    _, N, _ = gt_points.shape
    
    # Compute pairwise distances: (B, M, N)
    dists = torch.cdist(pred_points, gt_points, p=2)  # L2 distance
    
    # Soft assignment using softmin (temperature-scaled softmax of negative distances)
    # softmin(x) = softmax(-x / temperature)
    soft_assign = F.softmin(dists / temperature, dim=2)  # (B, M, N)
    
    # Weighted distance: for each predicted point, compute weighted distance to GT
    weighted_dists = (soft_assign * dists).sum(dim=2)  # (B, M)
    
    # Average over points
    emd_loss = weighted_dists.mean(dim=1)  # (B,)
    
    if reduction == 'mean':
        return emd_loss.mean()
    elif reduction == 'sum':
        return emd_loss.sum()
    elif reduction == 'none':
        return emd_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


@LOSSES.register_module()
class EMDLoss(nn.Module):
    """Approximate Earth Mover's Distance Loss for point cloud refinement.
    
    Args:
        temperature (float): Temperature for soft assignment. Lower = sharper assignment.
            Defaults to 0.1.
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
            Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """
    
    def __init__(self, temperature=0.1, reduction='mean', loss_weight=1.0):
        super(EMDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred_points, gt_points, reduction_override=None, **kwargs):
        """Forward function of loss calculation.
        
        Args:
            pred_points (torch.Tensor): Predicted point cloud with shape [B, M, C].
            gt_points (torch.Tensor): Ground truth point cloud with shape [B, N, C].
            reduction_override (str, optional): Method to reduce losses.
                Options: 'none', 'sum', 'mean'. Defaults to None.
        
        Returns:
            torch.Tensor: EMD loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss = emd_loss(
            pred_points, gt_points, 
            temperature=self.temperature, 
            reduction=reduction
        )
        
        return loss * self.loss_weight

