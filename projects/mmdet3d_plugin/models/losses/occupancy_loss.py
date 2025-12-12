import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class OccupancyLoss(nn.Module):
    """Binary Cross Entropy Loss for occupancy maps.
    
    Args:
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
            Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """
    
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(OccupancyLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred_occupancy, gt_occupancy, reduction_override=None, **kwargs):
        """Forward function of loss calculation.
        
        Args:
            pred_occupancy (torch.Tensor): Predicted occupancy map with shape [B, C, H, W].
            gt_occupancy (torch.Tensor): Ground truth occupancy map with shape [B, C, H, W].
            reduction_override (str, optional): Method to reduce losses.
                Options: 'none', 'sum', 'mean'. Defaults to None.
        
        Returns:
            torch.Tensor: Occupancy loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy(
            pred_occupancy, gt_occupancy, reduction='none'
        )
        
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        # 'none' case: return as-is
        
        return loss * self.loss_weight

