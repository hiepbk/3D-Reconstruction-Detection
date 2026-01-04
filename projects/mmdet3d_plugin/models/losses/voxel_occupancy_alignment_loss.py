"""
Voxel Occupancy Alignment Loss.

Computes Dice loss between pseudo and GT voxel occupancy masks.
This is the most important loss for fixing over-generation and density mismatch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class VoxelOccupancyAlignmentLoss(nn.Module):
    """
    Dice loss for voxel occupancy alignment.
    
    Compares which voxels are non-empty between pseudo and GT sparse indices.
    This directly addresses over-generation and density mismatch.
    
    Args:
        loss_type (str): Type of loss. Options: 'dice', 'bce', 'focal'.
            Defaults to 'dice' (recommended).
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
            Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        eps (float): Small epsilon for numerical stability. Defaults to 1e-6.
    """
    
    def __init__(
        self,
        loss_type='dice',
        reduction='mean',
        loss_weight=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
    
    def forward(
        self,
        pseudo_indices: torch.Tensor,  # [Np, 4] (batch, z, y, x)
        gt_indices: torch.Tensor,      # [Ng, 4] (batch, z, y, x)
        spatial_shape: list,           # [Z, Y, X]
        reduction_override=None,
    ) -> torch.Tensor:
        """
        Compute occupancy alignment loss.
        
        Args:
            pseudo_indices: Sparse voxel indices from pseudo point cloud
            gt_indices: Sparse voxel indices from GT point cloud
            spatial_shape: Spatial shape [Z, Y, X] of the sparse grid
            reduction_override: Override reduction method
        
        Returns:
            loss: Scalar loss value
        """
        reduction = reduction_override if reduction_override else self.reduction
        
        # Convert sparse indices to dense occupancy masks
        pseudo_occ = self._indices_to_occupancy_mask(pseudo_indices, spatial_shape)
        gt_occ = self._indices_to_occupancy_mask(gt_indices, spatial_shape)
        
        if self.loss_type == 'dice':
            loss = self._compute_dice_loss(pseudo_occ, gt_occ)
        elif self.loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(
                pseudo_occ.float(), 
                gt_occ.float(),
                reduction='none'
            )
        elif self.loss_type == 'focal':
            loss = self._compute_focal_loss(pseudo_occ, gt_occ)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Apply reduction
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        # 'none' case: return as-is
        
        return loss * self.loss_weight
    
    def _indices_to_occupancy_mask(
        self,
        indices: torch.Tensor,  # [N, 4]
        spatial_shape: list,     # [Z, Y, X]
    ) -> torch.Tensor:
        """
        Convert sparse indices to dense binary occupancy mask.
        
        Args:
            indices: [N, 4] tensor with (batch, z, y, x)
            spatial_shape: [Z, Y, X]
        
        Returns:
            occupancy_mask: [B, Z, Y, X] binary mask
        """
        if indices.shape[0] == 0:
            # Empty indices -> return zero mask
            # Need to infer batch size from indices or use default
            # For now, assume batch_size=1 if empty
            return torch.zeros((1, *spatial_shape), device=indices.device, dtype=torch.float32)
        
        batch_size = int(indices[:, 0].max().item()) + 1
        Z, Y, X = spatial_shape
        
        # Create dense mask
        occupancy_mask = torch.zeros(
            (batch_size, Z, Y, X),
            device=indices.device,
            dtype=torch.float32
        )
        
        # Set occupied voxels to 1
        batch_idx = indices[:, 0].long()
        z_idx = indices[:, 1].long()
        y_idx = indices[:, 2].long()
        x_idx = indices[:, 3].long()
        
        # Clamp indices to valid range
        z_idx = torch.clamp(z_idx, 0, Z - 1)
        y_idx = torch.clamp(y_idx, 0, Y - 1)
        x_idx = torch.clamp(x_idx, 0, X - 1)
        
        occupancy_mask[batch_idx, z_idx, y_idx, x_idx] = 1.0
        
        return occupancy_mask
    
    def _compute_dice_loss(
        self,
        pred: torch.Tensor,  # [B, Z, Y, X]
        target: torch.Tensor, # [B, Z, Y, X]
    ) -> torch.Tensor:
        """
        Compute Dice loss: 1 - Dice_coefficient
        
        Dice = (2 * intersection + eps) / (sum(pred) + sum(target) + eps)
        """
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.shape[0], -1)  # [B, Z*Y*X]
        target_flat = target.view(target.shape[0], -1)  # [B, Z*Y*X]
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)  # [B]
        pred_sum = pred_flat.sum(dim=1)  # [B]
        target_sum = target_flat.sum(dim=1)  # [B]
        
        # Dice coefficient
        dice = (2.0 * intersection + self.eps) / (pred_sum + target_sum + self.eps)
        
        # Dice loss = 1 - Dice
        loss = 1.0 - dice  # [B]
        
        return loss
    
    def _compute_focal_loss(
        self,
        pred: torch.Tensor,  # [B, Z, Y, X]
        target: torch.Tensor, # [B, Z, Y, X]
    ) -> torch.Tensor:
        """
        Compute focal loss for binary classification.
        """
        # Convert to probabilities (if not already)
        if pred.max() > 1.0 or pred.min() < 0.0:
            # Assume logits, convert to probs
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred
        
        # Focal loss parameters
        alpha = 0.25
        gamma = 2.0
        
        # BCE term
        bce = F.binary_cross_entropy(pred_prob, target, reduction='none')
        
        # Focal term
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_weight = (1 - p_t) ** gamma
        
        # Apply alpha
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        
        loss = alpha_t * focal_weight * bce
        
        return loss

