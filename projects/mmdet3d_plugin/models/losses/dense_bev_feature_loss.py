"""
Dense BEV Feature Alignment Loss.

Computes cosine similarity loss between pseudo and GT dense BEV features
(after SECOND backbone + SECONDFPN neck), with optional foreground masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from typing import Optional


@LOSSES.register_module()
class DenseBEVFeatureLoss(nn.Module):
    """
    Dense BEV feature alignment loss using cosine similarity.
    
    This loss aligns the final dense BEV features (after backbone + neck)
    that are consumed by the detection head. It serves as an auxiliary
    regularizer to complement sparse feature alignment.
    
    Args:
        loss_weight (float): Weight of loss. Defaults to 0.1 (auxiliary loss).
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
            Defaults to 'mean'.
        eps (float): Small epsilon for numerical stability. Defaults to 1e-6.
        use_foreground_mask (bool): Whether to apply foreground masking.
            Defaults to True.
        mask_threshold (float): Threshold for foreground mask (teacher energy).
            Defaults to 0.01.
        mask_type (str): Type of mask. Options: 'teacher_energy', 'topk'.
            Defaults to 'teacher_energy'.
        topk_ratio (float): Ratio of top-k pixels to keep (if mask_type='topk').
            Defaults to 0.1 (10%).
    """
    
    def __init__(
        self,
        loss_weight: float = 0.1,
        reduction: str = 'mean',
        eps: float = 1e-6,
        use_foreground_mask: bool = True,
        mask_threshold: float = 0.01,
        mask_type: str = 'teacher_energy',  # 'teacher_energy' or 'topk'
        topk_ratio: float = 0.1,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps
        self.use_foreground_mask = use_foreground_mask
        self.mask_threshold = mask_threshold
        self.mask_type = mask_type
        self.topk_ratio = topk_ratio
        
        if mask_type not in ['teacher_energy', 'topk']:
            raise ValueError(f"Unknown mask_type: {mask_type}. Must be 'teacher_energy' or 'topk'")
    
    def forward(
        self,
        pseudo_bev: torch.Tensor,  # [B, C, H, W] - Student (pseudo) dense BEV features
        gt_bev: torch.Tensor,       # [B, C, H, W] - Teacher (GT) dense BEV features
        reduction_override: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute dense BEV feature alignment loss.
        
        Args:
            pseudo_bev: Pseudo dense BEV features [B, C, H, W]
            gt_bev: GT dense BEV features [B, C, H, W]
            reduction_override: Override reduction method
        
        Returns:
            loss: Scalar loss value
        """
        reduction = reduction_override if reduction_override else self.reduction
        
        # Ensure same shape
        assert pseudo_bev.shape == gt_bev.shape, \
            f"Shape mismatch: pseudo_bev {pseudo_bev.shape} vs gt_bev {gt_bev.shape}"
        
        B, C, H, W = pseudo_bev.shape
        
        # Normalize channel-wise (per pixel)
        # Normalize along channel dimension: [B, C, H, W] -> [B, C, H, W] (L2 normalized per pixel)
        pseudo_bev_norm = F.normalize(pseudo_bev, p=2, dim=1, eps=self.eps)
        
        # Teacher features: stop gradient (no backprop to GT backbone)
        with torch.no_grad():
            gt_bev_norm = F.normalize(gt_bev, p=2, dim=1, eps=self.eps)
        
        # Compute cosine similarity per pixel: [B, H, W]
        cosine_sim = (pseudo_bev_norm * gt_bev_norm).sum(dim=1)  # [B, H, W]
        
        # Cosine distance loss: 1 - cosine_similarity
        loss = 1.0 - cosine_sim  # [B, H, W]
        
        # Apply foreground masking if enabled
        if self.use_foreground_mask:
            mask = self._compute_foreground_mask(gt_bev)  # [B, H, W]
            # Apply mask: only compute loss on foreground pixels
            loss = loss * mask  # [B, H, W]
            # Average over masked pixels (avoid division by zero)
            num_foreground = mask.sum()
            if num_foreground > 0:
                loss = loss.sum() / num_foreground
            else:
                # No foreground pixels, return zero loss
                loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        else:
            # No masking: average over all pixels
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()
            # 'none' case: return as-is [B, H, W]
        
        return loss * self.loss_weight
    
    def _compute_foreground_mask(
        self,
        gt_bev: torch.Tensor,  # [B, C, H, W]
    ) -> torch.Tensor:  # [B, H, W] - binary mask
        """
        Compute foreground mask from teacher BEV features.
        
        Args:
            gt_bev: GT dense BEV features [B, C, H, W]
        
        Returns:
            mask: Binary foreground mask [B, H, W] (1 = foreground, 0 = background)
        """
        if self.mask_type == 'teacher_energy':
            # Method 1: Teacher occupancy mask
            # Sum absolute values along channel dimension: [B, C, H, W] -> [B, H, W]
            energy = gt_bev.abs().sum(dim=1)  # [B, H, W]
            mask = (energy > self.mask_threshold).float()  # [B, H, W]
        
        elif self.mask_type == 'topk':
            # Method 2: Top-k energy pixels
            # Sum absolute values along channel dimension: [B, C, H, W] -> [B, H, W]
            energy = gt_bev.abs().sum(dim=1)  # [B, H, W]
            B, H, W = energy.shape
            k = max(1, int(self.topk_ratio * H * W))  # Top k pixels per batch
            
            # Flatten spatial dimensions: [B, H, W] -> [B, H*W]
            energy_flat = energy.view(B, -1)  # [B, H*W]
            
            # Get top-k indices per batch
            _, topk_indices = energy_flat.topk(k, dim=1)  # [B, k]
            
            # Create mask: [B, H*W]
            mask_flat = torch.zeros_like(energy_flat)
            mask_flat.scatter_(1, topk_indices, 1.0)  # Set top-k to 1
            
            # Reshape back: [B, H*W] -> [B, H, W]
            mask = mask_flat.view(B, H, W)
        
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")
        
        return mask

