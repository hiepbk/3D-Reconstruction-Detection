"""
Sparse Feature Alignment Loss.

Computes feature matching loss between pseudo and GT sparse features,
only at overlapping voxel locations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from typing import Tuple


@LOSSES.register_module()
class SparseFeatureAlignmentLoss(nn.Module):
    """
    Feature alignment loss for sparse voxel features.
    
    Matches features only at overlapping voxel coordinates.
    This aligns semantic meaning of voxels without requiring exact point matching.
    
    Args:
        loss_type (str): Type of loss. Options: 'l2', 'smooth_l1', 'cosine'.
            Defaults to 'l2'.
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
            Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 0.2.
        eps (float): Small epsilon for numerical stability. Defaults to 1e-6.
    """
    
    def __init__(
        self,
        loss_type='l2',
        reduction='mean',
        loss_weight=0.2,
        eps=1e-6,
        normalize_features=True,  # Normalize features before loss (prevents trivial solution)
        hard_mining_ratio=0.5,  # Use hardest N% of voxels (0.0 = use all, 1.0 = use hardest)
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.normalize_features = normalize_features
        self.hard_mining_ratio = hard_mining_ratio
    
    def forward(
        self,
        pseudo_features: torch.Tensor,  # [Np, C]
        pseudo_indices: torch.Tensor,    # [Np, 4]
        gt_features: torch.Tensor,       # [Ng, C]
        gt_indices: torch.Tensor,         # [Ng, 4]
        reduction_override=None,
    ) -> torch.Tensor:
        """
        Compute feature alignment loss at overlapping voxels.
        
        Args:
            pseudo_features: Sparse features from pseudo point cloud
            pseudo_indices: Voxel indices for pseudo features
            gt_features: Sparse features from GT point cloud
            gt_indices: Voxel indices for GT features
            reduction_override: Override reduction method
        
        Returns:
            loss: Scalar loss value (0.0 if no overlapping voxels)
        """
        reduction = reduction_override if reduction_override else self.reduction
        
        # Find overlapping voxels
        pseudo_feat_common, gt_feat_common = self._extract_overlapping_features(
            pseudo_features, pseudo_indices,
            gt_features, gt_indices
        )
        
        # If no overlap, return zero loss
        if pseudo_feat_common.shape[0] == 0:
            return torch.tensor(0.0, device=pseudo_features.device, requires_grad=True)
        
        # Normalize features (CRITICAL: prevents trivial solution)
        if self.normalize_features:
            pseudo_feat_common = F.normalize(pseudo_feat_common, p=2, dim=1, eps=self.eps)
            gt_feat_common = F.normalize(gt_feat_common, p=2, dim=1, eps=self.eps)
        
        # Hard voxel mining: keep only hardest voxels to maintain gradients
        if self.hard_mining_ratio > 0.0 and pseudo_feat_common.shape[0] > 1:
            # Compute per-voxel distances
            if self.loss_type == 'cosine' or self.normalize_features:
                # For normalized features, use cosine distance
                dist = 1.0 - (pseudo_feat_common * gt_feat_common).sum(dim=1)  # [N]
            else:
                # For raw features, use L2 distance
                dist = torch.norm(pseudo_feat_common - gt_feat_common, p=2, dim=1)  # [N]
            
            # Select hardest voxels
            k = max(1, int(self.hard_mining_ratio * len(dist)))
            hard_indices = dist.topk(k, largest=True).indices  # Hardest k voxels
            
            pseudo_feat_common = pseudo_feat_common[hard_indices]
            gt_feat_common = gt_feat_common[hard_indices]
        
        # Compute feature loss
        if self.loss_type == 'l2':
            loss = F.mse_loss(pseudo_feat_common, gt_feat_common, reduction='none')
            loss = loss.mean(dim=-1)  # Average over feature dimension
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pseudo_feat_common, gt_feat_common, reduction='none')
            loss = loss.mean(dim=-1)
        elif self.loss_type == 'cosine' or self.normalize_features:
            # Cosine distance = 1 - cosine_similarity
            # If features are normalized, this is equivalent to cosine loss
            cosine_sim = (pseudo_feat_common * gt_feat_common).sum(dim=1)  # [N]
            loss = 1.0 - cosine_sim
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Apply reduction
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        # 'none' case: return as-is
        
        return loss * self.loss_weight
    
    def _extract_overlapping_features(
        self,
        pseudo_features: torch.Tensor,  # [Np, C]
        pseudo_indices: torch.Tensor,    # [Np, 4]
        gt_features: torch.Tensor,       # [Ng, C]
        gt_indices: torch.Tensor,         # [Ng, 4]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features at overlapping voxel coordinates.
        
        Uses dictionary-based lookup for O(1) matching.
        
        Returns:
            pseudo_feat_common: [N_common, C] features at common voxels
            gt_feat_common: [N_common, C] features at common voxels
        """
        if pseudo_indices.shape[0] == 0 or gt_indices.shape[0] == 0:
            # No overlap possible
            device = pseudo_features.device
            feat_dim = pseudo_features.shape[1] if pseudo_features.shape[0] > 0 else gt_features.shape[1]
            return (
                torch.empty((0, feat_dim), device=device),
                torch.empty((0, feat_dim), device=device)
            )
        
        # Convert indices to tuples for dictionary lookup
        # Move to CPU for efficient Python dict operations
        def indices_to_dict(indices, features):
            """Create dict mapping voxel tuple to feature index."""
            voxel_to_idx = {}
            indices_np = indices.cpu().numpy()
            for i, idx in enumerate(indices_np):
                voxel_key = tuple(idx)
                # If duplicate voxel, keep first occurrence
                if voxel_key not in voxel_to_idx:
                    voxel_to_idx[voxel_key] = i
            return voxel_to_idx
        
        pseudo_voxel_dict = indices_to_dict(pseudo_indices, pseudo_features)
        gt_voxel_dict = indices_to_dict(gt_indices, gt_features)
        
        # Find common voxels
        common_voxels = set(pseudo_voxel_dict.keys()) & set(gt_voxel_dict.keys())
        
        if len(common_voxels) == 0:
            # No overlap
            device = pseudo_features.device
            feat_dim = pseudo_features.shape[1]
            return (
                torch.empty((0, feat_dim), device=device),
                torch.empty((0, feat_dim), device=device)
            )
        
        # Get indices for common voxels
        # Sort by voxel coordinates for consistent ordering
        common_voxels_sorted = sorted(common_voxels)
        
        pseudo_indices_list = [pseudo_voxel_dict[v] for v in common_voxels_sorted]
        gt_indices_list = [gt_voxel_dict[v] for v in common_voxels_sorted]
        
        # Convert to tensors
        pseudo_indices_tensor = torch.tensor(pseudo_indices_list, device=pseudo_features.device, dtype=torch.long)
        gt_indices_tensor = torch.tensor(gt_indices_list, device=gt_features.device, dtype=torch.long)
        
        # Extract features
        pseudo_feat_common = pseudo_features[pseudo_indices_tensor]
        gt_feat_common = gt_features[gt_indices_tensor]
        
        return pseudo_feat_common, gt_feat_common

