# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


def color_loss(pred_colors, gt_colors, mode='l1', reduction='mean', chunk_size=1024):
    """Color loss for point cloud refinement.
    
    Computes loss between predicted and ground truth colors.
    Uses Chamfer-like matching: for each predicted point, find closest GT point
    and compute color difference.
    
    Memory-efficient chunked computation to avoid creating full (B, M, N) tensors.
    
    Args:
        pred_colors (torch.Tensor): Predicted colors with shape [B, M, 3] (RGB).
        gt_colors (torch.Tensor): Ground truth colors with shape [B, N, 3] (RGB).
        mode (str): Loss mode. Options: 'l1', 'l2', 'smooth_l1'. Defaults to 'l1'.
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
        chunk_size (int): Chunk size for memory-efficient computation. Defaults to 1024.
    
    Returns:
        torch.Tensor: Color loss value.
    """
    B, M, _ = pred_colors.shape
    _, N, _ = gt_colors.shape
    
    # Use chunked computation to avoid creating full (B, M, N) tensor
    # Process predicted colors in chunks, and also chunk target colors to reduce memory
    min_dists_chunks = []
    
    # Also chunk target colors to avoid (B, chunk_size, N) where N=40000 is too large
    target_chunk_size = chunk_size  # Process target colors in chunks too
    
    for i in range(0, M, chunk_size):
        end_i = min(i + chunk_size, M)
        pred_chunk = pred_colors[:, i:end_i, :]  # (B, chunk_size, 3)
        
        # Process target colors in chunks and find minimum across all chunks
        min_dists_for_pred_chunk = []
        for j in range(0, N, target_chunk_size):
            end_j = min(j + target_chunk_size, N)
            gt_chunk = gt_colors[:, j:end_j, :]  # (B, target_chunk_size, 3)
            
            # Compute pairwise color distances for this chunk: (B, chunk_size, target_chunk_size)
            color_dists_chunk = torch.cdist(pred_chunk, gt_chunk, p=2)  # L2 distance in RGB space
            
            # For each predicted color, find closest GT color in this chunk
            min_dists_chunk, _ = color_dists_chunk.min(dim=2)  # (B, chunk_size)
            min_dists_for_pred_chunk.append(min_dists_chunk)
            
            # Free memory immediately
            del color_dists_chunk, min_dists_chunk
        
        # Take minimum across all target chunks (each pred color finds closest across all GT chunks)
        min_dists_pred_chunk = torch.stack(min_dists_for_pred_chunk, dim=0).min(dim=0).values  # (B, chunk_size)
        min_dists_chunks.append(min_dists_pred_chunk)
    
    # Concatenate all chunks: (B, M)
    min_dists = torch.cat(min_dists_chunks, dim=1)  # (B, M)
    
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
    
    def __init__(self, mode='l1', reduction='mean', loss_weight=1.0, chunk_size=1024):
        super(ColorLoss, self).__init__()
        assert mode in ['l1', 'l2', 'smooth_l1']
        assert reduction in ['none', 'sum', 'mean']
        self.mode = mode
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.chunk_size = chunk_size
    
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
            reduction=reduction,
            chunk_size=self.chunk_size
        )
        
        return loss * self.loss_weight

