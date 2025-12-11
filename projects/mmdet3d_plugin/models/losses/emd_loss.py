# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


def emd_loss(pred_points, gt_points, temperature=0.1, reduction='mean', chunk_size=1024):
    """Approximate Earth Mover's Distance using soft assignment.
    
    Uses a differentiable approximation via soft assignment (temperature-scaled softmin).
    This is more efficient than exact EMD and works well for point cloud refinement.
    
    Memory-efficient chunked computation to avoid creating full (B, M, N) tensors.
    
    Args:
        pred_points (torch.Tensor): Predicted point cloud with shape [B, M, C].
        gt_points (torch.Tensor): Ground truth point cloud with shape [B, N, C].
        temperature (float): Temperature for soft assignment. Lower = sharper assignment.
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
        chunk_size (int): Chunk size for memory-efficient computation. Defaults to 1024.
    
    Returns:
        torch.Tensor: EMD loss value.
    """
    B, M, C = pred_points.shape
    _, N, _ = gt_points.shape
    
    # Use highly memory-efficient double chunking to avoid creating large tensors
    # Process both predicted and target points in small chunks
    weighted_dists_chunks = []
    
    # Use smaller chunks for 40k points
    target_chunk_size = min(chunk_size, 256)  # Cap at 256 for very large point clouds
    pred_chunk_size = min(chunk_size, 256)
    
    for i in range(0, M, pred_chunk_size):
        end_i = min(i + pred_chunk_size, M)
        pred_chunk = pred_points[:, i:end_i, :]  # (B, pred_chunk_size, C)
        
        # Process target points in chunks too
        weighted_dists_for_pred_chunk = []
        for j in range(0, N, target_chunk_size):
            end_j = min(j + target_chunk_size, N)
            gt_chunk = gt_points[:, j:end_j, :]  # (B, target_chunk_size, C)
            
            # Compute pairwise distances for this small chunk: (B, pred_chunk_size, target_chunk_size)
            dists_chunk = torch.cdist(pred_chunk, gt_chunk, p=2)  # L2 distance
            
            # Soft assignment using softmin (temperature-scaled softmax of negative distances)
            # softmin(x) = softmax(-x / temperature)
            soft_assign_chunk = F.softmin(dists_chunk / temperature, dim=2)  # (B, pred_chunk_size, target_chunk_size)
            
            # Weighted distance: for each predicted point, compute weighted distance to GT chunk
            weighted_dists_chunk = (soft_assign_chunk * dists_chunk).sum(dim=2)  # (B, pred_chunk_size)
            weighted_dists_for_pred_chunk.append(weighted_dists_chunk)
            
            # Free memory immediately
            del dists_chunk, soft_assign_chunk, weighted_dists_chunk, gt_chunk
            torch.cuda.empty_cache()  # Aggressive memory clearing
        
        # Sum across target chunks (each pred point gets contribution from all target chunks)
        if len(weighted_dists_for_pred_chunk) > 0:
            weighted_dists_pred_chunk = torch.stack(weighted_dists_for_pred_chunk, dim=0).sum(dim=0)  # (B, pred_chunk_size)
            weighted_dists_chunks.append(weighted_dists_pred_chunk)
            # Free memory
            del weighted_dists_for_pred_chunk, weighted_dists_pred_chunk
        else:
            # Edge case: no target chunks, create zero tensor
            weighted_dists_chunks.append(torch.zeros(B, end_i - i, device=pred_points.device, dtype=pred_points.dtype))
        
        # Free memory
        del pred_chunk
        torch.cuda.empty_cache()
    
    # Concatenate all chunks: (B, M)
    weighted_dists = torch.cat(weighted_dists_chunks, dim=1)  # (B, M)
    
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
    
    def __init__(self, temperature=0.1, reduction='mean', loss_weight=1.0, chunk_size=1024):
        super(EMDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.chunk_size = chunk_size
    
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
            reduction=reduction,
            chunk_size=self.chunk_size
        )
        
        return loss * self.loss_weight

