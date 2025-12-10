"""
Point Cloud Refinement Module for ReconstructionBackbone.

This module learns to refine pseudo point clouds (from depth estimation) to match
ground-truth LiDAR point clouds. It includes:
- Point cloud sampling (FPS)
- Learnable refinement network (permutation-invariant)
- Multi-objective losses (Chamfer Distance, EMD, Feature Loss, Smoothness)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
from mmdet.models.builder import BACKBONES
from mmdet3d.ops import furthest_point_sample
from mmdet.models.builder import build_backbone


class PointNet2Layer(nn.Module):
    """PointNet++ style layer for point cloud feature extraction."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) point features
        Returns:
            (B, out_channels, N) features
        """
        return self.mlp(x)


@BACKBONES.register_module()
class PointNetRefinement(nn.Module):
    """Neural network for refining pseudo point clouds.
    
    Takes a pseudo point cloud and produces refined/corrected point cloud.
    Architecture: PointNet++ style with residual connections.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 4,
        output_mode: str = 'residual',  # 'residual' or 'direct'
    ):
        """
        Args:
            in_channels: Input point dimension (3 for XYZ)
            hidden_channels: Hidden feature dimension
            num_layers: Number of PointNet layers
            output_mode: 'residual' outputs offsets, 'direct' outputs refined points
        """
        super().__init__()
        self.output_mode = output_mode
        
        # Feature extraction layers
        layers = []
        prev_channels = in_channels
        for i in range(num_layers):
            layers.append(PointNet2Layer(prev_channels, hidden_channels))
            prev_channels = hidden_channels
        
        self.feature_layers = nn.ModuleList(layers)
        
        # Output layer: predict per-point residual offsets or refined points
        if output_mode == 'residual':
            self.output_layer = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
                nn.BatchNorm1d(hidden_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channels // 2, 3, 1),  # XYZ offsets
            )
        else:  # direct
            self.output_layer = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
                nn.BatchNorm1d(hidden_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channels // 2, 3, 1),  # Refined XYZ
            )
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) or (N, 3) point cloud
        Returns:
            If output_mode='residual': (B, N, 3) or (N, 3) offsets
            If output_mode='direct': (B, N, 3) or (N, 3) refined points
        """
        # Handle both batched and unbatched inputs
        is_batched = points.dim() == 3
        if not is_batched:
            points = points.unsqueeze(0)  # (1, N, 3)
        
        B, N, C = points.shape
        x = points.transpose(1, 2)  # (B, 3, N)
        
        # Extract features through layers
        for layer in self.feature_layers:
            x = layer(x)
        
        # Generate output
        output = self.output_layer(x)  # (B, 3, N)
        output = output.transpose(1, 2)  # (B, N, 3)
        
        if self.output_mode == 'residual':
            # Add residual to input
            # the final output will be refined by using the offsets from the output layer to adjust to match with real lidar point cloud
            refined = points + output
        else:
            refined = output
        
        if not is_batched:
            refined = refined.squeeze(0)  # (N, 3)
        
        return refined


def chamfer_distance_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        pred_points: (B, M, 3) or (M, 3) predicted point cloud
        gt_points: (B, N, 3) or (N, 3) ground truth point cloud
        reduction: 'mean' or 'sum'
    
    Returns:
        Scalar loss value
    """
    # Handle batched and unbatched
    is_batched = pred_points.dim() == 3
    if not is_batched:
        pred_points = pred_points.unsqueeze(0)
        gt_points = gt_points.unsqueeze(0)
    
    B, M, _ = pred_points.shape
    _, N, _ = gt_points.shape
    
    # Compute pairwise distances: (B, M, N)
    dists = torch.cdist(pred_points, gt_points)  # (B, M, N)
    
    # For each point in pred, find closest in gt
    dist_pred_to_gt = dists.min(dim=2).values  # (B, M)
    
    # For each point in gt, find closest in pred
    dist_gt_to_pred = dists.min(dim=1).values  # (B, N)
    
    # Chamfer distance
    cd_loss = dist_pred_to_gt.mean(dim=1) + dist_gt_to_pred.mean(dim=1)  # (B,)
    
    if reduction == 'mean':
        return cd_loss.mean()
    else:
        return cd_loss.sum()


def emd_loss_approximate(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Approximate Earth Mover's Distance using optimal transport.
    Uses a simplified version: Hungarian matching for small sets, or Sinkhorn for larger.
    
    Args:
        pred_points: (B, M, 3) or (M, 3) predicted point cloud
        gt_points: (B, N, 3) or (N, 3) ground truth point cloud
        reduction: 'mean' or 'sum'
    
    Returns:
        Scalar loss value
    """
    # Handle batched and unbatched
    is_batched = pred_points.dim() == 3
    if not is_batched:
        pred_points = pred_points.unsqueeze(0)
        gt_points = gt_points.unsqueeze(0)
    
    B, M, _ = pred_points.shape
    _, N, _ = gt_points.shape
    
    # For simplicity, use a differentiable approximation:
    # Compute pairwise distances and use soft assignment
    dists = torch.cdist(pred_points, gt_points)  # (B, M, N)
    
    # Soft assignment using softmin (temperature-scaled softmax of negative distances)
    temperature = 0.1
    soft_assign = F.softmin(dists / temperature, dim=2)  # (B, M, N)
    
    # Weighted distance
    weighted_dists = (soft_assign * dists).sum(dim=2)  # (B, M)
    
    # Average over points
    emd_loss = weighted_dists.mean(dim=1)  # (B,)
    
    if reduction == 'mean':
        return emd_loss.mean()
    else:
        return emd_loss.sum()


def smoothness_loss(
    refined_points: torch.Tensor,
    pseudo_points: torch.Tensor,
    k_neighbors: int = 10,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Smoothness regularization: penalize large local variations in corrections.
    
    Args:
        refined_points: (B, M, 3) or (M, 3) refined points
        pseudo_points: (B, M, 3) or (M, 3) original pseudo points
        k_neighbors: Number of neighbors for local smoothness
        reduction: 'mean' or 'sum'
    
    Returns:
        Scalar loss value
    """
    # Handle batched and unbatched
    is_batched = refined_points.dim() == 3
    if not is_batched:
        refined_points = refined_points.unsqueeze(0)
        pseudo_points = pseudo_points.unsqueeze(0)
    
    B, M, _ = refined_points.shape
    
    # Compute corrections (residuals)
    corrections = refined_points - pseudo_points  # (B, M, 3)
    
    # For each point, find k nearest neighbors in pseudo_points
    # Use simplified version: compute variance of corrections
    # More sophisticated: use k-NN graph
    
    # Simple smoothness: penalize large variance in corrections
    correction_var = corrections.var(dim=1).mean(dim=1)  # (B,)
    
    if reduction == 'mean':
        return correction_var.mean()
    else:
        return correction_var.sum()

@BACKBONES.register_module()
class PointCloudRefinement(nn.Module):
    """Complete point cloud refinement system.
    
    Includes:
    - FPS sampling for pseudo and GT point clouds
    - Refinement network
    - Loss computation (Chamfer, EMD, Feature, Smoothness)
    """
    
    def __init__(
        self,
        refinement_net: Optional[Dict] = None,
        loss_weights: Optional[Dict] = None,
    ):
        """
        Args:
            refinement_net: Config dict for refinement network
            loss_weights: Dict with keys 'chamfer', 'emd', 'feature', 'smoothness'
        """
        super().__init__()
        
        # # Build refinement network
        # self.refinement_net = PointNetRefinement(
        #     in_channels=3,
        #     hidden_channels=refinement_net.get('hidden_channels', 64),
        #     num_layers=refinement_net.get('num_layers', 4),
        #     output_mode=refinement_net.get('output_mode', 'residual'),
        # )
        
        self.refinement_net = build_backbone(refinement_net)
        self.loss_weights = loss_weights
    
    def sample_points_fps(
        self,
        points: torch.Tensor,
        num_samples: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Sample points using FPS.
        
        Args:
            points: (B, N, 3) or (N, 3) point cloud
            num_samples: Target number of points
            device: Device to run FPS on (needs CUDA)
        
        Returns:
            (B, num_samples, 3) or (num_samples, 3) sampled points
        """
        is_batched = points.dim() == 3
        if not is_batched:
            points = points.unsqueeze(0)
        
        if device is None:
            device = points.device
        
        B, N, _ = points.shape
        
        if N <= num_samples:
            # Not enough points, return as-is
            if not is_batched:
                return points.squeeze(0)
            return points
        
        # FPS requires CUDA
        if device.type != 'cuda':
            # Fallback: random sampling
            indices = torch.randperm(N, device=device)[:num_samples]
            sampled = points[:, indices, :]
        else:
            # Use FPS
            points_for_fps = points.to(device).contiguous()
            fps_indices = furthest_point_sample(points_for_fps, num_samples)  # (B, num_samples)
            # fps_indices is (B, num_samples), need to gather
            B_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_samples)
            sampled = points_for_fps[B_idx, fps_indices]
        
        if not is_batched:
            sampled = sampled.squeeze(0)
        
        return sampled
    
    def _pad_point_clouds(
        self,
        point_clouds: List[torch.Tensor],
        target_num_points: int,
        pad_value: float = 0.0
    ) -> torch.Tensor:
        """Pad variable-length point clouds to same size.
        
        Args:
            point_clouds: List of (N_i, C) tensors (C can be 3 or 6)
            target_num_points: Target number of points to pad to
            pad_value: Value to use for padding
        
        Returns:
            padded: (B, target_num_points, C) padded tensor
        """
        if len(point_clouds) == 0:
            return torch.empty(0, target_num_points, 3)
        
        device = point_clouds[0].device
        B = len(point_clouds)
        C = point_clouds[0].shape[1]  # 3 or 6
        
        # Pad all to target_num_points
        padded_list = []
        for pc in point_clouds:
            N = pc.shape[0]
            if N < target_num_points:
                padding = torch.full((target_num_points - N, C), pad_value, device=device, dtype=pc.dtype)
                padded_pc = torch.cat([pc, padding], dim=0)
            else:
                #using the furthest point sample to downsample the points to the target number of points
                fps_indices = furthest_point_sample(pc.unsqueeze(0), target_num_points).squeeze(0)
                padded_pc = pc[fps_indices]
            padded_list.append(padded_pc)
        
        padded = torch.stack(padded_list, dim=0)  # (B, target_num_points, C)
        return padded
    
    def _padding_samples(self, pseudo_points: List[torch.Tensor], 
                         gt_points: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        
        assert len(pseudo_points) == len(gt_points), "Pseudo and GT points must have the same batch size"
        batch_size = len(pseudo_points)
        
        # find the max number of points in pseudo and gt points
        max_num_points = max(len(pseudo_points[i]) for i in range(batch_size))
        max_num_points_gt = max(len(gt_points[i]) for i in range(batch_size))
        target_num_points = max(max_num_points, max_num_points_gt)
        
        # pad pseudo and gt points to the target number of points
        padded_pseudo_points = self._pad_point_clouds(pseudo_points, target_num_points)
        padded_gt_points = self._pad_point_clouds(gt_points, target_num_points)
        
        return padded_pseudo_points, padded_gt_points
    
    
    
    def forward(
        self,
        pseudo_points: Union[torch.Tensor, List[torch.Tensor]],
        gt_points: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        return_loss: bool = False,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            pseudo_points: (B, N, C) tensor where C is 3 or 6, or list of (N, C) tensors
                Pseudo point clouds (after post-processing, already fixed size N from FPSDownsample)
            gt_points: (B, M, 3) tensor or None
                Ground truth point clouds (optional, for training)
            return_loss: Whether to compute and return losses
        
        Returns:
            refined_points: (B, N, C) tensor (same format as input if tensor, else list)
            losses: Dict of loss values (if return_loss=True and gt_points provided)
        """
        
        pseudo_sampled, gt_sampled = self._padding_samples(pseudo_points, gt_points)
        B, N, C = pseudo_sampled.shape

        
        # Refine using network (batch operation)
        refined_xyz = self.refinement_net(pseudo_sampled) # (B, N, 3)
        
        # If original had colors, preserve them
        if C == 6:
            # Preserve colors from original
            if refined_xyz.shape[1] == N:
                # Same size, just concatenate colors
                refined_batch = torch.cat([refined_xyz, pseudo_batch[:, :, 3:]], dim=2)  # (B, N, 6)
            else:
                # Different size after sampling, need to handle colors differently
                # For now, just return XYZ (could interpolate colors in future)
                refined_batch = refined_xyz  # (B, sample_pseudo_to, 3)
        else:
            refined_batch = refined_xyz  # (B, N, 3) or (B, sample_pseudo_to, 3)
        
        # Compute losses if needed
        losses = None
        if return_loss and gt_points is not None:
            # Convert GT to batch tensor if needed
            if isinstance(gt_points, list):
                gt_batch = torch.stack([p.float().to(device) for p in gt_points if p is not None], dim=0)  # (B, M, 3)
            else:
                gt_batch = gt_points  # Already (B, M, 3)
            
            # Sample GT if needed
            if self.sample_gt_to is not None:
                gt_sampled = self.sample_points_fps(gt_batch, self.sample_gt_to, device)  # (B, sample_gt_to, 3)
            else:
                gt_sampled = gt_batch  # (B, M, 3)
            
            # Compute losses on batched data
            loss_dict = {}
            
            # Chamfer Distance
            if self.loss_weights.get('chamfer', 0) > 0:
                cd_loss = chamfer_distance_loss(refined_xyz, gt_sampled)
                loss_dict['chamfer'] = cd_loss * self.loss_weights['chamfer']
            
            # EMD
            if self.loss_weights.get('emd', 0) > 0:
                emd_loss = emd_loss_approximate(refined_xyz, gt_sampled)
                loss_dict['emd'] = emd_loss * self.loss_weights['emd']
            
            # Smoothness
            if self.loss_weights.get('smoothness', 0) > 0:
                smooth_loss = smoothness_loss(refined_xyz, pseudo_sampled)
                loss_dict['smoothness'] = smooth_loss * self.loss_weights['smoothness']
            
            # Feature loss (placeholder)
            if self.loss_weights.get('feature', 0) > 0:
                loss_dict['feature'] = torch.tensor(0.0, device=device)
            
            # Total loss
            total_loss = sum(loss_dict.values())
            loss_dict['total'] = total_loss
            
            losses = loss_dict
        
        # Return in same format as input
        return refined_batch, losses

