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
from mmdet.models.builder import BACKBONES, LOSSES, build_loss
from mmdet3d.ops import furthest_point_sample
from mmdet.models.builder import build_backbone
from mmdet3d.models.losses import ChamferDistance, chamfer_distance


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
    Handles both XYZ-only (3 channels) and XYZRGB (6 channels) inputs.
    Prioritizes XYZ refinement over color refinement.
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # Will auto-detect from input (3 or 6)
        hidden_channels: int = 64,
        num_layers: int = 4,
        output_mode: str = 'residual',  # 'residual' or 'direct'
        color_weight: float = 0.1,  # Weight for color features (lower = less influence)
    ):
        """
        Args:
            in_channels: Input point dimension (3 for XYZ, 6 for XYZRGB) - can be auto-detected
            hidden_channels: Hidden feature dimension
            num_layers: Number of PointNet layers
            output_mode: 'residual' outputs offsets, 'direct' outputs refined points
            color_weight: Weight for color features (0.0-1.0, lower means less influence)
        """
        super().__init__()
        self.output_mode = output_mode
        self.color_weight = color_weight
        self.in_channels = in_channels
        self._initialized = False
        
        # Main XYZ feature extraction layers (always processes 3 channels)
        xyz_layers = []
        prev_channels = 3  # Always 3 for XYZ
        for i in range(num_layers):
            xyz_layers.append(PointNet2Layer(prev_channels, hidden_channels))
            prev_channels = hidden_channels
        
        self.xyz_feature_layers = nn.ModuleList(xyz_layers)
        
        # Lightweight color feature extraction (initialize if in_channels indicates colors)
        self.has_color_branch = (in_channels == 6)
        self.color_feature_layers = None
        if self.has_color_branch:
            # Lightweight color processing (fewer layers, smaller channels)
            color_layers = []
            prev_channels = 3  # RGB channels
            for i in range(max(1, num_layers // 2)):  # Half the layers for color
                color_layers.append(PointNet2Layer(prev_channels, hidden_channels // 2))
                prev_channels = hidden_channels // 2
            self.color_feature_layers = nn.ModuleList(color_layers)
        
        # Output layer for XYZ: predict per-point residual offsets or refined points
        if output_mode == 'residual':
            self.xyz_output_layer = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
                nn.BatchNorm1d(hidden_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channels // 2, 3, 1),  # XYZ offsets
            )
        else:  # direct
            self.xyz_output_layer = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
                nn.BatchNorm1d(hidden_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channels // 2, 3, 1),  # Refined XYZ
            )
        
        # Output layer for color (if color branch exists)
        if self.has_color_branch:
            self.color_output_layer = nn.Sequential(
                nn.Conv1d(hidden_channels // 2, hidden_channels // 4, 1),
                nn.BatchNorm1d(hidden_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channels // 4, 3, 1),  # RGB offsets or refined RGB
            )
        else:
            self.color_output_layer = None
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, C) where C is 3 (XYZ) or 6 (XYZRGB)
        Returns:
            If output_mode='residual': (B, N, C) offsets (C matches input)
            If output_mode='direct': (B, N, C) refined points (C matches input)
        """
        # Handle both batched and unbatched inputs
        is_batched = points.dim() == 3
        if not is_batched:
            points = points.unsqueeze(0)  # (1, N, C)
        
        B, N, C = points.shape
        
        # Split XYZ and color if present
        xyz = points[:, :, :3]  # (B, N, 3)
        xyz_t = xyz.transpose(1, 2)  # (B, 3, N)
        
        # Process XYZ through main network (heavily weighted)
        xyz_features = xyz_t
        for layer in self.xyz_feature_layers:
            xyz_features = layer(xyz_features)
        
        # Generate XYZ output
        xyz_output = self.xyz_output_layer(xyz_features)  # (B, 3, N)
        xyz_output = xyz_output.transpose(1, 2)  # (B, N, 3)
        
        if self.output_mode == 'residual':
            refined_xyz = xyz + xyz_output
        else:
            refined_xyz = xyz_output
        
        # Process color if present
        if C == 6:
            colors = points[:, :, 3:6]  # (B, N, 3)
            if self.has_color_branch and self.color_feature_layers is not None and self.color_output_layer is not None:
                # Refine colors through lightweight network
                colors_t = colors.transpose(1, 2)  # (B, 3, N)
                
                # Process color through lightweight network
                color_features = colors_t
                for layer in self.color_feature_layers:
                    color_features = layer(color_features)
                
                # Generate color output
                color_output = self.color_output_layer(color_features)  # (B, 3, N)
                color_output = color_output.transpose(1, 2)  # (B, N, 3)
                
                if self.output_mode == 'residual':
                    # Apply color weight to color refinement (lighter influence)
                    refined_colors = colors + self.color_weight * color_output
                else:
                    # Blend original and refined colors
                    refined_colors = (1 - self.color_weight) * colors + self.color_weight * color_output
            else:
                # No color branch, preserve original colors
                refined_colors = colors
            
            # Concatenate refined XYZ and colors
            refined = torch.cat([refined_xyz, refined_colors], dim=2)  # (B, N, 6)
        else:
            refined = refined_xyz  # (B, N, 3)
        
        if not is_batched:
            refined = refined.squeeze(0)  # (N, C)
        
        return refined


# Import registered losses (must be imported to register them)
from projects.mmdet3d_plugin.models.losses import EMDLoss, SmoothnessLoss, ColorLoss

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
        loss_chamfer: Optional[Dict] = None,
        loss_emd: Optional[Dict] = None,
        loss_smoothness: Optional[Dict] = None,
        loss_color: Optional[Dict] = None,
    ):
        """
        Args:
            refinement_net: Config dict for refinement network
            loss_chamfer: Config dict for Chamfer Distance loss
            loss_emd: Config dict for EMD loss
            loss_smoothness: Config dict for Smoothness loss
            loss_color: Config dict for Color loss
        """
        super().__init__()
        
        self.refinement_net = build_backbone(refinement_net)
        
        # Build loss modules using registered losses
        # Chamfer Distance (use mmdet3d's built-in)
        if loss_chamfer is not None:
            self.loss_chamfer = build_loss(loss_chamfer)
        else:
            self.loss_chamfer = None
        
        # EMD Loss
        if loss_emd is not None:
            self.loss_emd = build_loss(loss_emd)
        else:
            self.loss_emd = None
        
        # Smoothness Loss
        if loss_smoothness is not None:
            self.loss_smoothness = build_loss(loss_smoothness)
        else:
            self.loss_smoothness = None
        
        # Color Loss
        if loss_color is not None:
            self.loss_color = build_loss(loss_color)
        else:
            self.loss_color = None
    
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
        
        # Convert to lists if needed for padding
        if isinstance(pseudo_points, torch.Tensor):
            pseudo_list = [pseudo_points[i] for i in range(pseudo_points.shape[0])]
        else:
            pseudo_list = pseudo_points
        
        if isinstance(gt_points, torch.Tensor):
            gt_list = [gt_points[i] for i in range(gt_points.shape[0])]
        else:
            gt_list = gt_points if gt_points is not None else None
        
        # Pad to same size
        if gt_list is not None:
            pseudo_sampled, gt_sampled = self._padding_samples(pseudo_list, gt_list)
        else:
            # No GT, just pad pseudo points
            if len(pseudo_list) > 0:
                max_n = max(pc.shape[0] for pc in pseudo_list)
                pseudo_sampled = self._pad_point_clouds(pseudo_list, max_n)
            else:
                pseudo_sampled = torch.empty(0, 0, 3)
            gt_sampled = None
        
        B, N, C = pseudo_sampled.shape
        device = pseudo_sampled.device
        
        # Refine using network (batch operation)
        # Network now handles both XYZ and XYZRGB internally
        refined_batch = self.refinement_net(pseudo_sampled)  # (B, N, C) where C matches input
        
        # Extract XYZ and colors for loss computation
        refined_xyz = refined_batch[:, :, :3]  # (B, N, 3)
        refined_colors = refined_batch[:, :, 3:6] if C == 6 else None  # (B, N, 3) or None
        
        # Compute losses if needed
        losses = None
        if return_loss and gt_sampled is not None:
            # Extract GT XYZ and colors
            gt_xyz = gt_sampled[:, :, :3]  # (B, M, 3)
            gt_colors = gt_sampled[:, :, 3:6] if gt_sampled.shape[2] >= 6 else None  # (B, M, 3) or None
            
            # Compute losses on batched data
            loss_dict = {}
            
            # Chamfer Distance (XYZ only - most important)
            if self.loss_chamfer is not None:
                loss_src, loss_dst = self.loss_chamfer(refined_xyz, gt_xyz)
                cd_loss = loss_src + loss_dst
                loss_dict['loss_chamfer'] = cd_loss
            
            # EMD (XYZ only)
            if self.loss_emd is not None:
                emd_loss_val = self.loss_emd(refined_xyz, gt_xyz)
                loss_dict['loss_emd'] = emd_loss_val
            
            # Smoothness (XYZ only)
            if self.loss_smoothness is not None:
                pseudo_xyz = pseudo_sampled[:, :, :3]  # (B, N, 3)
                smooth_loss_val = self.loss_smoothness(refined_xyz, pseudo_xyz)
                loss_dict['loss_smoothness'] = smooth_loss_val
            
            # Color loss (if both have colors)
            if self.loss_color is not None and refined_colors is not None and gt_colors is not None:
                color_loss_val = self.loss_color(refined_colors, gt_colors)
                loss_dict['loss_color'] = color_loss_val
            
            losses = loss_dict
        
        # Return in same format as input
        return refined_batch, losses

