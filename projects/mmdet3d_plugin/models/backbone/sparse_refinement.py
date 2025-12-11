"""
Sparse Voxel-Based Point Cloud Refinement.

This module uses sparse voxel convolutions to process point clouds efficiently.
Both pseudo and GT point clouds are voxelized and processed through sparse encoders,
then compared in feature space for refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
from mmdet.models.builder import BACKBONES, LOSSES, build_loss
from mmdet3d.ops import Voxelization
from mmdet3d.models import builder
from projects.mmdet3d_plugin.utils.live_visualizer import enqueue_live_frame


@BACKBONES.register_module()
class SparseRefinement(nn.Module):
    """Sparse voxel-based refinement network.
    
    Architecture:
    1. Voxelize both pseudo and GT point clouds
    2. Encode voxels using HardSimpleVFE
    3. Process through SparseEncoder (3D sparse convolutions)
    4. Compare features in sparse feature space
    5. Optionally decode back to point cloud
    """
    
    def __init__(
        self,
        pts_voxel_layer: Dict,
        pts_voxel_encoder: Dict,
        pts_middle_encoder: Dict,
        loss_feature: Optional[Dict] = None,
        loss_weight: float = 1.0,
        use_color: bool = False,
    ):
        """
        Args:
            pts_voxel_layer: Config for voxelization layer
            pts_voxel_encoder: Config for voxel encoder (e.g., HardSimpleVFE)
            pts_middle_encoder: Config for sparse middle encoder (e.g., SparseEncoder)
            loss_feature: Config for feature comparison loss
            loss_weight: Weight for the feature loss
            use_color: If True, use RGB colors in addition to XYZ
        """
        super().__init__()
        
        self.use_color = use_color
        self.loss_weight = loss_weight
        self.vis_grid_size = None  # optionally set by visualization hook
        
        # Build voxelization layer
        self.voxel_layer = Voxelization(**pts_voxel_layer)
        # Cache voxel meta for visualization
        self.voxel_size = torch.tensor(pts_voxel_layer['voxel_size'], dtype=torch.float32)
        self.point_cloud_range = torch.tensor(pts_voxel_layer['point_cloud_range'], dtype=torch.float32)
        
        # Build voxel encoder
        self.voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        
        # Build sparse middle encoder
        self.middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        
        # Build feature loss
        if loss_feature is not None:
            self.loss_feature = build_loss(loss_feature)
        else:
            self.loss_feature = None

        # Visualization caching flag
        self.enable_visual_debug = True
        self.last_vis_coors = None
        self.last_vis_coors_gt = None

    def _prepare_vis_coors(self, coors: torch.Tensor) -> torch.Tensor:
        """Detach, crop (if configured), and move coors to CPU for visualization."""
        if coors is None:
            return None
        coors_cpu = coors.detach().cpu()
        if self.vis_grid_size is None:
            return coors_cpu
        # coors format: [batch, z, y, x]
        voxel_size = self.voxel_size.cpu().numpy()
        pcr = self.point_cloud_range.cpu().numpy()
        gx = (pcr[3] - pcr[0]) / voxel_size[0]
        gy = (pcr[4] - pcr[1]) / voxel_size[1]
        gz = (pcr[5] - pcr[2]) / voxel_size[2]
        cx = gx * 0.5
        cy = gy * 0.5
        cz = gz * 0.5
        vx_half = self.vis_grid_size[0] * 0.5
        vy_half = self.vis_grid_size[1] * 0.5
        vz_half = self.vis_grid_size[2] * 0.5
        coors_np = coors_cpu.numpy()
        mask = (
            (coors_np[:, 3] >= cx - vx_half) & (coors_np[:, 3] <= cx + vx_half) &
            (coors_np[:, 2] >= cy - vy_half) & (coors_np[:, 2] <= cy + vy_half) &
            (coors_np[:, 1] >= cz - vz_half) & (coors_np[:, 1] <= cz + vz_half)
        )
        return torch.as_tensor(coors_np[mask], dtype=torch.int32)
    
    def _voxelize_and_encode(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Voxelize and encode batched point clouds.
        
        Args:
            points: (B, N, C) tensor
        
        Returns:
            voxel_features: Voxel features
            num_points: Number of points per voxel
            coors: Voxel coordinates (with batch index)
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)
        batch_size = points.shape[0]

        voxels_list, coors_list, num_points_list = [], [], []
        for b in range(batch_size):
            res = points[b]
            if not res.is_contiguous():
                res = res.contiguous()
            if not torch.is_floating_point(res):
                res = res.float()
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels_list.append(res_voxels)
            coors_list.append(res_coors)
            num_points_list.append(res_num_points)

        voxels = torch.cat(voxels_list, dim=0)
        num_points = torch.cat(num_points_list, dim=0)

        coors_batch = []
        for i, coor in enumerate(coors_list):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors = torch.cat(coors_batch, dim=0)

        voxel_features = self.voxel_encoder(voxels, num_points, coors)

        return voxel_features, num_points, coors
    
    def _process_sparse_features(
        self,
        voxel_features: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Process voxel features through sparse encoder.
        
        Args:
            voxel_features: Voxel features
            num_points: Number of points per voxel
            coors: Voxel coordinates
            batch_size: Batch size
        
        Returns:
            sparse_features: Sparse feature representation
        """
        # Process through sparse middle encoder
        sparse_features = self.middle_encoder(voxel_features, coors, batch_size)
        
        return sparse_features
    
    def forward(
        self,
        pseudo_points: torch.Tensor,
        gt_points: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            pseudo_points: (B, N, C) tensor
            gt_points: (B, N, C) tensor (optional)
            return_loss: Whether to compute losses
        
        Returns:
            refined_points: (B, N, C) refined point clouds (currently returns input)
            losses: Dict of loss values (if return_loss=True)
        """
        if pseudo_points.dim() == 2:
            pseudo_points = pseudo_points.unsqueeze(0)
        batch_size = pseudo_points.shape[0]

        pseudo_points_xyz = pseudo_points if self.use_color else pseudo_points[:, :, :3]
        
        # Voxelize and encode pseudo points
        pseudo_voxel_features, pseudo_num_points, pseudo_coors = self._voxelize_and_encode(pseudo_points_xyz)
        
        # Process through sparse encoder
        pseudo_sparse_features = self._process_sparse_features(
            pseudo_voxel_features, pseudo_num_points, pseudo_coors, batch_size
        )
        
        # Cache pseudo coors for visualization (only if enabled)
        if self.enable_visual_debug:
            self.last_coors = pseudo_coors.detach().cpu()
            self.last_vis_coors = self._prepare_vis_coors(pseudo_coors)

        # Compute losses if needed
        losses = None
        if return_loss and gt_points is not None:
            loss_dict = {}
            
            if gt_points.dim() == 2:
                gt_points = gt_points.unsqueeze(0)

            gt_points_xyz = gt_points if self.use_color else gt_points[:, :, :3]
            
            # Voxelize and encode GT points
            gt_voxel_features, gt_num_points, gt_coors = self._voxelize_and_encode(gt_points_xyz)
            
            # Process through sparse encoder
            gt_sparse_features = self._process_sparse_features(
                gt_voxel_features, gt_num_points, gt_coors, batch_size
            )

            # Cache GT coors for visualization (only if enabled)
            if self.enable_visual_debug:
                self.last_coors_gt = gt_coors.detach().cpu()
                self.last_vis_coors_gt = self._prepare_vis_coors(gt_coors)
                enqueue_live_frame(
                    dict(
                        pseudo_coors=self.last_vis_coors,
                        gt_coors=self.last_vis_coors_gt,
                        voxel_size=self.voxel_size.detach().cpu().numpy(),
                        pcr=self.point_cloud_range.detach().cpu().numpy(),
                    )
                )
            
            # Compute feature loss
            if self.loss_feature is not None:
                # Compare sparse features
                # SparseEncoder returns (B, C*D, H, W) feature maps
                # Both should have the same shape if voxelization params are the same
                if pseudo_sparse_features.shape == gt_sparse_features.shape:
                    # Direct feature map comparison
                    feature_loss = self.loss_feature(pseudo_sparse_features, gt_sparse_features)
                else:
                    # Handle shape mismatch by interpolating or cropping
                    # For now, use simple L2 on flattened features
                    pseudo_flat = pseudo_sparse_features.flatten(1)  # (B, C*D*H*W)
                    gt_flat = gt_sparse_features.flatten(1)  # (B, C*D*H*W)
                    min_size = min(pseudo_flat.shape[1], gt_flat.shape[1])
                    feature_loss = self.loss_feature(
                        pseudo_flat[:, :min_size], 
                        gt_flat[:, :min_size]
                    )
                loss_dict['loss_feature'] = feature_loss * self.loss_weight
            
            losses = loss_dict
        
        # For now, return pseudo_points as-is (refinement in feature space, not point space)
        # TODO: Add decoder to map features back to refined points
        refined_points = pseudo_points
        
        return refined_points, losses

