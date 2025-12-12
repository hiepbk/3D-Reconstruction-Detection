# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import DynamicScatter
from mmdet3d.models.builder import VOXEL_ENCODERS


@VOXEL_ENCODERS.register_module()
class HardVoxelOccupancyVFE(nn.Module):
    """Voxel occupancy encoder.
    
    Returns a binary occupancy value (1 = non-empty voxel, 0 = empty).
    """

    def __init__(self):
        super(HardVoxelOccupancyVFE, self).__init__()
        self.fp16_enabled = False

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """
        Args:
            features: (N, M, C) point features (unused)
            num_points: (N,) number of points in each voxel
            coors: (N, 3 or 4) voxel coords (unused here)

        Returns:
            torch.Tensor: Occupancy map of shape (N, 1)
        """

        # occupancy = 1 if voxel has any points, else 0
        occupancy = (num_points > 0).float().view(-1, 1)

        return occupancy.contiguous()
    
    
@VOXEL_ENCODERS.register_module()
class SoftVoxelOccupancyVFE(nn.Module):
    """Soft voxel occupancy encoder.

    Produces voxel occupancy probability in [0, 1] by combining:
     - number of points inside voxel
     - spatial distribution of points (xyz mean/variance)
     - learnable voting MLP
    """

    def __init__(self, max_points_norm=10):
        super(SoftVoxelOccupancyVFE, self).__init__()
        self.fp16_enabled = False
        self.max_points_norm = max_points_norm

        # MLP voting head: input 1 (density) + 1 (variance) + 3 (mean xyz) = 5 features
        self.mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """
        Args:
            features: (N, M, C) , typically xyz or xyz+other
            num_points: (N,)
            coors: (N, 3 or 4)

        Returns:
            (N,1) soft occupancy probability
        """

        N, M, C = features.shape

        # ---------------------
        # 1. Density-based probability
        # ---------------------
        # clamp to avoid extreme density
        clamped_pts = torch.clamp(num_points, max=self.max_points_norm)
        p_density = (clamped_pts / float(self.max_points_norm)).float()
        p_density = p_density.view(N, 1)

        # ---------------------
        # 2. Feature-based probability (xyz distribution)
        # ---------------------
        xyz = features[:, :, :3]                           # (N, M, 3)
        mask = (torch.arange(M, device=features.device).unsqueeze(0) < num_points.unsqueeze(1)).float()
        mask = mask.unsqueeze(-1)                          # (N, M, 1)

        # mean xyz
        masked_xyz = xyz * mask
        sum_xyz = masked_xyz.sum(dim=1)                   # (N, 3)
        mean_xyz = sum_xyz / torch.clamp(num_points.view(N, 1).float(), min=1)

        # variance xyz
        diff = (xyz - mean_xyz.unsqueeze(1)) * mask
        var_xyz = (diff ** 2).sum(dim=1) / torch.clamp(num_points.view(N, 1).float(), min=1)
        var_xyz_mean = var_xyz.mean(dim=1, keepdim=True)  # (N,1)

        # convert variance to probability
        # high variance → lower confidence
        p_var = torch.exp(-0.5 * var_xyz_mean)

        # ---------------------
        # 3. Voting MLP
        # ---------------------
        mlp_input = torch.cat([p_density, p_var, mean_xyz], dim=1)
        logits = self.mlp(mlp_input)                      # (N,1)
        occupancy = torch.sigmoid(logits)                 # convert to probability

        return occupancy.contiguous()
