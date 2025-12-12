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
    def forward(self, features, num_points, coors, occ_sparse_shape):
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
        
        # then, convert to the feature map with shape of occ_sparse_shape [Z, Y, X]
        occupancy = occupancy.view(-1, 1, occ_sparse_shape[0], occ_sparse_shape[1], occ_sparse_shape[2])

        return occupancy.contiguous()
    
    
@VOXEL_ENCODERS.register_module()
class SoftVoxelOccupancyVFE(nn.Module):
    """
    Deterministic soft voxel occupancy encoder.
    Uses:
        - number of points (n)
        - spatial variance of xyz (mean variance)
    to compute occupancy probability in [0, 1].

    Formula:
        p_occ = 1 - exp( -λ*n - γ*var )
    """

    def __init__(self, lambda_n=0.3, gamma_var=5.0, eps=1e-6, occ_sparse_shape=None):
        super().__init__()
        self.lambda_n = lambda_n
        self.gamma_var = gamma_var
        self.eps = eps
        self.fp16_enabled = False
        self.occ_sparse_shape = occ_sparse_shape

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """
        Args:
            features: (N, M, C) padded points, first 3 dims must be xyz
            num_points: (N,) number of valid points in each voxel
            coors: unused

        Returns:
            occupancy: (N, 1) soft occupancy probability
        """

        N, M, C = features.shape

        # ----- Extract xyz -----
        xyz = features[:, :, :3]

        # ----- Build mask for padded points -----
        mask = (torch.arange(M, device=xyz.device)
                .unsqueeze(0) < num_points.unsqueeze(1))  # (N, M)

        mask_exp = mask.unsqueeze(-1).float()            # (N, M, 1)

        # ----- Compute masked mean -----
        xyz_sum = (xyz * mask_exp).sum(dim=1)            # (N, 3)
        denom = num_points.unsqueeze(1).float() + self.eps
        xyz_mean = xyz_sum / denom                       # (N, 3)

        # ----- Compute variance -----
        diff = (xyz - xyz_mean.unsqueeze(1)) * mask_exp  # (N, M, 3)
        var = (diff.pow(2).sum(dim=1) / denom).mean(dim=1)   # (N,)

        # ----- Combined deterministic occupancy -----
        # p = 1 - exp(-λ*n - γ*var)
        n = num_points.float()
        occupancy = 1.0 - torch.exp(-self.lambda_n * n - self.gamma_var * var)
        
        # then, convert to the feature map with shape of occ_sparse_shape [Z, Y, X]
        occupancy = occupancy.view(-1, 1, self.occ_sparse_shape[0], self.occ_sparse_shape[1], self.occ_sparse_shape[2])

        return occupancy.view(-1, 1).contiguous()