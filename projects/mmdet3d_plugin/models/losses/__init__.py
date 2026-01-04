# Copyright (c) OpenMMLab. All rights reserved.
from .emd_loss import EMDLoss, emd_loss
from .smoothness_loss import SmoothnessLoss, smoothness_loss
from .color_loss import ColorLoss, color_loss
from .simple_l2_loss import SimpleL2Loss, simple_l2_loss
from .occupancy_loss import OccupancyLoss
from .voxel_occupancy_alignment_loss import VoxelOccupancyAlignmentLoss
from .sparse_feature_alignment_loss import SparseFeatureAlignmentLoss
from .dense_bev_feature_loss import DenseBEVFeatureLoss

__all__ = [
    'EMDLoss', 'emd_loss',
    'SmoothnessLoss', 'smoothness_loss',
    'ColorLoss', 'color_loss',
    'SimpleL2Loss', 'simple_l2_loss',
    'OccupancyLoss',
    'VoxelOccupancyAlignmentLoss',
    'SparseFeatureAlignmentLoss',
    'DenseBEVFeatureLoss',
]

