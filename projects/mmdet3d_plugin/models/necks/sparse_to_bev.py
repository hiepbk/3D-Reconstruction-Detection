"""
Sparse to BEV Feature Converter.

Converts sparse 3D features from SparseConv to dense BEV feature maps
for standard MMDetection3D detection heads.
"""

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import NECKS


@NECKS.register_module()
class SparseToDenseBEV(BaseModule):
    """
    Convert sparse 3D features to dense BEV feature maps.
    
    This module takes sparse features from SparseEncoder and converts them
    to dense BEV feature maps that can be fed into SECOND backbone.
    
    Input:
        sparse_features: [N, C] sparse features
        sparse_indices: [N, 4] voxel indices (batch, z, y, x)
        spatial_shape: [Z, Y, X] spatial shape of sparse grid
        batch_size: int, batch size
    
    Output:
        bev_features: [B, C*D, H, W] dense BEV feature map
        where D is the Z dimension (height), H=Y, W=X
    """
    
    def __init__(self, init_cfg=None):
        super(SparseToDenseBEV, self).__init__(init_cfg=init_cfg)
    
    def forward(self, sparse_features, sparse_indices, spatial_shape, batch_size):
        """
        Convert sparse features to dense BEV.
        
        Args:
            sparse_features: [N, C] tensor of sparse features
            sparse_indices: [N, 4] tensor of voxel indices (batch, z, y, x)
            spatial_shape: [Z, Y, X] list/tuple of spatial dimensions
            batch_size: int, batch size
        
        Returns:
            bev_features: [B, C*D, H, W] dense BEV feature map
        """
        import spconv
        
        # Create SparseConvTensor
        Z, Y, X = spatial_shape
        sparse_tensor = spconv.SparseConvTensor(
            features=sparse_features,
            indices=sparse_indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )
        
        # Convert to dense: [B, C, Z, Y, X]
        dense_features = sparse_tensor.dense()
        
        # Reshape to BEV: [B, C*Z, Y, X]
        # This is the standard format for SECOND backbone
        B, C, D, H, W = dense_features.shape
        bev_features = dense_features.view(B, C * D, H, W)
        
        return bev_features

