"""
Sparse Voxel-Based Point Cloud Refinement.

This module uses sparse voxel convolutions to process point clouds efficiently.
Both pseudo and GT point clouds are voxelized and processed through sparse encoders,
then compared in feature space for refinement.
"""

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
from mmdet.models.builder import BACKBONES, LOSSES, build_loss
from mmdet3d.ops import Voxelization
from mmdet3d.models import builder, FUSION_LAYERS, MIDDLE_ENCODERS
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule

import torch
import pickle

import matplotlib.pyplot as plt


# We will use the transforme
# @MIDDLE_ENCODERS.register_module()
# class Transformer



@MIDDLE_ENCODERS.register_module()
class BEVHeightOccupancy(BaseModule):
    """BEV height occupancy module using U-Net style 2D convolutions.
    
    Main purpose: Predict occupancy for volume [180, 180, 32] based on sparse features.
    Target: Same voxel grid [180, 180, 32] generated from real LiDAR scene.
    
    Takes sparse features (B, C, H, W) and outputs occupancy maps (B, occ_feature_shape[2], H, W)
    where each channel represents occupancy probability (0-1) at different height levels.
    
    Args:
        in_channels: Number of input channels (e.g., 256 from SparseEncoder)
        Unet_channels: List of channel sizes for U-Net [256, 512, 1024, 2048]
        occ_feature_shape: [X, Y, C] BEV feature shape of occupancy [180, 180, 32]
        use_residual: Whether to use residual connections in encoder/decoder
        use_attention: Whether to use attention mechanism
        norm_cfg: Config for normalization layer
        init_cfg: Config for initialization (default: Kaiming for Conv2d)
    Returns:
        occupancy_map: (B, occ_feature_shape[2], H, W) occupancy probability maps
    """
    def __init__(self, 
                 in_channels=256,
                 Unet_channels=[256, 512, 1024, 2048],
                 occ_feature_shape=[180, 180, 32],  # [X,Y,C] BEV feature of occupancy
                 use_residual=True,
                 use_attention=True,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        # Set default init_cfg if not provided
        if init_cfg is None:
            init_cfg = dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_out',
                nonlinearity='relu'
            )
        super(BEVHeightOccupancy, self).__init__(init_cfg=init_cfg)
        
        self.occ_feature_shape = occ_feature_shape
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Project input to Unet_channels[0]
        self.input_proj = nn.Conv2d(in_channels, Unet_channels[0], 1)
        
        # Build deeper encoder: 256 -> 512 -> 1024 -> 2048
        self.encoders = nn.ModuleList()
        self.encoder_residual_flags = []  # Track which encoders can use residual
        
        for i in range(len(Unet_channels) - 1):
            in_ch = Unet_channels[i]
            out_ch = Unet_channels[i + 1]
            
            # First conv: may use stride 2 for downsampling (except first encoder)
            encoder_block = []
            if i == 0:
                # First encoder: same spatial size, can use residual
                encoder_block.append(nn.Conv2d(in_ch, in_ch, 3, padding=1))
                encoder_block.append(build_norm_layer(norm_cfg, in_ch)[1])
                encoder_block.append(nn.ReLU(inplace=True))
                encoder_block.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
                # Can use residual if enabled and channels match (but first encoder changes channels, so no residual)
                self.encoder_residual_flags.append(False)
            else:
                # Subsequent encoders: stride 2 for downsampling
                encoder_block.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
                encoder_block.append(build_norm_layer(norm_cfg, out_ch)[1])
                encoder_block.append(nn.ReLU(inplace=True))
                encoder_block.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
                # Can use residual if enabled and channels match
                self.encoder_residual_flags.append(use_residual and in_ch == out_ch)
            
            encoder_block.append(build_norm_layer(norm_cfg, out_ch)[1])
            encoder_block.append(nn.ReLU(inplace=True))
            
            self.encoders.append(nn.Sequential(*encoder_block))
        
        # Build attention modules (if enabled) - one for each encoder output
        if use_attention:
            self.attention_modules = nn.ModuleList()
            for ch in Unet_channels[1:]:  # One attention module per encoder output
                # Simple channel attention: GlobalAvgPool -> FC -> Sigmoid
                self.attention_modules.append(
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(ch, ch // 4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ch // 4, ch, 1),
                        nn.Sigmoid()
                    )
                )
        else:
            self.attention_modules = None
        
        # Build deeper decoder: 2048 -> 1024 -> 512 -> 256
        # Decoders process: upsampled_features + encoder_skip_features -> output
        # Forward pass logic:
        # - decoder[0]: processes encoder_features[3] (2048) -> (1024)
        # - decoder[1]: receives upsampled decoder[0] output (1024) + encoder_features[2] (1024) = (2048) -> (512)
        # - decoder[2]: receives upsampled decoder[1] output (512) + encoder_features[1] (512) = (1024) -> (256)
        self.decoders = nn.ModuleList()
        for decoder_idx in range(len(Unet_channels) - 1):  # 0, 1, 2
            if decoder_idx == 0:
                # Bottom decoder: no skip connection, just process 2048 -> 1024
                in_ch = Unet_channels[-1]  # 2048 (from last encoder)
                out_ch = Unet_channels[-2]  # 1024
            else:
                # Middle decoders: concatenate upsampled decoder output with encoder skip features
                # decoder[decoder_idx-1] outputs Unet_channels[-(decoder_idx+1)] channels
                # We concatenate with encoder_features[-(decoder_idx+1)] which also has Unet_channels[-(decoder_idx+1)] channels
                # So: in_ch = 2 * Unet_channels[-(decoder_idx+1)]
                #     out_ch = Unet_channels[-(decoder_idx+2)]
                skip_ch = Unet_channels[-(decoder_idx + 1)]  # Channels in skip connection
                in_ch = 2 * skip_ch  # decoder output + skip feature
                out_ch = Unet_channels[-(decoder_idx + 2)]  # Output channels
            
            decoder_block = []
            decoder_block.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            decoder_block.append(build_norm_layer(norm_cfg, out_ch)[1])
            decoder_block.append(nn.ReLU(inplace=True))
            decoder_block.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
            decoder_block.append(build_norm_layer(norm_cfg, out_ch)[1])
            decoder_block.append(nn.ReLU(inplace=True))
            
            self.decoders.append(nn.Sequential(*decoder_block))
        
        # Final output: gradually compress channels from Unet_channels[0] to occ_feature_shape[2]
        target_channels = self.occ_feature_shape[2]  # e.g., 32
        input_channels = Unet_channels[0]  # e.g., 256
        
        # Build gradual compression path: divide by 2 until reaching target
        compression_layers = []
        current_channels = input_channels
        
        while current_channels > target_channels:
            next_channels = max(current_channels // 2, target_channels)
            compression_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, next_channels, 1),
                    build_norm_layer(norm_cfg, next_channels)[1],
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = next_channels
        
        # Final layer: ensure we reach exactly target_channels and apply sigmoid
        if current_channels != target_channels:
            compression_layers.append(
                nn.Conv2d(current_channels, target_channels, 1)
            )
        
        self.occupancy_head = nn.Sequential(*compression_layers)
        # Note: No sigmoid here - output logits instead of probabilities
        # Sigmoid will be applied in loss function (binary_cross_entropy_with_logits)
        # or during inference/visualization when needed
        
    # Note: No custom init_weights needed - default Kaiming initialization (from init_cfg) 
    # is sufficient for logits. No sigmoid means no special bias initialization required.
        
    def forward(self, sparse_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            sparse_features: (B, C, H, W) input feature map
            
        Returns:
            occupancy_logits: (B, self.occ_feature_shape[2], H, W) occupancy logits (not probabilities)
                Apply torch.sigmoid() to get probabilities [0, 1] if needed
        """
        # Project input to Unet_channels[0]
        x = self.input_proj(sparse_features)  # (B, 256, H, W)
        
        # Encoder path: store features for skip connections
        encoder_features = [x]  # Store input for first skip connection
        
        for i, encoder in enumerate(self.encoders):
            out = encoder(encoder_features[-1])
            
            # Apply attention if enabled (one attention module per encoder output)
            if self.use_attention and i < len(self.attention_modules):
                attn = self.attention_modules[i]
                out = out * attn(out)  # Channel attention
            
            # Residual connection (if enabled and channels match)
            if self.encoder_residual_flags[i] and encoder_features[-1].shape[1] == out.shape[1]:
                out = out + encoder_features[-1]
            
            encoder_features.append(out)
        
        # Decoder path: upsample and concatenate with encoder features
        # encoder_features: [input(256), e1_out(512), e2_out(1024), e3_out(2048)]
        # indices:            [0]          [1]          [2]          [3]
        x = encoder_features[-1]  # Start from deepest encoder output (2048, 45, 45)
        
        for i, decoder in enumerate(self.decoders):
            # i=0: decoder 3 (2048->1024), no skip, just process
            # i=1: decoder 2 (1024->512), upsample + skip from encoder_features[2] (1024)
            # i=2: decoder 1 (512->256), upsample + skip from encoder_features[1] (512)
            
            if i == 0:
                # First decoder (bottom): no skip connection, just process
                x = decoder(x)  # (2048, 45, 45) -> (1024, 45, 45)
            else:
                # Subsequent decoders: upsample then concatenate with skip connection
                # Get the corresponding encoder feature for skip connection
                # encoder_features[-(i+1)] gives us the right skip feature
                # For i=1: encoder_features[-(2)] = encoder_features[2] = e2_out (1024, 90, 90)
                # For i=2: encoder_features[-(3)] = encoder_features[1] = e1_out (512, 180, 180)
                skip_idx = len(encoder_features) - (i + 1)  # 3-2=1 for i=1, 3-3=0 for i=2
                skip_feat = encoder_features[skip_idx]  # Get skip feature
                
                # Upsample x to match skip feature's spatial size
                x = F.interpolate(x, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)
                
                # Concatenate: upsampled decoder output + encoder skip feature
                x = torch.cat([x, skip_feat], dim=1)
                
                # Process through decoder
                x = decoder(x)
        
        # Final occupancy logits: compress to target channels (no sigmoid)
        occupancy_logits = self.occupancy_head(x)  # (B, self.occ_feature_shape[2], H, W)
        
        return occupancy_logits  # Return logits, not probabilities

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
        sparse_refinement_transformer: Dict,
        loss_feature: Dict = None,
        loss_index: Dict = None,
        loss_weight: float = 1.0,
        use_color: bool = False,
        debug_viz: bool = False,
        debug_viz_dir: str = "debug_viz",
    ):
        """
        Args:
            pts_voxel_layer: Config for voxelization layer
            pts_voxel_encoder: Config for voxel encoder (e.g., HardSimpleVFE)
            pts_middle_encoder: Config for sparse middle encoder (e.g., SparseEncoderV2 with return_type='sparse')
            sparse_refinement_transformer: Config for ShapeFormer-style transformer
            loss_feature: Config for feature alignment loss (optional, transformer has its own losses)
            loss_index: Config for index alignment loss (optional)
            loss_weight: Weight for the losses
            use_color: If True, use RGB colors in addition to XYZ
        """
        super().__init__()
        
        self.use_color = use_color
        self.loss_weight = loss_weight
        self.debug_viz = debug_viz
        self.debug_viz_dir = debug_viz_dir

        
        # Build voxelization layer
        self.voxel_layer = Voxelization(**pts_voxel_layer)
        # Cache voxel meta for visualization
        self.voxel_size = torch.tensor(pts_voxel_layer['voxel_size'], dtype=torch.float32)
        self.point_cloud_range = torch.tensor(pts_voxel_layer['point_cloud_range'], dtype=torch.float32)
        
        # Build voxel encoder
        self.voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        
        # Build sparse middle encoder (should return sparse features + indices when return_type='sparse')
        self.middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        
        # Build pattern adaptation transformer (using builder pattern)
        self.pattern_adaptation = builder.build_middle_encoder(sparse_refinement_transformer)
        
        # Build losses (optional, transformer has its own losses)
        if loss_feature is not None:
            self.loss_feature = build_loss(loss_feature)
        else:
            self.loss_feature = None
        
        if loss_index is not None:
            self.loss_index = build_loss(loss_index)
        else:
            self.loss_index = None
        
        # Get sparse_shape from middle_encoder config
        self.sparse_shape = pts_middle_encoder.get('sparse_shape', [41, 1440, 1440])  # [Z, Y, X]

        # Visualization caching flag
        self.enable_visual_debug = False
        self.debug_counter = 0
        
        
        
    # ===== OLD OCCUPANCY METHODS (COMMENTED OUT) =====
    # def _build_occupancy_voxelization(self, occupancy_voxel_layer: Dict) -> Voxelization:
    #     """Build occupancy voxelization layer."""
    #     ...
    
    def _voxel_encoder(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    
    # ===== OLD OCCUPANCY METHODS (COMMENTED OUT) =====
    # All occupancy-related methods removed - using transformer approach instead
    
    def forward_train(
        self,
        pseudo_sparse_features: torch.Tensor,
        pseudo_sparse_indices: torch.Tensor,
        pseudo_sparse_spatial_shape: List[int],
        gt_points: Optional[torch.Tensor] = None,
        
        return_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        
        assert gt_points is not None, "GT points are required for training"
        
        batch_size = gt_points.shape[0]
        gt_voxel_features, gt_num_points, gt_coors = self._voxel_encoder(gt_points)
        gt_sparse_features, gt_sparse_indices, gt_sparse_spatial_shape = self.middle_encoder(
            gt_voxel_features, gt_coors, batch_size
        )
        
        refined_features, refined_indices, transformer_losses = self.pattern_adaptation(
            pseudo_sparse_features=pseudo_sparse_features,
            pseudo_sparse_indices=pseudo_sparse_indices,
            spatial_shape=pseudo_sparse_spatial_shape,
            gt_sparse_features=gt_sparse_features,
            gt_sparse_indices=gt_sparse_indices,
            return_loss=True,
        )
        losses = None
        if transformer_losses:
            losses = transformer_losses.copy()
            for k, v in losses.items():
                losses[k] = v * self.loss_weight
        return refined_features, refined_indices, losses

    
    def forward_test(
        self,
        pseudo_sparse_features: torch.Tensor,
        pseudo_sparse_indices: torch.Tensor,
        pseudo_sparse_spatial_shape: List[int],
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        refined_features, refined_indices, _ = self.pattern_adaptation(
            pseudo_sparse_features=pseudo_sparse_features,
            pseudo_sparse_indices=pseudo_sparse_indices,
            spatial_shape=pseudo_sparse_spatial_shape,
            gt_sparse_features=None,
            gt_sparse_indices=None,
            return_loss=False,
        )
        return refined_features, refined_indices, None

    
    
    def forward(
        self,
        pseudo_points: torch.Tensor,
        gt_points: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Unified forward: run shared ops, then branch to train/test."""
        if pseudo_points.dim() == 2:
            pseudo_points = pseudo_points.unsqueeze(0)
        batch_size = pseudo_points.shape[0]

        pseudo_points_xyz = pseudo_points if self.use_color else pseudo_points[:, :, :3]
        gt_points_xyz = None
        if gt_points is not None:
            if gt_points.dim() == 2:
                gt_points = gt_points.unsqueeze(0)
            gt_points_xyz = gt_points if self.use_color else gt_points[:, :, :3]

        # Pseudo branch (shared)
        pseudo_voxel_features, pseudo_num_points, pseudo_coors = self._voxel_encoder(pseudo_points_xyz)
        pseudo_sparse_features, pseudo_sparse_indices, pseudo_sparse_spatial_shape = self.middle_encoder(
            pseudo_voxel_features, pseudo_coors, batch_size
        )

        if return_loss:
            refined_features, refined_indices, losses = self.forward_train(
                pseudo_sparse_features=pseudo_sparse_features,
                pseudo_sparse_indices=pseudo_sparse_indices,
                pseudo_sparse_spatial_shape=pseudo_sparse_spatial_shape,
                gt_points=gt_points,
                return_loss=return_loss,
            )

            return refined_features, refined_indices, losses
        else:
            refined_features, refined_indices, _ = self.forward_test(
                pseudo_sparse_features=pseudo_sparse_features,
                pseudo_sparse_indices=pseudo_sparse_indices,
                pseudo_sparse_spatial_shape=pseudo_sparse_spatial_shape,
            )

            return refined_features, refined_indices, None

    
    def _compute_feature_loss(
        self,
        refined_features: torch.Tensor,
        gt_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature alignment loss."""
        # TODO: Implement proper matching (e.g., Hungarian matching, nearest neighbor)
        # For now, assume sizes match or truncate
        if refined_features.shape[0] == gt_features.shape[0]:
            return self.loss_feature(refined_features, gt_features)
        else:
            min_size = min(refined_features.shape[0], gt_features.shape[0])
            return self.loss_feature(refined_features[:min_size], gt_features[:min_size])
    
    def _compute_index_loss(
        self,
        refined_indices: torch.Tensor,
        gt_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute index alignment loss."""
        if self.loss_index is None:
            return torch.tensor(0.0, device=refined_indices.device)
        
        if refined_indices.shape[0] == gt_indices.shape[0]:
            return self.loss_index(refined_indices.float(), gt_indices.float())
        else:
            min_size = min(refined_indices.shape[0], gt_indices.shape[0])
            return self.loss_index(refined_indices[:min_size].float(), gt_indices[:min_size].float())

        
        
        

        
        

