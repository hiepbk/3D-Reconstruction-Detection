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
        bev_height_occupancy: Dict,
        occupancy_voxel_layer: Dict,
        occupancy_voxel_encoder: Dict,
        loss_occupancy: Dict,

        loss_weight: float = 1.0,
        use_color: bool = False,
        debug_viz: bool = False,
        debug_viz_dir: str = "debug_viz",
    ):
        """
        Args:
            pts_voxel_layer: Config for voxelization layer
            pts_voxel_encoder: Config for voxel encoder (e.g., HardSimpleVFE)
            pts_middle_encoder: Config for sparse middle encoder (e.g., SparseEncoder)
            bev_height_occupancy: Config for BEV height occupancy layer (e.g., BEVHeightOccupancy)
            occupancy_voxel_layer: Config for occupancy voxelization layer (e.g., Voxelization)
            occupancy_voxel_encoder: Config for occupancy voxel encoder (e.g., SoftVoxelOccupancyVFE)
            loss_occupancy: Config for occupancy loss (e.g., OccupancyLoss)
            loss_weight: Weight for the occupancy loss
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
        
        # Build sparse middle encoder
        self.middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        
        # (B, C*D, H, W) -> (B, C*D, H, W)
        # Build the BEV height refinement layer (multi-occupancy feature of height)
        
        self.bev_height_occupancy = builder.build_middle_encoder(bev_height_occupancy)
        
        
        self.gt_occupancty_layer = self._build_occupancy_voxelization(occupancy_voxel_layer)
        # build the voxelization layer for the occupancy ground truth
        self.gt_occupancty_voxel_encoder = builder.build_voxel_encoder(occupancy_voxel_encoder)
        # Build occupancy loss from config
        self.loss_occupancy = build_loss(loss_occupancy)
        
        # Projection layer to normalize feature dimensions (will be created on first forward)
        # sparse_features might have variable channels (C or C*D), project to 256
        self.feat_proj_conv = None  # Will be created dynamically based on input channels
        
        # Decoder: (M', 256) -> (M', 3) to decode features back to xyz coordinates
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # Output xyz
        )
        
        # Get sparse_shape from middle_encoder config for mapping back to full grid
        self.sparse_shape = pts_middle_encoder.get('sparse_shape', [41, 1440, 1440])  # [Z, Y, X]

        # Visualization caching flag
        self.enable_visual_debug = False
        self.debug_counter = 0
        
        
        
    def _build_occupancy_voxelization(self, occupancy_voxel_layer: Dict) -> Voxelization:
        """Build occupancy voxelization layer.
        
        Args:
            occupancy_voxel_layer: Config for occupancy voxelization layer (e.g., Voxelization)
        """
        
        point_cloud_range = occupancy_voxel_layer.get('point_cloud_range', None)
        self.occ_feature_shape = occupancy_voxel_layer.get('occ_feature_shape', None)
        if point_cloud_range is None or self.occ_feature_shape is None:
            raise ValueError("point_cloud_range and sparse_shape must be provided for occupancy voxelization")

        # Convert to tensors for arithmetic, then back to list for Voxelization
        pcr_tensor = torch.tensor(point_cloud_range, dtype=torch.float32)
        occ_shape_tensor = torch.tensor(self.occ_feature_shape, dtype=torch.float32)
        occ_voxel_size = (pcr_tensor[3:] - pcr_tensor[:3]) / occ_shape_tensor  # (3,)
        occ_voxel_size = occ_voxel_size.tolist()
        
        # update the occupancy_voxel_layer voxel_size and remove the occ_feature_shape
        occupancy_voxel_layer['voxel_size'] = occ_voxel_size
        occupancy_voxel_layer.pop('occ_feature_shape', None)
        return Voxelization(**occupancy_voxel_layer)
    
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
    
    def _generate_occupancy_map(
        self,
        voxel_features: torch.Tensor,
        num_points: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate occupancy map from voxel features (sparse conv + bev height occupancy).
        
        Args:
            voxel_features: Voxel features
            num_points: Number of points per voxel
            coors: Voxel coordinates
            batch_size: Batch size
        
        Returns:
            occupancy_logits: (B, num_height_levels, H, W) Occupancy logits (not probabilities)
                Apply torch.sigmoid() to get probabilities [0, 1] if needed
            sparse_features: (B, C, H, W) Sparse feature representation from SparseEncoder
        """
        # Process through sparse middle encoder
        sparse_features = self.middle_encoder(voxel_features, coors, batch_size) # (B, C, H, W)
        
        # Process through BEV height occupancy (returns logits, not probabilities)
        occupancy_logits = self.bev_height_occupancy(sparse_features) # (B, num_height_levels, H, W)
        
        return occupancy_logits, sparse_features
    

    
    def _gather_feats_from_occupancy(
        self,
        sparse_features: torch.Tensor,
        occupancy_logits: torch.Tensor,
        coors: torch.Tensor,
        bev_shape: Tuple[int, int],
        sparse_shape: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather features from BEV feature map using occupancy mask.
        
        Args:
            sparse_features: (B, C, H, W) sparse features from SparseEncoder
            occupancy_logits: (B, occupancy_channels, H, W) occupancy logits (not probabilities)
            coors: (M, 4) voxel coordinates [batch, z, y, x]
            bev_shape: (H, W) BEV feature map shape
            sparse_shape: [Z, Y, X] full grid shape
        
        Returns:
            gathered_feats: (M', C) gathered feature vectors
            new_coors: (M', 4) new voxel coordinates after occupancy filtering
        """
        if coors.shape[0] == 0:
            return torch.empty((0, sparse_features.shape[1]), device=sparse_features.device), \
                   torch.empty((0, 4), device=coors.device, dtype=coors.dtype)
        
        H, W = bev_shape
        B, C = sparse_features.shape[:2]
        occupancy_channels = occupancy_logits.shape[1]
        num_z_levels = sparse_shape[0]
        
        # Convert logits to probabilities for thresholding
        occupancy_prob = torch.sigmoid(occupancy_logits)  # (B, occupancy_channels, H, W)
        
        # Map Z dimension to occupancy channels proportionally
        z_to_channel_scale = occupancy_channels / num_z_levels
        
        # Calculate downsample ratios
        max_x = coors[:, 3].max().item()
        max_y = coors[:, 2].max().item()
        ratio_x = max(1, int((max_x + 1) / W))
        ratio_y = max(1, int((max_y + 1) / H))
        
        gathered_feats = []
        new_coors = []
        
        for coor in coors:
            b, z, y, x = coor.int().cpu().tolist()
            if b >= B:
                continue
            
            # Map to BEV indices
            bev_y = min(H - 1, y // ratio_y)
            bev_x = min(W - 1, x // ratio_x)
            
            # Map Z index to occupancy channel
            if 0 <= z < num_z_levels:
                channel_idx = int(z * z_to_channel_scale)
                channel_idx = min(occupancy_channels - 1, channel_idx)
                # Check occupancy probability at this location and mapped channel
                occ_prob = occupancy_prob[b, channel_idx, bev_y, bev_x]
                # Use threshold to filter (e.g., > 0.5)
                if occ_prob > 0.5:
                    # Gather feature from sparse_features
                    feat = sparse_features[b, :, bev_y, bev_x]  # (C,)
                    gathered_feats.append(feat)
                    new_coors.append(coor)
        
        if not gathered_feats:
            return torch.empty((0, C), device=sparse_features.device), \
                   torch.empty((0, 4), device=coors.device, dtype=coors.dtype)
        
        gathered_feats = torch.stack(gathered_feats, dim=0)  # (M', C)
        new_coors = torch.stack(new_coors, dim=0)  # (M', 4)
        
        return gathered_feats, new_coors
    
    
    def calculate_loss(
        self,
        pseudo_occupancy_logits: torch.Tensor,
        gt_occupancy_map: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate occupancy loss.
        
        Args:
            pseudo_occupancy_logits: (B, in_channels, H, W) pseudo occupancy logits (not probabilities)
            gt_occupancy_map: (B, in_channels, H, W) GT occupancy probability maps [0, 1]
        
        Returns:
            Loss value
        """
        # gt_occupancy_map is (B, in_channels, H, W) eg. [2, 32, 180, 180] - probabilities
        # pseudo_occupancy_logits is (B, in_channels, H, W) eg. [2, 32, 180, 180] - logits
        
        # Loss function will use binary_cross_entropy_with_logits internally (more stable)
        return self.loss_occupancy(pseudo_occupancy_logits, gt_occupancy_map, use_logits=True)
    
    def _generate_gt_occupancy_map(
        self,
        gt_points: torch.Tensor,
    ) -> torch.Tensor:
        """Generate GT occupancy map from GT points.
        
        Args:
            gt_points: (B, N, 3) GT point cloud coordinates [x, y, z]
            batch_size: Batch size
        """
        
        B = gt_points.shape[0]
        voxels_list, coors_list, num_points_list = [], [], []
        
        for b in range(B):
            res = gt_points[b]
            if not res.is_contiguous():
                res = res.contiguous()
            if not torch.is_floating_point(res):
                res = res.float()
            res_voxels, res_coors, res_num_points = self.gt_occupancty_layer(res)
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
        
        voxel_occupancy = self.gt_occupancty_voxel_encoder(voxels, num_points, coors)
        voxel_occupancy_per_batch = torch.split(
            voxel_occupancy, [c.shape[0] for c in coors_list], dim=0
        )

        X, Y, C = self.occ_feature_shape
        gt_occupancy_feature_map = torch.zeros(
            B, C, Y, X, device=voxel_occupancy.device
        )

        for b_idx in range(B):
            coors_b = coors_list[b_idx]                 # (Nb, 3) [z, y, x]
            occ_b   = voxel_occupancy_per_batch[b_idx] # (Nb, 1)

            z = coors_b[:, 0].long()
            y = coors_b[:, 1].long()
            x = coors_b[:, 2].long()

            gt_occupancy_feature_map[b_idx, z, y, x] = occ_b.squeeze(-1)

        return gt_occupancy_feature_map

    def _save_debug_data(
        self,
        pseudo_coors: torch.Tensor,
        gt_points: torch.Tensor,
        pseudo_occupancy_logits: torch.Tensor,
        gt_occupancy_map: torch.Tensor,
    ):
        """Save debug visualization data to pickle file.
        
        Args:
            pseudo_coors: (M, 4) pseudo voxel coordinates [batch, z, y, x]
            gt_points: (B, N, 3) GT point cloud
            pseudo_occupancy_logits: (B, C, H, W) pseudo occupancy logits (will be converted to probabilities)
            gt_occupancy_map: (B, C, H, W) GT occupancy map (probabilities)
        """
        import os
        
        # Create debug directory if it doesn't exist
        os.makedirs(self.debug_viz_dir, exist_ok=True)
        
        # Get GT coors by voxelizing GT points
        gt_coors = None
        if gt_points is not None:
            B = gt_points.shape[0]
            gt_coors_list = []
            for b in range(B):
                res = gt_points[b]
                if not res.is_contiguous():
                    res = res.contiguous()
                if not torch.is_floating_point(res):
                    res = res.float()
                _, res_coors, _ = self.voxel_layer(res)
                # Add batch index
                coor_pad = F.pad(res_coors, (1, 0), mode='constant', value=b)
                gt_coors_list.append(coor_pad)
            if gt_coors_list:
                gt_coors = torch.cat(gt_coors_list, dim=0)
        
        # Convert logits to probabilities for visualization (better for visualization)
        pseudo_occupancy_map = torch.sigmoid(pseudo_occupancy_logits) if pseudo_occupancy_logits is not None else None
        
        # Prepare data for saving (save probabilities, not logits, for visualization)
        debug_data = {
            "pseudo_coors": pseudo_coors.detach().cpu() if pseudo_coors is not None else None,
            "gt_coors": gt_coors.detach().cpu() if gt_coors is not None else None,
            "pseudo_occupancy_map": pseudo_occupancy_map.detach().cpu() if pseudo_occupancy_map is not None else None,
            "gt_occupancy_map": gt_occupancy_map.detach().cpu() if gt_occupancy_map is not None else None,
            "voxel_size": self.voxel_size.cpu().numpy().tolist(),
            "point_cloud_range": self.point_cloud_range.cpu().numpy().tolist(),
        }
        
        # Save to pickle file
        # Note: debug_counter is incremented in forward() before calling this function
        filename = f"debug_iter_{self.debug_counter:06d}.pkl"
        filepath = os.path.join(self.debug_viz_dir, filename)
        
        with open(filepath, "wb") as f:
            pickle.dump(debug_data, f)
        

 
    
    
    
    
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
        
        # Process through sparse encoder and BEV height occupancy
        pseudo_occupancy_logits, pseudo_sparse_features = self._generate_occupancy_map(
            pseudo_voxel_features, pseudo_num_points, pseudo_coors, batch_size
        )
        
        # Compute losses if needed
        losses = None
        gt_occupancy_map = None
        
        # Generate GT occupancy map if we have GT points (needed for loss and/or visualization)
        if gt_points is not None:
            gt_occupancy_map = self._generate_gt_occupancy_map(gt_points)
            
            if return_loss:
                # Calculate occupancy loss (reuse gt_occupancy_map to avoid regenerating)
                loss_value = self.calculate_loss(pseudo_occupancy_logits, gt_occupancy_map)
                

                self.debug_counter += 1
                # save the debug data here if viz flag is True and debug_counter is divisible by 300
                if self.debug_viz and self.debug_counter % 10 == 0:
                    self._save_debug_data(
                        pseudo_coors=pseudo_coors,
                        gt_points=gt_points,
                        pseudo_occupancy_logits=pseudo_occupancy_logits,
                        gt_occupancy_map=gt_occupancy_map,
                    )
                    
                # Increment debug counter every iteration (to track all iterations, not just saved ones)
                
                losses = dict(loss_occupancy=loss_value)
        
        
        # TODO: Implement refined_points generation from occupancy map
        # For now, return pseudo_points as placeholder
        refined_points = pseudo_points
        
        return refined_points, losses

