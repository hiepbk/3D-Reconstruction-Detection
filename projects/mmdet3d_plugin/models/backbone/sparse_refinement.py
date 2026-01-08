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
        loss_occupancy: Dict,
        loss_feature: Dict,
        loss_bev: Optional[Dict] = None,
        loss_weight: float = 1.0,
        loss_occupancy_weight: float = 1.0,
        loss_feature_weight: float = 0.2,
        loss_bev_weight: float = 0.1,
        use_color: bool = False,
        debug_viz: bool = False,
        debug_viz_dir: str = "debug_viz",
        teacher_checkpoint: Optional[str] = None,  # Path to pretrained teacher checkpoint
        training_phase: int = 2,  # 1 = Train teacher, 2 = Train student (default)
    ):
        """
        Feature-space domain adaptation for sparse voxel features.
        
        No transformer, no AR generation, just direct feature alignment.
        
        Args:
            pts_voxel_layer: Config for voxelization layer
            pts_voxel_encoder: Config for voxel encoder (e.g., HardSimpleVFE)
            pts_middle_encoder: Config for sparse middle encoder (e.g., SparseEncoderV2 with return_type='sparse')
            loss_occupancy: Config for voxel occupancy alignment loss (Dice loss recommended)
            loss_feature: Config for sparse feature alignment loss (L2 or cosine)
            loss_bev: Config for dense BEV feature alignment loss (cosine similarity, optional)
            loss_weight: Global weight multiplier for all losses
            loss_occupancy_weight: Weight for occupancy loss (default: 1.0)
            loss_feature_weight: Weight for feature loss (default: 0.2)
            loss_bev_weight: Weight for dense BEV loss (default: 0.1, auxiliary loss)
            use_color: If True, use RGB colors in addition to XYZ
        """
        super().__init__()
        
        self.use_color = use_color
        self.loss_weight = loss_weight
        self.loss_occupancy_weight = loss_occupancy_weight
        self.loss_feature_weight = loss_feature_weight
        self.loss_bev_weight = loss_bev_weight
        self.debug_viz = debug_viz
        self.debug_viz_dir = debug_viz_dir
        self.training_phase = training_phase  # 1 = Train teacher, 2 = Train student

        
        # Build voxelization layer
        self.voxel_layer = Voxelization(**pts_voxel_layer)
        # Cache voxel meta for visualization
        self.voxel_size = torch.tensor(pts_voxel_layer['voxel_size'], dtype=torch.float32)
        self.point_cloud_range = torch.tensor(pts_voxel_layer['point_cloud_range'], dtype=torch.float32)
        
        # Build voxel encoder
        self.voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        
        # Build SEPARATE sparse middle encoders for GT and Pseudo (NOT shared)
        # This prevents trivial identity mapping and enables proper feature distillation
        # GT branch (teacher)
        self.middle_encoder_gt = builder.build_middle_encoder(pts_middle_encoder)
        # Pseudo branch (student)
        self.middle_encoder_pseudo = builder.build_middle_encoder(pts_middle_encoder)
        
        # Load pretrained teacher weights if provided
        if teacher_checkpoint is not None:
            self._load_teacher_checkpoint(teacher_checkpoint)
        
        # Set training phase behavior
        if self.training_phase == 1:
            # Phase 1: Train teacher encoder (unfreeze, allow gradients)
            for param in self.middle_encoder_gt.parameters():
                param.requires_grad = True
            print("Phase 1: Teacher encoder is trainable (requires_grad=True)")
        elif self.training_phase == 2:
            # Phase 2: Freeze teacher encoder (no gradients)
            for param in self.middle_encoder_gt.parameters():
                param.requires_grad = False
            print("Phase 2: Teacher encoder is frozen (requires_grad=False)")
        else:
            raise ValueError(f"Invalid training_phase: {self.training_phase}. Must be 1 or 2.")
        
        # Build losses (direct feature alignment, no transformer)
        self.loss_occupancy = build_loss(loss_occupancy)
        self.loss_feature = build_loss(loss_feature)
        
        # Build dense BEV feature loss (optional, auxiliary loss)
        if loss_bev is not None:
            self.loss_bev = build_loss(loss_bev)
        else:
            self.loss_bev = None
        
        # Get sparse_shape from middle_encoder config
        # Convert to list to avoid dict_keys pickle issues
        sparse_shape_val = pts_middle_encoder.get('sparse_shape', [41, 1440, 1440])
        self.sparse_shape = list(sparse_shape_val) if isinstance(sparse_shape_val, (list, tuple)) else sparse_shape_val  # [Z, Y, X]

        # Visualization caching flag
        self.enable_visual_debug = False
        self.debug_counter = 0
    
    def _load_teacher_checkpoint(self, checkpoint_path: str):
        """
        Load pretrained weights for teacher encoder (middle_encoder_gt).
        
        Supports two checkpoint formats:
        1. Extracted teacher checkpoint (from load_pretrained_teacher.py):
           - Keys: middle_encoder_gt.*
        2. Full CenterPoint checkpoint:
           - Keys: pts_middle_encoder.* (will be mapped to middle_encoder_gt.*)
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
        
        print(f"Loading teacher encoder weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Map checkpoint keys to model keys
        teacher_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('middle_encoder_gt.'):
                # Already in correct format
                teacher_state_dict[key] = value
            elif key.startswith('pts_middle_encoder.'):
                # Map from CenterPoint format
                new_key = key.replace('pts_middle_encoder.', 'middle_encoder_gt.')
                teacher_state_dict[new_key] = value
            # Ignore other keys
        
        if len(teacher_state_dict) == 0:
            print(f"⚠️  Warning: No teacher encoder weights found in checkpoint!")
            print(f"Available keys (first 10):")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                print(f"  - {key}")
            return
        
        # Load weights into teacher encoder
        missing_keys, unexpected_keys = self.middle_encoder_gt.load_state_dict(
            teacher_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"⚠️  Warning: Missing keys in teacher encoder:")
            for key in missing_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"⚠️  Warning: Unexpected keys (ignored):")
            for key in unexpected_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")
        
        loaded_count = len(teacher_state_dict) - len(missing_keys)
        print(f"✓ Loaded {loaded_count}/{len(teacher_state_dict)} teacher encoder weights")
        
        # Freeze teacher encoder after loading pretrained weights
        for param in self.middle_encoder_gt.parameters():
            param.requires_grad = False
        print("✓ Teacher encoder frozen (requires_grad=False)")
        
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
        pseudo_dense_features: torch.Tensor,
        pseudo_sparse_features: torch.Tensor,
        pseudo_sparse_indices: torch.Tensor,
        pseudo_sparse_spatial_shape: List[int],
        gt_points: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Tuple[Dict, Optional[Dict[str, torch.Tensor]]]:
        """
        Training forward: compute feature alignment losses.
        
        No generation, no transformer, just direct loss computation.
        """
        assert gt_points is not None, "GT points are required for training"
        
        batch_size = gt_points.shape[0]
        
        # Phase 1: Train teacher encoder (use gradients)
        # Phase 2: Freeze teacher encoder (no gradients, use as target)
        if self.training_phase == 1:
            # Phase 1: Reuse GT features already computed in forward() (no redundant processing)
            # In forward(), we processed GT points and set pseudo_* = gt_* features
            # These features already have gradients flowing through teacher encoder
            gt_dense_features = pseudo_dense_features
            gt_sparse_features = pseudo_sparse_features
            gt_sparse_indices = pseudo_sparse_indices
            gt_sparse_spatial_shape = pseudo_sparse_spatial_shape
            # In Phase 1, we don't compute alignment losses (teacher is learning)
            # Detection loss will be computed in ResDet3D using GT branch features
            losses = {}
        else:
            # Phase 2: Freeze teacher encoder - no gradients, use as target
            with torch.no_grad():
                gt_voxel_features, gt_num_points, gt_coors = self._voxel_encoder(gt_points)
                gt_dense_features, gt_sparse_features, gt_sparse_indices, gt_sparse_spatial_shape = self.middle_encoder_gt(
                    gt_voxel_features, gt_coors, batch_size
                )
            
            # Detach GT features to prevent gradient flow into GT branch
            # But keep them as targets for pseudo branch to learn from
            gt_sparse_features = gt_sparse_features.detach()
            gt_sparse_indices = gt_sparse_indices.detach()
            
            # Phase 2: Compute alignment losses
            losses = {}
            
            # Loss 1: Voxel Occupancy Alignment (MOST IMPORTANT)
            loss_occupancy = self.loss_occupancy(
                pseudo_indices=pseudo_sparse_indices,
                gt_indices=gt_sparse_indices,
                spatial_shape=pseudo_sparse_spatial_shape,
            )
            losses['loss_occupancy'] = loss_occupancy * self.loss_occupancy_weight
            
            # Loss 2: Sparse Feature Alignment (only at overlapping voxels)
            loss_feature = self.loss_feature(
                pseudo_features=pseudo_sparse_features,
                pseudo_indices=pseudo_sparse_indices,
                gt_features=gt_sparse_features,
                gt_indices=gt_sparse_indices,
            )
            losses['loss_feature'] = loss_feature * self.loss_feature_weight
            
            # Loss 3: Dense BEV Feature Alignment (auxiliary loss)
            # Aligns dense BEV features [B, C*D, H, W] using cosine similarity with foreground masking
            if self.loss_bev is not None:
                loss_bev = self.loss_bev(
                    pseudo_bev=pseudo_dense_features,  # [B, C*D, H, W]
                    gt_bev=gt_dense_features,          # [B, C*D, H, W]
                )
                losses['loss_bev'] = loss_bev * self.loss_bev_weight
        
        # Apply global weight multiplier
        for k in losses:
            losses[k] = losses[k] * self.loss_weight
        
        # Return sparse features and dense features for both branches (for detection)
        # Store in a dict format that ResDet3D can use
        sparse_feat_dict = {
            'pseudo': {
                'dense_features': pseudo_dense_features,  # [B, C*D, H, W] dense BEV for detection
                'features': pseudo_sparse_features,       # [N, C] sparse features for loss
                'indices': pseudo_sparse_indices,         # [N, 4] sparse indices
                'spatial_shape': pseudo_sparse_spatial_shape,
            },
            'gt': {
                'dense_features': gt_dense_features,     # [B, C*D, H, W] dense BEV (for reference)
                'features': gt_sparse_features,           # [N, C] sparse features for loss
                'indices': gt_sparse_indices,             # [N, 4] sparse indices
                'spatial_shape': gt_sparse_spatial_shape,
            }
        }
        
        return sparse_feat_dict, losses

    
    def forward_test(
        self,
        pseudo_dense_features: torch.Tensor,
        pseudo_sparse_features: torch.Tensor,
        pseudo_sparse_indices: torch.Tensor,
        pseudo_sparse_spatial_shape: List[int],
        gt_points: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict, Optional[Dict[str, float]]]:
        """
        Test forward: return pseudo features directly, compute metrics if GT available.
        
        No refinement needed - pseudo features are already aligned during training.
        """
        batch_size = pseudo_dense_features.shape[0]
        
        # In inference, return sparse features and dense features for both branches
        sparse_feat_dict = {
            'pseudo': {
                'dense_features': pseudo_dense_features,  # [B, C*D, H, W] dense BEV for detection
                'features': pseudo_sparse_features,        # [N, C] sparse features
                'indices': pseudo_sparse_indices,          # [N, 4] sparse indices
                'spatial_shape': pseudo_sparse_spatial_shape,
            }
        }
        
        metrics = None
        if gt_points is not None:
            # GT branch for metrics computation (if available)
            gt_voxel_features, gt_num_points, gt_coors = self._voxel_encoder(gt_points)
            gt_dense_features, gt_sparse_features, gt_sparse_indices, _ = self.middle_encoder_gt(
                gt_voxel_features, gt_coors, batch_size
            )
            # Add GT features to dict
            sparse_feat_dict['gt'] = {
                'dense_features': gt_dense_features,       # [B, C*D, H, W] dense BEV (for reference)
                'features': gt_sparse_features,             # [N, C] sparse features
                'indices': gt_sparse_indices,               # [N, 4] sparse indices
                'spatial_shape': pseudo_sparse_spatial_shape,
            }
            # Compute metrics for monitoring (optional, for eval hook)
            metrics = self._compute_metrics(
                pseudo_sparse_features, pseudo_sparse_indices,
                gt_sparse_features, gt_sparse_indices
            )
        
        return sparse_feat_dict, metrics
    
    def _compute_metrics(
        self,
        pseudo_features: torch.Tensor,
        pseudo_indices: torch.Tensor,
        gt_features: torch.Tensor,
        gt_indices: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics (for monitoring, not used in loss)."""
        metrics = {}
        
        # Count metrics
        pseudo_count = pseudo_indices.shape[0]
        gt_count = gt_indices.shape[0]
        metrics['pseudo_count'] = float(pseudo_count)
        metrics['gt_count'] = float(gt_count)
        metrics['refined_count'] = float(pseudo_count)  # No refinement, same as pseudo
        metrics['count_diff'] = abs(pseudo_count - gt_count) / max(gt_count, 1)
        metrics['gen_ratio'] = pseudo_count / max(gt_count, 1)
        
        # Feature distance (if overlapping voxels exist)
        if pseudo_count > 0 and gt_count > 0:
            # Find overlapping voxels
            pseudo_voxels = set(tuple(idx.cpu().numpy()) for idx in pseudo_indices)
            gt_voxels = set(tuple(idx.cpu().numpy()) for idx in gt_indices)
            common_voxels = pseudo_voxels & gt_voxels
            
            if len(common_voxels) > 0:
                # Extract features at common voxels (simplified, for metrics only)
                # This is approximate - full matching done in loss
                feat_dist = torch.norm(
                    pseudo_features.mean(dim=0) - gt_features.mean(dim=0)
                ).item()
                metrics['feat_dist'] = feat_dist
            else:
                metrics['feat_dist'] = 0.0
        else:
            metrics['feat_dist'] = 0.0
        
        # Chamfer-like distance (simplified)
        metrics['chamfer_like_dist'] = 0.0  # Not computed for sparse features
        
        return metrics

    
    
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

        # Phase 1 optimization: Skip pseudo branch entirely (save memory/time)
        if self.training_phase == 1:
            # Phase 1: Only process GT points through teacher encoder
            if gt_points_xyz is None:
                raise ValueError("GT points are required for Phase 1 training")
            
            # Process GT points only
            gt_voxel_features, gt_num_points, gt_coors = self._voxel_encoder(gt_points_xyz)
            gt_dense_features, gt_sparse_features, gt_sparse_indices, gt_sparse_spatial_shape = self.middle_encoder_gt(
                gt_voxel_features, gt_coors, batch_size
            )
            
            # Create dummy pseudo features (same as GT) for return format consistency
            pseudo_dense_features = gt_dense_features
            pseudo_sparse_features = gt_sparse_features
            pseudo_sparse_indices = gt_sparse_indices
            pseudo_sparse_spatial_shape = gt_sparse_spatial_shape
        else:
            # Phase 2: Process both pseudo and GT branches
            # Pseudo branch: voxelize and encode through Pseudo SparseEncoder (separate from GT)
            pseudo_voxel_features, pseudo_num_points, pseudo_coors = self._voxel_encoder(pseudo_points_xyz)
            pseudo_dense_features, pseudo_sparse_features, pseudo_sparse_indices, pseudo_sparse_spatial_shape = self.middle_encoder_pseudo(
                pseudo_voxel_features, pseudo_coors, batch_size
            )

        if return_loss:
            sparse_feat_dict, losses = self.forward_train(
                pseudo_dense_features=pseudo_dense_features,
                pseudo_sparse_features=pseudo_sparse_features,
                pseudo_sparse_indices=pseudo_sparse_indices,
                pseudo_sparse_spatial_shape=pseudo_sparse_spatial_shape,
                gt_points=gt_points_xyz,  # Use xyz only
                return_loss=return_loss,
            )
            return sparse_feat_dict, losses
        else:
            # Test mode: return pseudo features, compute metrics if GT available
            sparse_feat_dict, metrics = self.forward_test(
                pseudo_dense_features=pseudo_dense_features,
                pseudo_sparse_features=pseudo_sparse_features,
                pseudo_sparse_indices=pseudo_sparse_indices,
                pseudo_sparse_spatial_shape=pseudo_sparse_spatial_shape,
                gt_points=gt_points_xyz,  # Pass GT points for metrics computation
            )

            return sparse_feat_dict, metrics


        
        
        

        
        

