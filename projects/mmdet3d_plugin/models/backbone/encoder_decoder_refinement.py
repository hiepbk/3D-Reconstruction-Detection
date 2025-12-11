"""
Encoder-Decoder Point Cloud Refinement with Latent-Level Supervision.

This module implements a two-stage refinement approach:
1. Encoder: Full pseudo point cloud (M×3) → latent vector (L-dim)
2. Latent-level supervision: EMD/Chamfer on downsampled 1024 points (cheap)
3. Decoder: Latent vector → refined point cloud (M×3)
4. Full-resolution loss: L2/smooth-L1 on full 40k points (cheap, no pairwise matrices)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union, List
from mmdet.models.builder import BACKBONES, LOSSES, build_loss
from mmdet3d.ops import furthest_point_sample
from mmdet3d.models.losses import ChamferDistance


class PointNetEncoder(nn.Module):
    """PointNet-style encoder for point cloud feature extraction.
    
    Can output either:
    - Per-point features: (B, N, L) for latent-level loss
    - Global feature: (B, L) for decoder
    """
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 1024, hidden_dims: list = [64, 128, 256, 512]):
        """
        Args:
            in_channels: Input point dimension (3 for XYZ)
            latent_dim: Output latent dimension (default 1024)
            hidden_dims: Hidden dimensions for each layer
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        # Point-wise MLP layers
        layers = []
        prev_dim = in_channels
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim
        
        self.point_layers = nn.Sequential(*layers)
        
        # Project to latent dimension (per-point features)
        self.latent_proj = nn.Sequential(
            nn.Conv1d(prev_dim, latent_dim, 1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )
        
        # Global feature extraction (for decoder)
        # Note: No BatchNorm here because batch_size=1 causes issues, and global feature is already normalized by max pooling
        self.global_layers = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, points: torch.Tensor, return_global: bool = True) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) point cloud
            return_global: If True, return global feature (B, L). If False, return per-point features (B, N, L).
        Returns:
            If return_global: (B, latent_dim) global feature vector
            If not return_global: (B, N, latent_dim) per-point features
        """
        B, N, C = points.shape
        
        # Transpose to (B, C, N) for Conv1d
        x = points.transpose(1, 2)  # (B, 3, N)
        
        # Point-wise feature extraction
        x = self.point_layers(x)  # (B, hidden_dim, N)
        
        # Project to latent dimension (per-point features)
        per_point_features = self.latent_proj(x)  # (B, latent_dim, N)
        
        if return_global:
            # Global max pooling
            global_feat = per_point_features.max(dim=2)[0]  # (B, latent_dim)
            # Additional global processing
            global_feat = self.global_layers(global_feat)  # (B, latent_dim)
            return global_feat
        else:
            # Return per-point features
            per_point_features = per_point_features.transpose(1, 2)  # (B, N, latent_dim)
            return per_point_features


class CoordinateBasedDecoder(nn.Module):
    """Memory-efficient coordinate-based decoder.
    
    Instead of generating all points at once (which requires huge weight matrices),
    this decoder takes a latent code + original point coordinates and outputs refined coordinates.
    This is similar to NeRF-style coordinate-based MLPs.
    
    Architecture: latent + coordinates → refined coordinates
    This avoids the need for a Linear(128, 40000*3) layer.
    """
    
    def __init__(self, latent_dim: int = 1024, hidden_dims: list = [512, 256, 128], chunk_size: int = 4096):
        """
        Args:
            latent_dim: Input latent dimension
            hidden_dims: Hidden dimensions for decoder MLP
            chunk_size: Process points in chunks to avoid memory spikes
        """
        super().__init__()
        self.chunk_size = chunk_size
        
        # Decoder MLP: (latent + coordinates) → refined coordinates
        # Input: latent (L-dim) + coordinates (3-dim) = (L+3)-dim
        # Output: refined coordinates (3-dim)
        layers = []
        input_dim = latent_dim + 3  # latent + xyz coordinates
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm works with batch_size=1
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim
        
        # Output layer: refined coordinates (3-dim)
        layers.append(nn.Linear(prev_dim, 3))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, latent: torch.Tensor, original_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim) latent vector
            original_points: (B, M, 3) original point coordinates
        Returns:
            refined_points: (B, M, 3) refined point cloud
        """
        B, M, _ = original_points.shape
        device = original_points.device
        
        # Expand latent to match number of points: (B, M, latent_dim)
        latent_expanded = latent.unsqueeze(1).expand(B, M, -1)  # (B, M, latent_dim)
        
        # Concatenate latent with coordinates: (B, M, latent_dim + 3)
        input_features = torch.cat([latent_expanded, original_points], dim=2)  # (B, M, latent_dim + 3)
        
        # Process in chunks to avoid memory spikes
        if M > self.chunk_size:
            refined_chunks = []
            for i in range(0, M, self.chunk_size):
                end_i = min(i + self.chunk_size, M)
                chunk = input_features[:, i:end_i, :]  # (B, chunk_size, latent_dim + 3)
                
                # Reshape for MLP: (B * chunk_size, latent_dim + 3)
                B_chunk = chunk.shape[0]
                chunk_flat = chunk.view(B_chunk * (end_i - i), -1)
                
                # Decode: (B * chunk_size, 3)
                refined_chunk_flat = self.decoder(chunk_flat)
                
                # Reshape back: (B, chunk_size, 3)
                refined_chunk = refined_chunk_flat.view(B_chunk, end_i - i, 3)
                refined_chunks.append(refined_chunk)
            
            refined_points = torch.cat(refined_chunks, dim=1)  # (B, M, 3)
        else:
            # Process all at once if small enough
            input_flat = input_features.view(B * M, -1)  # (B * M, latent_dim + 3)
            refined_flat = self.decoder(input_flat)  # (B * M, 3)
            refined_points = refined_flat.view(B, M, 3)  # (B, M, 3)
        
        return refined_points


@BACKBONES.register_module()
class EncoderDecoderRefinement(nn.Module):
    """Encoder-Decoder refinement network with latent-level supervision.
    
    Architecture:
    1. Encoder: pseudo (M×3) → latent (L-dim)
    2. Latent loss: EMD/Chamfer on 1024 downsampled points (cheap)
    3. Decoder: latent (L-dim) → refined (M×3)
    4. Full-resolution loss: L2/smooth-L1 on full points (cheap)
    """
    
    def __init__(
        self,
        encoder: Optional[Dict] = None,
        decoder: Optional[Dict] = None,
        latent_dim: int = 1024,
        latent_downsample: int = 1024,  # Number of points for latent-level loss
        loss_latent: Optional[Dict] = None,  # EMD/Chamfer on 1024 points
        loss_full: Optional[Dict] = None,  # L2/smooth-L1 on full points
        loss_smoothness: Optional[Dict] = None,
        loss_latent_weight: float = 1.0,
        loss_full_weight: float = 1.0,
        loss_full_sample_points: Optional[int] = None,  # Sample points for full-resolution loss (None = use all)
        use_color: bool = False,
    ):
        """
        Args:
            encoder: Config dict for encoder (PointNetEncoder)
            decoder: Config dict for decoder (PointNetDecoder)
            latent_dim: Latent dimension (default 1024)
            latent_downsample: Number of points to downsample for latent loss (default 1024)
            loss_latent: Config dict for latent-level loss (EMD/Chamfer on 1024 points)
            loss_full: Config dict for full-resolution loss (L2/smooth-L1)
            loss_smoothness: Config dict for smoothness regularization
            loss_latent_weight: Weight for latent-level loss
            loss_full_weight: Weight for full-resolution loss
            use_color: If False, only use XYZ (3 channels)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.latent_downsample = latent_downsample
        self.loss_latent_weight = loss_latent_weight
        self.loss_full_weight = loss_full_weight
        self.loss_full_sample_points = loss_full_sample_points
        self.use_color = use_color
        
        # Build encoder
        if encoder is None:
            encoder_config = dict(
                in_channels=3,
                latent_dim=latent_dim,
                hidden_dims=[64, 128, 256, 512],
            )
        else:
            encoder_config = encoder.copy() if isinstance(encoder, dict) else {}
            # Remove 'type' if present (not needed for direct instantiation)
            encoder_config.pop('type', None)
        
        self.encoder = PointNetEncoder(**encoder_config)
        
        # Build decoder (coordinate-based, doesn't need output size)
        if decoder is None:
            decoder_config = dict(
                latent_dim=latent_dim,
                hidden_dims=[512, 256, 128],
                chunk_size=4096,  # Process points in 4k chunks
            )
        else:
            decoder_config = decoder.copy()
            decoder_config['latent_dim'] = latent_dim
            decoder_config.pop('type', None)
            decoder_config.pop('output_points', None)  # Not needed for coordinate-based decoder
        
        self.decoder = CoordinateBasedDecoder(**decoder_config)
        
        # Build loss modules
        if loss_latent is not None:
            self.loss_latent = build_loss(loss_latent)
        else:
            self.loss_latent = None
        
        if loss_full is not None:
            self.loss_full = build_loss(loss_full)
        else:
            self.loss_full = None
        
        if loss_smoothness is not None:
            self.loss_smoothness = build_loss(loss_smoothness)
        else:
            self.loss_smoothness = None
    
    # Removed _build_decoder - decoder is built in __init__ now
    
    def _fps_downsample(self, points: torch.Tensor, num_samples: int, device: torch.device) -> torch.Tensor:
        """Downsample points using FPS."""
        is_batched = points.dim() == 3
        if not is_batched:
            points = points.unsqueeze(0)
        
        B, N, C = points.shape
        
        if N <= num_samples:
            # If N < num_samples, pad with last point repeated
            if N < num_samples:
                padding = points[:, -1:, :].repeat(1, num_samples - N, 1)  # (B, num_samples - N, C)
                points = torch.cat([points, padding], dim=1)  # (B, num_samples, C)
            return points if is_batched else points.squeeze(0)
        
        if device.type != 'cuda':
            # Fallback: random sampling
            indices = torch.randperm(N, device=device)[:num_samples]
            sampled = points[:, indices, :]
        else:
            points_for_fps = points.to(device).contiguous()
            fps_indices = furthest_point_sample(points_for_fps, num_samples)  # (B, num_samples)
            B_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_samples)
            sampled = points_for_fps[B_idx, fps_indices]
        
        return sampled if is_batched else sampled.squeeze(0)
    
    def forward(
        self,
        pseudo_points: Union[torch.Tensor, List[torch.Tensor]],
        gt_points: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        return_loss: bool = False,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            pseudo_points: (B, M, 3) tensor or list of (M, 3) tensors
            gt_points: (B, N, 3) tensor or list of (N, 3) tensors (optional)
            return_loss: Whether to compute losses
        
        Returns:
            refined_points: (B, M, 3) tensor or list of (M, 3) tensors
            losses: Dict of loss values (if return_loss=True)
        """
        # Handle list inputs (convert to tensor)
        if isinstance(pseudo_points, list):
            pseudo_points = torch.stack(pseudo_points, dim=0)  # (B, M, C)
        
        if isinstance(gt_points, list):
            gt_points = torch.stack(gt_points, dim=0)  # (B, N, C)
        
        B, M, C = pseudo_points.shape
        device = pseudo_points.device
        
        # Only use XYZ if not using colors
        if not self.use_color:
            pseudo_points = pseudo_points[:, :, :3]
            C = 3
        
        # 1. Encode pseudo point cloud to latent
        pseudo_latent = self.encoder(pseudo_points)  # (B, latent_dim)
        
        # 2. Decode latent + original points to refined points (coordinate-based)
        # This avoids the huge weight matrix of generating all points at once
        refined_points = self.decoder(pseudo_latent, pseudo_points)  # (B, M, 3)
        
        # Compute losses if needed
        losses = None
        if return_loss and gt_points is not None:
            loss_dict = {}
            
            # Only use XYZ for GT
            gt_xyz = gt_points[:, :, :3] if gt_points.shape[2] >= 3 else gt_points
            
            # 3. Latent-level loss (cheap: only on 1024 points)
            if self.loss_latent is not None:
                # Downsample both pseudo and GT to 1024 points
                pseudo_downsampled = self._fps_downsample(pseudo_points, self.latent_downsample, device)
                gt_downsampled = self._fps_downsample(gt_xyz, self.latent_downsample, device)
                
                # Encode downsampled points to per-point latent features
                pseudo_latent_features = self.encoder(pseudo_downsampled, return_global=False)  # (B, K, latent_dim)
                gt_latent_features = self.encoder(gt_downsampled, return_global=False)  # (B, K, latent_dim)
                
                # Compute latent-level loss (EMD/Chamfer on latent features)
                # Now we have (B, K, latent_dim) which can be treated as point clouds
                latent_loss = self.loss_latent(pseudo_latent_features, gt_latent_features)
                
                # ChamferDistance returns (loss_source, loss_target) tuple
                # Sum them to get a single scalar loss
                if isinstance(latent_loss, tuple):
                    latent_loss = latent_loss[0] + latent_loss[1]  # Sum source and target losses
                
                loss_dict['loss_latent'] = latent_loss * self.loss_latent_weight
            
            # 4. Full-resolution loss (cheap: L2/smooth-L1, no pairwise matrices)
            if self.loss_full is not None:
                # Sample points for loss computation to reduce memory during backward pass
                if self.loss_full_sample_points is not None and self.loss_full_sample_points < M:
                    # Sample both refined and GT points for loss computation
                    refined_for_loss = self._fps_downsample(refined_points, self.loss_full_sample_points, device)  # (B, K, 3)
                    
                    # Align GT to sampled refined points
                    gt_aligned_list = []
                    for b_idx in range(B):
                        gt_b = gt_xyz[b_idx]  # (N_b, 3)
                        # Sample GT to match sampled refined points
                        gt_b_aligned = self._fps_downsample(gt_b.unsqueeze(0), self.loss_full_sample_points, device)  # (1, K, 3)
                        gt_b_aligned = gt_b_aligned.squeeze(0)  # (K, 3)
                        gt_aligned_list.append(gt_b_aligned)
                    
                    gt_aligned = torch.stack(gt_aligned_list, dim=0)  # (B, K, 3)
                else:
                    # Use all points (might cause OOM during backward)
                    # Align GT to refined points (handle per-batch-item alignment)
                    gt_aligned_list = []
                    for b_idx in range(B):
                        gt_b = gt_xyz[b_idx]  # (N_b, 3) - might vary per batch
                        refined_b = refined_points[b_idx]  # (M, 3) - always M points
                        M_b = refined_b.shape[0]  # Should be M
                        N_b = gt_b.shape[0]
                        
                        if N_b != M_b:
                            # Align GT to M points using FPS (downsample or pad)
                            gt_b_aligned = self._fps_downsample(gt_b.unsqueeze(0), M_b, device)  # (1, M_b, 3)
                            gt_b_aligned = gt_b_aligned.squeeze(0)  # (M_b, 3)
                        else:
                            gt_b_aligned = gt_b
                        
                        gt_aligned_list.append(gt_b_aligned)
                    
                    # Stack to (B, M, 3) - all should have M points now
                    gt_aligned = torch.stack(gt_aligned_list, dim=0)  # (B, M, 3)
                    refined_for_loss = refined_points
                
                # Compute full-resolution loss (L2/smooth-L1)
                full_loss = self.loss_full(refined_for_loss, gt_aligned)
                loss_dict['loss_full'] = full_loss * self.loss_full_weight
            
            # 5. Smoothness regularization (optional)
            if self.loss_smoothness is not None:
                smooth_loss = self.loss_smoothness(refined_points, pseudo_points)
                loss_dict['loss_smoothness'] = smooth_loss
            
            losses = loss_dict
        
        # Return as tensor (B, M, 3) - reconstruction_backbone will convert to list if needed
        return refined_points, losses

