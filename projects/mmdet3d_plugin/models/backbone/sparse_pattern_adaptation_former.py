"""
SparsePatternAdaptationFormer: ShapeFormer-inspired transformer for sparse pattern adaptation.

Uses Vector Quantization (VQ) codebook and two-stage autoregressive transformer:
1. Coordinate Transformer: predicts next coordinate
2. Value Transformer: predicts codebook index for that coordinate

Training: Teacher forcing with [S_P, END, S_C, END]
Inference: Autoregressive generation from S_P only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from mmdet3d.models.builder import MIDDLE_ENCODERS
from .vector_quantizer import VectorQuantizer
import time


def row_major_key(indices: torch.Tensor, spatial_shape: List[int]) -> torch.Tensor:
    """Convert 3D voxel indices to row-major flattened coordinates.
    
    Args:
        indices: (N, 4) [batch_idx, z_idx, y_idx, x_idx]
        spatial_shape: [D, H, W] e.g., [2, 180, 180]
    
    Returns:
        coord_ids: (N,) flattened coordinates in row-major order
    """
    D, H, W = spatial_shape
    z = indices[:, 1].long()
    y = indices[:, 2].long()
    x = indices[:, 3].long()
    # Row-major: z * (H * W) + y * W + x
    coord_ids = z * (H * W) + y * W + x
    return coord_ids


def sort_by_row_major(
    features: torch.Tensor,
    indices: torch.Tensor,
    spatial_shape: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort sparse features and indices by row-major order.
    
    Args:
        features: (N, C) sparse features
        indices: (N, 4) [batch_idx, z_idx, y_idx, x_idx]
        spatial_shape: [D, H, W]
    
    Returns:
        sorted_features: (N, C) sorted features
        sorted_indices: (N, 4) sorted indices
        coord_ids: (N,) sorted coordinate IDs
    """
    coord_ids = row_major_key(indices, spatial_shape)
    sort_idx = torch.argsort(coord_ids)
    return features[sort_idx], indices[sort_idx], coord_ids[sort_idx]


class BlockCausalAttention(nn.Module):
    """KV-cached block-causal attention for efficient autoregressive decoding.
    
    Supports both:
    - Training: full sequence with explicit mask
    - Inference: incremental decoding with KV cache
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5
        
    def forward(
        self,
        x: torch.Tensor,
        window_size: int = 512,
        attn_mask: Optional[torch.Tensor] = None,  # Deprecated: kept for API compatibility, not used
    ) -> torch.Tensor:
        """Training forward: chunked sliding-window attention (O(T·K) memory).
        
        Processes sequence in chunks, computing attention only within sliding window.
        NEVER allocates full (T × T) attention matrix.
        
        Args:
            x: (B, T, d_model) input tokens
            window_size: Sliding window size K (each token attends to previous K tokens)
            attn_mask: Unused (kept for API compatibility)
        
        Returns:
            out: (B, T, d_model) output tokens
        """
        B, T, C = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.nhead, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = self.k_proj(x).view(B, T, self.nhead, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        v = self.v_proj(x).view(B, T, self.nhead, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        
        # Chunked sliding-window attention
        # Process sequence in chunks to avoid T×T allocation
        chunk_size = window_size  # Process chunks of size K
        out_chunks = []
        
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_len = end - start
            
            # Query chunk: tokens [start, end)
            q_chunk = q[:, :, start:end, :]  # (B, H, chunk_len, d_head)
            
            # Key/Value chunk: tokens [max(0, start-window_size), end)
            # This ensures each token in q_chunk can attend to previous K tokens
            kv_start = max(0, start - window_size)
            k_chunk = k[:, :, kv_start:end, :]  # (B, H, ≤(window_size+chunk_len), d_head)
            v_chunk = v[:, :, kv_start:end, :]  # (B, H, ≤(window_size+chunk_len), d_head)
            
            # Compute attention scores: (B, H, chunk_len, kv_len)
            kv_len = k_chunk.shape[2]
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale  # (B, H, chunk_len, kv_len)
            
            # Create block-causal sliding-window mask (vectorized)
            # For query token at position i_rel in q_chunk (full position = start + i_rel),
            # it can attend to key positions j_rel in k_chunk (full position = kv_start + j_rel)
            # where: max(0, start + i_rel - window_size) <= kv_start + j_rel <= start + i_rel
            offset = start - kv_start  # Offset between q_chunk and k_chunk start positions
            
            # Create position indices
            i_rel = torch.arange(chunk_len, device=scores.device).unsqueeze(1)  # (chunk_len, 1)
            j_rel = torch.arange(kv_len, device=scores.device).unsqueeze(0)  # (1, kv_len)
            
            # Full positions
            full_i = start + i_rel  # (chunk_len, 1)
            full_j = kv_start + j_rel  # (1, kv_len)
            
            # Causal: j <= i
            causal_mask = (full_j <= full_i)  # (chunk_len, kv_len)
            
            # Sliding window: j >= max(0, i - window_size)
            window_start = torch.clamp(full_i - window_size, min=0)  # (chunk_len, 1)
            window_mask = (full_j >= window_start)  # (chunk_len, kv_len)
            
            # Combine: can attend if both causal AND within window
            mask = causal_mask & window_mask  # (chunk_len, kv_len)
            
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Compute attention
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            out_chunk = torch.matmul(attn_weights, v_chunk)  # (B, H, chunk_len, d_head)
            out_chunks.append(out_chunk)
        
        # Concatenate all chunks
        out = torch.cat(out_chunks, dim=2)  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)
        out = self.out_proj(out)
        
        return out
    
    def forward_step(
        self,
        x_t: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        window_size: int = 512,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Inference forward: incremental decoding with KV cache.
        
        Args:
            x_t: (B, 1, d_model) current token
            past_kv: Optional tuple of (past_k, past_v), each (B, H, K, d_head)
            window_size: Sliding window size K
        
        Returns:
            out_t: (B, 1, d_model) output for current token
            new_kv: Tuple of (new_k, new_v), each (B, H, min(K+1, window_size), d_head)
        """
        B, _, C = x_t.shape
        q_t = self.q_proj(x_t).view(B, 1, self.nhead, self.d_head).transpose(1, 2)  # (B, H, 1, d_head)
        k_t = self.k_proj(x_t).view(B, 1, self.nhead, self.d_head).transpose(1, 2)  # (B, H, 1, d_head)
        v_t = self.v_proj(x_t).view(B, 1, self.nhead, self.d_head).transpose(1, 2)  # (B, H, 1, d_head)
        
        if past_kv is None:
            # First token: no past
            k_cache = k_t  # (B, H, 1, d_head)
            v_cache = v_t  # (B, H, 1, d_head)
        else:
            past_k, past_v = past_kv
            # Append new k/v to cache
            k_cache = torch.cat([past_k, k_t], dim=2)  # (B, H, K+1, d_head)
            v_cache = torch.cat([past_v, v_t], dim=2)  # (B, H, K+1, d_head)
            
            # Truncate to window_size (keep only last K tokens)
            if k_cache.shape[2] > window_size:
                k_cache = k_cache[:, :, -window_size:, :]
                v_cache = v_cache[:, :, -window_size:, :]
        
        # Attention: q_t attends to all cached k/v
        scores = torch.matmul(q_t, k_cache.transpose(-2, -1)) * self.scale  # (B, H, 1, K)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v_cache)  # (B, H, 1, d_head)
        out = out.transpose(1, 2).contiguous().view(B, 1, C)  # (B, 1, d_model)
        out = self.out_proj(out)
        
        return out, (k_cache, v_cache)


class BlockCausalTransformer(nn.Module):
    """Block-causal sliding-window attention for autoregressive decoding.
    
    Each token attends only to the previous K tokens (sliding window), enabling:
    - Training memory: O(T·K) instead of O(T²)
    - Fast inference: O(T) with KV cache support
    - Consistent train/inference behavior
    
    This is the correct architecture for autoregressive generation at long sequence
    lengths (1k-4k tokens), unlike Swin which is incompatible with AR decoding.
    
    Args:
        d_model: Feature dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        window_size: Sliding window size K (each token attends to previous K tokens).
        dropout: Dropout rate.
        dim_feedforward: FFN hidden dimension.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        window_size: int = 512,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_layers = num_layers

        self.attn_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_blocks.append(
                BlockCausalAttention(d_model, nhead, dropout)
            )
            self.ffn_blocks.append(
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout),
                )
            )
            self.norm1.append(nn.LayerNorm(d_model))
            self.norm2.append(nn.LayerNorm(d_model))

    def _create_block_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create block-causal sliding-window attention mask.
        
        Each token at position i can attend to tokens in [max(0, i-K), i].
        This creates a lower-triangular band mask with width K.
        
        Args:
            seq_len: Sequence length T.
            device: Device for mask tensor.
        
        Returns:
            mask: (T, T) boolean mask where True = can attend, False = masked out.
        """
        # Create position indices: (T, 1) and (1, T)
        i = torch.arange(seq_len, device=device).unsqueeze(1)  # (T, 1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
        
        # Standard causal: j <= i
        causal_mask = (j <= i)
        
        # Sliding window: j >= max(0, i - window_size)
        window_start = torch.clamp(i - self.window_size, min=0)
        window_mask = (j >= window_start)
        
        # Combine: can attend if both causal AND within window
        mask = causal_mask & window_mask
        
        return mask

    def forward(
        self,
        x: torch.Tensor,
        coord_ids: Optional[torch.Tensor] = None,
        grid_shape: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Training forward: chunked sliding-window attention (O(T·K) memory).
        
        Uses chunked attention to avoid allocating full (T × T) matrices.
        Each token attends to previous K tokens within sliding window.
        
        Args:
            x: (B, T, d_model) token features.
            coord_ids: Unused (kept for API compatibility).
            grid_shape: Unused (kept for API compatibility).
        
        Returns:
            x_out: (B, T, d_model) refined features.
        """
        assert x.dim() == 3, "Expected (B, T, C)"
        
        for layer_idx in range(self.num_layers):
            attn = self.attn_blocks[layer_idx]
            ffn = self.ffn_blocks[layer_idx]
            norm1 = self.norm1[layer_idx]
            norm2 = self.norm2[layer_idx]

            # Pre-norm
            x_ln = norm1(x)
            
            # Chunked block-causal attention: O(T·K) memory, never allocates T×T
            attn_out = attn(x_ln, window_size=self.window_size)
            x = x + attn_out

            # FFN with residual
            x_ffn_ln = norm2(x)
            x = x + ffn(x_ffn_ln)

        return x
    
    def forward_step(
        self,
        x_t: torch.Tensor,
        past_kv_list: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Inference forward: incremental decoding with KV cache.
        
        Args:
            x_t: (B, 1, d_model) current token
            past_kv_list: Optional list of (past_k, past_v) tuples, one per layer
        
        Returns:
            out_t: (B, 1, d_model) output for current token
            new_kv_list: List of (new_k, new_v) tuples, one per layer
        """
        if past_kv_list is None:
            past_kv_list = [None] * self.num_layers
        
        new_kv_list = []
        x = x_t
        
        for layer_idx in range(self.num_layers):
            attn = self.attn_blocks[layer_idx]
            ffn = self.ffn_blocks[layer_idx]
            norm1 = self.norm1[layer_idx]
            norm2 = self.norm2[layer_idx]
            
            # Pre-norm
            x_ln = norm1(x)
            
            # Incremental attention with KV cache
            attn_out, new_kv = attn.forward_step(
                x_ln,
                past_kv=past_kv_list[layer_idx],
                window_size=self.window_size,
            )
            x = x + attn_out
            new_kv_list.append(new_kv)
            
            # FFN with residual
            x_ffn_ln = norm2(x)
            x = x + ffn(x_ffn_ln)
        
        return x, new_kv_list


@MIDDLE_ENCODERS.register_module()
class SparsePatternAdaptationFormer(nn.Module):
    """SparsePatternAdaptationFormer: ShapeFormer-inspired transformer for pattern adaptation.
    
    Architecture:
    1. Vector Quantization: Quantize sparse features to codebook indices
    2. Coordinate Transformer: Autoregressively predict next coordinate
    3. Value Transformer: Predict codebook index for that coordinate
    
    Training: [S_P, END, S_C, END] with teacher forcing
    Inference: Autoregressive generation from S_P only
    
    Args:
        d_model: Model dimension (default: 512)
        nhead: Number of attention heads
        num_coord_layers: Number of layers in Coordinate Transformer
        num_value_layers: Number of layers in Value Transformer
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
        activation: Activation function ('relu' or 'gelu')
        codebook_size: Vocabulary size for VQ codebook (default: 4096)
        codebook_dim: Feature dimension for codebook (should match sparse features, default: 128)
        commitment_cost: Weight for VQ commitment loss
        spatial_shape: [D, H, W] spatial shape of sparse grid (e.g., [2, 180, 180])
        max_seq_length: Maximum sequence length for generation (default: 10000)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_coord_layers: int = 6,
        num_value_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        codebook_size: int = 4096,
        codebook_dim: int = 128,
        commitment_cost: float = 0.25,
        grid_shape: List[int] = [64, 64, 64],
        max_seq_length: int = 10000,
        coord_window_size: int = 1536,  # Coord transformer window (larger for global structure + END visibility)
        value_window_size: int = 512,  # Value transformer window (smaller, local context sufficient)
        run_inference_during_training: bool = False,  # Whether to run AR inference during training (slow!)
        inference_freq: int = 10,  # Run inference every N training calls (only if run_inference_during_training=True)
    ):
        super().__init__()
        
        self.d_model = d_model
        self.run_inference_during_training = run_inference_during_training
        self.inference_freq = inference_freq
        self._training_call_count = 0  # Counter for inference frequency
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.grid_shape = grid_shape
        self.max_seq_length = max_seq_length
        # Max tokens for each stream (pseudo / gt) before concatenation,
        # following ShapeFormer-style block_size//2 logic.
        # Reserve 1 slot in each stream for its END token.
        self.max_stream_len = max(1, self.max_seq_length // 2 - 1)
        
        D, H, W = grid_shape
        self.max_coord_id = D * H * W  # Maximum coordinate ID (for coordinate vocabulary)
        
        # Vector Quantization codebook
        self.vq = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=codebook_dim,
            commitment_cost=commitment_cost,
        )
        
        # Token embeddings
        # Coordinate embedding: maps coordinate IDs to d_model
        self.coord_embed = nn.Embedding(self.max_coord_id + 2, d_model)  # +2 for END and PAD tokens
        # Value embedding: maps codebook indices to d_model
        self.value_embed = nn.Embedding(codebook_size + 2, d_model)  # +2 for END and PAD tokens
        
        # Positional encoding (learned)
        self.pos_embed = nn.Embedding(max_seq_length, d_model)
        
        # Input projection: combine coord and value embeddings
        self.input_proj = nn.Linear(d_model * 2, d_model)  # coord + value -> d_model
        
        # Coordinate Transformer: predicts next coordinate (block-causal sliding-window attention)
        # coord_window_size: larger window for global structure and END token visibility
        self.coord_transformer = BlockCausalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_coord_layers,
            window_size=coord_window_size,  # Larger: 1536 for sequences up to 4096
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        # Value Transformer: predicts codebook index given coordinate (block-causal sliding-window attention)
        # value_window_size: smaller window sufficient for local appearance context
        self.value_transformer = BlockCausalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_value_layers,
            window_size=value_window_size,  # Smaller: 512 for local context
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        # Output heads
        # Coordinate head: predicts next coordinate ID
        self.coord_head = nn.Linear(d_model, self.max_coord_id + 2)  # +2 for END, PAD
        
        # Value head: predicts codebook index
        self.value_head = nn.Linear(d_model, codebook_size + 2)  # +2 for END, PAD
        
        # Special tokens
        self.END_COORD = self.max_coord_id
        self.END_VALUE = codebook_size
        self.PAD_COORD = self.max_coord_id + 1
        self.PAD_VALUE = codebook_size + 1
    
    def _embed_tokens(
        self,
        coord_ids: torch.Tensor,
        value_ids: torch.Tensor,
        pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed coordinate and value tokens.
        
        Args:
            coord_ids: (T,) coordinate IDs
            value_ids: (T,) codebook value IDs
            pos_ids: (T,) position IDs
        
        Returns:
            embedded: (T, d_model) embedded tokens
        """
        coord_emb = self.coord_embed(coord_ids)  # (T, d_model)
        value_emb = self.value_embed(value_ids)  # (T, d_model)
        pos_emb = self.pos_embed(pos_ids)  # (T, d_model)
        
        # Combine coord and value embeddings
        combined = torch.cat([coord_emb, value_emb], dim=-1)  # (T, 2*d_model)
        token_emb = self.input_proj(combined)  # (T, d_model)
        
        # Add positional encoding
        embedded = token_emb + pos_emb  # (T, d_model)
        
        return embedded
    def forward(
        self,
        pseudo_sparse_features: torch.Tensor,
        pseudo_sparse_indices: torch.Tensor,
        spatial_shape: List[int],
        gt_sparse_features: Optional[torch.Tensor] = None,
        gt_sparse_indices: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Parent forward: dispatch to train/test paths based on return_loss."""
        
        
        assert self.grid_shape == spatial_shape, "Grid shape mismatch"
        
        if return_loss:
            return self.forward_train(
                pseudo_sparse_features=pseudo_sparse_features,
                pseudo_sparse_indices=pseudo_sparse_indices,
                spatial_shape=spatial_shape,
                gt_sparse_features=gt_sparse_features,
                gt_sparse_indices=gt_sparse_indices,
            )
        return self.forward_test(
            pseudo_sparse_features=pseudo_sparse_features,
            pseudo_sparse_indices=pseudo_sparse_indices,
            spatial_shape=spatial_shape,
        )

    def forward_train(
        self,
        pseudo_sparse_features: torch.Tensor,
        pseudo_sparse_indices: torch.Tensor,
        spatial_shape: List[int],
        gt_sparse_features: Optional[torch.Tensor] = None,
        gt_sparse_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Training forward: shared pseudo ops + per-batch train."""
        device = pseudo_sparse_features.device
        # Always use model-configured spatial shape to keep coord ids in range
        spatial_shape_model = self.grid_shape

        batch_indices = pseudo_sparse_indices[:, 0].long()
        unique_batches = torch.unique(batch_indices).tolist()

        refined_features_list: List[torch.Tensor] = []
        refined_indices_list: List[torch.Tensor] = []
        losses_list: List[Dict[str, torch.Tensor]] = []

        for b_idx in unique_batches:
            batch_mask = batch_indices == b_idx
            pseudo_feat_b = pseudo_sparse_features[batch_mask]
            pseudo_idx_b = pseudo_sparse_indices[batch_mask]

            if pseudo_feat_b.numel() == 0:
                refined_features_list.append(torch.empty((0, self.codebook_dim), device=device))
                refined_indices_list.append(torch.empty((0, 4), device=device, dtype=torch.long))
                continue

            # Shared pseudo preprocessing with clamped coord ids
            pseudo_feat_sorted, pseudo_idx_sorted, pseudo_coords = sort_by_row_major(
                pseudo_feat_b, pseudo_idx_b, spatial_shape_model
            )
            _, vq_loss_pseudo, pseudo_value_ids = self.vq(pseudo_feat_sorted)
            pseudo_coord_ids = pseudo_coords.long().clamp(0, self.END_COORD)
        
            # Train branch per batch
            gt_batch_mask = gt_sparse_indices[:, 0] == b_idx
            gt_feat_b = gt_sparse_features[gt_batch_mask]
            gt_idx_b = gt_sparse_indices[gt_batch_mask]

            if gt_feat_b.numel() == 0:
                refined_features_list.append(torch.empty((0, self.codebook_dim), device=device))
                refined_indices_list.append(torch.empty((0, 4), device=device, dtype=torch.long))
                continue

            gt_feat_sorted, gt_idx_sorted, gt_coords = sort_by_row_major(
                gt_feat_b, gt_idx_b, spatial_shape_model
            )
            _, vq_loss_gt, gt_value_ids = self.vq(gt_feat_sorted)
            gt_coord_ids = gt_coords.long().clamp(0, self.END_COORD)

            # === Symmetric max-length handling for pseudo and GT (ShapeFormer-style) ===
            # Each stream is capped to self.max_stream_len tokens before adding END.
            if len(pseudo_coord_ids) > self.max_stream_len:
                # Uniform subsample to max_stream_len
                keep_idx = torch.linspace(
                    0,
                    len(pseudo_coord_ids) - 1,
                    steps=self.max_stream_len,
                    device=device,
                ).long()
                pseudo_coord_ids = pseudo_coord_ids[keep_idx]
                pseudo_value_ids = pseudo_value_ids[keep_idx]

            if len(gt_coord_ids) > self.max_stream_len:
                keep_idx = torch.linspace(
                    0,
                    len(gt_coord_ids) - 1,
                    steps=self.max_stream_len,
                    device=device,
                ).long()
                gt_coord_ids = gt_coord_ids[keep_idx]
                gt_value_ids = gt_value_ids[keep_idx]

            seq_coords = torch.cat([
                pseudo_coord_ids,
                torch.tensor([self.END_COORD], device=device),
                gt_coord_ids,
                torch.tensor([self.END_COORD], device=device),
            ])
            seq_values = torch.cat([
                pseudo_value_ids,
                torch.tensor([self.END_VALUE], device=device),
                gt_value_ids,
                torch.tensor([self.END_VALUE], device=device),
            ])

            seq_pos_ids = torch.arange(len(seq_coords), device=device).clamp(max=self.max_seq_length - 1)

            seq_embedded = self._embed_tokens(seq_coords, seq_values, seq_pos_ids).unsqueeze(0)
            coord_hidden = self.coord_transformer(seq_embedded)  # Block-causal attention
            coord_logits = self.coord_head(coord_hidden)

            coord_emb_for_value = self.coord_embed(seq_coords).unsqueeze(0)
            value_input = coord_hidden + coord_emb_for_value
            value_hidden = self.value_transformer(value_input)  # Block-causal attention
            value_logits = self.value_head(value_hidden)

            pseudo_len = len(pseudo_coord_ids) + 1  # include END
            gt_start = pseudo_len
            gt_end = pseudo_len + len(gt_coord_ids)

            target_coords = seq_coords[gt_start:gt_end + 1]
            coord_loss = F.cross_entropy(
                coord_logits[0, gt_start - 1:gt_end].contiguous().view(-1, coord_logits.shape[-1]),
                target_coords.contiguous().view(-1),
            )

            target_values = seq_values[gt_start:gt_end + 1]
            value_loss = F.cross_entropy(
                value_logits[0, gt_start - 1:gt_end].contiguous().view(-1, value_logits.shape[-1]),
                target_values.contiguous().view(-1),
            )

            losses_list.append({
                'loss_coord': coord_loss,
                'loss_value': value_loss,
                'loss_vq_pseudo': vq_loss_pseudo,
                'loss_vq_gt': vq_loss_gt,
            })

            # === Optional AR inference for monitoring (Fix 3: Training vs inference separation) ===
            # AR inference is slow (20-80s per sample), so we only run it occasionally for monitoring
            # Training uses teacher forcing, so AR inference is NOT needed for gradients
            # 
            # Default: run_inference_during_training=False (disabled for speed)
            # If enabled: run every inference_freq iterations for monitoring
            
            should_run_inference = (
                self.run_inference_during_training and 
                (self._training_call_count % self.inference_freq == 0)
            )
            
            if should_run_inference:
                with torch.no_grad():
                    # Measure the time of the forward_test
                    start_time = time.time()
                    
                    refined_feat_b, refined_idx_b, _ = self.forward_test(
                        pseudo_sparse_features=pseudo_feat_b,
                        pseudo_sparse_indices=pseudo_idx_b,
                        spatial_shape=spatial_shape_model,
                    )
                    
                    end_inference_time = time.time()
                    
                    # === Monitoring metrics (no gradients) ===
                    # Compare refined features with GT features for monitoring
                    # Target: refined should match GT count and features
                    gt_count = gt_feat_sorted.shape[0]
                    refined_count = refined_feat_b.shape[0]
                    pseudo_count = pseudo_feat_sorted.shape[0]  # Keep for reference
                    
                    # Count difference: refined vs GT (normalized)
                    count_diff = abs(gt_count - refined_count) / max(gt_count, 1)
                    
                    # Generation length ratio (refined / GT) - target is 1.0
                    gen_ratio = refined_count / max(gt_count, 1)
                    
                    # Feature statistics comparison: refined vs GT (if both non-empty)
                    if gt_count > 0 and refined_count > 0:
                        # Mean feature distance (simple L2 between mean features)
                        gt_mean = gt_feat_sorted.mean(dim=0)  # (C,)
                        refined_mean = refined_feat_b.mean(dim=0)  # (C,)
                        feature_dist = torch.norm(gt_mean - refined_mean).item()
                        
                        # Chamfer-like distance (sampled, to avoid OOM)
                        # Sample a subset for comparison
                        sample_size = min(100, gt_count, refined_count)
                        gt_sample_idx = torch.randperm(gt_count, device=device)[:sample_size]
                        refined_sample_idx = torch.randperm(refined_count, device=device)[:sample_size]
                        gt_sample = gt_feat_sorted[gt_sample_idx]  # (sample_size, C)
                        refined_sample = refined_feat_b[refined_sample_idx]  # (sample_size, C)
                        
                        # Compute pairwise distances (chunked to avoid OOM)
                        chunk_size = 50
                        min_dists = []
                        for i in range(0, sample_size, chunk_size):
                            chunk_end = min(i + chunk_size, sample_size)
                            refined_chunk = refined_sample[i:chunk_end]  # (chunk_len, C)
                            # Distance from refined_chunk to gt_sample
                            dists = torch.cdist(refined_chunk.unsqueeze(0), gt_sample.unsqueeze(0))  # (1, chunk_len, sample_size)
                            min_dists.append(dists.min(dim=-1)[0].mean().item())
                        chamfer_like_dist = sum(min_dists) / len(min_dists) if min_dists else 0.0
                    else:
                        feature_dist = 0.0
                        chamfer_like_dist = 0.0
                        
                    end_time = time.time()
                    
                    # Print monitoring metrics as table
                    infer_time = end_inference_time - start_time
                    stats_time = end_time - end_inference_time
                    total_time = end_time - start_time
                    

                    print(f"{'batch':<6} | {'infer_time':<10} | {'stats_time':<10} | {'total_time':<10} | "
                            f"{'refined':<8} | {'gt':<6} | {'pseudo':<8} | "
                            f"{'count_diff':<10} | {'gen_ratio':<10} | {'feat_dist':<10} | {'chamfer':<10}")
                    print("-" * 120)
                    
                    # Print data row
                    print(f"{b_idx:<6} | {infer_time:<10.4f} | {stats_time:<10.4f} | {total_time:<10.4f} | "
                          f"{refined_count:<8} | {gt_count:<6} | {pseudo_count:<8} | "
                          f"{count_diff:<10.4f} | {gen_ratio:<10.4f} | {feature_dist:<10.4f} | {chamfer_like_dist:<10.4f}")

                    refined_features_list.append(refined_feat_b.detach())
                    refined_indices_list.append(refined_idx_b.detach())
            else:
                # Skip AR inference: return empty tensors (not used for training anyway)
                # Training uses teacher forcing, so refined output is not needed
                refined_features_list.append(torch.empty((0, self.codebook_dim), device=device))
                refined_indices_list.append(torch.empty((0, 4), device=device, dtype=torch.long))
        
        # Increment training call counter
        self._training_call_count += 1

        refined_features = torch.cat(refined_features_list, dim=0) if refined_features_list else torch.empty((0, self.codebook_dim), device=device)
        refined_indices = torch.cat(refined_indices_list, dim=0) if refined_indices_list else torch.empty((0, 4), device=device, dtype=torch.long)

        losses = None
        if losses_list:
            losses = {
                'loss_coord': sum(l['loss_coord'] for l in losses_list) / len(losses_list),
                'loss_value': sum(l['loss_value'] for l in losses_list) / len(losses_list),
                'loss_vq_pseudo': sum(l['loss_vq_pseudo'] for l in losses_list) / len(losses_list),
                'loss_vq_gt': sum(l['loss_vq_gt'] for l in losses_list) / len(losses_list),
            }

        return refined_features, refined_indices, losses

    def forward_test(
        self,
        pseudo_sparse_features: torch.Tensor,
        pseudo_sparse_indices: torch.Tensor,
        spatial_shape: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Inference forward: shared pseudo ops + per-batch test."""
        device = pseudo_sparse_features.device
        spatial_shape_model = self.grid_shape

        batch_indices = pseudo_sparse_indices[:, 0].long()
        unique_batches = torch.unique(batch_indices).tolist()

        refined_features_list: List[torch.Tensor] = []
        refined_indices_list: List[torch.Tensor] = []

        for b_idx in unique_batches:
            batch_mask = batch_indices == b_idx
            pseudo_feat_b = pseudo_sparse_features[batch_mask]
            pseudo_idx_b = pseudo_sparse_indices[batch_mask]

            if pseudo_feat_b.numel() == 0:
                refined_features_list.append(torch.empty((0, self.codebook_dim), device=device))
                refined_indices_list.append(torch.empty((0, 4), device=device, dtype=torch.long))
                continue

            pseudo_feat_sorted, pseudo_idx_sorted, pseudo_coords = sort_by_row_major(
                pseudo_feat_b, pseudo_idx_b, spatial_shape_model
            )
            _, _, pseudo_value_ids = self.vq(pseudo_feat_sorted)
            pseudo_coord_ids = pseudo_coords.long().clamp(0, self.END_COORD)
        
            # === Initialize KV cache by processing pseudo tokens incrementally ===
            # Process all pseudo tokens incrementally to build KV cache (O(T_pseudo) done once)
            pseudo_seq_coords = torch.cat([
                pseudo_coord_ids,
                torch.tensor([self.END_COORD], device=device),
            ])
            pseudo_seq_values = torch.cat([
                pseudo_value_ids,
                torch.tensor([self.END_VALUE], device=device),
            ])
            
            # Initialize KV caches
            coord_kv_cache = None
            value_kv_cache = None
            last_coord_hidden = None
            
            # Process pseudo tokens incrementally to build KV cache
            for t in range(len(pseudo_seq_coords)):
                coord_id_t = pseudo_seq_coords[t].item()
                value_id_t = pseudo_seq_values[t].item()
                pos_id_t = t
                
                # Embed token
                coord_emb_t = self.coord_embed(torch.tensor([coord_id_t], device=device)).unsqueeze(0)  # (1, 1, d_model)
                value_emb_t = self.value_embed(torch.tensor([value_id_t], device=device)).unsqueeze(0)  # (1, 1, d_model)
                pos_emb_t = self.pos_embed(torch.tensor([pos_id_t], device=device).clamp(max=self.max_seq_length - 1)).unsqueeze(0)  # (1, 1, d_model)
                
                combined_t = torch.cat([coord_emb_t, value_emb_t], dim=-1)  # (1, 1, 2*d_model)
                token_emb_t = self.input_proj(combined_t)  # (1, 1, d_model)
                token_t = token_emb_t + pos_emb_t  # (1, 1, d_model)
                
                # Update coord KV cache
                coord_hidden_t, coord_kv_cache = self.coord_transformer.forward_step(token_t, coord_kv_cache)
                last_coord_hidden = coord_hidden_t
                
                # Update value KV cache
                coord_emb_for_value = self.coord_embed(torch.tensor([coord_id_t], device=device)).unsqueeze(0)  # (1, 1, d_model)
                value_input_t = coord_hidden_t + coord_emb_for_value  # (1, 1, d_model)
                _, value_kv_cache = self.value_transformer.forward_step(value_input_t, value_kv_cache)
            
            # === Autoregressive generation with KV cache (O(T) total) ===
            # CORRECT AR CAUSALITY:
            # coord_t → coord_transformer → coord_hidden_t → coord_{t+1}
            # coord_hidden_t → value_transformer → value_t
            # (coord_{t+1}, value_t) is used for NEXT timestep input
            # 
            # CRITICAL: Never feed value_t back into coord_transformer at timestep t
            # This breaks causality and causes generation length explosion
            
            generated_indices = []
            generated_values = []
            # Adaptive max length: use GT count as reference (if available) or pseudo * 1.5
            max_gen_length = min(self.max_seq_length, int(len(pseudo_coord_ids) * 1.5))
            
            # Start generation from position after pseudo sequence
            current_pos = len(pseudo_seq_coords)
            # Use last coord hidden state to predict first new coord
            coord_logits = self.coord_head(last_coord_hidden)  # (1, 1, vocab_size)
            coord_probs = F.softmax(coord_logits, dim=-1)
            next_coord = torch.multinomial(coord_probs.view(-1), 1)[0].item()  # Stochastic sampling

            for step in range(max_gen_length):
                if next_coord == self.END_COORD:
                    break
                
                # Step 1: Embed ONLY coord (no value yet) and process through coord transformer
                # This maintains strict causality: coord_t → coord_hidden_t → coord_{t+1}
                coord_emb_t = self.coord_embed(torch.tensor([next_coord], device=device)).unsqueeze(0)  # (1, 1, d_model)
                # Use END_VALUE as placeholder for value embedding (needed for input_proj)
                value_emb_t = self.value_embed(torch.tensor([self.END_VALUE], device=device)).unsqueeze(0)  # (1, 1, d_model)
                pos_emb_t = self.pos_embed(torch.tensor([current_pos], device=device).clamp(max=self.max_seq_length - 1)).unsqueeze(0)  # (1, 1, d_model)
                
                combined_t = torch.cat([coord_emb_t, value_emb_t], dim=-1)  # (1, 1, 2*d_model)
                token_emb_t = self.input_proj(combined_t)  # (1, 1, d_model)
                token_t = token_emb_t + pos_emb_t  # (1, 1, d_model)
                
                # Update coord KV cache: coord_t enters coord transformer
                coord_hidden_t, coord_kv_cache = self.coord_transformer.forward_step(token_t, coord_kv_cache)
                
                # Step 2: Predict value from coord_hidden_t (value is conditioned on coord structure)
                next_coord_emb = self.coord_embed(torch.tensor([next_coord], device=device)).unsqueeze(0)  # (1, 1, d_model)
                value_input_t = coord_hidden_t + next_coord_emb  # (1, 1, d_model)
                
                # Update value KV cache
                value_hidden_t, value_kv_cache = self.value_transformer.forward_step(value_input_t, value_kv_cache)
                value_logits = self.value_head(value_hidden_t)  # (1, 1, vocab_size)
                value_probs = F.softmax(value_logits, dim=-1)
                next_value = torch.multinomial(value_probs.view(-1), 1)[0].item()
                
                if next_value == self.END_VALUE:
                    break
                
                # Store generated output
                coord_3d = self._coord_id_to_3d(next_coord, spatial_shape_model).to(device)
                coord_4d = torch.cat([
                    torch.tensor([b_idx], device=device).unsqueeze(0),
                    coord_3d.unsqueeze(0)
                ], dim=1)
                generated_indices.append(coord_4d)
                generated_values.append(next_value)
                
                # Step 3: Predict next coord from coord_hidden_t (NOT from coord+value feedback!)
                # This is the key fix: coord_{t+1} depends only on coord_hidden_t, not on value_t
                coord_logits = self.coord_head(coord_hidden_t)  # (1, 1, vocab_size)
                coord_probs = F.softmax(coord_logits, dim=-1)
                next_coord = torch.multinomial(coord_probs.view(-1), 1)[0].item()
                
                # Update for next iteration
                current_pos += 1
                last_coord_hidden = coord_hidden_t  # Use coord_hidden_t, not coord_hidden_t2

            if len(generated_values) > 0:
                generated_value_tensor = torch.tensor(generated_values, device=device)
                generated_features = self.vq.embedding(generated_value_tensor)
                generated_indices_tensor = torch.cat(generated_indices, dim=0)
                refined_features_list.append(generated_features)
                refined_indices_list.append(generated_indices_tensor)
            else:
                refined_features_list.append(torch.empty((0, self.codebook_dim), device=device))
                refined_indices_list.append(torch.empty((0, 4), device=device, dtype=torch.long))

        refined_features = torch.cat(refined_features_list, dim=0) if refined_features_list else torch.empty((0, self.codebook_dim), device=device)
        refined_indices = torch.cat(refined_indices_list, dim=0) if refined_indices_list else torch.empty((0, 4), device=device, dtype=torch.long)

        return refined_features, refined_indices, None

    
    def _coord_id_to_3d(self, coord_id: int, spatial_shape: List[int]) -> torch.Tensor:
        """Convert flattened coordinate ID back to 3D (z, y, x).
        
        Args:
            coord_id: Flattened coordinate ID
            spatial_shape: [D, H, W]
        
        Returns:
            coord_3d: (3,) tensor [z, y, x]
        """
        D, H, W = spatial_shape
        z = coord_id // (H * W)
        remainder = coord_id % (H * W)
        y = remainder // W
        x = remainder % W
        return torch.tensor([z, y, x], dtype=torch.long)
