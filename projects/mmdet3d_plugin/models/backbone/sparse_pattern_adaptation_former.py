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


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only transformer block (GPT-style) with causal self-attention.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
        activation: Activation function
    """
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Build decoder layers (causal self-attention only)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with causal self-attention.
        
        Args:
            x: (B, T, d_model) input sequence
            attn_mask: (T, T) causal mask (optional, will be created if None)
        
        Returns:
            output: (B, T, d_model) output sequence
        """
        B, T, _ = x.shape
        
        # Create causal mask if not provided
        if attn_mask is None:
            # Causal mask: upper triangular (True = masked, cannot attend)
            # Lower triangular (False = can attend to past and present)
            attn_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        
        # Apply decoder layers (self-attention only, no cross-attention)
        # Use memory=x for self-attention (query, key, value all from x)
        for layer in self.layers:
            x = layer(x, x, tgt_mask=attn_mask)  # self-attention: query=x, key=x, value=x
        
        return x


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
        spatial_shape: List[int] = [2, 180, 180],
        max_seq_length: int = 10000,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.spatial_shape = spatial_shape
        self.max_seq_length = max_seq_length
        
        D, H, W = spatial_shape
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
        
        # Coordinate Transformer: predicts next coordinate
        self.coord_transformer = DecoderOnlyTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_coord_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        
        # Value Transformer: predicts codebook index given coordinate
        self.value_transformer = DecoderOnlyTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_value_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
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
        spatial_shape_model = self.spatial_shape

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
        
            # Build sequence with max length handling: keep full pseudo, truncate GT tail if needed
            pseudo_len = len(pseudo_coord_ids) + 1  # include END
            # Reserve 1 slot for final END
            available_gt_slots = max(0, self.max_seq_length - pseudo_len - 1)
            if available_gt_slots < len(gt_coord_ids):
                gt_coord_ids = gt_coord_ids[:available_gt_slots]
                gt_value_ids = gt_value_ids[:available_gt_slots]

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
            coord_hidden = self.coord_transformer(seq_embedded)
            coord_logits = self.coord_head(coord_hidden)

            coord_emb_for_value = self.coord_embed(seq_coords).unsqueeze(0)
            value_input = coord_hidden + coord_emb_for_value
            value_hidden = self.value_transformer(value_input)
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

            refined_features_list.append(gt_feat_sorted)
            refined_indices_list.append(gt_idx_sorted)

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
        spatial_shape_model = self.spatial_shape

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
        
            # Test branch per batch
            seq_coords = torch.cat([
                pseudo_coord_ids,
                torch.tensor([self.END_COORD], device=device),
            ])
            seq_values = torch.cat([
                pseudo_value_ids,
                torch.tensor([self.END_VALUE], device=device),
            ])

            generated_indices = []
            generated_values = []
            max_gen_length = min(self.max_seq_length, len(pseudo_coord_ids) * 2)

            for _ in range(max_gen_length):
                # Clamp pos ids to embedding range
                seq_pos_ids = torch.arange(len(seq_coords), device=device).clamp(max=self.max_seq_length - 1)
                seq_embedded = self._embed_tokens(seq_coords, seq_values, seq_pos_ids).unsqueeze(0)

                coord_hidden = self.coord_transformer(seq_embedded)
                coord_logits = self.coord_head(coord_hidden[:, -1:, :])
                coord_probs = F.softmax(coord_logits, dim=-1)
                next_coord = torch.multinomial(coord_probs.view(-1), 1)[0].item()
                if next_coord == self.END_COORD:
                    break

                coord_emb = self.coord_embed(torch.tensor([next_coord], device=device)).unsqueeze(0)
                value_input_seq = coord_hidden[:, -1:, :] + coord_emb
                value_hidden = self.value_transformer(value_input_seq)
                value_logits = self.value_head(value_hidden)
                value_probs = F.softmax(value_logits, dim=-1)
                next_value = torch.multinomial(value_probs.view(-1), 1)[0].item()
                if next_value == self.END_VALUE:
                    break

                seq_coords = torch.cat([seq_coords, torch.tensor([next_coord], device=device)])
                seq_values = torch.cat([seq_values, torch.tensor([next_value], device=device)])

                coord_3d = self._coord_id_to_3d(next_coord, spatial_shape_model)
                coord_4d = torch.cat([
                    torch.tensor([b_idx], device=device).unsqueeze(0),
                    coord_3d.unsqueeze(0)
                ], dim=1)
                generated_indices.append(coord_4d)
                generated_values.append(next_value)

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
