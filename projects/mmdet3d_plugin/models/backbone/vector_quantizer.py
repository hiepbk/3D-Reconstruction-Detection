"""
Vector Quantization (VQ) Module for Sparse Pattern Adaptation.

Learns a codebook of prototype feature vectors and quantizes continuous features
to discrete codebook indices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class VectorQuantizer(nn.Module):
    """Vector Quantization layer for sparse features.
    
    Learns a codebook of prototype vectors and quantizes continuous features
    to discrete indices. Uses straight-through estimator for gradients.
    
    Args:
        num_embeddings: Vocabulary size (number of codebook entries), e.g., 4096
        embedding_dim: Feature dimension, e.g., 128
        commitment_cost: Weight for commitment loss (default: 0.25)
    """
    
    def __init__(
        self,
        num_embeddings: int = 4096,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: learnable prototype vectors [V, D]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize uniformly
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input features.
        
        Args:
            inputs: (N, D) continuous features
        
        Returns:
            quantized: (N, D) quantized features (prototypes)
            vq_loss: Scalar VQ loss (codebook + commitment)
            encoding_indices: (N,) discrete codebook indices
        """
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)  # (N, D)
        
        # Compute distances to all codebook entries
        # distances[i, j] = ||flat_input[i] - embedding[j]||^2
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)  # (N, 1)
            + torch.sum(self.embedding.weight ** 2, dim=1)  # (V,)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())  # (N, V)
        )
        
        # Find nearest codebook entry (argmin)
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)
        
        # Quantize: replace features with nearest prototypes
        quantized = self.embedding(encoding_indices)  # (N, D)
        
        # Straight-through estimator: use quantized in forward, gradients flow to inputs
        quantized = inputs + (quantized - inputs).detach()
        
        # VQ losses
        # Codebook loss: pull prototypes toward assigned features
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        
        # Commitment loss: push encoder features toward assigned prototypes
        q_latent_loss = F.mse_loss(inputs, quantized.detach())
        
        vq_loss = e_latent_loss + self.commitment_cost * q_latent_loss
        
        # Preserve original shape
        quantized = quantized.view_as(inputs)
        encoding_indices = encoding_indices.view(inputs.shape[:-1])
        
        return quantized, vq_loss, encoding_indices

