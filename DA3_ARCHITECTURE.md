# Depth Anything 3 (DA3) Architecture Diagram

## Overview
Depth Anything 3 is a multi-view depth estimation model that uses a Vision Transformer backbone (DinoV2) and a convolutional decoder head (DualDPT).

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT                                             │
│                    Multi-view Images                                        │
│                    (B, N, 3, H, W)                                         │
│                    + Optional: Extrinsics, Intrinsics                      │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTIONAL: Camera Encoder (CameraEnc)                     │
│                    ┌─────────────────────────────────────┐                  │
│                    │  Input: Extrinsics (B,N,4,4)       │                  │
│                    │         Intrinsics (B,N,3,3)        │                  │
│                    │         Image Size (H, W)            │                  │
│                    └──────────────┬──────────────────────┘                  │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                          │
│                    │  Pose Encoding              │                          │
│                    │  (c2w, intrinsics → 9D)      │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                          │
│                    │  MLP Projection             │                          │
│                    │  (9D → dim_out)             │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                          │
│                    │  Transformer Trunk          │                          │
│                    │  (4x Transformer Blocks)    │                          │
│                    │  - Self-Attention            │                          │
│                    │  - MLP                      │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                          │
│                    │  Camera Tokens               │                          │
│                    │  (B, N, dim_out=768)         │                          │
│                    └──────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             │ (cam_token)
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKBONE: DinoV2 (Vision Transformer)                   │
│                    ┌─────────────────────────────────────┐                  │
│                    │  Vision Transformer (ViT-Base)      │                  │
│                    │  - Patch Embedding (14x14 patches)   │                  │
│                    │  - 12 Transformer Blocks             │                  │
│                    │    * Self-Attention                  │                  │
│                    │    * MLP                              │                  │
│                    │    * LayerNorm                        │                  │
│                    │  - Optional: Camera Token Injection  │                  │
│                    └──────────────┬──────────────────────┘                  │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                          │
│                    │  Multi-Scale Feature Extract│                          │
│                    │  Layers: [5, 7, 9, 11]       │                          │
│                    │  Output: 4 feature maps      │                          │
│                    │  Shape: [B*S, N_patch, C]    │                          │
│                    │  C = 1536 (concatenated)      │                          │
│                    └──────────────┬──────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             │ (feats: List of 4 feature maps)
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HEAD: DualDPT (Convolutional Decoder)                    │
│                    ┌─────────────────────────────────────┐                  │
│                    │  Stage 1: Token → Spatial            │                  │
│                    │  - LayerNorm                         │                  │
│                    │  - Reshape: (B*S, N, C) → (B, C, H, W)│                  │
│                    │  - 1x1 Conv Projections (4 scales)   │                  │
│                    │  - Spatial Resize (x4, x2, x1, /2)   │                  │
│                    └──────────────┬──────────────────────┘                  │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                          │
│                    │  Stage 2: Feature Fusion   │                          │
│                    │  ┌─────────────────────┐  │                          │
│                    │  │ Main Branch:          │  │                          │
│                    │  │ - RefineNet4 → 3     │  │                          │
│                    │  │ - RefineNet3 → 2     │  │                          │
│                    │  │ - RefineNet2 → 1     │  │                          │
│                    │  │ - RefineNet1 (final) │  │                          │
│                    │  │                       │  │                          │
│                    │  │ Aux Branch:           │  │                          │
│                    │  │ - Same fusion chain   │  │                          │
│                    │  │ - Multi-level (4)     │  │                          │
│                    │  └─────────────────────┘  │                          │
│                    │  (All Conv2D operations)   │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                          │
│                    │  Stage 3: Output Heads      │                          │
│                    │  ┌─────────────────────┐  │                          │
│                    │  │ Main Head:           │  │                          │
│                    │  │ - Conv3x3 → Conv1x1  │  │                          │
│                    │  │ - Output: depth (1D) │  │                          │
│                    │  │          conf (1D)    │  │                          │
│                    │  │                       │  │                          │
│                    │  │ Aux Head:             │  │                          │
│                    │  │ - 5x Conv3x3 blocks   │  │                          │
│                    │  │ - Conv3x3 → Conv1x1  │  │                          │
│                    │  │ - Output: ray (7D)    │  │                          │
│                    │  │          ray_conf (1D)│  │                          │
│                    │  └─────────────────────┘  │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                          │
│                    │  Output:                      │                          │
│                    │  - depth: (B, S, H, W)        │                          │
│                    │  - depth_conf: (B, S, H, W)  │                          │
│                    │  - ray: (B, S, 7, H, W)       │                          │
│                    │  - ray_conf: (B, S, H, W)    │                          │
│                    └──────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             │ (output with depth, ray, etc.)
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTIONAL: Camera Pose Estimation                        │
│                    ┌─────────────────────────────────────┐                  │
│                    │  Option A: Ray-based Pose            │                  │
│                    │  - Use ray + ray_conf from DualDPT   │                  │
│                    │  - Compute extrinsics & intrinsics   │                  │
│                    │                                      │                  │
│                    │  Option B: Camera Decoder (CameraDec) │                  │
│                    │  ┌──────────────────────────────┐  │                  │
│                    │  │ Input: Last feature map        │  │                  │
│                    │  │ (B*N, C=1536)                  │  │                  │
│                    │  └──────────┬─────────────────────┘  │                  │
│                    │             │                         │                  │
│                    │  ┌──────────▼─────────────────────┐  │                  │
│                    │  │ MLP Backbone                   │  │                  │
│                    │  │ (Linear → ReLU → Linear)        │  │                  │
│                    │  └──────────┬─────────────────────┘  │                  │
│                    │             │                         │                  │
│                    │  ┌──────────▼─────────────────────┐  │                  │
│                    │  │ Output Heads:                   │  │                  │
│                    │  │ - Translation (3D)              │  │                  │
│                    │  │ - Quaternion (4D)               │  │                  │
│                    │  │ - FOV (2D)                       │  │                  │
│                    │  └──────────┬─────────────────────┘  │                  │
│                    │             │                         │                  │
│                    │  ┌──────────▼─────────────────────┐  │                  │
│                    │  │ Convert to Extrinsics &         │  │                  │
│                    │  │ Intrinsics (4x4, 3x3)           │  │                  │
│                    │  └─────────────────────────────────┘  │                  │
│                    └─────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING                                          │
│                    ┌─────────────────────────────────────┐                  │
│                    │  Sky Estimation                      │                  │
│                    │  - Filter sky regions                │                  │
│                    │  - Set sky to max depth              │                  │
│                    │                                      │                  │
│                    │  Optional: Gaussian Splatting       │                  │
│                    │  - GS-DPT head                      │                  │
│                    │  - 3DGS parameters                  │                  │
│                    └─────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                            │
│                    ┌─────────────────────────────────────┐                  │
│                    │  - depth: (B, S, H, W)              │                  │
│                    │  - depth_conf: (B, S, H, W)        │                  │
│                    │  - extrinsics: (B, N, 4, 4)         │                  │
│                    │  - intrinsics: (B, N, 3, 3)         │                  │
│                    │  - sky: (B, S, 1, H, W) [optional] │                  │
│                    │  - gaussians: [optional]            │                  │
│                    └─────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. **DinoV2 Backbone** (Pure Transformer)
- **Type**: Vision Transformer (ViT-Base)
- **Architecture**: 
  - Patch Embedding: 14×14 patches
  - 12 Transformer Blocks (layers 0-11)
  - Each block: Self-Attention + MLP + LayerNorm
  - Output layers: [5, 7, 9, 11] → 4 multi-scale feature maps
  - Feature dimension: 1536 (concatenated from multiple layers)

### 2. **DualDPT Head** (Pure Convolutional)
- **Type**: Convolutional Decoder
- **Architecture**:
  - **Token → Spatial**: Reshape transformer tokens to spatial feature maps
  - **Projection**: 1×1 convolutions to project features
  - **Resize**: ConvTranspose2D (upsample) and Conv2D (downsample) to align scales
  - **Fusion**: Feature pyramid fusion using RefineNet blocks (all Conv2D)
  - **Output**: Two heads (main + aux) with Conv2D layers

### 3. **Camera Encoder** (Optional, Transformer-based)
- **Input**: Extrinsics + Intrinsics
- **Architecture**: 
  - MLP to project pose encoding to tokens
  - 4 Transformer blocks (self-attention)
  - Output: Camera tokens injected into backbone

### 4. **Camera Decoder** (Optional, MLP-based)
- **Input**: Last feature map from backbone
- **Architecture**: 
  - MLP backbone (Linear → ReLU → Linear)
  - Three output heads: Translation, Quaternion, FOV
  - Converts to extrinsics and intrinsics

## Key Characteristics

1. **Backbone**: Pure Transformer (DinoV2) - uses self-attention
2. **Head**: Pure Convolutional (DualDPT) - uses Conv2D operations only
3. **Hybrid Architecture**: Transformer encoder + Convolutional decoder
4. **Multi-scale Features**: Extracts features from 4 different transformer layers
5. **Dual Output**: Main depth head + Auxiliary ray head
6. **Optional Components**: Camera pose estimation, Gaussian Splatting, Sky detection

## Data Flow Summary

```
Images → [Optional: CameraEnc] → DinoV2 Backbone → DualDPT Head → Depth + Ray
                                                                    ↓
                                                          [Optional: CameraDec] → Poses
                                                                    ↓
                                                          [Optional: Sky/GS] → Final Output
```

