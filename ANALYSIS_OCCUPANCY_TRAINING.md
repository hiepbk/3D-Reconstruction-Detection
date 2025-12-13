# Analysis: Occupancy Prediction Training Issues

## Current Training Behavior

From the training logs:
- **Initial loss**: ~1.35 (iteration 1)
- **Plateau loss**: ~1.19-1.20 (after ~300 iterations)
- **Gradient norm**: Very small (0.0026-0.1925), mostly < 0.1
- **Loss reduction**: Only ~11% reduction, then stagnation

## Root Cause Analysis

### 1. **Network Architecture Issues** ⚠️ **PRIMARY CONCERN**

#### Problem: Shallow U-Net with Limited Capacity
- **Current architecture**: Only 2 encoder/decoder levels (128→256→512, then back)
- **Issue**: Too shallow to capture complex spatial patterns in occupancy maps
- **Compression bottleneck**: Final compression 128→64→32 might lose important features

#### Specific Issues:
```python
# Current: Very shallow encoder
encoder1: 128 → 128 → 256  # Only 2 conv layers
encoder2: 256 → 512        # Only 2 conv layers
decoder2: 768 → 256        # Processes concatenated features
decoder1: 256 → 128        # Final decoder
occupancy_head: 128 → 64 → 32  # Aggressive compression
```

**Recommendations:**
1. **Deeper U-Net**: Add more encoder/decoder levels (3-4 levels instead of 2)
2. **More channels**: Increase channel capacity (e.g., [256, 512, 1024, 2048])
3. **Residual connections**: Add residual blocks in encoder/decoder
4. **Attention mechanisms**: Add spatial attention to focus on occupied regions

### 2. **Training Schedule Issues** ⚠️ **SECONDARY CONCERN**

#### Problem: Learning Rate Too Low / Gradient Clipping Too Aggressive
- **Current LR**: 0.0001 (1e-4) with AdamW
- **Gradient clipping**: max_norm=0.1 (very aggressive)
- **Gradient norms**: 0.0026-0.1925 (mostly < 0.1, suggesting gradients are being clipped)

**Evidence:**
- Very small gradient norms suggest either:
  1. Learning rate is too low (model learns very slowly)
  2. Gradient clipping is too aggressive (cutting off useful gradients)
  3. Model is stuck in a local minimum

**Recommendations:**
1. **Increase learning rate**: Try 0.001 (1e-3) or 0.0005 (5e-4) for the refinement network
2. **Relax gradient clipping**: Increase max_norm to 1.0 or 10.0
3. **Warmup**: Add learning rate warmup (gradually increase LR in first few epochs)
4. **Separate LR for refinement**: Use higher LR for refinement network than DA3

### 3. **Loss Function Issues** ⚠️ **MODERATE CONCERN**

#### Problem: Class Imbalance Not Fully Addressed
- **Current**: BCE + Dice (dice_weight=0.5)
- **Issue**: Occupancy maps have severe class imbalance (~30,000 empty vs ~2,400 occupied per channel)
- **Loss value**: ~1.2 suggests predictions are not very accurate

**Recommendations:**
1. **Use Focal Loss**: Better handles class imbalance
   ```python
   loss_type='focal',
   focal_alpha=0.25,  # Weight for positive class
   focal_gamma=2.0,    # Focusing parameter
   ```
2. **Increase Dice weight**: Try dice_weight=1.0 or 2.0
3. **Channel weighting**: Weight lower height levels more (ground is more important)

### 4. **Initialization Issues** ⚠️ **MODERATE CONCERN**

#### Problem: No Explicit Weight Initialization
- **Current**: Using default PyTorch initialization
- **Issue**: May lead to poor initial gradients or vanishing gradients

**Recommendations:**
1. **Xavier/Kaiming initialization**: For conv layers
2. **Small initial bias**: For final sigmoid layer, initialize bias to predict ~0.1 (expected occupancy rate)
3. **Layer-wise learning rate**: Lower LR for early layers, higher for later layers

### 5. **Data/Preprocessing Issues** ⚠️ **LOW CONCERN**

#### Potential Issues:
- **GT occupancy generation**: SoftVoxelOccupancyVFE might produce soft targets that are hard to learn
- **Voxel size mismatch**: Different voxel sizes for pseudo vs GT might cause misalignment

**Recommendations:**
1. **Verify GT occupancy**: Check if GT occupancy maps are reasonable
2. **Hard vs Soft targets**: Try HardVoxelOccupancyVFE (binary targets) vs SoftVoxelOccupancyVFE
3. **Data augmentation**: Add rotation/flipping augmentation for occupancy maps

## Recommended Fixes (Priority Order)

### Priority 1: Fix Training Schedule
```python
# In config file:
optimizer = dict(
    type='AdamW', 
    lr=0.001,  # Increase from 0.0001 to 0.001
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'refinement': dict(lr_mult=10.0),  # 10x LR for refinement network
        }
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))  # Increase from 0.1 to 1.0
```

### Priority 2: Improve Network Architecture
```python
# Deeper U-Net with more capacity
bev_height_occupancy=dict(
    type='BEVHeightOccupancy',
    in_channels=256,
    Unet_channels=[256, 512, 1024, 2048],  # More channels, deeper
    occ_feature_shape=[180, 180, 32],
    use_residual=True,  # Add residual connections
    use_attention=True,  # Add attention mechanism
)
```

### Priority 3: Switch to Focal Loss
```python
loss_occupancy=dict(
    type='OccupancyLoss',
    loss_type='focal',  # Change from 'bce_dice' to 'focal'
    focal_alpha=0.25,
    focal_gamma=2.0,
    reduction='mean',
    loss_weight=1.0,
)
```

### Priority 4: Add Weight Initialization
```python
# In BEVHeightOccupancy.__init__:
init_cfg=dict(
    type='Kaiming',
    layer='Conv2d',
    mode='fan_out',
    nonlinearity='relu'
)
```

## Expected Improvements

After implementing these fixes:
- **Loss reduction**: Should see 30-50% reduction (from 1.2 to 0.6-0.8)
- **Gradient norms**: Should increase to 0.5-2.0 range
- **Convergence**: Should see steady decrease instead of plateau
- **Training time**: May need more epochs but should converge better

## Diagnostic Steps

1. **Check GT occupancy maps**: Visualize GT occupancy to ensure they're reasonable
2. **Monitor per-channel loss**: Check if certain height levels are harder to learn
3. **Gradient flow**: Check if gradients are flowing through all layers
4. **Prediction quality**: Visualize predicted occupancy maps to see what the model is learning

