# Feature-Space Domain Adaptation: Two-Branch Feature Distillation

## 🎯 Core Goal

**Train two separate sparse convolution backbones:**
- **GT branch (teacher)**: LiDAR ground-truth points
- **Pseudo branch (student)**: Camera-derived pseudo points

Both branches feed into the same 3D detection task, and the pseudo branch is explicitly taught to match GT sparse features.

**Success criterion:** Pseudo branch alone produces nearly identical 3D detection results as the GT branch.

## 🧠 Overall Architecture

```
GT points ──► SparseConv_GT ──► Neck ──► Head ──► 3D detection loss
                    │
                    └────────► Feature distillation loss
                    ▲
Pseudo points ─► SparseConv_Pseudo
```

**Key design choices:**
- ✅ GT and Pseudo sparse conv backbones are **NOT shared** (separate parameters)
- ✅ Detection head + neck **ARE shared**
- ✅ GT branch is the **teacher**
- ✅ Pseudo branch is the **student**

---

## 🧩 Model Components

### 1️⃣ GT Branch (Teacher)

**Input:** `gt_points`

**Pipeline:**
```
gt_points
  → voxelization
  → SparseConv_GT backbone
  → Neck
  → Detection Head
  → Standard 3D detection losses
```

**Characteristics:**
- This branch defines the upper performance bound
- Backbone parameters: **Trainable**
- Later can be EMA or frozen (optional)

### 2️⃣ Pseudo Branch (Student)

**Input:** `pseudo_points` (from camera → DepthAnything3)

**Pipeline:**
```
pseudo_points
  → voxelization
  → SparseConv_Pseudo backbone
  → Neck (shared)
  → Detection Head (shared)
  → Detection loss (optional / weaker)
```

**Characteristics:**
- Learns from feature distillation loss
- Backbone parameters: **Trainable** (learns to match GT features)
- Detection loss is optional and typically disabled early in training

## 🔁 Feature Distillation (Core Idea)

### Where to Match Features

Extract sparse voxel features from the **last SparseConv stage** of both backbones.

Match only **overlapping voxel coordinates**.

**Feature tensors:**
```
gt_feat:     [Ng, C]
gt_indices:  [Ng, 4]

pseudo_feat:    [Np, C]
pseudo_indices: [Np, 4]
```

### Matching Rule

**Intersect voxel coordinates:**
```python
matched_voxels = intersect(gt_indices, pseudo_indices)
```

**Gather matched features:**
```python
gt_f     = gt_feat[idx_gt]
pseudo_f = pseudo_feat[idx_pseudo]
```

## 📉 Feature Alignment Loss (REQUIRED)

**Use cosine similarity loss:**
```python
gt_f = F.normalize(gt_f, dim=1)
pseudo_f = F.normalize(pseudo_f, dim=1)

loss_feat = 1 - (gt_f * pseudo_f).sum(dim=1)
loss_feat = loss_feat.mean()
```

**Why cosine?**
- Prevents feature magnitude collapse
- Enforces semantic alignment
- Stable gradients

## ⚖️ Total Training Loss

```
loss_total = loss_det_gt + λ * loss_feat
```

**Recommended:**
- `λ = 1.0` initially
- Do NOT backprop detection loss from pseudo branch early

## 🧪 Training Procedure

**During training (per iteration):**

1. **Forward GT branch** → detection loss
2. **Forward Pseudo branch** → feature extraction
3. **Compute feature distillation loss**
4. **Backprop:**
   - GT backbone: detection loss only
   - Pseudo backbone: feature loss (+ optional weak detection loss)

## 📊 Evaluation Strategy (CRITICAL)

**After each epoch, run two inference passes:**

### 🔍 Evaluation A — GT backbone
```
gt_points
 → SparseConv_GT
 → Neck
 → Head
 → 3D detection output
```

**Record:**
- mAP / NDS / AP3D

**This is your oracle baseline**

### 🔍 Evaluation B — Pseudo backbone
```
pseudo_points
 → SparseConv_Pseudo
 → Neck (same weights)
 → Head (same weights)
 → 3D detection output
```

**Compare against GT evaluation:**
- AP drop
- Recall difference
- Box consistency

## ✅ Success Criteria

**You succeed only if:**

Pseudo-only inference produces:
- Similar AP / NDS as GT branch
- Similar box geometry and confidence
- No GT points used during pseudo inference

**If pseudo ≈ GT in detection → feature alignment is correct.**

## 🚫 Important Constraints

- ❌ No transformers
- ❌ No shared sparse backbone
- ❌ No GT points at pseudo inference time
- ❌ No shortcut concatenation

## 🚀 Optional Improvements (Later)

- EMA GT backbone instead of raw training weights
- Hard voxel mining (top-k cosine error)
- Weak detection loss on pseudo branch after warm-up
- Freeze GT backbone after N epochs

## 🧠 Final Note

This design:
- Directly optimizes what you care about
- Avoids proxy metrics
- Gives a binary success signal
- Scales to real deployment

**You're no longer guessing whether features are "good" — the detector itself tells you.**

---

## 🏗️ Architecture Design (Legacy - For Reference)

### Option A: Direct Feature Alignment (Previous Approach - Shared Backbone)

```
Training:
┌─────────────────┐         ┌─────────────────┐
│ Camera Images   │         │ LiDAR Points    │
│   (6 views)     │         │   (GT, 40K)     │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│ DepthAnything3  │         │ Direct Input    │
│  → Pseudo PC    │         │                 │
│  (500K points)  │         │                 │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────┐
│     Voxelization (same grid)            │
│  voxel_size=[0.075, 0.075, 0.2]         │
│  pc_range=[-54, -54, -5, 54, 54, 3]     │
└────────┬───────────────────┬─────────────┘
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│ Voxel Encoder   │  │ Voxel Encoder   │
│ (HardSimpleVFE) │  │ (HardSimpleVFE) │
└────────┬────────┘  └────────┬────────┘
         │                     │
         ▼                     ▼
┌─────────────────┐  ┌─────────────────┐
│ SparseEncoder   │  │ SparseEncoder   │
│ (Shared weights)│  │ (Shared weights)│
│  → F_pseudo     │  │  → F_gt         │
└────────┬────────┘  └────────┬────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
         ┌──────────────────────┐
         │  Feature Alignment   │
         │  Loss Computation    │
         └──────────────────────┘

Inference:
Camera → DepthAnything3 → Pseudo PC → Voxelization → SparseEncoder → Detection Head
(No GT, No refinement needed)
```

**Key Points:**
- ✅ Shared SparseEncoder weights (same backbone for both)
- ✅ No decoder needed
- ✅ No AR generation
- ✅ Direct feature matching

### Option B: Feature Refinement Network (More Complex)

Add a small refinement network that takes pseudo features and refines them:

```
F_pseudo → RefinementNet → F_refined
                              ↓
                         Loss vs F_gt
```

**When to use:** If direct alignment doesn't work well, add a learnable refinement step.

---

## 📊 Loss Functions (Priority Order)

### 1. **Occupancy Alignment Loss** (MOST IMPORTANT)

**Purpose:** Match which voxels are non-empty between pseudo and GT.

**Implementation:**
```python
# After sparse conv, get occupancy masks
pseudo_occ_mask = (pseudo_sparse_indices is not None)  # Binary: voxel exists or not
gt_occ_mask = (gt_sparse_indices is not None)

# Convert to dense occupancy maps (B, Z, Y, X)
pseudo_occ_dense = sparse_to_dense_occupancy(pseudo_sparse_indices, spatial_shape)
gt_occ_dense = sparse_to_dense_occupancy(gt_sparse_indices, spatial_shape)

# BCE loss on occupancy
loss_occupancy = F.binary_cross_entropy_with_logits(
    pseudo_occ_dense.float(), 
    gt_occ_dense.float()
)
```

**Why this matters:**
- Fixes over-generation (500K → 40K mismatch)
- Ensures same sparsity pattern
- Most important for downstream detector

### 2. **Feature Alignment Loss** (Core Matching)

**Purpose:** Match feature distributions at shared voxels.

**Option 2a: L2/SmoothL1 (Simple)**
```python
# Find shared voxels (same indices)
shared_indices = find_intersection(pseudo_indices, gt_indices)

# Extract features at shared voxels
pseudo_shared_feat = extract_features_at_indices(pseudo_features, pseudo_indices, shared_indices)
gt_shared_feat = extract_features_at_indices(gt_features, gt_indices, shared_indices)

# L2 loss
loss_feature = F.mse_loss(pseudo_shared_feat, gt_shared_feat)
```

**Option 2b: Cosine Distance (Better for scale-invariant)**
```python
loss_feature = 1 - F.cosine_similarity(
    pseudo_shared_feat, 
    gt_shared_feat, 
    dim=-1
).mean()
```

**Option 2c: Sinkhorn/OT Distance (Best, but expensive)**
```python
# Optimal transport matching
loss_feature = sinkhorn_distance(pseudo_features, gt_features)
```

**Recommendation:** Start with L2, upgrade to cosine if needed.

### 3. **Multi-Scale Feature Loss** (Important for Rich Features)

**Purpose:** Match features at multiple scales (like FPN).

**Implementation:**
```python
# Extract features from multiple encoder layers
pseudo_feat_layers = [layer1, layer2, layer3, layer4]  # Multi-scale
gt_feat_layers = [layer1, layer2, layer3, layer4]

losses = []
for p_feat, g_feat in zip(pseudo_feat_layers, gt_feat_layers):
    loss = compute_feature_loss(p_feat, g_feat)  # L2 or cosine
    losses.append(loss)

loss_multiscale = sum(losses) / len(losses)
```

### 4. **Density Regularization** (Cheap but Effective)

**Purpose:** Penalize large differences in non-empty voxel count.

```python
pseudo_count = pseudo_sparse_indices.shape[0]
gt_count = gt_sparse_indices.shape[0]

loss_count = F.l1_loss(
    torch.tensor(pseudo_count, dtype=torch.float32),
    torch.tensor(gt_count, dtype=torch.float32)
) / max(gt_count, 1)  # Normalize by GT count
```

### 5. **Feature Distribution Loss** (Advanced)

**Purpose:** Match statistical properties (mean, std) of features.

```python
pseudo_mean = pseudo_features.mean(dim=0)
gt_mean = gt_features.mean(dim=0)
loss_mean = F.mse_loss(pseudo_mean, gt_mean)

pseudo_std = pseudo_features.std(dim=0)
gt_std = gt_features.std(dim=0)
loss_std = F.mse_loss(pseudo_std, gt_std)

loss_distribution = loss_mean + loss_std
```

---

## 🔧 Implementation Strategy

### Phase 1: Replace AR Transformer with Feature Matcher

**Current Flow:**
```
SparseRefinement.forward_train()
  → pattern_adaptation (AR transformer)
  → AR generation losses
```

**New Flow:**
```
SparseRefinement.forward_train()
  → Extract sparse features (pseudo + GT)
  → Compute feature alignment losses
  → Return losses (no generation needed)
```

### Phase 2: Add Occupancy Prediction

**New Component:**
```python
class OccupancyPredictor(nn.Module):
    """Predict occupancy mask from sparse features."""
    def forward(self, sparse_features, sparse_indices, spatial_shape):
        # Convert sparse to dense occupancy
        # Return occupancy logits (B, Z, Y, X)
        pass
```

### Phase 3: Multi-Scale Matching

Extract features from multiple SparseEncoder layers and match at each scale.

---

## 🎨 Code Structure

### New Module: `FeatureDomainAdapter`

```python
@BACKBONES.register_module()
class FeatureDomainAdapter(nn.Module):
    """
    Feature-space domain adaptation between pseudo and GT point clouds.
    
    No generation, no AR, just feature matching.
    """
    
    def __init__(
        self,
        # Shared encoder (same for pseudo and GT)
        pts_voxel_layer: Dict,
        pts_voxel_encoder: Dict,
        pts_middle_encoder: Dict,  # SparseEncoder
        
        # Loss configs
        loss_occupancy: Dict,
        loss_feature: Dict,
        loss_count: Dict = None,
        loss_multiscale: bool = True,
        
        loss_weight: float = 1.0,
    ):
        # Build shared components
        self.voxel_layer = Voxelization(**pts_voxel_layer)
        self.voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
        self.middle_encoder = build_middle_encoder(pts_middle_encoder)
        
        # Build losses
        self.loss_occupancy = build_loss(loss_occupancy)
        self.loss_feature = build_loss(loss_feature)
        self.loss_count = build_loss(loss_count) if loss_count else None
        
        self.loss_weight = loss_weight
        self.loss_multiscale = loss_multiscale
    
    def forward_train(
        self,
        pseudo_points: torch.Tensor,  # (B, N, C)
        gt_points: torch.Tensor,       # (B, M, C)
    ) -> Dict[str, torch.Tensor]:
        """Training: compute feature alignment losses."""
        
        # 1. Voxelize both
        pseudo_voxel_feat, _, pseudo_coors = self._voxelize_and_encode(pseudo_points)
        gt_voxel_feat, _, gt_coors = self._voxelize_and_encode(gt_points)
        
        # 2. Sparse conv (shared weights)
        pseudo_sparse_feat, pseudo_sparse_idx, pseudo_spatial = self.middle_encoder(
            pseudo_voxel_feat, pseudo_coors, batch_size
        )
        gt_sparse_feat, gt_sparse_idx, gt_spatial = self.middle_encoder(
            gt_voxel_feat, gt_coors, batch_size
        )
        
        # 3. Compute losses
        losses = {}
        
        # Occupancy loss (most important)
        loss_occ = self._compute_occupancy_loss(
            pseudo_sparse_idx, gt_sparse_idx, pseudo_spatial
        )
        losses['loss_occupancy'] = loss_occ
        
        # Feature alignment loss
        loss_feat = self._compute_feature_loss(
            pseudo_sparse_feat, pseudo_sparse_idx,
            gt_sparse_feat, gt_sparse_idx
        )
        losses['loss_feature'] = loss_feat
        
        # Count regularization
        if self.loss_count:
            loss_cnt = self._compute_count_loss(
                pseudo_sparse_idx, gt_sparse_idx
            )
            losses['loss_count'] = loss_cnt
        
        # Multi-scale (if enabled)
        if self.loss_multiscale:
            # Extract from multiple layers
            # ... (implementation details)
            pass
        
        # Apply weights
        for k in losses:
            losses[k] = losses[k] * self.loss_weight
        
        return losses
    
    def forward_test(
        self,
        pseudo_points: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: just return features, no GT needed."""
        pseudo_voxel_feat, _, pseudo_coors = self._voxelize_and_encode(pseudo_points)
        pseudo_sparse_feat, pseudo_sparse_idx, _ = self.middle_encoder(
            pseudo_voxel_feat, pseudo_coors, batch_size
        )
        return pseudo_sparse_feat, pseudo_sparse_idx
    
    def _compute_occupancy_loss(self, pseudo_idx, gt_idx, spatial_shape):
        """Convert sparse indices to dense occupancy maps and compute BCE."""
        # Implementation: sparse_to_dense + BCE
        pass
    
    def _compute_feature_loss(self, pseudo_feat, pseudo_idx, gt_feat, gt_idx):
        """Match features at shared voxels."""
        # Find intersection of indices
        # Extract features at shared locations
        # Compute L2 or cosine loss
        pass
```

---

## 🔑 Key Design Decisions

### Decision 1: Shared vs Separate Encoders?

**✅ CURRENT APPROACH: Separate Encoders (Required)**
- **GT and Pseudo use independent SparseConv backbones**
- Prevents trivial identity mapping
- Allows pseudo branch to learn domain-specific features
- More parameters but necessary for proper distillation

**❌ Previous Approach: Shared Encoder (Caused Collapse)**
- Same SparseEncoder for both pseudo and GT
- Led to feature loss collapse (trivial solution)
- Model found shortcut: "Just copy backbone statistics → zero loss"
- This is why we switched to separate backbones

**Recommendation:** Always use separate encoders for two-branch distillation.

### Decision 2: How to Handle Index Mismatch?

**Problem:** Pseudo and GT have different non-empty voxels.

**Solution Options:**

1. **Match only shared voxels** (Recommended)
   - Find intersection of indices
   - Only compute loss on shared locations
   - Simple and effective

2. **Nearest neighbor matching**
   - For each pseudo voxel, find nearest GT voxel
   - More expensive but handles misalignment

3. **Optimal transport**
   - Best matching but very expensive
   - Use only if needed

**Recommendation:** Start with shared voxels, upgrade if needed.

### Decision 3: Multi-Scale or Single Scale?

**Single Scale:**
- Match only final SparseEncoder output
- Simpler, faster

**Multi-Scale:**
- Match at multiple encoder layers
- Richer supervision, better alignment

**Recommendation:** Start single-scale, add multi-scale if needed.

### Decision 4: Occupancy Prediction Method?

**Option A: Direct from Sparse Indices**
- Convert sparse indices → dense occupancy map
- Simple, no learnable parameters

**Option B: Learnable Occupancy Head**
- Small network predicts occupancy from features
- More flexible but adds complexity

**Recommendation:** Start with Option A (direct conversion).

---

## 📝 Implementation Checklist

### Step 1: Create FeatureDomainAdapter Module
- [ ] Create new `FeatureDomainAdapter` class
- [ ] Implement `_voxelize_and_encode` helper
- [ ] Implement `_compute_occupancy_loss`
- [ ] Implement `_compute_feature_loss`
- [ ] Implement `_compute_count_loss` (optional)

### Step 2: Replace AR Transformer in SparseRefinement
- [ ] Remove `pattern_adaptation` (AR transformer)
- [ ] Replace with `FeatureDomainAdapter`
- [ ] Update `forward_train` to use new losses
- [ ] Update `forward_test` to return features directly

### Step 3: Update Loss Configs
- [ ] Add `loss_occupancy` config (BCE)
- [ ] Add `loss_feature` config (L2 or cosine)
- [ ] Add `loss_count` config (optional, L1)

### Step 4: Update Config File
- [ ] Replace `sparse_refinement_transformer` with `feature_domain_adapter`
- [ ] Update loss configs
- [ ] Remove AR-related configs

### Step 5: Testing
- [ ] Verify losses compute correctly
- [ ] Check gradient flow
- [ ] Monitor training stability
- [ ] Compare feature distributions

---

## 🚨 Potential Pitfalls & Solutions

### Pitfall 1: Index Mismatch Causes Empty Loss

**Problem:** If no shared voxels, feature loss is 0.

**Solution:** 
- Always compute occupancy loss (works even with no overlap)
- Add count loss to penalize mismatch
- Use nearest neighbor if needed

### Pitfall 2: Feature Scale Mismatch

**Problem:** Pseudo and GT features have different scales.

**Solution:**
- Use cosine distance instead of L2
- Normalize features before loss
- Add feature distribution loss

### Pitfall 3: Memory Issues with Dense Occupancy

**Problem:** Converting sparse → dense for large grids is expensive.

**Solution:**
- Use sparse operations when possible
- Downsample occupancy map if needed
- Process in chunks

### Pitfall 4: Training Instability

**Problem:** Losses might still be unstable.

**Solution:**
- Start with occupancy loss only
- Gradually add feature loss
- Use gradient clipping
- Lower learning rate for refinement module

---

## 🎯 Success Metrics

**Training:**
- ✅ Losses decrease smoothly (no explosions)
- ✅ Occupancy loss → 0 (perfect voxel matching)
- ✅ Feature loss decreases (features align)
- ✅ Count loss decreases (density matches)

**Inference:**
- ✅ Detection accuracy on camera-only input matches LiDAR baseline
- ✅ Feature distributions are similar
- ✅ No over-generation (reasonable point counts)

---

## 🔄 Migration Path

### Phase 1: Proof of Concept (1-2 days)
1. Create minimal `FeatureDomainAdapter`
2. Implement occupancy loss only
3. Test training stability

### Phase 2: Full Implementation (3-5 days)
1. Add feature alignment loss
2. Add count regularization
3. Replace AR transformer in config
4. Full training run

### Phase 3: Optimization (as needed)
1. Add multi-scale matching
2. Experiment with loss combinations
3. Fine-tune hyperparameters

---

## 💡 Additional Ideas

### Idea 1: Progressive Training
- Start with occupancy loss only
- Gradually add feature loss
- Helps with stability

### Idea 2: Feature Normalization
- Normalize features to unit sphere
- Makes cosine distance more meaningful
- Helps with scale mismatch

### Idea 3: Attention-Based Matching
- Use attention to find correspondences
- More flexible than exact index matching
- Can handle small misalignments

### Idea 4: Adversarial Training
- Add discriminator to distinguish pseudo vs GT features
- Forces better alignment
- More complex but potentially better

---

## 📚 References & Inspiration

- **BEVDepth**: Uses feature alignment for depth estimation
- **BEVFormer**: Feature-space matching in BEV
- **SparseOcc**: Occupancy prediction from sparse features
- **Domain Adaptation**: Standard feature alignment techniques

---

## ✅ Implementation Checklist

### Phase 1: Separate Backbones (Current Priority)

- [ ] Create separate SparseConv_GT and SparseConv_Pseudo backbones
- [ ] Ensure both have identical architecture but separate parameters
- [ ] Update model to forward both branches during training
- [ ] Implement feature extraction from last SparseConv stage

### Phase 2: Feature Distillation Loss

- [ ] Extract sparse features from both backbones
- [ ] Match features at overlapping voxel coordinates
- [ ] Implement cosine similarity loss with normalization
- [ ] Add hard voxel mining (top-k hardest voxels)

### Phase 3: Training Loop Updates

- [ ] Forward GT branch → compute detection loss
- [ ] Forward Pseudo branch → extract features
- [ ] Compute feature distillation loss
- [ ] Backprop: GT backbone (detection loss), Pseudo backbone (feature loss)

### Phase 4: Evaluation Strategy

- [ ] Implement GT branch evaluation (oracle baseline)
- [ ] Implement Pseudo branch evaluation (camera-only)
- [ ] Compare detection metrics (mAP, NDS, AP3D)
- [ ] Report AP drop as success metric

### Phase 5: Monitoring & Debugging

- [ ] Monitor `loss_occupancy` and `loss_feature` during training
- [ ] Track `grad_norm` to ensure learning is active
- [ ] Compare GT vs Pseudo detection performance
- [ ] Adjust loss weights if needed

---

## 📚 Key Differences from Previous Approaches

### Previous Approach (Shared Backbone + Transformer)

**Architecture:**
- ❌ Shared SparseConv backbone for both GT and Pseudo
- ❌ AR transformer for sequence generation
- ❌ Complex evaluation hooks

**Problems:**
- ❌ Feature loss collapsed to near-zero (trivial solution)
- ❌ Training instability (exploding losses)
- ❌ Over-generation of points
- ❌ Gradient vanishing

### Current Approach (Separate Backbones + Feature Distillation)

**Architecture:**
- ✅ Separate SparseConv backbones (GT and Pseudo)
- ✅ Direct feature alignment (no generation, no transformers)
- ✅ Shared neck and head for detection

**Benefits:**
- ✅ Prevents trivial solutions (separate backbones)
- ✅ Stable training (no AR instability)
- ✅ Direct optimization (detection performance)
- ✅ Clear success signal (AP drop metric)

**Why This Works:**
1. **No identity mapping**: Separate backbones prevent model from copying statistics
2. **Direct optimization**: Optimizes detection performance, not proxy metrics
3. **Stable training**: No AR instability, no exploding losses
4. **Clear success signal**: Detection performance gap directly measures success

---

**Remember:** The goal is to make pseudo branch produce identical detection results as GT branch. Success is measured by detection performance, not feature similarity alone.

