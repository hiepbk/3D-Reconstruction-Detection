# Ray Map to Camera Pose: Short Formulation

## Ray Map Structure
For each pixel `p`, the ray map stores: **r = (t, d) ∈ R⁶**
- **Channels 0-2**: `t ∈ R³` - Ray origin (camera position)
- **Channels 3-5**: `d ∈ R³` - Ray direction in world frame, where `d = RK⁻¹p`

## Camera Pose Estimation from Ray Map

### Step 1: Extract Ray Directions (Channels 0-2)
```
d_predicted = ray_map[:, :, :3]  # (N, 3) - predicted ray directions in world frame
```

### Step 2: Canonical Reference Rays
```
I_K = identity camera intrinsics (normalized, imw=imh=2.0)
d_canonical = unproject(pixels, I_K)  # (N, 3) - canonical rays from identity camera
```

### Step 3: Find Homography H
Find optimal homography `H` that maps canonical rays to predicted rays:
```
H = argmin_H Σ w_i ||H · d_canonical[i] - d_predicted[i]||²
```
Solved via **RANSAC-weighted homography estimation**.

### Step 4: QL Decomposition
Decompose homography: **H = Q · L**
- **Q** → Rotation matrix **R** (3×3)
- **L** → Lower triangular matrix containing intrinsics:
  - `f = (L[0,0], L[1,1])` → focal lengths
  - `pp = (L[2,0], L[2,1])` → principal point

### Step 5: Extract Translation (Channels 3-5)
```
T = weighted_average(ray_map[:, :, 3:], confidence)
```

## Final Output
- **Extrinsics**: `[R | T]` (rotation + translation)
- **Intrinsics**: `K` from `f` and `pp`

## Key Insight
The **first 3 channels** (ray directions) encode the camera **orientation** and **intrinsics** through the homography between canonical and predicted rays. The **last 3 channels** directly provide the camera **translation**.

