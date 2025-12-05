"""
Inference configuration file for nuScenes inference script.
Modify these settings to control model behavior and post-processing.
"""

# ============================================================================
# Camera Configuration
# ============================================================================

# Camera types to process (in order)
CAM_TYPES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# ============================================================================
# Model Configuration
# ============================================================================

# Model name or path
MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE"

# Device to use for inference
# Options: "cuda" or "cpu"
DEVICE = "cuda"  # Will auto-fallback to "cpu" if CUDA not available

# Export format for model outputs
# Options: "glb", "npz", "mini_npz", or combinations like "glb-npz"
EXPORT_FORMAT = "glb"

# Reference view selection strategy
# Options: "first", "middle", "saddle_balanced", "saddle_sim_range"
REF_VIEW_STRATEGY = "saddle_balanced"

# Use ray-based pose estimation (more accurate but slower)
USE_RAY_POSE = False

# Maximum number of points in point cloud
MAX_POINTS = 1_000_000


# ============================================================================
# GLB Export Configuration
# ============================================================================

GLB_CONFIG = {
    "sky_depth_def": 98.0,  # Percentile used to fill sky pixels with plausible depth values
    "conf_thresh_percentile": 30.0,  # Lower percentile for adaptive confidence threshold
    "filter_black_bg": False,  # Filter near-black background pixels
    "filter_white_bg": False,  # Filter near-white background pixels
    "max_depth": 100.0,  # Maximum depth threshold to filter far objects/infinity (in meters, None = no limit)
}


# ============================================================================
# Point Cloud Post-Processing Configuration
# ============================================================================

# Voxel size for point cloud downsampling (in meters)
# Smaller values preserve more detail but result in more points
# Recommended: 0.05-0.2 for nuScenes
# Set to None to disable downsampling
DOWNSAMPLE_VOXEL_SIZE = 0.1  # None to disable

# Apply Furthest Point Sampling (FPS) after voxel downsampling for uniform distribution
DOWNSAMPLE_USE_FPS = True

# Number of points to sample with FPS (required if DOWNSAMPLE_USE_FPS is True)
DOWNSAMPLE_FPS_NUM_POINTS = 40000  # e.g., 100000

# Point cloud range for voxelization [x_min, y_min, z_min, x_max, y_max, z_max]
#point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# Set to None to auto-compute from point cloud
DOWNSAMPLE_POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]  # e.g., [-50, -50, -5, 50, 50, 3]

