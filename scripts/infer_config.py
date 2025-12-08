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


SHOW_GT_BOXES = False

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
# DOWNSAMPLE_POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]  # e.g., [-50, -50, -5, 50, 50, 3]
# DOWNSAMPLE_POINT_CLOUD_RANGE = None  # e.g., [-50, -50, -5, 50, 50, 3]
DOWNSAMPLE_POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]  # e.g., [-50, -50, -5, 50, 50, 3]

# ============================================================================
# Ball Query Configuration (Density-Aware Sampling)
# ============================================================================

# Use ball_query for density-aware downsampling (preserves local structure better than pure FPS)
# Strategy: Use FPS to get anchor points, then ball_query to find neighbors within radius
# This preserves more points in dense regions and fewer in sparse regions
DOWNSAMPLE_USE_BALL_QUERY = True  # Set to True to enable ball_query

# Ball query parameters (only used if DOWNSAMPLE_USE_BALL_QUERY is True)
DOWNSAMPLE_BALL_QUERY_MIN_RADIUS = 0.0  # Minimum radius (meters) - typically 0.0
DOWNSAMPLE_BALL_QUERY_MAX_RADIUS = 0.5  # Maximum radius (meters) - adjust based on scene scale
DOWNSAMPLE_BALL_QUERY_SAMPLE_NUM = 16  # Maximum number of neighbors per anchor point
DOWNSAMPLE_BALL_QUERY_ANCHOR_POINTS = 25000  # Number of FPS anchor points for ball_query

# ============================================================================
# Post-processing pipeline (mimic mmdet3d style)
# Each step receives/returns a dict with at least: points, colors, polygon_mask
# ============================================================================
POST_PROCESSING_PIPELINE = [
    # Voxel downsample (always runs if voxel_size is not None)
    dict(
        type="VoxelDownsample",
        voxel_size=DOWNSAMPLE_VOXEL_SIZE,
        point_cloud_range=DOWNSAMPLE_POINT_CLOUD_RANGE,
    ),
    # Density-aware ball query (optional)
    dict(
        type="BallQueryDownsample",
        enabled=DOWNSAMPLE_USE_BALL_QUERY,
        min_radius=DOWNSAMPLE_BALL_QUERY_MIN_RADIUS,
        max_radius=DOWNSAMPLE_BALL_QUERY_MAX_RADIUS,
        sample_num=DOWNSAMPLE_BALL_QUERY_SAMPLE_NUM,
        anchor_points=DOWNSAMPLE_BALL_QUERY_ANCHOR_POINTS,
    ),
    # Uniform cap with FPS (optional)
    dict(
        type="FPSDownsample",
        enabled=DOWNSAMPLE_USE_FPS,
        num_points=DOWNSAMPLE_FPS_NUM_POINTS,
    ),
]

