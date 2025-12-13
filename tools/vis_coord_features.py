import argparse
import glob
import os
import pickle
import numpy as np
import open3d as o3d

# CROP_SIZE = [500, 500, 20]  # [X,Y,Z] voxels (centered crop); set None to disable
CROP_SIZE = None

# Channel to visualize (None = all channels, otherwise specific channel index)
OCCUPANCY_CHANNEL = 16

# Occupancy threshold: minimum probability to visualize a voxel (0.0-1.0)
OCCUPANCY_THRESHOLD = 0.9

# Probability ranges for intensity (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
PROB_RANGES = [(i * 0.1, (i + 1) * 0.1) for i in range(10)]
# Intensity: lower probability = darker color (appears more transparent)
# Range: 0.1 (very dark/transparent) to 1.0 (full brightness/opaque)
INTENSITY_MAP = [0.1 + i * 0.09 for i in range(10)]  # 0.1 to 1.0


DISPLAY_FLAGS = {
    "pseudo_coors": False,
    "gt_coors": False,
    "pseudo_occupancy_map": True,
    "gt_occupancy_map": True,
    
}




def _make_voxel_mesh(voxel_indices: np.ndarray, voxel_size: np.ndarray, pcr: np.ndarray, 
                     base_color: tuple, intensity: float = 1.0):
    """Create voxel meshes with color intensity representing probability.
    
    Args:
        voxel_indices: (N, 4) voxel coordinates [batch, z, y, x]
        voxel_size: [vx, vy, vz] voxel sizes
        pcr: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        base_color: (R, G, B) base color tuple (0-1 range)
        intensity: Color intensity multiplier (0.0 = dark, 1.0 = full brightness)
                   Lower intensity = lower probability = more transparent appearance
    
    Returns:
        List of TriangleMesh objects
    """
    meshes = []
    # Adjust color by intensity (darker = lower probability)
    adjusted_color = tuple(c * intensity for c in base_color)
    
    for idx in voxel_indices:
        x0 = pcr[0] + idx[3] * voxel_size[0]
        y0 = pcr[1] + idx[2] * voxel_size[1]
        z0 = pcr[2] + idx[1] * voxel_size[2]
        x1 = x0 + voxel_size[0]
        y1 = y0 + voxel_size[1]
        z1 = z0 + voxel_size[2]
        
        # Create box mesh
        box = o3d.geometry.TriangleMesh.create_box(
            width=x1 - x0,
            height=y1 - y0,
            depth=z1 - z0
        )
        box.translate([x0, y0, z0])
        
        # Set color with intensity adjustment (lower intensity = appears more transparent)
        box.paint_uniform_color(adjusted_color)
        box.compute_vertex_normals()
        
        meshes.append(box)
    
    return meshes


def _make_voxel_lines(voxel_indices: np.ndarray, voxel_size: np.ndarray, pcr: np.ndarray, color: tuple):
    """Create voxel wireframes (for non-empty voxels from coors)."""
    lines = []
    colors = []
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for idx in voxel_indices:
        x0 = pcr[0] + idx[3] * voxel_size[0]
        y0 = pcr[1] + idx[2] * voxel_size[1]
        z0 = pcr[2] + idx[1] * voxel_size[2]
        x1 = x0 + voxel_size[0]
        y1 = y0 + voxel_size[1]
        z1 = z0 + voxel_size[2]
        corners = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])
        for e in edges:
            lines.append(corners[[e[0], e[1]]])
            colors.append(color)
    if not lines:
        return None, None
    return np.stack(lines, axis=0), np.stack(colors, axis=0)


def _make_voxel_center_points(voxel_indices: np.ndarray, voxel_size: np.ndarray, pcr: np.ndarray, color: tuple):
    """Create center points of voxels.
    
    Args:
        voxel_indices: (N, 4) voxel coordinates [batch, z, y, x]
        voxel_size: [vx, vy, vz] voxel sizes
        pcr: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        color: (R, G, B) color tuple (0-1 range)
    
    Returns:
        o3d.geometry.PointCloud object
    """
    if len(voxel_indices) == 0:
        return None
    
    points = []
    colors_list = []
    
    for idx in voxel_indices:
        # Calculate voxel center position
        x_center = pcr[0] + (idx[3] + 0.5) * voxel_size[0]
        y_center = pcr[1] + (idx[2] + 0.5) * voxel_size[1]
        z_center = pcr[2] + (idx[1] + 0.5) * voxel_size[2]
        
        points.append([x_center, y_center, z_center])
        colors_list.append(color)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors_list))
    
    return pcd


def _make_bbox_lines(pcr: np.ndarray, color: tuple):
    """Create bounding box wireframe."""
    x0, y0, z0, x1, y1, z1 = pcr
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    lines = []
    colors = []
    for e in edges:
        lines.append(corners[[e[0], e[1]]])
        colors.append(color)
    return np.stack(lines, axis=0), np.stack(colors, axis=0)


def occupancy_channel_to_voxels(
    occupancy_map: np.ndarray,
    channel_idx: int,
    pcr: np.ndarray,
    voxel_size: np.ndarray,  # Original voxel size (not used for occupancy)
    prob_ranges: list,
    intensity_map: list,
    threshold: float = 0.1,
    z_level: float = None,
):
    """Convert occupancy map channel to voxel coordinates.
    
    Uses occupancy grid (180x180x32) with its own voxel size, not the original grid.
    
    Args:
        occupancy_map: (B, C, H, W) occupancy probability maps, where H=W=180, C=32
        channel_idx: Channel index (Z level in occupancy grid, 0-31)
        pcr: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz] original voxel sizes (not used, kept for compatibility)
        prob_ranges: List of (min_prob, max_prob) tuples for probability ranges
        intensity_map: List of intensity values corresponding to prob_ranges
        threshold: Minimum occupancy probability to visualize (default: 0.1)
        z_level: Z level in occupancy grid (if None, uses channel_idx directly)
    
    Returns:
        Tuple of (list of voxel groups, count of non-empty voxels)
        Voxel coordinates are in occupancy grid: [batch, z_occ, y_occ, x_occ]
        where z_occ = channel_idx (0-31), y_occ = 0-179, x_occ = 0-179
    """
    if occupancy_map is None:
        return [], 0
    
    occ_np = np.asarray(occupancy_map)
    if occ_np.ndim != 4:
        return [], 0
    
    B, C, H, W = occ_np.shape  # H=W=180, C=32 (occupancy grid)
    
    if channel_idx >= C:
        print(f"Warning: channel_idx {channel_idx} >= C {C}, using channel {C-1}")
        channel_idx = C - 1
    
    # Calculate occupancy voxel size from occupancy grid shape
    # Occupancy grid: X=180, Y=180, Z=32
    occ_voxel_size = np.array([
        (pcr[3] - pcr[0]) / W,  # X: e.g., 108 / 180 = 0.6
        (pcr[4] - pcr[1]) / H,  # Y: e.g., 108 / 180 = 0.6
        (pcr[5] - pcr[2]) / C,  # Z: e.g., 11 / 32 = 0.34375
    ], dtype=np.float32)
    
    # Z level in occupancy grid is the channel index directly
    if z_level is None:
        z_level = channel_idx  # Direct mapping: channel 0 = z_level 0 in occupancy grid
    
    # Extract channel occupancy map: (B, H, W)
    channel_occ = occ_np[:, channel_idx, :, :]  # (B, 180, 180)
    
    # Count non-empty voxels (above threshold)
    non_empty_mask = channel_occ > threshold
    num_non_empty = int(non_empty_mask.sum())
    
    # Group voxels by probability range (only for non-empty voxels)
    voxel_groups = [[] for _ in prob_ranges]
    
    for b in range(B):
        for bev_y in range(H):  # 0 to 179 (occupancy grid Y)
            for bev_x in range(W):  # 0 to 179 (occupancy grid X)
                prob = channel_occ[b, bev_y, bev_x]
                
                # Skip if probability is below threshold
                if prob < threshold:
                    continue
                
                # Find which probability range this belongs to
                range_idx = min(int(prob * 10), len(prob_ranges) - 1)
                
                # Use occupancy grid coordinates directly: [batch, z_occ, y_occ, x_occ]
                # z_occ = channel_idx (0-31), y_occ = bev_y (0-179), x_occ = bev_x (0-179)
                voxel_groups[range_idx].append([b, int(z_level), bev_y, bev_x])
    
    # Create result for each probability range
    result = []
    for range_idx, (min_prob, max_prob) in enumerate(prob_ranges):
        if len(voxel_groups[range_idx]) == 0:
            continue
        
        voxel_indices = np.array(voxel_groups[range_idx], dtype=np.int32)
        intensity = intensity_map[range_idx]
        
        # Color will be set by caller (red for pseudo, green for GT)
        # Intensity represents probability: lower = darker (appears more transparent)
        result.append((voxel_indices, intensity, min_prob, max_prob))
    
    return result, num_non_empty


def visualize_file(path: str, channel_idx: int = OCCUPANCY_CHANNEL, coors_mode: str = "voxel"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Only extract what we need: coors and occupancy maps
    pseudo_coors = data.get("pseudo_coors")
    gt_coors = data.get("gt_coors")
    pseudo_occupancy_map = data.get("pseudo_occupancy_map")  # (B, C, H, W)
    gt_occupancy_map = data.get("gt_occupancy_map")  # (B, C, H, W)
    voxel_size = np.asarray(data.get("voxel_size"), dtype=np.float32)
    pcr = np.asarray(data.get("point_cloud_range"), dtype=np.float32)
    
    def to_np(x):
        if x is None:
            return None
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)
    
    pseudo_coors = to_np(pseudo_coors)
    gt_coors = to_np(gt_coors)
    pseudo_occupancy_map = to_np(pseudo_occupancy_map)
    gt_occupancy_map = to_np(gt_occupancy_map)
    
    def crop_coors(coors: np.ndarray):
        """Crop coors if CROP_SIZE is set."""
        if coors is None or CROP_SIZE is None:
            return coors
        gx = (pcr[3] - pcr[0]) / voxel_size[0]
        gy = (pcr[4] - pcr[1]) / voxel_size[1]
        gz = (pcr[5] - pcr[2]) / voxel_size[2]
        cx = gx * 0.5
        cy = gy * 0.5
        cz = gz * 0.5
        vx_half = CROP_SIZE[0] * 0.5
        vy_half = CROP_SIZE[1] * 0.5
        vz_half = CROP_SIZE[2] * 0.5
        mask = (
            (coors[:, 3] >= cx - vx_half) & (coors[:, 3] <= cx + vx_half) &
            (coors[:, 2] >= cy - vy_half) & (coors[:, 2] <= cy + vy_half) &
            (coors[:, 1] >= cz - vz_half) & (coors[:, 1] <= cz + vz_half)
        )
        return coors[mask]
    
    if pseudo_coors is not None:
        pseudo_coors = crop_coors(pseudo_coors)
    if gt_coors is not None:
        gt_coors = crop_coors(gt_coors)
    
    print(f"[INFO] {path}")
    print(f"  pseudo_coors: {None if pseudo_coors is None else pseudo_coors.shape}")
    print(f"  gt_coors: {None if gt_coors is None else gt_coors.shape}")
    print(f"  pseudo_occupancy_map: {None if pseudo_occupancy_map is None else pseudo_occupancy_map.shape}")
    print(f"  gt_occupancy_map: {None if gt_occupancy_map is None else gt_occupancy_map.shape}")
    if channel_idx is None:
        print(f"  Visualizing all channels")
    else:
        print(f"  Visualizing channel {channel_idx}")
    import sys
    sys.stdout.flush()
    
    # Calculate occupancy grid info and voxel size
    occ_voxel_size = None
    if gt_occupancy_map is not None:
        occ_np = np.asarray(gt_occupancy_map)
        if occ_np.ndim == 4:
            B, C, H, W = occ_np.shape  # C=32, H=W=180 (occupancy grid)
            
            # Calculate occupancy voxel size from occupancy grid
            occ_voxel_size = np.array([
                (pcr[3] - pcr[0]) / W,  # X: e.g., 108 / 180 = 0.6
                (pcr[4] - pcr[1]) / H,  # Y: e.g., 108 / 180 = 0.6
                (pcr[5] - pcr[2]) / C,  # Z: e.g., 11 / 32 = 0.34375
            ], dtype=np.float32)
            
            print(f"  Occupancy map shape: (B={B}, C={C}, H={H}, W={W})")
            print(f"  Occupancy grid: X={W}, Y={H}, Z={C}")
            print(f"  Occupancy voxel size: ({occ_voxel_size[0]:.4f}, {occ_voxel_size[1]:.4f}, {occ_voxel_size[2]:.4f})")
            sys.stdout.flush()
        else:
            print(f"  GT occupancy map has wrong shape: {occ_np.shape}")
    else:
        print(f"  GT occupancy map is None")
    
    # Use occupancy voxel size if available, otherwise fall back to original
    if occ_voxel_size is None:
        occ_voxel_size = voxel_size
    
    # Convert occupancy maps to voxel meshes with intensity-based transparency
    # Only visualize voxels above threshold to avoid heavy rendering
    threshold = OCCUPANCY_THRESHOLD
    
    # Process pseudo occupancy map if flag is enabled
    pseudo_voxel_groups = []
    total_pseudo_non_empty = 0
    if DISPLAY_FLAGS.get("pseudo_occupancy_map", True) and pseudo_occupancy_map is not None:
        occ_np = np.asarray(pseudo_occupancy_map)
        if occ_np.ndim == 4:
            B, C, H, W = occ_np.shape
            if channel_idx is None:
                # Process all channels
                print(f"  Visualizing ALL {C} channels of pseudo occupancy map")
                print(f"  Occupancy threshold: {threshold}")
                for ch_idx in range(C):
                    groups, num_non_empty = occupancy_channel_to_voxels(
                        pseudo_occupancy_map, ch_idx, pcr, occ_voxel_size,
                        PROB_RANGES, INTENSITY_MAP, threshold=threshold
                    )
                    pseudo_voxel_groups.extend([(ch_idx, g) for g in groups])
                    total_pseudo_non_empty += num_non_empty
                    
                    if ch_idx < 5 or ch_idx % 50 == 0 or ch_idx == C - 1:
                        total_voxels = sum(len(voxels) for voxels, _, _, _ in groups)
                        print(f"    Channel {ch_idx}: {num_non_empty} non-empty voxels, {total_voxels} to visualize")
            else:
                # Process specific channel
                if channel_idx >= C:
                    print(f"  Warning: channel_idx {channel_idx} >= C {C}, using channel {C-1}")
                    channel_idx = C - 1
                print(f"  Visualizing channel {channel_idx} of pseudo occupancy map")
                print(f"  Occupancy threshold: {threshold}")
                groups, num_non_empty = occupancy_channel_to_voxels(
                    pseudo_occupancy_map, channel_idx, pcr, occ_voxel_size,
                    PROB_RANGES, INTENSITY_MAP, threshold=threshold
                )
                pseudo_voxel_groups = [(channel_idx, g) for g in groups]
                total_pseudo_non_empty = num_non_empty
                total_voxels = sum(len(voxels) for voxels, _, _, _ in groups)
                print(f"    Channel {channel_idx}: {num_non_empty} non-empty voxels, {total_voxels} to visualize")
            
            if channel_idx is None:
                print(f"  Total pseudo occupancy - Non-empty voxels (>{threshold}): {total_pseudo_non_empty}")
            else:
                print(f"  Pseudo occupancy channel {channel_idx} - Non-empty voxels (>{threshold}): {total_pseudo_non_empty}")
            total_pseudo_voxels = sum(
                len(voxel_indices) for _, (voxel_indices, _, _, _) in pseudo_voxel_groups
            )
            print(f"  Total pseudo voxels to visualize: {total_pseudo_voxels}")
    
    # Visualize GT occupancy map (all channels or specific channel) if flag is enabled
    all_gt_voxel_groups = []
    if DISPLAY_FLAGS.get("gt_occupancy_map", True) and gt_occupancy_map is not None:
        occ_np = np.asarray(gt_occupancy_map)
        if occ_np.ndim == 4:
            B, C, H, W = occ_np.shape
            
            # Determine which channels to visualize
            if channel_idx is None:
                # Visualize ALL channels
                channels_to_vis = list(range(C))
                print(f"  Visualizing ALL {C} channels of GT occupancy map")
            else:
                # Visualize specific channel only
                if channel_idx >= C:
                    print(f"  Warning: channel_idx {channel_idx} >= C {C}, using channel {C-1}")
                    channel_idx = C - 1
                channels_to_vis = [channel_idx]
                print(f"  Visualizing channel {channel_idx} of GT occupancy map")
            
            print(f"  Occupancy threshold: {threshold}")
            
            all_gt_voxel_groups = []
            total_gt_non_empty = 0
            
            for ch_idx in channels_to_vis:
                gt_voxel_groups, gt_num_non_empty = occupancy_channel_to_voxels(
                    gt_occupancy_map, ch_idx, pcr, occ_voxel_size,
                    PROB_RANGES, INTENSITY_MAP, threshold=threshold
                )
                all_gt_voxel_groups.append((ch_idx, gt_voxel_groups, gt_num_non_empty))
                total_gt_non_empty += gt_num_non_empty
                
                if len(channels_to_vis) <= 5 or ch_idx < 5 or ch_idx % 50 == 0 or ch_idx == channels_to_vis[-1]:
                    total_voxels = sum(len(voxels) for voxels, _, _, _ in gt_voxel_groups)
                    print(f"    Channel {ch_idx}: {gt_num_non_empty} non-empty voxels, {total_voxels} to visualize")
            
            print(f"  Total GT occupancy - Non-empty voxels (>{threshold}): {total_gt_non_empty}")
            total_gt_voxels = sum(
                sum(len(voxels) for voxels, _, _, _ in groups) 
                for _, groups, _ in all_gt_voxel_groups
            )
            print(f"  Total GT voxels to visualize: {total_gt_voxels}")
        else:
            print(f"  GT occupancy map has wrong shape: {occ_np.shape}")
    else:
        if not DISPLAY_FLAGS.get("gt_occupancy_map", True):
            print(f"  GT occupancy map visualization disabled by DISPLAY_FLAGS")
        else:
            print(f"  GT occupancy map is None")
    
    sys.stdout.flush()
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=os.path.basename(path), width=1920, height=1080, visible=True)
    
    # Add coordinate frame and bounding box
    extent = max(pcr[3] - pcr[0], pcr[4] - pcr[1], pcr[5] - pcr[2])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=extent * 0.05)
    bbox_lines, bbox_colors = _make_bbox_lines(pcr, (1, 1, 0))
    ls_bbox = o3d.geometry.LineSet()
    pts_bbox = bbox_lines.reshape(-1, 3)
    lines_idx = np.arange(len(pts_bbox), dtype=np.int32).reshape(-1, 2)
    ls_bbox.points = o3d.utility.Vector3dVector(pts_bbox)
    ls_bbox.lines = o3d.utility.Vector2iVector(lines_idx)
    ls_bbox.colors = o3d.utility.Vector3dVector(np.tile(bbox_colors[0:1], (lines_idx.shape[0], 1)))
    vis.add_geometry(axis)
    vis.add_geometry(ls_bbox)
    
    # Add non-empty voxels from coors (wireframes or center points)
    if DISPLAY_FLAGS.get("pseudo_coors", True) or DISPLAY_FLAGS.get("gt_coors", True):
        if coors_mode == "voxel":
            # Draw wireframe boxes
            if DISPLAY_FLAGS.get("pseudo_coors", True) and pseudo_coors is not None:
                pseudo_lines, pseudo_colors = _make_voxel_lines(pseudo_coors, voxel_size, pcr, (1, 0.5, 0))  # Orange
                if pseudo_lines is not None:
                    ls_p = o3d.geometry.LineSet()
                    pts = pseudo_lines.reshape(-1, 3)
                    lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
                    ls_p.points = o3d.utility.Vector3dVector(pts)
                    ls_p.lines = o3d.utility.Vector2iVector(lines_idx)
                    ls_p.colors = o3d.utility.Vector3dVector(np.tile(pseudo_colors[0:1], (lines_idx.shape[0], 1)))
                    vis.add_geometry(ls_p)
            
            if DISPLAY_FLAGS.get("gt_coors", True) and gt_coors is not None:
                gt_lines, gt_colors = _make_voxel_lines(gt_coors, voxel_size, pcr, (0, 0.5, 1))  # Blue
                if gt_lines is not None:
                    ls_g = o3d.geometry.LineSet()
                    pts = gt_lines.reshape(-1, 3)
                    lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
                    ls_g.points = o3d.utility.Vector3dVector(pts)
                    ls_g.lines = o3d.utility.Vector2iVector(lines_idx)
                    ls_g.colors = o3d.utility.Vector3dVector(np.tile(gt_colors[0:1], (lines_idx.shape[0], 1)))
                    vis.add_geometry(ls_g)
        
        elif coors_mode == "point":
            # Draw center points only
            if DISPLAY_FLAGS.get("pseudo_coors", True) and pseudo_coors is not None:
                pcd_pseudo = _make_voxel_center_points(pseudo_coors, voxel_size, pcr, (1, 0.5, 0))  # Orange
                if pcd_pseudo is not None:
                    vis.add_geometry(pcd_pseudo)
            
            if DISPLAY_FLAGS.get("gt_coors", True) and gt_coors is not None:
                pcd_gt = _make_voxel_center_points(gt_coors, voxel_size, pcr, (0, 0.5, 1))  # Blue
                if pcd_gt is not None:
                    vis.add_geometry(pcd_gt)
    
    # Add occupancy-based voxel wireframes (lines) for better performance
    # Pseudo occupancy map visualization
    if DISPLAY_FLAGS.get("pseudo_occupancy_map", True) and pseudo_voxel_groups:
        for ch_idx, (voxel_indices, intensity, min_prob, max_prob) in pseudo_voxel_groups:
            # Adjust color intensity based on probability
            color = tuple(c * intensity for c in (1.0, 0.0, 0.0))  # Red for pseudo
            # Use occupancy voxel size for visualization (occupancy grid coordinates)
            lines, colors = _make_voxel_lines(voxel_indices, occ_voxel_size, pcr, color)
            if lines is not None:
                ls = o3d.geometry.LineSet()
                pts = lines.reshape(-1, 3)
                lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
                ls.points = o3d.utility.Vector3dVector(pts)
                ls.lines = o3d.utility.Vector2iVector(lines_idx)
                ls.colors = o3d.utility.Vector3dVector(np.tile(colors[0:1], (lines_idx.shape[0], 1)))
                vis.add_geometry(ls)
    
    # GT occupancy map visualization
    if DISPLAY_FLAGS.get("gt_occupancy_map", True):
        for ch_idx, gt_voxel_groups, _ in all_gt_voxel_groups:
            for voxel_indices, intensity, min_prob, max_prob in gt_voxel_groups:
                # Adjust color intensity based on probability
                color = tuple(c * intensity for c in (0.0, 1.0, 0.0))  # Green for GT
                # Use occupancy voxel size for visualization (occupancy grid coordinates)
                lines, colors = _make_voxel_lines(voxel_indices, occ_voxel_size, pcr, color)
                if lines is not None:
                    ls = o3d.geometry.LineSet()
                    pts = lines.reshape(-1, 3)
                    lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
                    ls.points = o3d.utility.Vector3dVector(pts)
                    ls.lines = o3d.utility.Vector2iVector(lines_idx)
                    ls.colors = o3d.utility.Vector3dVector(np.tile(colors[0:1], (lines_idx.shape[0], 1)))
                    vis.add_geometry(ls)
    
    # Note: Using color intensity to represent probability/transparency
    # Lower intensity = darker color = appears more transparent
    # Higher intensity = brighter color = appears more opaque
    
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Visualize occupancy maps with transparency.")
    parser.add_argument("--dir", default="work_dirs/resdet3d_nuscenes_mini/debug_viz", 
                       type=str, required=True, help="Directory containing *.pkl debug files.")
    parser.add_argument("--coors-mode", type=str, default="voxel", choices=["voxel", "point"],
                       help="Visualization mode for coors: 'voxel' (wireframe boxes) or 'point' (center points only)")
    args = parser.parse_args()
    
    files = sorted(glob.glob(os.path.join(args.dir, "*.pkl")))
    if not files:
        print(f"No pickle files found in {args.dir}")
        return
    
    for fpath in files:
        print(f"Showing {fpath} (close window to continue)")
        visualize_file(fpath, channel_idx=OCCUPANCY_CHANNEL, coors_mode=args.coors_mode)


if __name__ == "__main__":
    main()
