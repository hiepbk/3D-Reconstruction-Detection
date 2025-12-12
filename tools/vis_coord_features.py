import argparse
import glob
import os
import pickle
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt


# CROP_SIZE = [500, 500, 20]  # [X,Y,Z] voxels (centered crop); set None to disable
CROP_SIZE = None


def _make_voxel_lines(voxel_indices: np.ndarray, voxel_size: np.ndarray, pcr: np.ndarray, color: tuple):
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


def _make_bbox_lines(pcr: np.ndarray, color: tuple):
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


def visualize_file(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)

    pseudo_coors = data.get("pseudo_coors")
    gt_coors = data.get("gt_coors")
    pseudo_voxel_feats = data.get("pseudo_voxel_features")  # Per-voxel features from VFE (N, C)
    gt_voxel_feats = data.get("gt_voxel_features")  # Per-voxel features from VFE (N, C)
    pseudo_occupancy_map = data.get("pseudo_occupancy_map")  # (B, C, H, W) occupancy probability maps
    gt_occupancy_map = data.get("gt_occupancy_map")  # (B, C, H, W) occupancy probability maps
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
    pseudo_voxel_feats = to_np(pseudo_voxel_feats)
    gt_voxel_feats = to_np(gt_voxel_feats)
    pseudo_occupancy_map = to_np(pseudo_occupancy_map)
    gt_occupancy_map = to_np(gt_occupancy_map)

    def crop_coors_and_feats(coors: np.ndarray, feats: np.ndarray):
        """Crop coors and corresponding voxel_features using the same mask."""
        if coors is None or CROP_SIZE is None:
            return coors, feats
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
        coors_cropped = coors[mask]
        # Only crop feats if they exist and match the coors length
        if feats is not None and len(feats.shape) >= 1 and feats.shape[0] == coors.shape[0]:
            feats_cropped = feats[mask]
        else:
            feats_cropped = feats
        return coors_cropped, feats_cropped

    if pseudo_coors is not None:
        pseudo_coors, pseudo_voxel_feats = crop_coors_and_feats(pseudo_coors, pseudo_voxel_feats)
    if gt_coors is not None:
        gt_coors, gt_voxel_feats = crop_coors_and_feats(gt_coors, gt_voxel_feats)

    print(f"[INFO] {path}")
    print(f"  pseudo_coors: {None if pseudo_coors is None else pseudo_coors.shape}")
    print(f"  pseudo_voxel_feats: {None if pseudo_voxel_feats is None else pseudo_voxel_feats.shape}")
    print(f"  pseudo_occupancy_map: {None if pseudo_occupancy_map is None else pseudo_occupancy_map.shape}")
    print(f"  gt_coors: {None if gt_coors is None else gt_coors.shape}")
    print(f"  gt_voxel_feats: {None if gt_voxel_feats is None else gt_voxel_feats.shape}")
    print(f"  gt_occupancy_map: {None if gt_occupancy_map is None else gt_occupancy_map.shape}")
    import sys
    sys.stdout.flush()
    
    # Visualize occupancy maps and their difference
    # The shape of occupancy maps is (B, C, H, W) where C is the number of channels (e.g., 256)
    def display_occupancy_difference(pseudo_occ, gt_occ):
        if pseudo_occ is None or gt_occ is None:
            print("  Occupancy maps not available for visualization")
            return
        
        for b in range(pseudo_occ.shape[0]):
            differences = np.abs(pseudo_occ[b] - gt_occ[b])
            num_channels = pseudo_occ.shape[1]
            
            # Display a few representative channels (e.g., first, middle, last)
            channels_to_show = [0, num_channels // 4, num_channels // 2, 3 * num_channels // 4, num_channels - 1]
            channels_to_show = [c for c in channels_to_show if c < num_channels]
            
            for i in channels_to_show:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(pseudo_occ[b, i], cmap='hot', vmin=0, vmax=1)
                plt.colorbar()
                plt.title(f"Pseudo Occupancy Channel {i}")
                plt.subplot(1, 3, 2)
                plt.imshow(gt_occ[b, i], cmap='hot', vmin=0, vmax=1)
                plt.colorbar()
                plt.title(f"GT Occupancy Channel {i}")
                plt.subplot(1, 3, 3)
                plt.imshow(differences[i], cmap='hot')
                plt.colorbar()
                plt.title(f"Difference Channel {i}")
                plt.tight_layout()
                plt.show()
            
            # Also show mean across all channels
            pseudo_mean = pseudo_occ[b].mean(axis=0)
            gt_mean = gt_occ[b].mean(axis=0)
            diff_mean = np.abs(pseudo_mean - gt_mean)
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(pseudo_mean, cmap='hot', vmin=0, vmax=1)
            plt.colorbar()
            plt.title("Pseudo Occupancy (Mean across channels)")
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mean, cmap='hot', vmin=0, vmax=1)
            plt.colorbar()
            plt.title("GT Occupancy (Mean across channels)")
            plt.subplot(1, 3, 3)
            plt.imshow(diff_mean, cmap='hot')
            plt.colorbar()
            plt.title("Difference (Mean across channels)")
            plt.tight_layout()
            plt.show()

    # Display occupancy maps and their difference
    # display_occupancy_difference(pseudo_occupancy_map, gt_occupancy_map)
    
    def occupancy_map_to_voxels(occupancy_map, pcr, voxel_size, occupancy_threshold=0.5):
        """Convert occupancy map (B, C, H, W) to voxel coordinates in original grid.
        
        Args:
            occupancy_map: (B, C, H, W) occupancy probability maps, where H=W=180
            pcr: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
            voxel_size: [vx, vy, vz] voxel sizes
            occupancy_threshold: Threshold for considering a voxel as occupied
            num_z_levels: Number of Z levels in original grid (41)
        
        Returns:
            voxel_coors: (N, 4) voxel coordinates [batch, z, y, x] in original grid
        """
        if occupancy_map is None:
            return None
        
        occ_np = np.asarray(occupancy_map)
        if occ_np.ndim != 4:
            return None
        
        B, C, H, W = occ_np.shape  # H=W=180, C=256 (channels)
        
        # Calculate original grid size from point cloud range and voxel size
        grid_size_x = int((pcr[3] - pcr[0]) / voxel_size[0])  # e.g., 1440
        grid_size_y = int((pcr[4] - pcr[1]) / voxel_size[1])  # e.g., 1440
        grid_size_z = int((pcr[5] - pcr[2]) / voxel_size[2])
        
        # Calculate mapping from BEV map (H, W) to original grid (grid_size_y, grid_size_x)
        # Each cell in BEV map corresponds to multiple cells in original grid
        scale_x = grid_size_x / W  # e.g., 1440 / 180 = 8
        scale_y = grid_size_y / H  # e.g., 1440 / 180 = 8
        scale_z = grid_size_z / C
        
        voxel_coors = []
        
        for b in range(B):
            for bev_y in range(H):
                for bev_x in range(W):
                    # Get occupancy for all channels at this (bev_y, bev_x) location
                    z_occ_at_loc = occ_np[b, :, bev_y, bev_x]  # (C,)
                    
                    # Find channels with occupancy above threshold
                    z_occupied_channels = np.where(z_occ_at_loc > occupancy_threshold)[0]
                    
                    # This layer of height does not have any occupied voxels
                    if len(z_occupied_channels) == 0:
                        continue
                    
                    # For each occupied channel, map to Z level and create ONE voxel at the center of the 8x8 region
                    for ch in z_occupied_channels:
                        z_level = ch / C * grid_size_z
                        
                        # Map BEV position to original grid
                        # Each 180x180 cell corresponds to 8x8 region in 1440x1440
                        # Center of 8x8 region is at offset (4, 4) within that block
                        x_start = int(bev_x * scale_x)  # Start of 8x8 block
                        y_start = int(bev_y * scale_y)  # Start of 8x8 block
                        x_center = x_start + int(scale_x / 2)  # Center = start + 4 (since scale_x = 8)
                        y_center = y_start + int(scale_y / 2)  # Center = start + 4 (since scale_y = 8)
                        
                        # Clamp to grid boundaries
                        x_center = min(grid_size_x - 1, max(0, x_center))
                        y_center = min(grid_size_y - 1, max(0, y_center))
                        
                        # Create ONE voxel at the center of the 8x8 region at this Z level
                        voxel_coors.append([b, z_level, y_center, x_center])
        
        if len(voxel_coors) == 0:
            return None
        
        return np.array(voxel_coors, dtype=np.int32)
    
    # Calculate grid size for info
    if gt_occupancy_map is not None:
        occ_np = np.asarray(gt_occupancy_map)
        if occ_np.ndim == 4:
            B, C, H, W = occ_np.shape
            grid_size_x = int((pcr[3] - pcr[0]) / voxel_size[0])
            grid_size_y = int((pcr[4] - pcr[1]) / voxel_size[1])
            grid_size_z = int((pcr[5] - pcr[2]) / voxel_size[2])
            scale_x = grid_size_x / W
            scale_y = grid_size_y / H
            scale_z = grid_size_z / C
            print(f"  Occupancy map shape: (B={B}, C={C}, H={H}, W={W})")
            print(f"  Original grid size: ({grid_size_x}, {grid_size_y}, {grid_size_z})")
            print(f"  Scale factor: ({scale_x:.2f}, {scale_y:.2f}, {scale_z:.2f}) - each BEV cell = {scale_x:.0f}x{scale_y:.0f}x{scale_z:.0f} grid cells")
            import sys
            sys.stdout.flush()
    
    # Convert occupancy maps to voxel coordinates
    gt_occ_voxels = occupancy_map_to_voxels(gt_occupancy_map, pcr, voxel_size, occupancy_threshold=0.5)
    # pseudo_occ_voxels = occupancy_map_to_voxels(pseudo_occupancy_map, pcr, voxel_size, occupancy_threshold=0.5, num_z_levels=41)
    
    print(f"  GT occupancy voxels: {None if gt_occ_voxels is None else gt_occ_voxels.shape}")
    # print(f"  Pseudo occupancy voxels: {None if pseudo_occ_voxels is None else pseudo_occ_voxels.shape}")
    import sys
    sys.stdout.flush()
    
    # Create voxel wireframes from occupancy maps
    gt_occ_lines = gt_occ_colors = None
    pseudo_occ_lines = pseudo_occ_colors = None
    
    if gt_occ_voxels is not None:
        gt_occ_voxel_size = np.array([scale_x*voxel_size[0], scale_y*voxel_size[1], scale_z])
        gt_occ_lines, gt_occ_colors = _make_voxel_lines(gt_occ_voxels, gt_occ_voxel_size, pcr, (0, 1, 0))  # Green for GT
    # if pseudo_occ_voxels is not None:
    #     pseudo_occ_lines, pseudo_occ_colors = _make_voxel_lines(pseudo_occ_voxels, voxel_size, pcr, (1, 0, 0))  # Red for pseudo
    
    # # Original voxel wireframes from coordinates (if available)
    pseudo_lines = pseudo_colors = None
    gt_lines = gt_colors = None
    # if pseudo_coors is not None:
    #     pseudo_lines, pseudo_colors = _make_voxel_lines(pseudo_coors, voxel_size, pcr, (1, 0.5, 0))  # Orange for original pseudo
    if gt_coors is not None:
        gt_lines, gt_colors = _make_voxel_lines(gt_coors, voxel_size, pcr, (0, 0.5, 1))  # Blue for original GT


    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=os.path.basename(path), width=1280, height=720, visible=True)

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

    # if pseudo_lines is not None:
    #     ls_p = o3d.geometry.LineSet()
    #     pts = pseudo_lines.reshape(-1, 3)
    #     lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
    #     ls_p.points = o3d.utility.Vector3dVector(pts)
    #     ls_p.lines = o3d.utility.Vector2iVector(lines_idx)
    #     ls_p.colors = o3d.utility.Vector3dVector(np.tile(pseudo_colors[0:1], (lines_idx.shape[0], 1)))
    #     vis.add_geometry(ls_p)

    # if gt_lines is not None:
    #     ls_g = o3d.geometry.LineSet()
    #     pts = gt_lines.reshape(-1, 3)
    #     lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
    #     ls_g.points = o3d.utility.Vector3dVector(pts)
    #     ls_g.lines = o3d.utility.Vector2iVector(lines_idx)
    #     ls_g.colors = o3d.utility.Vector3dVector(np.tile(gt_colors[0:1], (lines_idx.shape[0], 1)))
    #     vis.add_geometry(ls_g)
    
    # Add occupancy-based voxels
    if gt_occ_lines is not None:
        ls_gt_occ = o3d.geometry.LineSet()
        pts = gt_occ_lines.reshape(-1, 3)
        lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
        ls_gt_occ.points = o3d.utility.Vector3dVector(pts)
        ls_gt_occ.lines = o3d.utility.Vector2iVector(lines_idx)
        ls_gt_occ.colors = o3d.utility.Vector3dVector(np.tile(gt_occ_colors[0:1], (lines_idx.shape[0], 1)))
        vis.add_geometry(ls_gt_occ)
    
    # if pseudo_occ_lines is not None:
    #     ls_pseudo_occ = o3d.geometry.LineSet()
    #     pts = pseudo_occ_lines.reshape(-1, 3)
    #     lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
    #     ls_pseudo_occ.points = o3d.utility.Vector3dVector(pts)
    #     ls_pseudo_occ.lines = o3d.utility.Vector2iVector(lines_idx)
    #     ls_pseudo_occ.colors = o3d.utility.Vector3dVector(np.tile(pseudo_occ_colors[0:1], (lines_idx.shape[0], 1)))
    #     vis.add_geometry(ls_pseudo_occ)

    vis.run()
    vis.destroy_window()

# python -m tools.vis_coord_features --dir work_dirs/resdet3d_nuscenes_mini/debug_viz
def main():
    parser = argparse.ArgumentParser(description="Offline visualize voxel coords/features from pickles.")
    parser.add_argument("--dir", default="work_dirs/resdet3d_nuscenes_mini/debug_viz",type=str, required=True, help="Directory containing *.pkl debug files.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.pkl")))
    if not files:
        print(f"No pickle files found in {args.dir}")
        return

    for fpath in files:
        print(f"Showing {fpath} (close window to continue)")
        visualize_file(fpath)


if __name__ == "__main__":
    main()

