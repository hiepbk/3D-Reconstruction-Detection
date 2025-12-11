import numpy as np
import torch
from mmcv.runner import HOOKS, Hook


def _make_voxel_lines(voxel_indices: np.ndarray, voxel_size: np.ndarray, pcr: np.ndarray, color: tuple):
    """Create line segments (wireframe cube) for each voxel index."""
    lines = []
    colors = []
    # edges of a unit cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom square
        (4, 5), (5, 6), (6, 7), (7, 4),  # top square
        (0, 4), (1, 5), (2, 6), (3, 7)   # verticals
    ]
    for idx in voxel_indices:
        # idx = [batch, z, y, x]
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
        offset = len(lines) * 0  # not used, edges use local indices
        for e in edges:
            lines.append(corners[[e[0], e[1]]])
            colors.append(color)
    if not lines:
        return None, None
    lines = np.stack(lines, axis=0)  # (L, 2, 3)
    colors = np.stack(colors, axis=0)
    return lines, colors


def _make_bbox_lines(pcr: np.ndarray, color: tuple):
    """Create wireframe lines for the overall bounding box."""
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
    lines = np.stack(lines, axis=0)
    colors = np.stack(colors, axis=0)
    return lines, colors


@HOOKS.register_module()
class VisualDebugSparseHook(Hook):
    """Open3D visualization of non-empty voxels for pseudo vs GT.
    
    Draws wireframe cubes (red = pseudo, green = GT) for batch 0, plus bbox and axis.
    """

    def __init__(self, interval=500, max_voxels=20000):
        self.interval = interval
        self.max_voxels = max_voxels

    def after_train_iter(self, runner):
        if not torch.cuda.is_available():
            return
        curr_iter = runner.iter
        if self.interval <= 0 or (curr_iter + 1) % self.interval != 0:
            return

        # Get model and refinement
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        rb = getattr(model, 'reconstruction_backbone', None)
        if rb is None or getattr(rb, 'refinement', None) is None:
            return
        ref = rb.refinement

        # Enable caching for this iteration
        ref.enable_visual_debug = True

        # Need cached coors from last forward
        if not hasattr(ref, 'last_coors'):
            return

        coors = ref.last_coors  # (N, 4) [batch,z,y,x]
        voxel_size = ref.voxel_size.cpu().numpy()
        pcr = ref.point_cloud_range.cpu().numpy()

        # Filter batch 0 and cap count
        coors_b0 = coors[coors[:, 0] == 0].numpy()
        if coors_b0.shape[0] == 0:
            return
        coors_b0 = coors_b0[:self.max_voxels]

        # If GT coors cached, show green; otherwise only pseudo (red)
        pseudo_lines, pseudo_colors = _make_voxel_lines(coors_b0, voxel_size, pcr, (1, 0, 0))

        # Try to fetch GT coors if stored separately (optional)
        gt_lines = gt_colors = None
        if hasattr(ref, 'last_coors_gt'):
            coors_gt = ref.last_coors_gt
            coors_gt_b0 = coors_gt[coors_gt[:, 0] == 0].numpy()
            coors_gt_b0 = coors_gt_b0[:self.max_voxels]
            gt_lines, gt_colors = _make_voxel_lines(coors_gt_b0, voxel_size, pcr, (0, 1, 0))

        try:
            import open3d as o3d
        except ImportError:
            runner.logger.warning("VisualDebugSparseHook: open3d not installed; skipping visualization.")
            return

        # Build line sets
        geometries = []
        # Bounding box
        bbox_lines, bbox_colors = _make_bbox_lines(pcr, (1, 1, 0))
        ls_bbox = o3d.geometry.LineSet()
        pts = bbox_lines.reshape(-1, 3)
        unique_pts, inv = np.unique(pts, axis=0, return_inverse=True)
        ls_bbox.points = o3d.utility.Vector3dVector(unique_pts)
        lines_idx = inv.reshape(-1, 2)
        ls_bbox.lines = o3d.utility.Vector2iVector(lines_idx)
        ls_bbox.colors = o3d.utility.Vector3dVector(bbox_colors)
        geometries.append(ls_bbox)

        if pseudo_lines is not None:
            ls_p = o3d.geometry.LineSet()
            pts = pseudo_lines.reshape(-1, 3)
            # unique points and remap
            unique_pts, inv = np.unique(pts, axis=0, return_inverse=True)
            ls_p.points = o3d.utility.Vector3dVector(unique_pts)
            lines_idx = inv.reshape(-1, 2)
            ls_p.lines = o3d.utility.Vector2iVector(lines_idx)
            ls_p.colors = o3d.utility.Vector3dVector(pseudo_colors)
            geometries.append(ls_p)

        if gt_lines is not None:
            ls_g = o3d.geometry.LineSet()
            pts = gt_lines.reshape(-1, 3)
            unique_pts, inv = np.unique(pts, axis=0, return_inverse=True)
            ls_g.points = o3d.utility.Vector3dVector(unique_pts)
            lines_idx = inv.reshape(-1, 2)
            ls_g.lines = o3d.utility.Vector2iVector(lines_idx)
            ls_g.colors = o3d.utility.Vector3dVector(gt_colors)
            geometries.append(ls_g)

        if not geometries:
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Sparse Voxels Iter {curr_iter+1}", width=1280, height=720)
        # Add axis at origin, size proportional to scene extent
        extent = max(pcr[3] - pcr[0], pcr[4] - pcr[1], pcr[5] - pcr[2])
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=extent * 0.05)
        vis.add_geometry(axis)
        for g in geometries:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.destroy_window()

        # Disable caching after use
        ref.enable_visual_debug = False

