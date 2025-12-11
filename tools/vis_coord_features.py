import argparse
import glob
import os
import pickle
import numpy as np
import open3d as o3d


CROP_SIZE = [500, 500, 20]  # [X,Y,Z] voxels (centered crop); set None to disable


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
    pseudo_feats = data.get("pseudo_features")
    gt_feats = data.get("gt_features")
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
    pseudo_feats = to_np(pseudo_feats)
    gt_feats = to_np(gt_feats)

    def crop_coors(coors: np.ndarray):
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

    def gather_feats(coors: np.ndarray, feats):
        if coors is None or feats is None:
            return None
        feats_np = np.asarray(feats)
        ndim = feats_np.ndim
        # derive downsample ratios so we can map voxel indices to feature map
        if coors is not None and coors.size > 0:
            max_x = coors[:, 3].max()
            max_y = coors[:, 2].max()
            max_z = coors[:, 1].max()
        else:
            max_x = max_y = max_z = 0
        out = []
        for idx in coors:
            b, z, y, x = idx.astype(int)
            if b >= feats_np.shape[0]:
                continue
            if ndim == 5:
                # Expect (B, C, Z, Y, X), map voxel indices to feature grid
                ratio_x = max(1, int(np.ceil((max_x + 1) / feats_np.shape[4])))
                ratio_y = max(1, int(np.ceil((max_y + 1) / feats_np.shape[3])))
                ratio_z = max(1, int(np.ceil((max_z + 1) / feats_np.shape[2])))
                fx = min(feats_np.shape[4] - 1, x // ratio_x)
                fy = min(feats_np.shape[3] - 1, y // ratio_y)
                fz = min(feats_np.shape[2] - 1, z // ratio_z)
                out.append(feats_np[b, :, fz, fy, fx])
            elif ndim == 4:
                # Expect (B, C, H, W) BEV; ignore z, use y,x with downsample ratio
                ratio_x = max(1, int(np.ceil((max_x + 1) / feats_np.shape[3])))
                ratio_y = max(1, int(np.ceil((max_y + 1) / feats_np.shape[2])))
                fx = min(feats_np.shape[3] - 1, x // ratio_x)
                fy = min(feats_np.shape[2] - 1, y // ratio_y)
                out.append(feats_np[b, :, fy, fx])
            else:
                # Unsupported shape
                continue
        if not out:
            return None
        return np.stack(out, axis=0)

    pseudo_feat_vecs = gather_feats(pseudo_coors, pseudo_feats)
    gt_feat_vecs = gather_feats(gt_coors, gt_feats)

    print(f"[INFO] {path}")
    print(f"  pseudo_coors: {None if pseudo_coors is None else pseudo_coors.shape}")
    print(f"  pseudo_feats: {None if pseudo_feats is None else pseudo_feats.shape}")
    print(f"  gathered pseudo_feat_vecs: {None if pseudo_feat_vecs is None else pseudo_feat_vecs.shape}")
    print(f"  gt_coors: {None if gt_coors is None else gt_coors.shape}")
    print(f"  gt_feats: {None if gt_feats is None else gt_feats.shape}")
    print(f"  gathered gt_feat_vecs: {None if gt_feat_vecs is None else gt_feat_vecs.shape}")

    # Print a few coordinate/feature pairs for inspection
    def print_samples(name, coors, feats):
        if coors is None or feats is None:
            return
        n = min(5, coors.shape[0], feats.shape[0])
        print(f"{name}: coors {coors.shape}, feats {feats.shape}")
        print(f"{name} sample pairs (coord -> first 5 dims of feature):")
        for i in range(n):
            print(f"  {coors[i].tolist()} -> {feats[i][:5].tolist()}")
        norms = np.linalg.norm(feats, axis=1)
        print(f"{name} feat norm stats: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        import sys
        sys.stdout.flush()

    print_samples("Pseudo", pseudo_coors, pseudo_feat_vecs)
    print_samples("GT", gt_coors, gt_feat_vecs)
    import sys
    sys.stdout.flush()

    pseudo_lines = pseudo_colors = None
    gt_lines = gt_colors = None
    if pseudo_coors is not None:
        pseudo_lines, pseudo_colors = _make_voxel_lines(pseudo_coors, voxel_size, pcr, (1, 0, 0))
    if gt_coors is not None:
        gt_lines, gt_colors = _make_voxel_lines(gt_coors, voxel_size, pcr, (0, 1, 0))

    def make_feat_pcd(coors, feats, base_color):
        if coors is None or feats is None:
            return None
        # map feature norm to intensity
        centers = np.zeros((coors.shape[0], 3), dtype=np.float32)
        centers[:, 0] = pcr[0] + (coors[:, 3] + 0.5) * voxel_size[0]
        centers[:, 1] = pcr[1] + (coors[:, 2] + 0.5) * voxel_size[1]
        centers[:, 2] = pcr[2] + (coors[:, 1] + 0.5) * voxel_size[2]
        # Same color as voxel wireframe (uniform)
        colors = np.tile(np.asarray(base_color, dtype=np.float32), (centers.shape[0], 1))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # Build feature point clouds (norm-coded)
    pcd_pseudo = make_feat_pcd(pseudo_coors, pseudo_feat_vecs, (1, 0, 0))
    pcd_gt = make_feat_pcd(gt_coors, gt_feat_vecs, (0, 1, 0))

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

    if pseudo_lines is not None:
        ls_p = o3d.geometry.LineSet()
        pts = pseudo_lines.reshape(-1, 3)
        lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
        ls_p.points = o3d.utility.Vector3dVector(pts)
        ls_p.lines = o3d.utility.Vector2iVector(lines_idx)
        ls_p.colors = o3d.utility.Vector3dVector(np.tile(pseudo_colors[0:1], (lines_idx.shape[0], 1)))
        vis.add_geometry(ls_p)

    if gt_lines is not None:
        ls_g = o3d.geometry.LineSet()
        pts = gt_lines.reshape(-1, 3)
        lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
        ls_g.points = o3d.utility.Vector3dVector(pts)
        ls_g.lines = o3d.utility.Vector2iVector(lines_idx)
        ls_g.colors = o3d.utility.Vector3dVector(np.tile(gt_colors[0:1], (lines_idx.shape[0], 1)))
        vis.add_geometry(ls_g)

    if pcd_pseudo is not None:
        vis.add_geometry(pcd_pseudo)
    if pcd_gt is not None:
        vis.add_geometry(pcd_gt)

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

