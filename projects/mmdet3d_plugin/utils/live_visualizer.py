import threading
import queue
import numpy as np
import open3d as o3d
from typing import Optional


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


class LiveVoxelVisualizer:
    """Lightweight live Open3D visualizer fed by a global queue."""

    def __init__(self, window_name="Training", width=1280, height=720):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.queue = queue.Queue(maxsize=2)
        self.thread = None
        self.running = False
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.axis = None
        self.ls_bbox = None
        self.ls_pseudo = None
        self.ls_gt = None
        self.last_frame = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None

    def enqueue(self, frame: dict):
        if frame is None:
            return
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass

    def _loop(self):
        while self.running:
            try:
                frame = self.queue.get(timeout=0.1)
                self.last_frame = frame
            except queue.Empty:
                frame = self.last_frame
            if frame is None:
                continue
            self._render(frame)

    def _render(self, frame: dict):
        pseudo_coors = frame.get("pseudo_coors")
        gt_coors = frame.get("gt_coors")
        voxel_size = frame.get("voxel_size", np.array([1.0, 1.0, 1.0], dtype=np.float32))
        pcr = frame.get("pcr", np.array([-1, -1, -1, 1, 1, 1], dtype=np.float32))

        pseudo_lines = pseudo_colors = None
        gt_lines = gt_colors = None

        if pseudo_coors is not None:
            pseudo_lines, pseudo_colors = _make_voxel_lines(pseudo_coors, voxel_size, pcr, (1, 0, 0))
        if gt_coors is not None:
            gt_lines, gt_colors = _make_voxel_lines(gt_coors, voxel_size, pcr, (0, 1, 0))

        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(self.window_name, self.width, self.height, visible=True)
            extent = max(pcr[3] - pcr[0], pcr[4] - pcr[1], pcr[5] - pcr[2])
            self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=extent * 0.05)
            bbox_lines, bbox_colors = _make_bbox_lines(pcr, (1, 1, 0))
            ls_bbox = o3d.geometry.LineSet()
            pts_bbox = bbox_lines.reshape(-1, 3)
            lines_idx = np.arange(len(pts_bbox), dtype=np.int32).reshape(-1, 2)
            ls_bbox.points = o3d.utility.Vector3dVector(pts_bbox)
            ls_bbox.lines = o3d.utility.Vector2iVector(lines_idx)
            ls_bbox.colors = o3d.utility.Vector3dVector(
                np.tile(bbox_colors[0:1], (lines_idx.shape[0], 1)))
            self.vis.add_geometry(self.axis)
            self.vis.add_geometry(ls_bbox)
            self.ls_bbox = ls_bbox

        if pseudo_lines is not None:
            pts = pseudo_lines.reshape(-1, 3)
            lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
            if self.ls_pseudo is None:
                self.ls_pseudo = o3d.geometry.LineSet()
                self.vis.add_geometry(self.ls_pseudo, reset_bounding_box=False)
            self.ls_pseudo.points = o3d.utility.Vector3dVector(pts)
            self.ls_pseudo.lines = o3d.utility.Vector2iVector(lines_idx)
            self.ls_pseudo.colors = o3d.utility.Vector3dVector(
                np.tile(pseudo_colors[0:1], (lines_idx.shape[0], 1)))
            self.vis.update_geometry(self.ls_pseudo)

        if gt_lines is not None:
            pts = gt_lines.reshape(-1, 3)
            lines_idx = np.arange(len(pts), dtype=np.int32).reshape(-1, 2)
            if self.ls_gt is None:
                self.ls_gt = o3d.geometry.LineSet()
                self.vis.add_geometry(self.ls_gt, reset_bounding_box=False)
            self.ls_gt.points = o3d.utility.Vector3dVector(pts)
            self.ls_gt.lines = o3d.utility.Vector2iVector(lines_idx)
            self.ls_gt.colors = o3d.utility.Vector3dVector(
                np.tile(gt_colors[0:1], (lines_idx.shape[0], 1)))
            self.vis.update_geometry(self.ls_gt)

        self.vis.poll_events()
        self.vis.update_renderer()


_live_vis: Optional[LiveVoxelVisualizer] = None


def get_live_visualizer():
    global _live_vis
    if _live_vis is None:
        _live_vis = LiveVoxelVisualizer()
        _live_vis.start()
    return _live_vis


def enqueue_live_frame(frame: dict):
    vis = get_live_visualizer()
    vis.enqueue(frame)

