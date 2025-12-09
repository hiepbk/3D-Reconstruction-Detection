"""
Reconstruction Point Cloud Post-processing Pipeline (mmdet3d-style).

This module provides pipeline transforms for post-processing point clouds
generated from DepthAnything3 reconstruction.

Each transform is registered with @PIPELINES.register_module() and can be used
directly in mmdet3d's data pipeline configuration.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from scipy.spatial import cKDTree
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmdet3d.ops import Voxelization, furthest_point_sample, ball_query


@PIPELINES.register_module()
class VoxelDownsample:
    """Voxel-based downsampling of point clouds.
    
    Divides space into voxels and represents each voxel by its centroid.
    """
    def __init__(self, voxel_size=None, point_cloud_range=None, device: torch.device = None):
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.device = device

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.voxel_size is None:
            return data
        
        points = data['points']
        colors = data.get('colors')
        polygon_mask = data.get('polygon_mask')

        points_tensor = torch.from_numpy(points).float()

        if self.point_cloud_range is None:
            x_min, y_min, z_min = points.min(axis=0) - 1.0
            x_max, y_max, z_max = points.max(axis=0) + 1.0
            pcr = [x_min, y_min, z_min, x_max, y_max, z_max]
        else:
            pcr = self.point_cloud_range

        voxel_size = self.voxel_size
        if isinstance(voxel_size, (int, float)):
            voxel_size = [voxel_size, voxel_size, voxel_size]

        voxel_layer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pcr,
            max_num_points=100,
            max_voxels=200000
        )
        voxel_layer.eval()

        with torch.no_grad():
            voxels, coors, num_points_per_voxel = voxel_layer(points_tensor)

        num_voxels = voxels.shape[0]
        voxel_centers = []
        for i in range(num_voxels):
            n_points = num_points_per_voxel[i].item()
            if n_points > 0:
                voxel_points = voxels[i, :n_points, :]
                center = voxel_points.mean(dim=0).numpy()
                voxel_centers.append(center)

        if len(voxel_centers) == 0:
            return {
                'points': points,
                'colors': colors,
                'polygon_mask': polygon_mask,
                'indices': np.arange(len(points))
            }

        voxel_centers = np.array(voxel_centers)

        if colors is not None or polygon_mask is not None:
            tree = cKDTree(points)
            _, closest_indices = tree.query(voxel_centers, k=1)
            voxel_colors = colors[closest_indices] if colors is not None else None
            voxel_polygon_mask = polygon_mask[closest_indices] if polygon_mask is not None else None
        else:
            voxel_colors = None
            voxel_polygon_mask = None
            closest_indices = np.arange(len(voxel_centers))

        return {
            'points': voxel_centers,
            'colors': voxel_colors,
            'polygon_mask': voxel_polygon_mask,
            'indices': closest_indices,
        }


@PIPELINES.register_module()
class BallQueryDownsample:
    """Density-aware downsampling using ball query.
    
    Uses FPS to get anchor points, then ball_query to find neighbors within radius.
    This preserves more points in dense regions and fewer in sparse regions.
    """
    def __init__(self, enabled=True, min_radius=0.0, max_radius=0.5, sample_num=16, anchor_points=None, device: torch.device = None):
        self.enabled = enabled
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.sample_num = sample_num
        self.anchor_points = anchor_points
        self.device = device

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.anchor_points is None:
            return data
        if self.device is None or self.device.type != "cuda":
            print(f"  Warning: BallQueryDownsample requires CUDA but device is {self.device}, skipping...")
            return data

        points = data['points']
        colors = data.get('colors')
        polygon_mask = data.get('polygon_mask')

        points_tensor = torch.from_numpy(points).float()
        device = self.device

        if len(points) <= self.anchor_points:
            return {
                'points': points,
                'colors': colors,
                'polygon_mask': polygon_mask,
                'indices': np.arange(len(points))
            }

        # FPS to get anchors
        points_for_fps = points_tensor.to(device).unsqueeze(0)
        anchor_indices = furthest_point_sample(points_for_fps, self.anchor_points).squeeze(0)
        anchor_points = points[anchor_indices.cpu().numpy()]

        # ball_query to gather neighbors
        points_tensor_gpu = points_tensor.to(device).unsqueeze(0).contiguous()
        anchor_tensor_gpu = torch.from_numpy(anchor_points).float().to(device).unsqueeze(0).contiguous()
        with torch.no_grad():
            ball_query_indices = ball_query(
                self.min_radius,
                self.max_radius,
                self.sample_num,
                points_tensor_gpu,
                anchor_tensor_gpu
            )
        ball_query_indices = ball_query_indices.squeeze(0).cpu().numpy().flatten()
        valid_mask = (ball_query_indices >= 0) & (ball_query_indices < len(points))
        unique_indices = np.unique(ball_query_indices[valid_mask])
        anchor_indices_np = anchor_indices.cpu().numpy()
        final_indices = np.unique(np.concatenate([unique_indices, anchor_indices_np]))

        downsampled_points = points[final_indices]
        downsampled_colors = colors[final_indices] if colors is not None else None
        downsampled_polygon_mask = polygon_mask[final_indices] if polygon_mask is not None else None

        return {
            'points': downsampled_points,
            'colors': downsampled_colors,
            'polygon_mask': downsampled_polygon_mask,
            'indices': final_indices
        }


@PIPELINES.register_module()
class FPSDownsample:
    """Furthest Point Sampling (FPS) downsampling.
    
    Selects points such that they are maximally distant from each other,
    ensuring uniform coverage of the point cloud.
    """
    def __init__(self, enabled=True, num_points=None, device: torch.device = None):
        self.enabled = enabled
        self.num_points = num_points
        self.device = device

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.num_points is None:
            return data

        points = data['points']
        colors = data.get('colors')
        polygon_mask = data.get('polygon_mask')

        if len(points) <= self.num_points:
            return {
                'points': points,
                'colors': colors,
                'polygon_mask': polygon_mask,
                'indices': np.arange(len(points))
            }

        device = self.device if self.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points_tensor = torch.from_numpy(points).float().to(device).unsqueeze(0)
        fps_indices = furthest_point_sample(points_tensor, self.num_points).squeeze(0).cpu().numpy()

        downsampled_points = points[fps_indices]
        downsampled_colors = colors[fps_indices] if colors is not None else None
        downsampled_polygon_mask = polygon_mask[fps_indices] if polygon_mask is not None else None

        return {
            'points': downsampled_points,
            'colors': downsampled_colors,
            'polygon_mask': downsampled_polygon_mask,
            'indices': fps_indices
        }


@PIPELINES.register_module()
class ResPointCloudPipeline:
    """Reconstruction Point Cloud Post-processing Pipeline.
    
    This pipeline wraps multiple post-processing transforms (VoxelDownsample,
    BallQueryDownsample, FPSDownsample) into a single transform that can be
    used in mmdet3d's data pipeline.
    
    Mimics MultiScaleFlipAug3D's initialization strategy - simply uses Compose
    to handle all transforms, since they're already registered with PIPELINES.
    
    Args:
        transforms (list[dict]): List of transform configs to apply sequentially.
            Each transform should be a dict with 'type' key and transform-specific args.
    
    Example:
        >>> pipeline = ResPointCloudPipeline(
        ...     transforms=[
        ...         dict(type='VoxelDownsample', voxel_size=0.1),
        ...         dict(type='BallQueryDownsample', enabled=True, anchor_points=25000),
        ...         dict(type='FPSDownsample', enabled=True, num_points=40000),
        ...     ]
        ... )
        >>> result = pipeline({'points': points, 'colors': colors})
    """
    
    def __init__(self, transforms: List[Dict[str, Any]]):
        """Initialize ResPointCloudPipeline.
        
        Args:
            transforms: List of transform config dicts
        """
        self.transforms = Compose(transforms)
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the post-processing pipeline to data.
        
        Args:
            data: Dict containing at least 'points' key (np.ndarray, N, 3).
                May also contain 'colors' (N, 3) and 'polygon_mask' (N,).
        
        Returns:
            Dict with same keys as input, but with processed 'points' and
            optionally updated 'colors' and 'polygon_mask'.
        """
        if data is None or 'points' not in data or data['points'] is None:
            return data
        
        # Initialize indices if not present
        if 'indices' not in data or data['indices'] is None:
            data['indices'] = np.arange(len(data['points']))
        
        # Run pipeline (Compose handles everything)
        result = self.transforms(data)
        
        # Ensure result has all expected keys
        if result is None:
            return data
        
        # Ensure indices are present
        if 'indices' not in result:
            result['indices'] = np.arange(len(result.get('points', [])))
        
        return result

