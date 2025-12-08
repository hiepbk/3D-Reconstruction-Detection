"""
Post-processing pipeline utilities (mmdet3d-style).

Each step is a small processor class with a __call__ that takes a data dict:
    {
        'points': np.ndarray (N, 3),
        'colors': np.ndarray (N, 3) or None,
        'polygon_mask': np.ndarray (N,) or None,
        'indices': np.ndarray (N,) mapping to original inputs
    }

Processors should return an updated data dict with the same keys.
"""

from typing import List, Dict, Any, Callable
import numpy as np
import torch


class BaseProcessor:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class VoxelDownsample(BaseProcessor):
    def __init__(self, voxel_size, point_cloud_range, downsample_fn: Callable):
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.downsample_fn = downsample_fn

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.voxel_size is None:
            return data
        return self.downsample_fn(
            data['points'],
            colors=data.get('colors'),
            polygon_mask=data.get('polygon_mask'),
            voxel_size=self.voxel_size,
            use_fps=False,
            fps_num_points=None,
            point_cloud_range=self.point_cloud_range,
            use_ball_query=False,
        )


class BallQueryDownsample(BaseProcessor):
    def __init__(self, cfg: Dict[str, Any], downsample_fn: Callable, device: torch.device):
        self.enabled = cfg.get('enabled', True)
        self.min_radius = cfg.get('min_radius', 0.0)
        self.max_radius = cfg.get('max_radius', 0.5)
        self.sample_num = cfg.get('sample_num', 16)
        self.anchor_points = cfg.get('anchor_points', None)
        self.downsample_fn = downsample_fn
        self.device = device

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.anchor_points is None:
            return data
        if self.device.type != "cuda":
            print(f"  Warning: BallQueryDownsample requires CUDA but device is {self.device.type}, skipping...")
            return data
        return self.downsample_fn(
            data['points'],
            colors=data.get('colors'),
            polygon_mask=data.get('polygon_mask'),
            voxel_size=None,
            use_fps=False,
            fps_num_points=None,
            point_cloud_range=None,
            use_ball_query=True,
            ball_query_min_radius=self.min_radius,
            ball_query_max_radius=self.max_radius,
            ball_query_sample_num=self.sample_num,
            ball_query_anchor_points=self.anchor_points,
        )


class FPSDownsample(BaseProcessor):
    def __init__(self, cfg: Dict[str, Any], downsample_fn: Callable):
        self.enabled = cfg.get('enabled', True)
        self.num_points = cfg.get('num_points', None)
        self.downsample_fn = downsample_fn

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.num_points is None:
            return data
        return self.downsample_fn(
            data['points'],
            colors=data.get('colors'),
            polygon_mask=data.get('polygon_mask'),
            voxel_size=None,
            use_fps=True,
            fps_num_points=self.num_points,
            point_cloud_range=None,
            use_ball_query=False,
        )


def build_post_processing_pipeline(pipeline_cfg: List[Dict[str, Any]], downsample_fn: Callable, device: torch.device):
    """
    Build processors by matching step['type'] to the corresponding class.
    No elif chain; just a registry-style mapping.
    """
    registry = {
        "VoxelDownsample": lambda step: VoxelDownsample(
            voxel_size=step.get('voxel_size', None),
            point_cloud_range=step.get('point_cloud_range', None),
            downsample_fn=downsample_fn,
        ),
        "BallQueryDownsample": lambda step: BallQueryDownsample(
            cfg=step,
            downsample_fn=downsample_fn,
            device=device,
        ),
        "FPSDownsample": lambda step: FPSDownsample(
            cfg=step,
            downsample_fn=downsample_fn,
        ),
    }

    processors = []
    for step in pipeline_cfg or []:
        step_type = step.get('type')
        if not step_type:
            continue
        builder = registry.get(step_type)
        if builder is None:
            print(f"  Warning: Unknown post-processing step type '{step_type}', skipping...")
            continue
        processors.append(builder(step))
    return processors


def run_post_processing_pipeline(data: Dict[str, Any], processors: List[BaseProcessor]) -> Dict[str, Any]:
    """
    Run a list of processors sequentially, preserving index mapping.
    """
    if data is None or 'points' not in data or data['points'] is None:
        return data

    result = {
        'points': data['points'],
        'colors': data.get('colors', None),
        'polygon_mask': data.get('polygon_mask', None),
        'indices': data.get('indices', None),
    }
    if result['indices'] is None:
        result['indices'] = np.arange(len(result['points']))

    for proc in processors:
        res = proc(result)
        if res is None:
            continue

        base_indices = result.get('indices', np.arange(len(result['points'])))
        step_indices = res.get('indices', np.arange(len(res.get('points', []))))
        mapped_indices = base_indices[step_indices] if len(base_indices) > 0 else step_indices

        new_colors = res.get('colors', None)
        if new_colors is None and result.get('colors') is not None and step_indices is not None:
            new_colors = result['colors'][step_indices]

        new_polygon_mask = res.get('polygon_mask', None)
        if new_polygon_mask is None and result.get('polygon_mask') is not None and step_indices is not None:
            new_polygon_mask = result['polygon_mask'][step_indices]

        result = {
            'points': res.get('points', result['points']),
            'colors': new_colors,
            'polygon_mask': new_polygon_mask,
            'indices': mapped_indices
        }

    return result


