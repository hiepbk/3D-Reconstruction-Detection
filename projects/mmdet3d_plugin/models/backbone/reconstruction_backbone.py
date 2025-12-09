"""
Reconstruction Backbone for ResDet3D.
Wraps DepthAnything3 to generate point clouds from multi-view images.
"""

import numpy as np
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from mmdet.models.builder import BACKBONES
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import Compose


from projects.mmdet3d_plugin.models.depth_anything_3.api import DepthAnything3
from projects.mmdet3d_plugin.models.depth_anything_3.specs import Prediction
from projects.mmdet3d_plugin.datasets.pipelines.respoint_post_processing import ResPointCloudPipeline


@BACKBONES.register_module()
class ReconstructionBackbone(nn.Module):
    """Reconstruction backbone that generates point clouds from multi-view images.
    
    This backbone:
    1. Takes multi-view images from mmdet3d data pipeline
    2. Runs DepthAnything3 forward to get depth maps
    3. Back-projects depth maps to 3D point clouds
    4. Transforms points from camera to LiDAR coordinates using lidar2img
    5. Applies post-processing pipeline (voxel, ball_query, FPS)
    6. Returns point cloud in same format as bin file (numpy array or tensor)
    """
    
    def __init__(
        self,
        pretrained: str,
        cache_dir: Optional[str] = None,
        respoint_post_processing_pipeline: Optional[List[Dict]] = None,
        glb_config: Optional[Dict] = None,
        ref_view_strategy: str = "saddle_balanced",
        use_ray_pose: bool = False,
        max_points: int = 1_000_000,
        filter_sky: bool = True,
        max_depth: Optional[float] = None,
        conf_thresh_percentile: Optional[float] = None,
    ):
        """Initialize ReconstructionBackbone.
        
        Args:
            pretrained: Pretrained DepthAnything3 model name or path
            cache_dir: Cache directory for model
            post_processing_pipeline: List of post-processing step configs
            glb_config: GLB export config (for filtering)
            ref_view_strategy: Reference view selection strategy
            use_ray_pose: Use ray-based pose estimation
            max_points: Maximum number of points
            filter_sky: Filter sky regions
            max_depth: Maximum depth threshold
            conf_thresh_percentile: Confidence threshold percentile
        """
        super().__init__()
        
        # Store cache directory
        self.cache_dir = cache_dir
        
        # Build DepthAnything3 model
        print(f"[DEBUG] ReconstructionBackbone: Loading DepthAnything3 model from {pretrained}")
        print(f"[DEBUG] ReconstructionBackbone: cache_dir = {cache_dir}")
        self.da3_model = DepthAnything3.from_pretrained(pretrained, cache_dir=cache_dir)
        print(f"[DEBUG] ReconstructionBackbone: DepthAnything3 model loaded successfully")
        self.da3_model.eval()
        print(f"[DEBUG] ReconstructionBackbone: Model set to eval mode")
        print(f"[DEBUG] ReconstructionBackbone: Cache directory = {cache_dir}")
        
        # Store config
        self.respoint_post_processing_pipeline_cfg = respoint_post_processing_pipeline
        print(f"[DEBUG] ReconstructionBackbone: Building post-processing pipeline...")
        print(f"[DEBUG] ReconstructionBackbone: Pipeline config length = {len(self.respoint_post_processing_pipeline_cfg) if self.respoint_post_processing_pipeline_cfg else 0}")
        # post-processing pipeline will be built on first forward
        self.post_pipeline = Compose(self.respoint_post_processing_pipeline_cfg)
        print(f"[DEBUG] ReconstructionBackbone: Post-processing pipeline built successfully")
        self.glb_config = glb_config or {}
        self.ref_view_strategy = ref_view_strategy
        self.use_ray_pose = use_ray_pose
        self.max_points = max_points
        self.filter_sky = filter_sky
        self.max_depth = max_depth or self.glb_config.get('max_depth', None)
        self.conf_thresh_percentile = conf_thresh_percentile or self.glb_config.get('conf_thresh_percentile', None)
        

    
    
    def _extract_images_from_data(self, img: torch.Tensor) -> torch.Tensor:
        """Extract images from mmdet3d data format.
        
        Args:
            img: Image tensor from mmdet3d data (could be DataContainer or tensor)
                Shape: (B, N, 3, H, W) or (N, 3, H, W) after unwrapping
        
        Returns:
            Image tensor with shape (B, N, 3, H, W) where B=1
        """
        # Handle DataContainer
        if isinstance(img, DC):
            img = img.data[0]  # Get first batch item
        
        # Ensure batch dimension
        if img.dim() == 4:  # (N, 3, H, W)
            img = img.unsqueeze(0)  # (1, N, 3, H, W)
        
        return img
    
    def _extract_lidar2img_from_data(self, img_metas: List[Dict]) -> List[np.ndarray]:
        """Extract lidar2img transformations from mmdet3d data.
        
        Args:
            img_metas: List of image metadata dicts
        
        Returns:
            List of lidar2img 4x4 matrices (one per camera)
        """
        if not img_metas or len(img_metas) == 0:
            return []
        
        # Handle DataContainer
        if isinstance(img_metas, DC):
            img_metas = img_metas.data[0]
        
        # Get lidar2img from first metadata (should be same for all in batch)
        meta = img_metas[0] if isinstance(img_metas, list) else img_metas
        lidar2img = meta.get('lidar2img', [])
        
        if isinstance(lidar2img, list):
            return [np.array(l2i) for l2i in lidar2img]
        return []
    
    def _backproject_depth_to_points(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        max_depth: Optional[float] = None,
        conf: Optional[np.ndarray] = None,
        conf_thresh: Optional[float] = None,
        sky_mask: Optional[np.ndarray] = None,
        filter_sky: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Back-project depth map to 3D points in camera coordinates.
        
        Args:
            depth: Depth map (H, W)
            intrinsics: Camera intrinsics (3, 3)
            max_depth: Maximum depth threshold
            conf: Confidence map (H, W)
            conf_thresh: Confidence threshold
            sky_mask: Sky mask (H, W), True=sky
            filter_sky: Whether to filter sky
        
        Returns:
            Tuple of (points_cam, colors) where:
            - points_cam: (N, 3) points in camera coordinates
            - colors: (N, 3) colors or None
        """
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Back-project to 3D
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        points_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # Filter valid points
        valid = depth.flatten() > 0
        valid = valid & np.isfinite(depth.flatten())
        
        if max_depth is not None:
            valid = valid & (depth.flatten() <= max_depth)
        
        if conf is not None and conf_thresh is not None:
            conf_flat = conf.flatten()
            valid = valid & (conf_flat >= conf_thresh)
        
        if filter_sky and sky_mask is not None:
            sky_flat = sky_mask.flatten()
            valid = valid & (~sky_flat)  # Keep non-sky
        
        points_cam_valid = points_cam[valid]
        
        # Colors would come from processed_images if available
        colors = None
        
        return points_cam_valid, colors
    
    def _transform_points_cam_to_lidar(
        self,
        points_cam: np.ndarray,
        lidar2img: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        """Transform points from camera to LiDAR coordinates.
        
        Args:
            points_cam: Points in camera coordinates (N, 3)
            lidar2img: LiDAR to image transformation (4, 4)
            intrinsics: Camera intrinsics (3, 3)
        
        Returns:
            Points in LiDAR coordinates (N, 3)
        """
        # From nuscenes_dataset.py:
        # lidar2cam_r = inv(sensor2lidar_rotation)
        # lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
        # lidar2cam_rt[:3, :3] = lidar2cam_r.T
        # lidar2cam_rt[3, :3] = -lidar2cam_t
        # viewpad[:3, :3] = intrinsic
        # lidar2img = viewpad @ lidar2cam_rt.T
        
        # To extract cam2lidar:
        # 1. Extract lidar2cam_rt from lidar2img
        viewpad = np.eye(4)
        viewpad[:3, :3] = intrinsics
        viewpad_inv = np.linalg.inv(viewpad)
        lidar2cam_rt_T = viewpad_inv @ lidar2img
        lidar2cam_rt = lidar2cam_rt_T.T
        
        # 2. Extract rotation and translation from lidar2cam_rt
        # lidar2cam_rt[:3, :3] = lidar2cam_r.T
        # lidar2cam_rt[3, :3] = -lidar2cam_t
        lidar2cam_r_T = lidar2cam_rt[:3, :3]
        lidar2cam_r = lidar2cam_r_T.T
        lidar2cam_t = -lidar2cam_rt[3, :3]
        
        # 3. Compute cam2lidar
        cam2lidar_r = lidar2cam_r.T
        cam2lidar_t = -lidar2cam_r.T @ lidar2cam_t
        
        # Transform points: points_lidar = points_cam @ R.T + T
        points_lidar = points_cam @ cam2lidar_r.T + cam2lidar_t
        
        return points_lidar
    
    def forward(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Forward pass: generate point cloud from images.
        
        Args:
            img: Multi-view images (B, N, 3, H, W) or DataContainer
            img_metas: Image metadata list
            return_loss: Whether to return loss (unused, for compatibility)
        
        Returns:
            Point cloud tensor (N, 3) or (N, 4/5) in LiDAR coordinates
            Same format as point cloud from bin file
        """
        device = next(self.parameters()).device
        
        # Extract images
        img_tensor = self._extract_images_from_data(img)
        B, N, C, H, W = img_tensor.shape
        assert B == 1, "Batch size must be 1"
        
        # Extract lidar2img transformations
        lidar2img_list = self._extract_lidar2img_from_data(img_metas)
        
        # Run DepthAnything3 forward
        # DepthAnything3.forward expects (B, N, 3, H, W) on model device
        da3_device = next(self.da3_model.parameters()).device
        img_for_da3 = img_tensor.to(da3_device)
        
        with torch.no_grad():
            da3_output = self.da3_model.forward(
                image=img_for_da3,
                extrinsics=None,  # Let model estimate
                intrinsics=None,  # Let model estimate
                use_ray_pose=self.use_ray_pose,
                ref_view_strategy=self.ref_view_strategy,
            )
        
        # Convert output to Prediction object
        # da3_output is a dict, we need to create Prediction
        # Use the same method as DepthAnything3.inference
        prediction = self.da3_model._convert_to_prediction(da3_output)
        
        # Compute confidence threshold if needed
        conf_thresh = None
        if self.conf_thresh_percentile is not None and hasattr(prediction, 'conf') and prediction.conf is not None:
            sky = getattr(prediction, 'sky', None)
            if sky is not None and (~sky).sum() > 10:
                conf_pixels = prediction.conf[~sky]
            else:
                conf_pixels = prediction.conf.flatten()
            conf_thresh = np.percentile(conf_pixels, self.conf_thresh_percentile)
        
        # Back-project depth maps to point clouds and transform to LiDAR
        all_points_lidar = []
        all_colors = []
        
        for i in range(len(prediction.depth)):
            depth = prediction.depth[i]
            intrinsics = prediction.intrinsics[i] if hasattr(prediction, 'intrinsics') else None
            
            if intrinsics is None:
                raise ValueError(f"Intrinsics not available for view {i}")
            
            # Get sky mask and confidence for this view
            sky_mask = prediction.sky[i] if hasattr(prediction, 'sky') and prediction.sky is not None else None
            conf = prediction.conf[i] if hasattr(prediction, 'conf') and prediction.conf is not None else None
            
            # Back-project to camera coordinates
            points_cam, colors = self._backproject_depth_to_points(
                depth,
                intrinsics,
                max_depth=self.max_depth,
                conf=conf,
                conf_thresh=conf_thresh,
                sky_mask=sky_mask,
                filter_sky=self.filter_sky,
            )
            
            if len(points_cam) == 0:
                raise ValueError(f"No points were generated for view {i}")
            
            # Transform to LiDAR coordinates
            if i < len(lidar2img_list):
                points_lidar = self._transform_points_cam_to_lidar(
                    points_cam,
                    lidar2img_list[i],
                    intrinsics,
                )
            else:
                raise ValueError(f"No lidar2img transformation available for view {i}")
            
            all_points_lidar.append(points_lidar)
            if colors is not None:
                all_colors.append(colors)
        
        if not all_points_lidar:
            raise ValueError("No points were generated")
        
        # Concatenate all points
        combined_points = np.concatenate(all_points_lidar, axis=0)
        combined_colors = None
        if all_colors and all(all_colors):
            combined_colors = np.concatenate(all_colors, axis=0)
        
        # Build post-processing pipeline if needed
        self._build_post_pipeline(device)
        
        # Apply post-processing
        if self.post_pipeline is not None:
            pipeline_input = {
                'points': combined_points,
                'colors': combined_colors,
                'polygon_mask': None,
                'indices': None,
            }
            # ResPointCloudPipeline is callable directly
            pipeline_output = self.post_pipeline(pipeline_input)
            combined_points = pipeline_output['points']
            combined_colors = pipeline_output.get('colors')
        
        # Convert to tensor and return in same format as bin file
        # Bin file format: (N, 3) for xyz or (N, 4) for xyz+intensity or (N, 5) for xyz+intensity+ring
        points_tensor = torch.from_numpy(combined_points).float().to(device)
        
        # If colors available, could append as intensity
        # For now, return just xyz (N, 3) to match standard point cloud format
        return points_tensor

