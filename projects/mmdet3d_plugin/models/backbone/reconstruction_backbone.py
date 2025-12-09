"""
Reconstruction Backbone for ResDet3D.
Inherits from DepthAnything3 to generate point clouds from multi-view images.
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
class ReconstructionBackbone(DepthAnything3):
    """Reconstruction backbone that generates point clouds from multi-view images.
    
    Inherits from DepthAnything3 to reuse its functionality (input_processor, model, etc.)
    and overrides forward() to work with mmdet3d's data format.
    
    This backbone:
    1. Takes multi-view images from mmdet3d data pipeline
    2. Uses DepthAnything3's forward() to get depth maps
    3. Back-projects depth maps to 3D point clouds
    4. Transforms points from camera to LiDAR coordinates using lidar2img
    5. Applies post-processing pipeline (voxel, ball_query, FPS)
    6. Returns point cloud in same format as bin file (numpy array or tensor)
    """
    
    def __new__(cls, pretrained: str, cache_dir: Optional[str] = None, **kwargs):
        """Create instance using from_pretrained to avoid temporary instance."""
        # Use parent's from_pretrained to create the instance with pretrained weights
        # This already calls __init__ on the DepthAnything3 instance
        instance = DepthAnything3.from_pretrained(pretrained, cache_dir=cache_dir)
        # Change the class to ReconstructionBackbone
        instance.__class__ = cls
        return instance
    
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
            pretrained: Pretrained DepthAnything3 model name or path (HuggingFace Hub identifier)
            cache_dir: Cache directory for model
            respoint_post_processing_pipeline: List of post-processing step configs
            glb_config: GLB export config (for filtering)
            ref_view_strategy: Reference view selection strategy
            use_ray_pose: Use ray-based pose estimation
            max_points: Maximum number of points
            filter_sky: Filter sky regions
            max_depth: Maximum depth threshold
            conf_thresh_percentile: Confidence threshold percentile
        """
        # Note: __new__ already created and initialized the instance using from_pretrained
        # So self.model, self.config, self.input_processor, etc. are already set up
        # We just need to set ReconstructionBackbone-specific attributes
        # Check if already initialized (from __new__) to avoid re-initialization
        if hasattr(self, 'model') and hasattr(self, 'input_processor'):
            # Already initialized by from_pretrained in __new__
            print(f"[DEBUG] ReconstructionBackbone: Model already loaded from {pretrained}")
        else:
            # This shouldn't happen, but handle it just in case
            raise RuntimeError("ReconstructionBackbone instance not properly initialized")
        
        self.eval()  # Ensure eval mode
        print(f"[DEBUG] ReconstructionBackbone: Model set to eval mode")
        
        # Store ReconstructionBackbone-specific config
        self.respoint_post_processing_pipeline_cfg = respoint_post_processing_pipeline
        print(f"[DEBUG] ReconstructionBackbone: Building post-processing pipeline...")
        print(f"[DEBUG] ReconstructionBackbone: Pipeline config length = {len(self.respoint_post_processing_pipeline_cfg) if self.respoint_post_processing_pipeline_cfg else 0}")
        # Build post-processing pipeline using Compose
        self.post_pipeline = Compose(self.respoint_post_processing_pipeline_cfg) if self.respoint_post_processing_pipeline_cfg else None
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
            Image tensor with shape (B, N, 3, H, W)
        """
        # Handle DataContainer
        if isinstance(img, DC):
            # DataContainer.data is a list of tensors, one per batch item
            # Stack them to get (B, N, 3, H, W)
            img_list = img.data
            if isinstance(img_list, list):
                img = torch.stack(img_list, dim=0)  # (B, N, 3, H, W)
            else:
                img = img_list  # Already a tensor
        
        # Ensure batch dimension
        if img.dim() == 4:  # (N, 3, H, W)
            img = img.unsqueeze(0)  # (1, N, 3, H, W)
        
        return img
    
    def _extract_lidar2img_from_meta(self, meta: Dict) -> List[np.ndarray]:
        """Extract lidar2img transformations from a single metadata dict.
        
        Args:
            meta: Image metadata dict for one sample
        
        Returns:
            List of lidar2img 4x4 matrices (one per camera)
        """
        if not meta:
            return []
        
        lidar2img = meta.get('lidar2img', [])
        
        if isinstance(lidar2img, list):
            return [np.array(l2i) for l2i in lidar2img]
        return []
    
    def _backproject_depth_to_points(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        image: Optional[np.ndarray] = None,
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
            image: Processed image (H, W, 3) in [0, 255] range, optional
            max_depth: Maximum depth threshold
            conf: Confidence map (H, W)
            conf_thresh: Confidence threshold
            sky_mask: Sky mask (H, W), True=sky
            filter_sky: Whether to filter sky
        
        Returns:
            Tuple of (points_cam, colors) where:
            - points_cam: (N, 3) points in camera coordinates
            - colors: (N, 3) colors in [0, 1] range or None
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
        
        # Extract colors from image if available
        colors = None
        if image is not None:
            # image shape should be (H, W, 3) in [0, 255] range
            # Reshape to (H*W, 3) and filter by valid mask
            colors_flat = image.reshape(-1, 3)[valid]
            # Normalize to [0, 1] range
            colors = colors_flat.astype(np.float32) / 255.0
        
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
        use_image_paths: bool = False,  # Debug mode: use image paths instead of preprocessed tensors
    ) -> List[torch.Tensor]:
        """Forward pass: generate point cloud from images.
        
        Overrides DepthAnything3.forward() to work with mmdet3d's data format.
        Uses parent's forward() method to get depth maps, then back-projects to point clouds.
        Supports batch size > 1 for training.
        
        Args:
            img: Multi-view images (B, N, 3, H, W) or DataContainer
            img_metas: Image metadata list (one dict per batch item)
            return_loss: Whether to return loss (unused, for compatibility)
            use_image_paths: If True, extract image paths and use inference() method directly
                            (bypasses preprocessing, useful for debugging)
        
        Returns:
            List of point cloud tensors, one per batch item
            Each tensor has shape (N_points, 3) in LiDAR coordinates
            Same format as point cloud from bin file
        """
        device = next(self.parameters()).device
        
        # Handle DataContainer for img_metas
        if isinstance(img_metas, DC):
            img_metas = img_metas.data
        
        # Debug mode: use image paths directly with inference() method
        if use_image_paths:
            return self._forward_with_image_paths(img_metas, device)
        
        # Extract images from mmdet3d data format
        img_tensor = self._extract_images_from_data(img)
        B, N, C, H, W = img_tensor.shape
        
        # Process each batch item separately
        batch_point_clouds = []
        
        # Process all batch items at once using tensor path (more efficient)
        # InputProcessor now supports tensor inputs directly
        # Note: input_processor returns tensors on the same device as input
        imgs_processed, _, _ = self.input_processor(
            image=img_tensor,  # Pass tensor directly: (B, N, 3, H, W)
            extrinsics=None,
            intrinsics=None,
            process_res=504,  # Default processing resolution
            process_res_method="upper_bound_resize",
        )
        # imgs_processed shape: (B, N, 3, H_new, W_new) on same device as input
        
        # Process each batch item separately for point cloud generation
        for b_idx in range(B):
            # Extract images for this batch item
            imgs_batch = imgs_processed[b_idx]  # (N, 3, H_new, W_new)
            
            # Store CPU version for adding processed images later (matches inference())
            # _add_processed_images expects CPU tensors
            imgs_batch_cpu = imgs_batch.cpu() if imgs_batch.is_cuda else imgs_batch
            
            # Prepare model inputs exactly like inference() does:
            # 1. Move to model device
            # 2. Add batch dimension [None] to make (1, N, 3, H, W)
            # 3. Convert to float
            device = next(self.parameters()).device
            imgs_for_da3 = imgs_batch.to(device, non_blocking=True)[None].float()  # (1, N, 3, H, W)
            
            # Extract metadata for this batch item
            meta_batch = img_metas[b_idx] if isinstance(img_metas, list) else img_metas
            
            # Extract lidar2img transformations for this batch item
            lidar2img_list = self._extract_lidar2img_from_meta(meta_batch)
            
            # Run DepthAnything3 forward (parent's forward method)
            # This matches what _run_model_forward does in inference()
            with torch.no_grad():
                da3_output = super().forward(
                    image=imgs_for_da3,
                    extrinsics=None,  # Let model estimate
                    intrinsics=None,  # Let model estimate
                    export_feat_layers=[],  # No feature export needed
                    infer_gs=False,  # No Gaussian Splatting
                    use_ray_pose=self.use_ray_pose,
                    ref_view_strategy=self.ref_view_strategy,
                )
            
            # Convert output to Prediction object (matches inference())
            prediction = self._convert_to_prediction(da3_output)
            
            # Add processed images to prediction (matches inference())
            # _add_processed_images expects (N, 3, H, W) on CPU
            prediction = self._add_processed_images(prediction, imgs_batch_cpu)
            
            # Get processed images for color extraction
            processed_images = None
            if hasattr(prediction, 'processed_images') and prediction.processed_images is not None:
                processed_images = prediction.processed_images  # (N, H, W, 3) in [0, 255]
            
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
                    raise ValueError(f"Intrinsics not available for view {i} in batch {b_idx}")
                
                # Get sky mask and confidence for this view
                sky_mask = prediction.sky[i] if hasattr(prediction, 'sky') and prediction.sky is not None else None
                conf = prediction.conf[i] if hasattr(prediction, 'conf') and prediction.conf is not None else None
                
                # Get image for color extraction
                image = None
                if processed_images is not None and i < len(processed_images):
                    image = processed_images[i]  # (H, W, 3) in [0, 255]
                
                # Back-project to camera coordinates
                points_cam, colors = self._backproject_depth_to_points(
                    depth,
                    intrinsics,
                    image=image,  # Pass image for color extraction
                    max_depth=self.max_depth,
                    conf=conf,
                    conf_thresh=conf_thresh,
                    sky_mask=sky_mask,
                    filter_sky=self.filter_sky,
                )
                
                if len(points_cam) == 0:
                    raise ValueError(f"No points were generated for view {i} in batch {b_idx}")
                
                # Transform to LiDAR coordinates
                if i < len(lidar2img_list):
                    points_lidar = self._transform_points_cam_to_lidar(
                        points_cam,
                        lidar2img_list[i],
                        intrinsics,
                    )
                else:
                    raise ValueError(f"No lidar2img transformation available for view {i} in batch {b_idx}")
                
                all_points_lidar.append(points_lidar)
                if colors is not None:
                    all_colors.append(colors)
            
            if not all_points_lidar:
                raise ValueError(f"No points were generated for batch {b_idx}")
            
            # Concatenate all points for this batch item
            combined_points = np.concatenate(all_points_lidar, axis=0)
            combined_colors = None
            # Check if all_colors has elements and all are not None
            if all_colors and all(c is not None for c in all_colors):
                combined_colors = np.concatenate(all_colors, axis=0)
            
            # Apply post-processing pipeline
            if self.post_pipeline is not None:
                pipeline_input = {
                    'points': combined_points,
                    'colors': combined_colors,
                    'polygon_mask': None,
                    'indices': None,
                }
                pipeline_output = self.post_pipeline(pipeline_input)
                combined_points = pipeline_output['points']
                combined_colors = pipeline_output.get('colors')
            
            # Convert to tensor and add to batch results
            points_tensor = torch.from_numpy(combined_points).float().to(device)
            batch_point_clouds.append(points_tensor)
            
            # Store colors if available (for later access in inference)
            if not hasattr(self, '_batch_colors'):
                self._batch_colors = []
            if combined_colors is not None:
                # Store as numpy array (will be converted to tensor later if needed)
                self._batch_colors.append(combined_colors)
            else:
                self._batch_colors.append(None)
        
        # Return list of point clouds, one per batch item
        # This matches mmdet3d's convention: list[torch.Tensor] where each tensor is (N_points, 3)
        return batch_point_clouds
