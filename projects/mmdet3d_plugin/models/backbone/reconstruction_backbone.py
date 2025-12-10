"""
Reconstruction Backbone for ResDet3D.
Wraps DepthAnything3 to generate point clouds from multi-view images.
"""

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from mmdet.models.builder import BACKBONES
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import Compose


from projects.mmdet3d_plugin.models.depth_anything_3.api import DepthAnything3
from projects.mmdet3d_plugin.models.depth_anything_3.specs import Prediction
from projects.mmdet3d_plugin.datasets.pipelines.respoint_post_processing import ResPointCloudPipeline




# for debubggin
import matplotlib.pyplot as plt
import PIL.Image as Image
import os

@BACKBONES.register_module()
class ReconstructionBackbone(nn.Module):
    """Reconstruction backbone that generates point clouds from multi-view images.
    
    Wraps DepthAnything3 (composition) instead of inheriting from it.
    This avoids method signature conflicts and provides cleaner separation.
    
    This backbone:
    1. Takes multi-view images from mmdet3d data pipeline
    2. Uses DepthAnything3's forward() to get depth maps
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
        super(ReconstructionBackbone, self).__init__()
        
        # Create wrapped DepthAnything3 model
        print(f"[DEBUG] ReconstructionBackbone: Loading DepthAnything3 model from {pretrained}")
        print(f"[DEBUG] ReconstructionBackbone: cache_dir = {cache_dir}")
        self.da3_model = DepthAnything3.from_pretrained(pretrained, cache_dir=cache_dir)
        self.da3_model.eval()
        print(f"[DEBUG] ReconstructionBackbone: DepthAnything3 model loaded successfully")
        
        # Set to eval mode
        self.eval()
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
    
    @property
    def input_processor(self):
        """Access wrapped model's input_processor."""
        return self.da3_model.input_processor
    
    @property
    def output_processor(self):
        """Access wrapped model's output_processor."""
        return self.da3_model.output_processor
    
    def _convert_to_prediction(self, raw_output: dict[str, torch.Tensor], return_torch: bool = False) -> Prediction:
        """Convert raw model output to Prediction object."""
        return self.da3_model._convert_to_prediction(raw_output, return_torch=return_torch)
    
    def _add_processed_images(self, prediction: Prediction, imgs_cpu: torch.Tensor) -> Prediction:
        """Add processed images to prediction for visualization."""
        return self.da3_model._add_processed_images(prediction, imgs_cpu)
    
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

        # Convert BGR -> RGB (mmcv loads images in BGR by default).
        # img shape is (B, N, 3, H, W); channel dimension is index 2
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0], ...]

        return img
    
    def _extract_cam2lidar_rts_from_meta(self, meta: Dict, device: torch.device) -> torch.Tensor:
        """Extract cam2lidar_rts transformations from a single metadata dict.
        
        Args:
            meta: Image metadata dict for one sample
        
        Returns:
            Torch tensor of shape (B, N, 4, 4)
        """
        B = len(meta)
        cam2lidar_rts_list = []
        for b_idx, meta_batch in enumerate(meta):
            cam2lidar_rts = meta_batch.get('cam2lidar_rts', None)
            cam2lidar_rts_list.append(torch.tensor(cam2lidar_rts, device=device))
        multi_batch_cam2lidar_rts = torch.stack(cam2lidar_rts_list, dim=0).to(device=device, dtype=torch.float32)
        return multi_batch_cam2lidar_rts


    
    def _backproject_depth_to_points(
        self,
        multi_batch_depths: torch.Tensor,           # (B, N, H, W)
        multi_batch_intrinsics: torch.Tensor,       # (B, N, 3, 3)
        multi_batch_ori_imgs: Optional[torch.Tensor] = None,  # (B, N, 3, H, W)
        multi_batch_cam2lidar_rts: Optional[torch.Tensor] = None, # (B, N, 4, 4)
        
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        
        Args:
            multi_batch_depths: (B, N, H, W)
            multi_batch_intrinsics: (B, N, 3, 3)
            multi_batch_ori_imgs: (B, N, 3, H, W)
            multi_batch_cam2lidar_rts: (B, N, 4, 4)
        
        Returns:
            all_points_batch: list of length B, each item is a tensor of arbitrary shape (P, 3)
            all_colors_batch: list of length B, each item is a tensor of arbitrary shape (P, 3) or None
        """
        device = multi_batch_depths.device
        B, N, H, W = multi_batch_depths.shape

        # points_batch = []
        # colors_batch = []

        # precompute grids
        u = torch.arange(W, device=device, dtype=multi_batch_depths.dtype)
        v = torch.arange(H, device=device, dtype=multi_batch_depths.dtype)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # (H,W)
        
        all_points_batch = []
        all_colors_batch = []

        for batch_idx in range(B):
            points_batch = torch.zeros((0, 3), device=device, dtype=multi_batch_depths.dtype)
            colors_batch = torch.zeros((0, 3), device=device, dtype=multi_batch_depths.dtype)
            for cam_idx in range(N):
                depth = multi_batch_depths[batch_idx, cam_idx]  # (H,W)
                intr = multi_batch_intrinsics[batch_idx, cam_idx]  # (3,3)

                fx, fy = intr[0, 0], intr[1, 1]
                cx, cy = intr[0, 2], intr[1, 2]

                z = depth
                x = (uu - cx) * z / fx
                y = (vv - cy) * z / fy

                pts = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=1)

                valid = (z.reshape(-1) > 0) & torch.isfinite(z.reshape(-1))
                if self.max_depth is not None:
                    valid = valid & (z.reshape(-1) <= self.max_depth)
                # if self.conf_thresh_percentile is not None and self.conf_thresh_percentile > 0:
                #     conf_flat = multi_batch_confs[batch_idx, cam_idx].reshape(-1)
                #     valid = valid & (conf_flat >= self.conf_thresh_percentile)
                # if self.filter_sky and multi_batch_sky_masks is not None:
                #     sky_flat = multi_batch_sky_masks[batch_idx, cam_idx].reshape(-1)
                #     valid = valid & (~sky_flat)

                pts = pts[valid]

                cols = None
                if multi_batch_ori_imgs is not None:
                    img = multi_batch_ori_imgs[batch_idx, cam_idx]
                    if img.dtype != torch.float:
                        img = img.float()
                    if img.shape[1] != H or img.shape[2] != W:
                        img = F.interpolate(
                            img.unsqueeze(0),
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze(0)
                    img_flat = img.permute(1, 2, 0).reshape(-1, 3)
                    cols = img_flat[valid]
                    if cols.numel() > 0 and cols.max() > 1.5:
                        cols = cols / 255.0

                
                # pts_list.append(pts)
                # col_list.append(cols)
                # convert pts in camera coordinates to lidar coordinates
                pts = pts @ multi_batch_cam2lidar_rts[batch_idx, cam_idx][:3, :3].T + multi_batch_cam2lidar_rts[batch_idx, cam_idx][3, :3]
                
                points_batch = torch.cat([points_batch, pts], dim=0) if pts is not None else points_batch
                colors_batch = torch.cat([colors_batch, cols], dim=0) if cols is not None else colors_batch
            
            all_points_batch.append(points_batch)
            all_colors_batch.append(colors_batch)

        return all_points_batch, all_colors_batch
    
    def _transform_points_cam_to_lidar(
        self,
        points_cam: torch.Tensor,
        colors_cam: Optional[torch.Tensor],
        cam2lidar_rt: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Transform points from camera to LiDAR coordinates (torch)."""
        device = points_cam.device
        R = cam2lidar_rt[:3, :3]
        t = cam2lidar_rt[3, :3]
        if not torch.is_tensor(R):
            R = torch.as_tensor(R, device=device, dtype=points_cam.dtype)
        else:
            R = R.to(device=device, dtype=points_cam.dtype)
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=device, dtype=points_cam.dtype)
        else:
            t = t.to(device=device, dtype=points_cam.dtype)
        points_lidar = points_cam @ R.T + t
        return points_lidar, colors_cam
    
    def _extract_image_paths_from_meta(self, img_meta: Dict) -> List[str]:
        """Extract image file paths from img_meta.
        
        Args:
            img_meta: Image metadata dict from mmdet3d pipeline
        
        Returns:
            List of image file paths
        """
        # Try different possible keys where filenames might be stored
        if 'filename' in img_meta:
            filenames = img_meta['filename']
            if isinstance(filenames, list):
                return filenames
            elif isinstance(filenames, str):
                return [filenames]
        elif 'img_filename' in img_meta:
            filenames = img_meta['img_filename']
            if isinstance(filenames, list):
                return filenames
            elif isinstance(filenames, str):
                return [filenames]
        
        raise ValueError(f"Could not find image paths in img_meta. Available keys: {list(img_meta.keys())}")
    
    # def _forward_with_image_paths(self, img_metas: List[Dict], device: torch.device) -> List[torch.Tensor]:
    #     """Forward pass using image paths directly (debug mode).
        
    #     This bypasses all preprocessing and uses DepthAnything3's inference() method
    #     which handles image loading and preprocessing itself.
        
    #     Args:
    #         img_metas: Image metadata list (one dict per batch item)
    #         device: Device to place tensors on
        
    #     Returns:
    #         List of point cloud tensors, one per batch item
    #     """
    #     batch_point_clouds = []
        
    #     for b_idx, meta_batch in enumerate(img_metas):
    #         # Extract image paths
    #         image_paths = self._extract_image_paths_from_meta(meta_batch)
    #         print(f"[DEBUG] Using image paths mode for batch {b_idx}: {len(image_paths)} images")
            
    #         # Extract lidar2img transformations
    #         lidar2img_list = self._extract_lidar2img_from_meta(meta_batch)
            
    #         # Use DepthAnything3's inference() method directly
    #         # This handles all preprocessing internally
    #         with torch.no_grad():
    #             prediction = self.da3_model.inference(
    #                 image=image_paths,  # List of image paths
    #                 extrinsics=None,  # Let model estimate
    #                 intrinsics=None,  # Let model estimate
    #                 use_ray_pose=self.use_ray_pose,
    #                 ref_view_strategy=self.ref_view_strategy,
    #                 process_res=504,
    #                 process_res_method="upper_bound_resize",
    #             )
            
    #         # Get processed images for color extraction (already added by inference())
    #         processed_images = None
    #         if hasattr(prediction, 'processed_images') and prediction.processed_images is not None:
    #             processed_images = prediction.processed_images  # (N, H, W, 3) in [0, 255]
            
    #         # Compute confidence threshold if needed
    #         conf_thresh = None
    #         if self.conf_thresh_percentile is not None and hasattr(prediction, 'conf') and prediction.conf is not None:
    #             sky = getattr(prediction, 'sky', None)
    #             if sky is not None and (~sky).sum() > 10:
    #                 conf_pixels = prediction.conf[~sky]
    #             else:
    #                 conf_pixels = prediction.conf.flatten()
    #             conf_thresh = np.percentile(conf_pixels, self.conf_thresh_percentile)
            
    #         # Back-project depth maps to point clouds (debug: keep camera-frame points)
    #         all_points_lidar = []
    #         all_colors = []
            
    #         # Debug: use only the first camera and keep camera-frame points (no cam->lidar transform)
    #         for i in range(len(prediction.depth)):
                
    #             # debug 
    #             i=0
                
    #             depth = prediction.depth[i]
    #             intrinsics = prediction.intrinsics[i] if hasattr(prediction, 'intrinsics') else None
                
    #             if intrinsics is None:
    #                 raise ValueError(f"Intrinsics not available for view {i} in batch {b_idx}")
                
    #             # Get sky mask and confidence for this view
    #             sky_mask = prediction.sky[i] if hasattr(prediction, 'sky') and prediction.sky is not None else None
    #             conf = prediction.conf[i] if hasattr(prediction, 'conf') and prediction.conf is not None else None
                
    #             # Get image for color extraction
    #             image = None
    #             if processed_images is not None and i < len(processed_images):
    #                 image = processed_images[i]  # (H, W, 3) in [0, 255]
                
    #             # Back-project to camera coordinates
    #             points_cam, colors = self._backproject_depth_to_points(
    #                 depth,
    #                 intrinsics,
    #                 image=image,  # Pass image for color extraction
    #                 max_depth=self.max_depth,
    #                 conf=conf,
    #                 conf_thresh=conf_thresh,
    #                 sky_mask=sky_mask,
    #                 filter_sky=self.filter_sky,
    #             )
                
    #             if len(points_cam) == 0:
    #                 raise ValueError(f"No points were generated for view {i} in batch {b_idx}")
                
    #             # Debug: do not transform to lidar; use camera-frame points directly
    #             points_cam_dbg = points_cam
    #             all_points_lidar.append(points_cam_dbg)
    #             if colors is not None:
    #                 all_colors.append(colors)

    #             # Debug: only use the first camera/view
    #             break
            
    #         if not all_points_lidar:
    #             raise ValueError(f"No points were generated for batch {b_idx}")
            
    #         # Concatenate all points for this batch item
    #         combined_points = np.concatenate(all_points_lidar, axis=0)
    #         combined_colors = None
    #         # Check if all_colors has elements and all are not None
    #         if all_colors and all(c is not None for c in all_colors):
    #             combined_colors = np.concatenate(all_colors, axis=0)
            
    #         # Apply post-processing pipeline
    #         if self.post_pipeline is not None:
    #             pipeline_input = {
    #                 'points': combined_points,
    #                 'colors': combined_colors,
    #                 'polygon_mask': None,
    #                 'indices': None,
    #             }
    #             pipeline_output = self.post_pipeline(pipeline_input)
    #             combined_points = pipeline_output['points']
    #             combined_colors = pipeline_output.get('colors')
            
    #         # Convert to tensor and add to batch results
    #         points_tensor = torch.from_numpy(combined_points).float().to(device)
    #         batch_point_clouds.append(points_tensor)
            
    #         # Store colors if available (for later access in inference)
    #         if not hasattr(self, '_batch_colors'):
    #             self._batch_colors = []
    #         if combined_colors is not None:
    #             # Store as numpy array (will be converted to tensor later if needed)
    #             self._batch_colors.append(combined_colors)
    #         else:
    #             self._batch_colors.append(None)
        
    #     # Return list of point clouds, one per batch item
    #     return batch_point_clouds
    
    def forward(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        return_loss: bool = False,
        # use_image_paths: bool = False,  # Debug mode: use image paths instead of preprocessed tensors
    ) -> List[torch.Tensor]:
        """Forward pass: generate point cloud from images.
        
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
        
        # Extract images from mmdet3d data format
        multi_batch_ori_imgs = self._extract_images_from_data(img)
        B, N, C, H, W = multi_batch_ori_imgs.shape
        
        
        batch_point_clouds = []

        # Run DA3 forward once for the whole batch
        imgs_processed, _, _ = self.input_processor(
            image=multi_batch_ori_imgs,  # (B, N, 3, H, W)
            extrinsics=None,
            intrinsics=None,
            process_res=504,
            process_res_method="upper_bound_resize",
        )
        imgs_for_da3 = imgs_processed.to(device, non_blocking=True).float()
        with torch.no_grad():
            da3_output = self.da3_model.forward(
                image=imgs_for_da3,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=self.use_ray_pose,
                ref_view_strategy=self.ref_view_strategy,
            )
        prediction = self._convert_to_prediction(da3_output, return_torch=True)
        
        # prediction.depth.shape: (B, N, H, W)
        # prediction.intrinsics.shape: (B, N, 3, 3)
        # prediction.sky.shape: (B, N, H, W)
        # prediction.conf.shape: (B, N, H, W)
        # prediction.extrinsics.shape: (B, N, 4, 4)
        # prediction.processed_images shape: (B, N, H, W, 3)
        # prediction.gaussians.shape: (B, N, H, W, 3)
        # prediction.aux.shape: (B, N, H, W, 3)
        # prediction.scale_factor.shape: (B, N, H, W, 3)
        
        
        
        if prediction is not None:
            # Back-project depth maps to point clouds
            all_points_lidar = []
            all_colors_lidar = []


            multi_batch_depths = prediction.depth
            multi_batch_intrinsics = prediction.intrinsics
            multi_batch_cam2lidar_rts = self._extract_cam2lidar_rts_from_meta(img_metas, device=device)

            
            # modify this function to handle batch processing
            all_points_batch, all_colors_batch = self._backproject_depth_to_points(
                multi_batch_depths,
                multi_batch_intrinsics,
                multi_batch_ori_imgs,
                multi_batch_cam2lidar_rts,
            )
            
            if self.post_pipeline is not None:
                
                pipeline_input = {
                    'points': all_points_batch,
                    'colors': all_colors_batch,
                    'polygon_mask': None,
                    'indices': None,
                }
                pipeline_output = self.post_pipeline(pipeline_input)
                all_points_batch = pipeline_output['points']
                all_colors_batch = pipeline_output.get('colors', all_colors_batch)
            
        
        
        
        
        
        
        
        # Process each batch item separately for point cloud generation
        # for b_idx in range(B):
        #     # Extract images for this batch item
        #     imgs_batch = imgs_processed[b_idx]  # (N, 3, H_new, W_new)
            
        #     # Prepare model inputs exactly like inference() does:
        #     # 1. Move to model device
        #     # 2. Add batch dimension [None] to make (1, N, 3, H, W)
        #     # 3. Convert to float
        #     device = next(self.parameters()).device
        #     imgs_for_da3 = imgs_batch.to(device, non_blocking=True)[None].float()
            
        #     # Run DepthAnything3 forward (wrapped model's forward method)
        #     # This matches what _run_model_forward does in inference()
        #     with torch.no_grad():
        #         da3_output = self.da3_model.forward(
        #             image=imgs_for_da3,
        #             extrinsics=None,  # Let model estimate
        #             intrinsics=None,  # Let model estimate
        #             export_feat_layers=[],  # No feature export needed
        #             infer_gs=False,  # No Gaussian Splatting
        #             use_ray_pose=self.use_ray_pose,
        #             ref_view_strategy=self.ref_view_strategy,
        #         )
            
        #     # Convert output to Prediction object (matches inference())
        #     prediction = self._convert_to_prediction(da3_output, return_torch=True)
            
        #     # Compute confidence threshold if needed (torch only)
        #     conf_thresh = None
        #     if self.conf_thresh_percentile is not None and hasattr(prediction, 'conf') and prediction.conf is not None:
        #         sky = getattr(prediction, 'sky', None)
        #         conf_map = prediction.conf
        #         if sky is not None and sky.dtype != torch.bool:
        #             sky = sky >= 0.5
        #         if sky is not None and (~sky).sum() > 10:
        #             conf_pixels = conf_map[~sky]
        #         else:
        #             conf_pixels = conf_map.reshape(-1)
        #         if conf_pixels.numel() > 0:
        #             conf_thresh = torch.quantile(conf_pixels, self.conf_thresh_percentile / 100.0)
                
        #         # Back-project depth maps to point clouds
        #         all_points_lidar = []
        #         all_colors_lidar = []
        #         cam2lidar_rts = img_metas[b_idx]['cam2lidar_rts']
        #         for cam_idx in range(len(prediction.depth)):
        #             depth = prediction.depth[cam_idx]
        #             intrinsics = prediction.intrinsics[cam_idx] if hasattr(prediction, 'intrinsics') else None
                    
        #             if intrinsics is None:
        #                 raise ValueError(f"Intrinsics not available for view {cam_idx} in batch {b_idx}")
                    
        #             # Get sky mask and confidence for this view
        #             sky_mask = prediction.sky[cam_idx] if hasattr(prediction, 'sky') and prediction.sky is not None else None
        #             conf = prediction.conf[cam_idx] if hasattr(prediction, 'conf') and prediction.conf is not None else None
                    
        #             # Back-project to camera coordinates (returns xyz + colors)
        #             points_cam, colors_cam = self._backproject_depth_to_points(
        #                 depth,
        #                 intrinsics,
        #                 image=img_tensor[b_idx, cam_idx],
        #                 max_depth=self.max_depth,
        #                 conf=conf,
        #                 conf_thresh=conf_thresh,
        #                 sky_mask=sky_mask,
        #                 filter_sky=self.filter_sky,
        #             )
                    
        #             if len(points_cam) == 0:
        #                 print(f"[WARN] No points generated for view {cam_idx} in batch {b_idx}; skipping this view.")
        #                 continue

        #             # Transform to LiDAR coordinates
        #             points_lidar, colors_lidar = self._transform_points_cam_to_lidar(
        #                 points_cam,
        #                 colors_cam,
        #                 cam2lidar_rts[cam_idx],
        #                 intrinsics,
        #             )
                    
        #             all_points_lidar.append(points_lidar.clone())
        #             all_colors_lidar.append(colors_lidar.clone() if colors_lidar is not None else None)
                
        #         if not all_points_lidar:
        #             raise ValueError(f"No points were generated for batch {b_idx} (all views empty after filtering)")
                
        #         # Concatenate all points for this batch item
        #         combined_points = torch.cat(all_points_lidar, dim=0)
        #         combined_colors = None
        #         if all_colors_lidar:
        #             valid_color_lists = [c for c in all_colors_lidar if c is not None and c.numel() > 0]
        #             if valid_color_lists:
        #                 combined_colors = torch.cat(valid_color_lists, dim=0)
                
        #         # Apply post-processing pipeline
        #         if self.post_pipeline is not None:
        #             pipeline_input = {
        #                 'points': combined_points,
        #                 'colors': combined_colors,
        #                 'polygon_mask': None,
        #                 'indices': None,
        #             }
        #             pipeline_output = self.post_pipeline(pipeline_input)
        #             combined_points = pipeline_output['points']
        #             combined_colors = pipeline_output.get('colors', combined_colors)
                    
                    
        #             # merged points and colors to (N,6) xyzrgb
    
        #         # merged points and colors to (N,6) xyzrgb
        #         if combined_colors is not None and combined_colors.shape[0] == combined_points.shape[0]:
        #             merged = torch.cat([combined_points, combined_colors], dim=1)
        #         else:
        #             merged = combined_points
                
        #         points_tensor = merged.float().to(device)
        #         batch_point_clouds.append(points_tensor)

        return batch_point_clouds
