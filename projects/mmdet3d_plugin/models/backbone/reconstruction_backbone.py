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
from mmdet.models.builder import build_backbone

from projects.mmdet3d_plugin.models.backbone.depth_anything_3.api import DepthAnything3
from projects.mmdet3d_plugin.models.backbone.depth_anything_3.specs import Prediction
from projects.mmdet3d_plugin.datasets.pipelines.respoint_post_processing import DepthAnything3Filter
from projects.mmdet3d_plugin.models.backbone.point_cloud_refinement import PointCloudRefinement




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
        rescon_pipeline: Optional[List[Dict]] = None,
        glb_config: Optional[Dict] = None,
        ref_view_strategy: str = "saddle_balanced",
        use_ray_pose: bool = False,
        max_points: int = 1_000_000,
        filter_sky: bool = True,
        max_depth: Optional[float] = None,
        conf_thresh_percentile: Optional[float] = None,
        freeze_da3: bool = True,  # Freeze DepthAnything3 model (recommended)
        refinement: Optional[Dict] = None,  # Point cloud refinement config
    ):
        """Initialize ReconstructionBackbone.
        
        Args:
            pretrained: Pretrained DepthAnything3 model name or path (HuggingFace Hub identifier)
            cache_dir: Cache directory for model
            rescon_pipeline: List of post-processing step configs
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
        
        # Freeze DA3 model if requested (recommended for training)
        self.freeze_da3 = freeze_da3
        if self.freeze_da3:
            for param in self.da3_model.parameters():
                param.requires_grad = False
            print(f"[DEBUG] ReconstructionBackbone: DepthAnything3 model frozen (requires_grad=False)")
        else:
            print(f"[DEBUG] ReconstructionBackbone: DepthAnything3 model trainable (requires_grad=True)")
        
        # Set to eval mode (but can be switched to train mode for refinement)
        # self.eval()  # Don't force eval mode, let training script control it
        print(f"[DEBUG] ReconstructionBackbone: Model initialized")
        
        # Store ReconstructionBackbone-specific config
        self.rescon_pipeline_cfg = rescon_pipeline
        
        self.da3_pipeline_cfg = [cfg for cfg in rescon_pipeline if cfg['type'] == 'DepthAnything3Filter']
        self.refinement_pipeline_cfg = [cfg for cfg in rescon_pipeline if cfg['type'] == 'RefinementProcessor']
        
        self.da3_pipeline = Compose(self.da3_pipeline_cfg) if self.da3_pipeline_cfg else None
        self.refinement_pipeline = Compose(self.refinement_pipeline_cfg) if self.refinement_pipeline_cfg else None


        
        self.glb_config = glb_config or {}
        self.ref_view_strategy = ref_view_strategy
        self.use_ray_pose = use_ray_pose
        self.max_points = max_points
        self.filter_sky = filter_sky
        self.max_depth = max_depth or self.glb_config.get('max_depth', None)
        self.conf_thresh_percentile = conf_thresh_percentile or self.glb_config.get('conf_thresh_percentile', None)
        
        # Build point cloud refinement module
        if refinement is not None:
            print(f"[DEBUG] ReconstructionBackbone: Building point cloud refinement module...")
            self.refinement = build_backbone(refinement)
            print(f"[DEBUG] ReconstructionBackbone: Refinement module built successfully")
        else:
            self.refinement = None
            print(f"[DEBUG] ReconstructionBackbone: No refinement module configured")
    
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

    def _extract_lidar2img_from_meta(self, meta: Dict, device: torch.device) -> torch.Tensor:
        """Extract lidar2img matrices from metadata (must exist)."""
        if meta is None:
            raise ValueError("img_metas is required and must include lidar2img")
        lidar2img_list = []
        for meta_batch in meta:
            if 'lidar2img' not in meta_batch or meta_batch['lidar2img'] is None:
                raise ValueError("lidar2img missing in img_metas; required for GT colorization")
            lidar2img_list.append(torch.tensor(meta_batch['lidar2img'], device=device))
        return torch.stack(lidar2img_list, dim=0).to(device=device, dtype=torch.float32)

    def _get_gt_color_points(
        self,
        gt_points_list: List[torch.Tensor],
        multi_batch_ori_imgs: torch.Tensor,
        multi_batch_lidar2img: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Colorize GT points (LiDAR frame) by projecting into multi-view images using lidar2img.
        """
        B = len(gt_points_list)
        _, N, _, H, W = multi_batch_ori_imgs.shape  # (B, N, 3, H, W)

        gt_color_points_list: List[torch.Tensor] = []
        for b_idx in range(B):
            pts_lidar = gt_points_list[b_idx]  # (P, 3)
            colors = torch.zeros((pts_lidar.shape[0], 3), device=pts_lidar.device, dtype=pts_lidar.dtype)
            filled = torch.zeros((pts_lidar.shape[0],), device=pts_lidar.device, dtype=torch.bool)

            lidar2img = multi_batch_lidar2img[b_idx]  # (N,4,4)

            for cam_idx in range(min(N, lidar2img.shape[0])):
                pts_h = torch.cat(
                    [pts_lidar, torch.ones((pts_lidar.shape[0], 1), device=pts_lidar.device, dtype=pts_lidar.dtype)],
                    dim=1,
                )
                proj = pts_h @ lidar2img[cam_idx].T  # (P,4)
                z = proj[:, 2]
                u = proj[:, 0] / z
                v = proj[:, 1] / z
                valid = (z > 0) & (u >= 0) & (u <= (W - 1)) & (v >= 0) & (v <= (H - 1))

                if not valid.any():
                    continue

                idx = valid & (~filled)
                if not idx.any():
                    continue

                u_idx = u[idx].long()
                v_idx = v[idx].long()

                img = multi_batch_ori_imgs[b_idx, cam_idx]  # (3, H, W)
                img_hw3 = img.permute(1, 2, 0)  # (H, W, 3)
                sampled = img_hw3[v_idx, u_idx]
                if not torch.is_floating_point(sampled):
                    sampled = sampled.float()
                if sampled.max() > 1.5:
                    sampled = sampled / 255.0

                colors[idx] = sampled
                filled[idx] = True

            gt_color_points_list.append(torch.cat([pts_lidar, colors], dim=1))

        return gt_color_points_list
    
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
    
    def forward(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        return_loss: bool = False,
        points: Optional[torch.Tensor] = None,  # GT point cloud for training
        freeze_da3_override: Optional[bool] = None,  # Override freeze_da3 setting for this forward pass
    ) -> Tuple[List[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Forward pass: generate point cloud from images.
        
        Routes to forward_train or forward_test based on return_loss flag.
        Both paths apply refinement network.
        
        Args:
            img: Multi-view images (B, N, 3, H, W) or DataContainer
            img_metas: Image metadata list (one dict per batch item)
            return_loss: Whether to return loss (True=train, False=test)
            points: Ground truth point clouds (B, N, 3) or list of (N, 3) tensors for training
            freeze_da3_override: Override freeze_da3 setting for this forward pass (train mode only)
        
        Returns:
            batch_point_clouds: List of point cloud tensors, one per batch item
                Each tensor has shape (N_points, 3) or (N_points, 6) if colors included
            losses: Dict of loss values (if return_loss=True and refinement enabled)
        """
        if return_loss:
            return self.forward_train(img, img_metas, points, freeze_da3_override)
        else:
            return self.forward_test(img, img_metas)
    
    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        points: Optional[torch.Tensor] = None,
        freeze_da3_override: Optional[bool] = None,
    ) -> Tuple[List[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Forward pass for training mode.
        
        Args:
            img: Multi-view images (B, N, 3, H, W) or DataContainer
            img_metas: Image metadata list (one dict per batch item)
            points: Ground truth point clouds (B, N, 3) or list of (N, 3) tensors
            freeze_da3_override: Override freeze_da3 setting (None = use self.freeze_da3)
        
        Returns:
            batch_point_clouds: List of refined point cloud tensors
            losses: Dict of loss values (if refinement enabled)
        """
        # Determine whether to freeze DA3 for this forward pass
        freeze_da3 = self.freeze_da3 if freeze_da3_override is None else freeze_da3_override
        
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
        
        # Use no_grad context if DA3 is frozen
        da3_context = torch.no_grad() if freeze_da3 else torch.enable_grad()
        with da3_context:
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
        
        if prediction is not None:
            # Back-project depth maps to point clouds (batched)
            multi_batch_depths = prediction.depth
            multi_batch_intrinsics = prediction.intrinsics
            multi_batch_cam2lidar_rts = self._extract_cam2lidar_rts_from_meta(img_metas, device=device)

            # Back-project all batch items at once (returns lists of length B)
            all_points_batch, all_colors_batch = self._backproject_depth_to_points(
                multi_batch_depths,
                multi_batch_intrinsics,
                multi_batch_ori_imgs,
                multi_batch_cam2lidar_rts,
            )
            
            # Process each batch item separately through the post-processing pipeline
            pseudo_points_list = []
            
            for b_idx in range(B):
                points_b = all_points_batch[b_idx]  # (P, 3) tensor
                colors_b = all_colors_batch[b_idx] if all_colors_batch[b_idx] is not None and all_colors_batch[b_idx].numel() > 0 else None
                
                if points_b.shape[0] == 0:
                    raise ValueError(f"No points were generated for batch {b_idx} (all views empty after filtering)")
                
                # Apply post-processing pipeline
                if self.da3_pipeline is not None:
                    pipeline_input = {
                        'points': points_b,
                        'colors': colors_b,
                        'indices': None,
                    }
                    pipeline_output = self.da3_pipeline(pipeline_input)
                    points_b = pipeline_output['points']
                    colors_b = pipeline_output.get('colors', colors_b)
                
                # Merge points and colors to (N,6) xyzrgb format (or keep as (N,3))
                if colors_b is not None and colors_b.shape[0] == points_b.shape[0]:
                    merged = torch.cat([points_b, colors_b], dim=1)  # (N, 6)
                else:
                    merged = points_b  # (N, 3)
                
                pseudo_points_list.append(merged.float().to(device))
            

            
            # Prepare GT points in batch format (list of tensors)
            gt_points_list = None
            if points is not None:
                if isinstance(points, list):
                    gt_points_list = [p.float().to(device) for p in points if p is not None]
                elif isinstance(points, torch.Tensor):
                    if points.dim() == 3:  # (B, N, 3)
                        gt_points_list = [points[i].float().to(device) for i in range(B)]
                    else:  # (N, 3) - single point cloud, expand to batch
                        gt_points_list = [points.unsqueeze(0).expand(B, -1, -1).float().to(device)]

            # Colorize GT points if available using lidar2img
            if gt_points_list is not None:
                multi_batch_lidar2img = self._extract_lidar2img_from_meta(img_metas, device=device)
                gt_points_list = self._get_gt_color_points(
                    gt_points_list=gt_points_list,
                    multi_batch_ori_imgs=multi_batch_ori_imgs,
                    multi_batch_lidar2img=multi_batch_lidar2img,
                )
                
                
            display_point_cloud(pseudo_points_list[0].cpu().numpy(), colors=gt_points_list[0].cpu().numpy()[:, 3:], gt_bboxes_3d=None)
            
            # Apply refinement in batch mode (if enabled)
            if self.refinement is not None:
                # Refine entire batch at once
                refined_batch, refinement_losses = self.refinement(
                    pseudo_points=pseudo_points_list,  # list of (N, C) tensors
                    gt_points=gt_points_list,  # list of (N, 3 or 6) tensors
                    return_loss=True,  # Always compute loss in training
                )
                
                # refined_batch is (B, N, C) tensor, convert to list
                batch_point_clouds = [refined_batch[i] for i in range(B)]
                losses = refinement_losses
            else:
                # No refinement, convert batch tensor to list
                batch_point_clouds = [pseudo_points_batch[i] for i in range(B)]
                losses = None
        
        return batch_point_clouds, losses
    
    def forward_test(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
    ) -> Tuple[List[torch.Tensor], None]:
        """Forward pass for test/inference mode.
        
        Args:
            img: Multi-view images (B, N, 3, H, W) or DataContainer
            img_metas: Image metadata list (one dict per batch item)
        
        Returns:
            batch_point_clouds: List of refined point cloud tensors
            losses: Always None in test mode
        """
        device = next(self.parameters()).device
        
        # Handle DataContainer for img_metas
        if isinstance(img_metas, DC):
            img_metas = img_metas.data
        
        # Extract images from mmdet3d data format
        multi_batch_ori_imgs = self._extract_images_from_data(img)
        B, N, C, H, W = multi_batch_ori_imgs.shape
        
        batch_point_clouds = []

        # Run DA3 forward once for the whole batch (always frozen in test mode)
        imgs_processed, _, _ = self.input_processor(
            image=multi_batch_ori_imgs,  # (B, N, 3, H, W)
            extrinsics=None,
            intrinsics=None,
            process_res=504,
            process_res_method="upper_bound_resize",
        )
        imgs_for_da3 = imgs_processed.to(device, non_blocking=True).float()
        
        # Always use no_grad in test mode
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
        
        if prediction is not None:
            # Back-project depth maps to point clouds (batched)
            multi_batch_depths = prediction.depth
            multi_batch_intrinsics = prediction.intrinsics
            multi_batch_cam2lidar_rts = self._extract_cam2lidar_rts_from_meta(img_metas, device=device)

            # Back-project all batch items at once (returns lists of length B)
            all_points_batch, all_colors_batch = self._backproject_depth_to_points(
                multi_batch_depths,
                multi_batch_intrinsics,
                multi_batch_ori_imgs,
                multi_batch_cam2lidar_rts,
            )
            
            # Process each batch item separately through the post-processing pipeline
            pseudo_points_list = []
            
            for b_idx in range(B):
                points_b = all_points_batch[b_idx]  # (P, 3) tensor
                colors_b = all_colors_batch[b_idx] if all_colors_batch[b_idx] is not None and all_colors_batch[b_idx].numel() > 0 else None
                
                if points_b.shape[0] == 0:
                    raise ValueError(f"No points were generated for batch {b_idx} (all views empty after filtering)")
                
                # Apply post-processing pipeline
                if self.da3_pipeline is not None:
                    pipeline_input = {
                        'points': points_b,
                        'colors': colors_b,
                        'indices': None,
                    }
                    pipeline_output = self.da3_pipeline(pipeline_input)
                    points_b = pipeline_output['points']
                    colors_b = pipeline_output.get('colors', colors_b)
                
                # Merge points and colors to (N,6) xyzrgb format (or keep as (N,3))
                if colors_b is not None and colors_b.shape[0] == points_b.shape[0]:
                    merged = torch.cat([points_b, colors_b], dim=1)  # (N, 6)
                else:
                    merged = points_b  # (N, 3)
                
                pseudo_points_list.append(merged.float().to(device))
            
            # Apply refinement in batch mode (if enabled)
            if self.refinement is not None:
                # Refine entire batch at once (no GT in test mode)
                refined_batch, _ = self.refinement(
                    pseudo_points=pseudo_points_list,  # list of (N, C) tensors
                    gt_points=None,  # No GT in test mode
                    return_loss=False,  # No loss computation in test
                )
                
                # refined_batch is (B, N, C) tensor, convert to list
                batch_point_clouds = [refined_batch[i] for i in range(B)]
            else:
                # No refinement, convert batch tensor to list
                batch_point_clouds = [pseudo_points_list[i] for i in range(B)]
        
        return batch_point_clouds, None
        



# add visualizatin function here for debugging purposes
def display_point_cloud(points, colors=None, gt_bboxes_3d=None):
    import open3d as o3d
    """Display point cloud using open3d.
    
    Args:
        points (np.ndarray): Point cloud as numpy array of shape (N, 3)
        colors (np.ndarray, optional): Colors as numpy array of shape (N, 3) in [0, 1]
        gt_bboxes_3d (list, optional): List of ground truth 3D bounding boxes
    """
    if points is None or len(points) == 0:
        print(f"  Warning: No point cloud to display")
        return
    
    print(f"  Displaying point cloud with {len(points)} points...")
    print(f"  Press 'Q' or close the window to continue")
    
    # Convert numpy to open3d PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Set colors if provided
    if colors is not None:
        if colors.max() > 1.0:
            # Assume colors are in [0, 255], normalize to [0, 1]
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default gray color (white on white background is invisible)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud", width=1920, height=1080)
    vis.add_geometry(pcd)
    
    # Calculate point cloud center and bounds for proper view setup
    points_array = np.asarray(pcd.points)
    if len(points_array) > 0:
        center = points_array.mean(axis=0)
        bounds = points_array.max(axis=0) - points_array.min(axis=0)
        max_bound = bounds.max()
    else:
        center = np.array([0, 0, 0])
        max_bound = 1.0
    
    # Set up view to look at the point cloud center
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, -1])
    view_ctl.set_lookat(center)
    view_ctl.set_up([0, -1, 0])
    # Set zoom based on point cloud size
    if max_bound > 0:
        # Zoom to fit the point cloud (smaller zoom = wider view)
        zoom = 0.3 if max_bound > 50 else 0.7
    else:
        zoom = 0.7
    view_ctl.set_zoom(zoom)
    
    # Update renderer to apply view changes
    vis.poll_events()
    vis.update_renderer()
    
    # Draw the axis of the point cloud
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(axis)
    
    # Draw the gt_bboxes_3d on the point cloud
    if gt_bboxes_3d is not None and len(gt_bboxes_3d) > 0:
        print(f"  Adding {len(gt_bboxes_3d)} bounding boxes to visualization")
        for gt_bbox_3d in gt_bboxes_3d:
            # Extract bbox information
            # gt_bboxes_3d from mmdet3d are typically in LiDARBox3D format
            if hasattr(gt_bbox_3d, 'tensor'):
                # mmdet3d LiDARBox3D format: [x, y, z, w, l, h, yaw]
                bbox_tensor = gt_bbox_3d.tensor.cpu().numpy()
                if len(bbox_tensor.shape) == 2:
                    bbox_tensor = bbox_tensor[0]  # Take first box if batched
                
                center = bbox_tensor[:3]  # x, y, z
                size = bbox_tensor[3:6]  # w, l, h
                yaw = bbox_tensor[6]  # yaw angle
                
                # Create rotation matrix from yaw
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                rotation_matrix = np.array([
                    [cos_yaw, -sin_yaw, 0],
                    [sin_yaw, cos_yaw, 0],
                    [0, 0, 1]
                ])
            else:
                # Fallback: assume dict or other format
                center = np.array(gt_bbox_3d.get('center', [0, 0, 0]), dtype=np.float64)
                size = np.array(gt_bbox_3d.get('size', [1, 1, 1]), dtype=np.float64)
                rotation_matrix = gt_bbox_3d.get('rotation_matrix', np.eye(3))
            
            # Create OrientedBoundingBox
            obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, size)
            obb.color = [1, 0, 0]  # Red color for boxes
            vis.add_geometry(obb)
            
            # Change the color of points which are in this box
            indices = obb.get_point_indices_within_bounding_box(pcd.points)
            if len(indices) > 0:
                # Convert colors to numpy array, modify, then assign back
                colors_array = np.asarray(pcd.colors)
                colors_array[indices] = [1, 0, 0]  # Red color for points in box
                pcd.colors = o3d.utility.Vector3dVector(colors_array)
                vis.update_geometry(pcd)
            
            # Find the center of front face (heading direction)
            heading_dir = rotation_matrix[:2, 0]  # x, y components of heading
            yaw = np.arctan2(heading_dir[1], heading_dir[0])
            
            # Connect the bbox center with the front center -> heading direction
            front_center = center + size[0] * np.array([np.cos(yaw), np.sin(yaw), 0])
            # Append geometry line set from center to front center
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([center, front_center])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0]])  # Red color for heading line
            vis.add_geometry(line_set)
    
    vis.run()
    vis.destroy_window()