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
from projects.mmdet3d_plugin.models.backbone.depth_anything_3.utils.export.glb import export_to_glb
# import Lidainsatance box from mmdet3d
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes




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
        ensure_thresh_percentile: Optional[float] = None,
        base_conf_thresh: Optional[float] = None,
        freeze_da3: bool = True,  # Freeze DepthAnything3 model (recommended)
        refinement: Optional[Dict] = None,  # Point cloud refinement config
        export_glb: bool = False,  # Enable GLB export for debugging
        glb_export_dir: str = "output",  # Directory for GLB export
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
            conf_thresh_percentile: Confidence threshold percentile (lower bound)
            ensure_thresh_percentile: Upper percentile clamp for confidence threshold
            base_conf_thresh: Base confidence threshold value
        """
        super(ReconstructionBackbone, self).__init__()
        
        # Check training phase from refinement config (if available)
        training_phase = refinement.get('training_phase', 2) if refinement is not None else 2
        
        # Only initialize DA3 if not Phase 1 (to save memory)
        if training_phase == 1:
            self.da3_model = None
            print(f"[ReconstructionBackbone] Phase 1: Skipping DA3 initialization to save memory")
        else:
            # Phase 2: Initialize DA3 normally
            self.da3_model = DepthAnything3.from_pretrained(pretrained, cache_dir=cache_dir)
            self.da3_model.eval()
            self.da3_model = self.da3_model.float()  # Ensure float32 to avoid dtype mismatches
            
            self.freeze_da3 = freeze_da3
            if self.freeze_da3:
                for param in self.da3_model.parameters():
                    param.requires_grad = False
            print(f"[ReconstructionBackbone] Phase 2: DA3 model initialized and {'frozen' if self.freeze_da3 else 'trainable'}")
        
        self.freeze_da3 = freeze_da3
        
        # Build pipelines from config (don't store config dicts to avoid pickle issues)
        if rescon_pipeline is not None:
            da3_pipeline_cfg = [cfg for cfg in rescon_pipeline if cfg.get('type') == 'DepthAnything3Filter']
            refinement_pipeline_cfg = [cfg for cfg in rescon_pipeline if cfg.get('type') == 'RefinementProcessor']
            self.da3_pipeline = Compose(da3_pipeline_cfg) if da3_pipeline_cfg else None
            self.refinement_pipeline = Compose(refinement_pipeline_cfg) if refinement_pipeline_cfg else None
        else:
            self.da3_pipeline = None
            self.refinement_pipeline = None
        
        # Extract values from glb_config (don't store the dict itself)
        if glb_config is not None:
            self.max_depth = max_depth or glb_config.get('max_depth', None)
            self.conf_thresh_percentile = conf_thresh_percentile or glb_config.get('conf_thresh_percentile', None)
            self.ensure_thresh_percentile = ensure_thresh_percentile or glb_config.get('ensure_thresh_percentile', 90.0)
            self.base_conf_thresh = base_conf_thresh or glb_config.get('base_conf_thresh', 1.05)
        else:
            self.max_depth = max_depth
            self.conf_thresh_percentile = conf_thresh_percentile
            self.ensure_thresh_percentile = ensure_thresh_percentile if ensure_thresh_percentile is not None else 90.0
            self.base_conf_thresh = base_conf_thresh if base_conf_thresh is not None else 1.05
        
        self.ref_view_strategy = ref_view_strategy
        self.use_ray_pose = use_ray_pose
        self.max_points = max_points
        self.filter_sky = filter_sky
        self.export_glb = export_glb
        self.glb_export_dir = glb_export_dir
        
        # Build point cloud refinement module
        if refinement is not None:
            self.refinement = build_backbone(refinement)
        else:
            self.refinement = None
    
    @property
    def input_processor(self):
        """Access wrapped model's input_processor."""
        return self.da3_model.input_processor
    
    @property
    def output_processor(self):
        """Access wrapped model's output_processor."""
        return self.da3_model.output_processor
    
    @property
    def training_phase(self):
        """int: Current training phase (1 = train teacher, 2 = train student)."""
        if self.refinement is not None:
            return getattr(self.refinement, 'training_phase', 2)
        return 2  # Default to Phase 2
    
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
    
    def _extract_lidar2cam_rts_from_meta(self, meta: Dict, device: torch.device) -> torch.Tensor:
        """Extract lidar2cam_rts from metadata, convert to standard format, and normalize relative to first camera.
        
        Returns:
            Tensor of shape (B, N, 4, 4) in DA3's normalized format (camera 0 = identity), or None if not available.
        """
        batch_list = []
        for meta_batch in meta:
            lidar2cam_rts = meta_batch.get('lidar2cam_rts', None)
            if lidar2cam_rts is None:
                return None
            
            # Convert to numpy array and ensure shape (N, 4, 4)
            if isinstance(lidar2cam_rts, list):
                lidar2cam_rts = np.array(lidar2cam_rts)
            if lidar2cam_rts.ndim == 2:
                lidar2cam_rts = lidar2cam_rts[None, :, :]
            
            # Convert from non-standard format [[R.T, 0], [t, 1]] to standard [[R, t], [0, 1]]
            N = lidar2cam_rts.shape[0]
            lidar2cam_std = np.zeros((N, 4, 4), dtype=lidar2cam_rts.dtype)
            lidar2cam_std[:, :3, :3] = lidar2cam_rts[:, :3, :3].transpose(0, 2, 1)  # R = R.T.T
            lidar2cam_std[:, :3, 3] = lidar2cam_rts[:, 3, :3]  # t from row 3 to column 3
            lidar2cam_std[:, 3, 3] = 1.0
            
            batch_list.append(torch.tensor(lidar2cam_std, device=device, dtype=torch.float32))
        
        multi_batch = torch.stack(batch_list, dim=0)  # (B, N, 4, 4)
        
        # Normalize: make camera 0 identity, others relative to it (matching DA3 format)
        B, N = multi_batch.shape[:2]
        normalized = []
        for b in range(B):
            lidar2cam_0 = multi_batch[b, 0]  # (4, 4)
            R_0, t_0 = lidar2cam_0[:3, :3], lidar2cam_0[:3, 3:4]
            cam0_to_lidar = torch.eye(4, device=device, dtype=multi_batch.dtype)
            cam0_to_lidar[:3, :3] = R_0.T
            cam0_to_lidar[:3, 3:4] = -R_0.T @ t_0
            normalized.append(multi_batch[b] @ cam0_to_lidar)  # (N, 4, 4)
        
        return torch.stack(normalized, dim=0)  # (B, N, 4, 4)
    
    def _extract_intrinsics_from_meta(self, meta: Dict, device: torch.device) -> torch.Tensor:
        """Extract camera intrinsics from metadata.
        
        Args:
            meta: Image metadata dict (list of batch items)
            device: Target device
        
        Returns:
            Torch tensor of shape (B, N, 3, 3) or None if not available
        """
        B = len(meta)
        intrinsics_list = []
        for b_idx, meta_batch in enumerate(meta):
            # Only use cam_intrinsic from dataset (no fallback)
            if 'cam_intrinsic' not in meta_batch:
                return None
            
            # cam_intrinsic is a list of (3, 3) matrices, one per camera
            intrinsics = meta_batch['cam_intrinsic']
            
            if intrinsics is None:
                return None
            
            # Convert to numpy array if it's a list
            if isinstance(intrinsics, list):
                # Each element should be (3, 3)
                intrinsics = np.array(intrinsics)  # (N, 3, 3)
            
            # Ensure shape (N, 3, 3)
            if intrinsics.ndim == 2:  # Single camera (3, 3)
                intrinsics = intrinsics[None, :, :]  # (1, 3, 3)
            
            intrinsics_list.append(torch.tensor(intrinsics, device=device))
        
        multi_batch_intrinsics = torch.stack(intrinsics_list, dim=0).to(device=device, dtype=torch.float32)
        return multi_batch_intrinsics
    
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
        multi_batch_confs: Optional[torch.Tensor] = None,  # (B, N, H, W)
        conf_thr: Optional[float] = None,  # Confidence threshold
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
        # print('depth shape of _backproject_depth_to_points: H and W: ', H, W)
        # points_batch = []
        # colors_batch = []

        # precompute grids
        u = torch.arange(W, device=device, dtype=multi_batch_depths.dtype)
        v = torch.arange(H, device=device, dtype=multi_batch_depths.dtype)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # (H,W)
        
        all_points_batch = []
        all_colors_batch = []

        for batch_idx in range(B):
            # Collect points and colors in lists first to avoid quadratic memory growth
            # from repeated torch.cat operations
            points_list = []
            colors_list = []
            
            for cam_idx in range(N):
                depth = multi_batch_depths[batch_idx, cam_idx]  # (H,W)
                intr = multi_batch_intrinsics[batch_idx, cam_idx]  # (3,3)

                # Use ray-based method similar to _depths_to_world_points_with_colors
                # Create pixel coordinates (u, v, 1) for all pixels
                pix = torch.stack([uu.reshape(-1), vv.reshape(-1), torch.ones_like(uu).reshape(-1)], dim=1)  # (H*W, 3)
                
                # Get valid depth mask
                d_flat = depth.reshape(-1)  # (H*W,)
                valid = (d_flat > 0) & torch.isfinite(d_flat)
                if self.max_depth is not None:
                    valid = valid & (d_flat <= self.max_depth)
                
                # Filter by confidence threshold (same as GLB export)
                if multi_batch_confs is not None and conf_thr is not None:
                    conf = multi_batch_confs[batch_idx, cam_idx]  # (H, W)
                    conf_flat = conf.reshape(-1)  # (H*W,)
                    valid = valid & (conf_flat >= conf_thr)
                
                if not torch.any(valid):
                    continue
                
                # Get valid pixel indices
                vidx = torch.nonzero(valid, as_tuple=False).squeeze(-1)  # (M,)
                
                # Compute inverse intrinsics
                K_inv = torch.inverse(intr)  # (3, 3)
                
                # Transform pixels to rays: rays = K_inv @ pix[valid].T
                pix_valid = pix[vidx].T  # (3, M)
                rays = K_inv @ pix_valid  # (3, M)
                
                # Scale rays by depth to get points in camera coordinates
                d_valid = d_flat[vidx]  # (M,)
                Xc = rays * d_valid[None, :]  # (3, M)
                pts = Xc.T  # (M, 3) - points in camera coordinates
                

                

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
                    cols = img_flat[vidx]
                    if cols.numel() > 0 and cols.max() > 1.5:
                        cols = cols / 255.0
                        
                

                # Convert pts in camera coordinates to lidar coordinates
                if pts.numel() > 0:
                    pts = pts @ multi_batch_cam2lidar_rts[batch_idx, cam_idx][:3, :3].T + multi_batch_cam2lidar_rts[batch_idx, cam_idx][3, :3]
                    points_list.append(pts)
                    if cols is not None and cols.numel() > 0:
                        colors_list.append(cols)
                        

            
            # Concatenate all points/colors once at the end (much more memory efficient)
            if points_list:
                points_batch = torch.cat(points_list, dim=0)
                colors_batch = torch.cat(colors_list, dim=0) if colors_list else None
            else:
                points_batch = torch.zeros((0, 3), device=device, dtype=multi_batch_depths.dtype)
                colors_batch = None
            
            all_points_batch.append(points_batch)
            all_colors_batch.append(colors_batch)
            
        # # Visualize points from first batch item (for debugging - same as GLB export visualization)
        # # Note: This visualizes points in camera coordinates before lidar transformation
        # if len(all_points_batch) > 0 and all_points_batch[0].numel() > 0:
        #     import open3d as o3d
        #     pcd = o3d.geometry.PointCloud()
        #     # Visualize first batch item only
        #     points_vis = all_points_batch[0].cpu().numpy()
        #     pcd.points = o3d.utility.Vector3dVector(points_vis)
        #     if all_colors_batch[0] is not None and all_colors_batch[0].numel() > 0:
        #         colors_vis = all_colors_batch[0].cpu().numpy()
        #         # Normalize colors to [0, 1] if needed
        #         if colors_vis.max() > 1.0:
        #             colors_vis = colors_vis / 255.0
        #         pcd.colors = o3d.utility.Vector3dVector(colors_vis)
        #     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        #     geometries = [pcd, axis]
        #     o3d.visualization.draw_geometries(geometries, window_name="Pseudo Point Cloud (Camera Coords)")

        return all_points_batch, all_colors_batch

    def _padding_samples(
        self,
        pseudo_points: List[torch.Tensor],
        gt_points: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad pseudo and GT point clouds in the batch to the same number of points.
        
        Returns batched tensors with unified point count: (B, N, C).
        """
        if gt_points is not None:
            assert len(pseudo_points) == len(gt_points), "Pseudo and GT points must have the same batch size"
        batch_size = len(pseudo_points)

        # Determine target number of points across pseudo and gt
        max_num_points = max(len(pseudo_points[i]) for i in range(batch_size))
        if gt_points is not None:
            max_num_points_gt = max(len(gt_points[i]) for i in range(batch_size))
            target_num_points = max(max_num_points, max_num_points_gt)
        else:
            target_num_points = max_num_points

        # Pad to target size
        padded_pseudo_points = self._pad_point_clouds(pseudo_points, target_num_points)
        if gt_points is not None:
            padded_gt_points = self._pad_point_clouds(gt_points, target_num_points)
        else:
            padded_gt_points = None

        # Stack to tensors (B, N, C)
        padded_pseudo_points = torch.stack(padded_pseudo_points, dim=0)
        if gt_points is not None:
            padded_gt_points = torch.stack(padded_gt_points, dim=0)
        else:
            padded_gt_points = None

        return padded_pseudo_points, padded_gt_points

    def _pad_point_clouds(
        self,
        points_list: List[torch.Tensor],
        target_num_points: int,
    ) -> List[torch.Tensor]:
        """Pad a list of point clouds to the target number of points.
        
        Pads by repeating the last point if needed.
        """
        padded = []
        for pts in points_list:
            if pts.dim() != 2:
                print(f"pts.shape: {pts.shape}")
                raise ValueError(f"pts.shape: {pts.shape} is not 2D")
            n, c = pts.shape
            if n == target_num_points:
                padded.append(pts)
                continue
            if n == 0:
                # create zeros if empty
                pad_pts = torch.zeros((target_num_points, c), device=pts.device, dtype=pts.dtype)
                padded.append(pad_pts)
                continue
            if n < target_num_points:
                pad_count = target_num_points - n
                pad = pts[-1:].repeat(pad_count, 1)
                pad_pts = torch.cat([pts, pad], dim=0)
            else:
                pad_pts = pts[:target_num_points]
            padded.append(pad_pts)
        return padded
    
    def _export_glb_debug(
        self,
        prediction,
        multi_batch_ori_imgs: torch.Tensor,
        multi_batch_extrinsics_gt: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Debug helper: Export GLB file and return point cloud for visualization.
        
        To enable/disable: comment/uncomment the call to this function in forward_train/forward_test.
        
        Returns:
            glb_points_np: (N, 3) numpy array of points, or None if export disabled/failed
            glb_colors_np: (N, 3) numpy array of colors, or None if export disabled/failed
        """
        if not self.export_glb or prediction is None:
            return None, None
        
        from copy import deepcopy
        import torch.nn.functional as F
        
        # Create a copy of prediction for GLB export to avoid modifying the original
        prediction_glb = deepcopy(prediction)
        
        # Use original RGB images (not normalized) for GLB export
        # Resize original images to match depth map resolution (same as _backproject_depth_to_points does)
        # multi_batch_ori_imgs is (B, N, 3, H_orig, W_orig) in RGB format
        # prediction.depth is (N, H, W) or (B, N, H, W) at processed resolution
        # We need (N, H, W, 3) uint8 RGB for GLB export, matching depth map size
        
        # Get depth map size (processed resolution)
        if isinstance(prediction_glb.depth, torch.Tensor):
            depth_shape = prediction_glb.depth.shape
        else:
            depth_shape = np.array(prediction_glb.depth).shape
        
        # Remove batch dimension if present: (B, N, H, W) -> (N, H, W)
        if len(depth_shape) == 4:  # (B, N, H, W)
            H_proc, W_proc = depth_shape[2], depth_shape[3]
        else:  # (N, H, W)
            H_proc, W_proc = depth_shape[1], depth_shape[2]
        
        # Resize original images to depth map resolution (same approach as _backproject_depth_to_points)
        # multi_batch_ori_imgs[0] is (N, 3, H_orig, W_orig)
        ori_imgs_resized = F.interpolate(
            multi_batch_ori_imgs[0],  # (N, 3, H_orig, W_orig)
            size=(H_proc, W_proc),
            mode='bilinear',
            align_corners=False
        )  # (N, 3, H_proc, W_proc)
        
        # Convert to numpy and permute to (N, H, W, 3)
        imgs_numpy = ori_imgs_resized.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)
        
        # Ensure uint8 format [0, 255] and clip to valid range
        if imgs_numpy.dtype != np.uint8:
            if imgs_numpy.max() <= 1.0:
                imgs_numpy = (imgs_numpy * 255.0).clip(0, 255).astype(np.uint8)
            else:
                imgs_numpy = imgs_numpy.clip(0, 255).astype(np.uint8)
        else:
            imgs_numpy = np.clip(imgs_numpy, 0, 255).astype(np.uint8)
        
        prediction_glb.processed_images = imgs_numpy
        
        # Convert prediction fields to numpy if they're torch tensors
        # Extract first batch if batched (B, N, ...) -> (N, ...)
        # The export function expects (N, H, W) not (B, N, H, W)
        if isinstance(prediction_glb.depth, torch.Tensor):
            depth_np = prediction_glb.depth.cpu().numpy()
        else:
            depth_np = prediction_glb.depth
        # Remove batch dimension if present: (B, N, H, W) -> (N, H, W)
        if depth_np.ndim == 4:  # (B, N, H, W)
            depth_np = depth_np[0]  # (N, H, W)
        prediction_glb.depth = depth_np
            
        if isinstance(prediction_glb.intrinsics, torch.Tensor):
            intrinsics_np = prediction_glb.intrinsics.cpu().numpy()
        else:
            intrinsics_np = prediction_glb.intrinsics
        # Remove batch dimension if present: (B, N, 3, 3) -> (N, 3, 3)
        if intrinsics_np.ndim == 4:  # (B, N, 3, 3)
            intrinsics_np = intrinsics_np[0]  # (N, 3, 3)
        prediction_glb.intrinsics = intrinsics_np
            
        # Use normalized extrinsics (relative to first camera) for GLB export
        # This matches DA3's format: camera 0 is identity, others are relative to it
        # Whether we use GT or predicted extrinsics, they're in the same normalized format
        if isinstance(prediction_glb.extrinsics, torch.Tensor):
            extrinsics_np = prediction_glb.extrinsics.cpu().numpy()
        else:
            extrinsics_np = prediction_glb.extrinsics
        # Remove batch dimension if present: (B, N, 4, 4) -> (N, 4, 4)
        if extrinsics_np.ndim == 4:  # (B, N, 4, 4)
            extrinsics_np = extrinsics_np[0]  # (N, 4, 4)
        prediction_glb.extrinsics = extrinsics_np
            
        if isinstance(prediction_glb.conf, torch.Tensor):
            conf_np = prediction_glb.conf.cpu().numpy()
        else:
            conf_np = prediction_glb.conf
        # Remove batch dimension if present: (B, N, H, W) -> (N, H, W)
        if conf_np is not None and conf_np.ndim == 4:  # (B, N, H, W)
            conf_np = conf_np[0]  # (N, H, W)
        prediction_glb.conf = conf_np
        
        # Create export directory
        os.makedirs(self.glb_export_dir, exist_ok=True)
        
        # Prepare GT extrinsics for GLB export (if available)
        # Use GT extrinsics to fix wrong camera poses, but keep prediction.intrinsics for metric scale
        gt_extrinsics_for_glb = None
        if multi_batch_extrinsics_gt is not None:
            if isinstance(multi_batch_extrinsics_gt, torch.Tensor):
                gt_extrinsics_np = multi_batch_extrinsics_gt.cpu().numpy()
            else:
                gt_extrinsics_np = multi_batch_extrinsics_gt
            # Remove batch dimension if present: (B, N, 4, 4) -> (N, 4, 4)
            if gt_extrinsics_np.ndim == 4:  # (B, N, 4, 4)
                gt_extrinsics_for_glb = gt_extrinsics_np[0]  # (N, 4, 4)
            else:
                gt_extrinsics_for_glb = gt_extrinsics_np  # (N, 4, 4)
        
        # Export GLB and get points/colors directly
        glb_path, glb_points_np, glb_colors_np = export_to_glb(
            prediction=prediction_glb,
            export_dir=self.glb_export_dir,
            gt_extrinsics=gt_extrinsics_for_glb,  # Use GT extrinsics if available
            num_max_points=self.max_points,
            conf_thresh_percentile=self.conf_thresh_percentile or 40.0,
            show_cameras=True,
        )
        
        # Handle empty point cloud and validate colors
        if glb_points_np.shape[0] == 0:
            glb_points_np = None
            glb_colors_np = None
        elif glb_colors_np is not None:
            # Validate color shape matches points
            if glb_colors_np.shape[0] != glb_points_np.shape[0]:
                glb_colors_np = None
            elif glb_colors_np.shape[0] == 0:
                glb_colors_np = None
            else:
                # Ensure colors are in correct format (uint8 [0, 255])
                if glb_colors_np.dtype != np.uint8:
                    if glb_colors_np.max() <= 1.0:
                        glb_colors_np = (glb_colors_np * 255.0).astype(np.uint8)
                    else:
                        glb_colors_np = glb_colors_np.astype(np.uint8)
        
        return glb_points_np, glb_colors_np
    
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
        points: Optional[torch.Tensor] = None,  # GT point cloud for training
        return_loss: bool = False,
    ) -> Tuple[dict, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass: generate point cloud from images.
        
        Routes to forward_train or forward_test based on return_loss flag.
        Both paths apply refinement network.
        
        Args:
            img: Multi-view images (B, N, 3, H, W) or DataContainer
            img_metas: Image metadata list (one dict per batch item)
            points: Ground truth point clouds (B, N, 3) or list of (N, 3) tensors for training
            return_loss: Whether to return loss (True=train, False=test)
        
        Returns:
            pts_feat: Dict of refined point cloud features and indices
            losses: Dict of loss values (if return_loss=True and refinement enabled)
        """
        if return_loss:
            return self.forward_train(img, img_metas, points)
        else:
            return self.forward_test(img, img_metas, points)
    
    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        points: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass for training mode.
        
        Args:
            img: Multi-view images (B, N, 3, H, W) or DataContainer
            img_metas: Image metadata list (one dict per batch item)
            points: Ground truth point clouds (B, N, 3) or list of (N, 3) tensors
        
        Returns:
            pts_feat: Dict of refined point cloud features and indices
            losses: Dict of loss values (if refinement enabled)
        """
        
        device = next(self.parameters()).device
        
        # Handle DataContainer for img_metas
        if isinstance(img_metas, DC):
            img_metas = img_metas.data
        
        pts_feat = dict()
        losses = None
        
        # Phase 1 optimization: Skip DA3 entirely, use GT points directly
        training_phase = getattr(self.refinement, 'training_phase', 2) if self.refinement is not None else 2
        if training_phase == 1:
            # Phase 1: Direct GT points → teacher encoder (skip DA3, save memory/time)
            if points is None:
                raise ValueError("GT points are required for Phase 1 training")
            
            # Prepare GT points in batch format (list of tensors)
            if isinstance(points, torch.Tensor):
                if points.dim() == 2:
                    # Single point cloud, wrap in list
                    gt_points_list = [points]
                else:
                    # Batched: (B, N, 3) -> list of (N, 3)
                    gt_points_list = [points[b] for b in range(points.shape[0])]
            elif isinstance(points, list):
                gt_points_list = points
            else:
                raise ValueError(f"Unexpected points type: {type(points)}")
            
            # Pad GT points for batch processing
            padded_gt, _ = self._padding_samples(gt_points_list, None)
            
            # Process GT points through refinement (teacher encoder only in Phase 1)
            # Pass dummy pseudo points (same as GT) - refinement will only use GT branch in Phase 1
            sparse_feat_dict, refinement_losses = self.refinement(
                pseudo_points=padded_gt,  # Dummy (not used in Phase 1)
                gt_points=padded_gt,      # GT points for teacher encoder
                return_loss=True,
            )
            
            # Store sparse features dict for detection pipeline
            pts_feat['rescon_features'] = sparse_feat_dict
            losses = refinement_losses
            
            return pts_feat, losses
        
        # Phase 2: Normal flow with DA3 (generate pseudo points from images)
        # Extract images from mmdet3d data format
        multi_batch_ori_imgs = self._extract_images_from_data(img)
        B, N, C, H, W = multi_batch_ori_imgs.shape

        # Run DA3 forward once for the whole batch
        # Extract intrinsics and extrinsics from img_metas if available
        multi_batch_intrinsics_gt = self._extract_intrinsics_from_meta(img_metas, device=device)
        # Use lidar2cam_rts directly from dataset (no conversion needed)
        multi_batch_extrinsics_gt = self._extract_lidar2cam_rts_from_meta(img_metas, device=device)
        
        # Prepare extrinsics and intrinsics for DA3 input processor
        # Since image is a torch.Tensor, input processor expects torch.Tensors for extrinsics/intrinsics too
        # ALWAYS let DA3 predict extrinsics first (for consistent depth scale)
        # Then align depth to GT extrinsics using Umeyama alignment
        extrinsics_for_da3 = None  # Let DA3 predict extrinsics for consistent depth scale
        intrinsics_for_da3 = multi_batch_intrinsics_gt if multi_batch_intrinsics_gt is not None else None
        
        imgs_processed, extrinsics_processed, intrinsics_processed = self.input_processor(
            image=multi_batch_ori_imgs,  # (B, N, 3, H, W) torch.Tensor
            extrinsics=extrinsics_for_da3,  # (B, N, 4, 4) torch.Tensor or None
            intrinsics=intrinsics_for_da3,  # (B, N, 3, 3) torch.Tensor or None
            process_res=504,
            process_res_method="upper_bound_resize",
        )
        # Ensure images are in float32 (DA3's autocast will handle mixed precision internally)
        imgs_for_da3 = imgs_processed.to(device, non_blocking=True).float()
        
        # Convert extrinsics and intrinsics to float32 if they exist
        if extrinsics_processed is not None:
            extrinsics_processed = extrinsics_processed.to(device=device, dtype=torch.float32)
        if intrinsics_processed is not None:
            intrinsics_processed = intrinsics_processed.to(device=device, dtype=torch.float32)
        
        # Use inference_mode (more memory efficient than no_grad) if DA3 is frozen
        if self.freeze_da3:
            # Set to eval mode and use inference_mode for maximum memory savings
            self.da3_model.eval()
            with torch.inference_mode():
                da3_output = self.da3_model.forward(
                    image=imgs_for_da3,
                    extrinsics=extrinsics_processed,
                    intrinsics=intrinsics_processed,
                    export_feat_layers=[],
                    infer_gs=False,
                    use_ray_pose=self.use_ray_pose,
                    ref_view_strategy=self.ref_view_strategy,
                )
            # Clear cache after DA3 forward to free up memory
            torch.cuda.empty_cache()
        else:
            self.da3_model.train()
            da3_output = self.da3_model.forward(
                image=imgs_for_da3,
                extrinsics=extrinsics_processed,
                intrinsics=intrinsics_processed,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=self.use_ray_pose,
                ref_view_strategy=self.ref_view_strategy,
            )
        prediction = self._convert_to_prediction(da3_output, return_torch=True)
        
        # Skip alignment - DA3's predicted depth and extrinsics are already self-consistent and close to metric
        # When DA3 predicts extrinsics (extrinsics_for_da3=None), the depth is already in a scale that's close to metric
        # Aligning to GT extrinsics causes incorrect scaling and rotation issues (too small, sloped ground, wrong alignment)
        # Instead, we use DA3's predicted extrinsics and depth as-is for backprojection
        # The backprojection will use metric cam2lidar_rts from the dataset, which should work correctly
        # with DA3's learned depth scale (which is already close to metric, as observed when extrinsics=None)
        
        # GLB Export (for debugging) - comment/uncomment to enable/disable
        # glb_points_np, glb_colors_np = self._export_glb_debug(
        #     prediction=prediction,
        #     multi_batch_ori_imgs=multi_batch_ori_imgs,
        #     multi_batch_extrinsics_gt=multi_batch_extrinsics_gt,
        # )
        # GLB Export (for debugging) - comment/uncomment to enable/disable
        # glb_points_np, glb_colors_np = self._export_glb_debug(
        #     prediction=prediction,
        #     multi_batch_ori_imgs=multi_batch_ori_imgs,
        #     multi_batch_extrinsics_gt=multi_batch_extrinsics_gt,
        # )
        glb_points_np = None
        glb_colors_np = None
        
        if prediction is not None:
            # Back-project depth maps to point clouds (batched)
            # Convert depth to torch tensor if needed and ensure correct shape (B, N, H, W)
            if isinstance(prediction.depth, np.ndarray):
                depth_torch = torch.from_numpy(prediction.depth).to(device=device, dtype=torch.float32)
            elif isinstance(prediction.depth, torch.Tensor):
                depth_torch = prediction.depth.to(device=device, dtype=torch.float32)
            else:
                depth_torch = prediction.depth
            
            # Add batch dimension if needed: (N, H, W) -> (1, N, H, W)
            if depth_torch.ndim == 3:  # (N, H, W)
                depth_torch = depth_torch.unsqueeze(0)  # (1, N, H, W)
            
            multi_batch_depths = depth_torch
            B, N, H, W = multi_batch_depths.shape  # Get actual batch size
            
            # Convert intrinsics to torch tensor if needed and ensure correct shape (B, N, 3, 3)
            if isinstance(prediction.intrinsics, np.ndarray):
                intrinsics_torch = torch.from_numpy(prediction.intrinsics).to(device=device, dtype=torch.float32)
            elif isinstance(prediction.intrinsics, torch.Tensor):
                intrinsics_torch = prediction.intrinsics.to(device=device, dtype=torch.float32)
            else:
                intrinsics_torch = prediction.intrinsics
            
            # Add batch dimension if needed: (N, 3, 3) -> (B, N, 3, 3)
            if intrinsics_torch.ndim == 3:  # (N, 3, 3)
                intrinsics_torch = intrinsics_torch.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, 3, 3)
            elif intrinsics_torch.ndim == 4 and intrinsics_torch.shape[0] == 1:  # (1, N, 3, 3) but B > 1
                intrinsics_torch = intrinsics_torch.expand(B, -1, -1, -1)  # (B, N, 3, 3)
            
            multi_batch_intrinsics = intrinsics_torch
            multi_batch_cam2lidar_rts = self._extract_cam2lidar_rts_from_meta(img_metas, device=device)

            # Extract confidence maps and calculate threshold (same as GLB export)
            multi_batch_confs = None
            conf_thr = None
            if prediction.conf is not None and self.conf_thresh_percentile is not None:
                # Convert confidence to torch tensor if needed
                if isinstance(prediction.conf, np.ndarray):
                    conf_torch = torch.from_numpy(prediction.conf).to(device=device, dtype=torch.float32)
                elif isinstance(prediction.conf, torch.Tensor):
                    conf_torch = prediction.conf.to(device=device, dtype=torch.float32)
                else:
                    conf_torch = prediction.conf
                
                # Add batch dimension if needed: (N, H, W) -> (B, N, H, W)
                if conf_torch.ndim == 3:  # (N, H, W)
                    conf_torch = conf_torch.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, H, W)
                elif conf_torch.ndim == 4 and conf_torch.shape[0] == 1:  # (1, N, H, W) but B > 1
                    conf_torch = conf_torch.expand(B, -1, -1, -1)  # (B, N, H, W)
                
                multi_batch_confs = conf_torch
                
                # Calculate confidence threshold using percentile (same as GLB export)
                conf_flat = conf_torch.reshape(-1).cpu().numpy()
                lower = np.percentile(conf_flat, self.conf_thresh_percentile)
                upper = np.percentile(conf_flat, self.ensure_thresh_percentile)
                conf_thr = float(min(max(self.base_conf_thresh, lower), upper))
                # print(f"[DEBUG] Confidence threshold: {conf_thr:.4f} (percentile: {self.conf_thresh_percentile}%, ensure: {self.ensure_thresh_percentile}%, base: {self.base_conf_thresh})")

            # Back-project all batch items at once (returns lists of length B)
            all_points_batch, all_colors_batch = self._backproject_depth_to_points(
                multi_batch_depths,
                multi_batch_intrinsics,
                multi_batch_ori_imgs,
                multi_batch_cam2lidar_rts,
                multi_batch_confs,
                conf_thr,
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
            # Each element should be (N, 3), not (B, N, 3)
            gt_points_list = None
            if points is not None:
                if isinstance(points, list):
                    gt_points_list = [p.float().to(device) for p in points if p is not None]
                elif isinstance(points, torch.Tensor):
                    if points.dim() == 3:  # (B, N, 3)
                        gt_points_list = [points[i].float().to(device) for i in range(points.shape[0])]
                    elif points.dim() == 2:  # (N, 3) - single point cloud, expand to batch
                        # Expand to (B, N, 3) then split into list of (N, 3) tensors (one per batch item)
                        points_expanded = points.unsqueeze(0).expand(B, -1, -1).float().to(device)
                        gt_points_list = [points_expanded[i] for i in range(B)]
                    else:
                        raise ValueError(f"Unexpected points tensor dimension: {points.dim()}, shape: {points.shape}")

            # Colorize GT points if available using lidar2img (only if refinement uses colors)
            if gt_points_list is not None:
                # Check if refinement module uses colors
                use_color = getattr(self.refinement, 'use_color', False) if self.refinement is not None else False
                if use_color:
                    multi_batch_lidar2img = self._extract_lidar2img_from_meta(img_metas, device=device)
                    gt_points_list = self._get_gt_color_points(
                        gt_points_list=gt_points_list,
                        multi_batch_ori_imgs=multi_batch_ori_imgs,
                        multi_batch_lidar2img=multi_batch_lidar2img,
                    )
                # If not using colors, gt_points_list remains as XYZ only (B, N, 3)
            # comment or uncomment for visual debugging purposes
            for b_idx in range(B):
                # print(img_metas[b_idx]['filename'])
                
                # Extract GT bboxes from img_metas for visualization (if available)
                # Note: gt_bboxes_3d should be in main data dict, not img_metas
                # This code is for backward compatibility if it's still in img_metas
                gt_bboxes_3d_batch = None
                if 'gt_bboxes_3d' in img_metas[b_idx]:
                    gt_bboxes_3d_batch = img_metas[b_idx]['gt_bboxes_3d']
                    # Handle DataContainer wrapper
                    if isinstance(gt_bboxes_3d_batch, DC):
                        gt_bboxes_3d_batch = gt_bboxes_3d_batch.data
                    # Handle list/tuple wrapping (DataContainer.data might be a list)
                    if isinstance(gt_bboxes_3d_batch, (list, tuple)) and len(gt_bboxes_3d_batch) > 0:
                        # If it's a list, take the first element (batch item)
                        gt_bboxes_3d_batch = gt_bboxes_3d_batch[0] if len(gt_bboxes_3d_batch) == 1 else gt_bboxes_3d_batch[b_idx]
                    
                    if gt_bboxes_3d_batch is not None:
                        # Convert to LiDARInstance3DBoxes and convert to original gravity
                        from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
                        
                        # Get tensor from various formats
                        if isinstance(gt_bboxes_3d_batch, LiDARInstance3DBoxes):
                            bbox_tensor = gt_bboxes_3d_batch.tensor
                        elif isinstance(gt_bboxes_3d_batch, torch.Tensor):
                            bbox_tensor = gt_bboxes_3d_batch
                        elif isinstance(gt_bboxes_3d_batch, np.ndarray):
                            bbox_tensor = torch.from_numpy(gt_bboxes_3d_batch)
                        elif hasattr(gt_bboxes_3d_batch, 'tensor'):
                            bbox_tensor = gt_bboxes_3d_batch.tensor
                        else:
                            print(f"[WARNING] Unknown bbox format: {type(gt_bboxes_3d_batch)}, skipping visualization")
                            gt_bboxes_3d_batch = None
                            continue
                        
                        # Create LiDARInstance3DBoxes with origin (0.5, 0.5, 0.5) and convert to LIDAR
                        gt_bboxes_3d_ori = LiDARInstance3DBoxes(
                            bbox_tensor,
                            box_dim=bbox_tensor.shape[-1],
                            origin=(0.5, 0.5, -0.5)
                        ).convert_to(Box3DMode.LIDAR)
                        
                        # Extract tensor and apply transformations
                        gt_boxes_3d_ori = gt_bboxes_3d_ori.tensor.cpu().numpy()
                        
                        # Apply transformations for visualization
                        # Flip yaw angle
                        gt_boxes_3d_ori[:, 6] = -gt_boxes_3d_ori[:, 6] - np.pi / 2
                        
                        # Swap width and length for visualization
                        gt_boxes_3d_ori[..., [3, 4]] = gt_boxes_3d_ori[..., [4, 3]]
                        
                        # Replace the original with transformed version
                        gt_bboxes_3d_batch = gt_boxes_3d_ori
                        
                        bbox_count = len(gt_bboxes_3d_batch)
                        # print(f"[DEBUG] Found {bbox_count} GT bounding boxes for batch {b_idx} (converted to original gravity)")
                
                # COMMENTED OUT: Disable visualization during training
                # # Only visualize if GT points are available
                # if gt_points_list is not None and len(gt_points_list) > b_idx:
                #     gt_points_np = gt_points_list[b_idx].cpu().numpy()
                #     
                #     # Extract colors only if points have more than 3 columns
                #     if gt_points_np.shape[1] > 3:
                #         gt_colors = gt_points_np[:, 3:6]  # Extract RGB (assuming 6 columns: xyzrgb)
                #         # Ensure colors are in correct shape (N, 3)
                #         if gt_colors.shape[1] != 3:
                #             gt_colors = None
                #     else:
                #         gt_colors = None
                #     
                #     if gt_colors is not None:
                #         print(f"[DEBUG] GT point cloud: {len(gt_points_np)} points with colors shape {gt_colors.shape}")
                #     else:
                #         print(f"[DEBUG] GT point cloud: {len(gt_points_np)} points without colors (will use gray)")
                #     
                #     display_point_cloud(gt_points_np, colors=gt_colors, gt_bboxes_3d=gt_bboxes_3d_batch, window_name=f"GT Point Cloud ({len(gt_points_np):,} points)")
                # 
                # # Always visualize pseudo points (they should always be available)
                # if len(pseudo_points_list) > b_idx:
                #     pseudo_points_np = pseudo_points_list[b_idx].cpu().numpy()
                #     pseudo_colors = pseudo_points_np[:, 3:] if pseudo_points_np.shape[1] > 3 else None
                #     display_point_cloud(pseudo_points_np, colors=pseudo_colors, gt_bboxes_3d=gt_bboxes_3d_batch, window_name=f"Pseudo Point Cloud ({len(pseudo_points_np):,} points)")
                # 
                # # Display GLB point cloud if available (3rd visualization)
                # if glb_points_np is not None:
                #     if glb_colors_np is not None:
                #         print(f"[DEBUG] GLB point cloud: {len(glb_points_np)} points with colors shape {glb_colors_np.shape}, dtype={glb_colors_np.dtype}, range=[{glb_colors_np.min():.1f}, {glb_colors_np.max():.1f}]")
                #     else:
                #         print(f"[DEBUG] GLB point cloud: {len(glb_points_np)} points without colors (will use gray)")
                #     display_point_cloud(glb_points_np, colors=glb_colors_np, gt_bboxes_3d=gt_bboxes_3d_batch, window_name=f"GLB Point Cloud ({len(glb_points_np):,} points)")
                
            # Apply refinement in batch mode (if enabled)

            # Refine entire batch at once
            if gt_points_list is None:
                raise ValueError("GT points are required for refinement")

            # Pad pseudo and GT to the same number of points per batch
            padded_pseudo, padded_gt = self._padding_samples(pseudo_points_list, gt_points_list)

            sparse_feat_dict, refinement_losses = self.refinement(
                pseudo_points=padded_pseudo,  # (B, N, C) tensor
                gt_points=padded_gt,         # (B, N, C) tensor
                return_loss=True,            # Always compute loss in training
            )
            
            # Store sparse features dict for detection pipeline
            # Format: {'pseudo': {...}, 'gt': {...}}
            pts_feat['rescon_features'] = sparse_feat_dict
            losses = refinement_losses

        
        return pts_feat, losses
    
    def forward_test(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        points: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, Optional[Dict[str, float]]]:
        """Forward pass for test/inference mode.
        
        Args:
            img: Multi-view images (B, N, 3, H, W) or DataContainer
            img_metas: Image metadata list (one dict per batch item) or single dict (will be wrapped)
        
        Returns:
            pts_feat: Dict of refined point cloud features and indices
            losses: Always None in test mode
        """
        device = next(self.parameters()).device
        
        # Handle DataContainer for img_metas
        if isinstance(img_metas, DC):
            img_metas = img_metas.data
        
        # Handle case where img_metas is a single dict (from simple_test) or other formats
        if not isinstance(img_metas, list):
            if isinstance(img_metas, dict):
                img_metas = [img_metas]
            else:
                # Unexpected format
                raise TypeError(f"img_metas must be a list of dicts or a single dict, got {type(img_metas)}: {img_metas}")
        
        # Ensure all elements are dicts (not strings or other types)
        # This validation happens BEFORE we pass img_metas to extraction functions
        if len(img_metas) == 0:
            raise ValueError("img_metas is empty")
        
        for i, meta_item in enumerate(img_metas):
            if not isinstance(meta_item, dict):
                raise TypeError(
                    f"img_metas[{i}] must be a dict, got {type(meta_item)}: {meta_item}. "
                    f"Full img_metas type: {type(img_metas)}, length: {len(img_metas)}"
                )
        
        device = next(self.parameters()).device
        
        # Phase 1 optimization: Skip DA3 entirely, use GT points directly (same as forward_train)
        training_phase = getattr(self.refinement, 'training_phase', 2) if self.refinement is not None else 2
        if training_phase == 1:
            # Phase 1: Direct GT points → teacher encoder (skip DA3, save memory/time)
            if points is None:
                raise ValueError("GT points are required for Phase 1 evaluation")
            
            # Prepare GT points in batch format (list of tensors)
            if isinstance(points, torch.Tensor):
                if points.dim() == 2:
                    # Single point cloud, wrap in list
                    gt_points_list = [points]
                else:
                    # Batched: (B, N, 3) -> list of (N, 3)
                    gt_points_list = [points[b] for b in range(points.shape[0])]
            elif isinstance(points, list):
                gt_points_list = points
            else:
                raise ValueError(f"Unexpected points type: {type(points)}")
            
            # Pad GT points for batch processing
            padded_gt, _ = self._padding_samples(gt_points_list, None)
            
            # Process GT points through refinement (teacher encoder only in Phase 1)
            # Pass dummy pseudo points (same as GT) - refinement will only use GT branch in Phase 1
            sparse_feat_dict, metrics = self.refinement(
                pseudo_points=padded_gt,  # Dummy (not used in Phase 1)
                gt_points=padded_gt,      # GT points for teacher encoder
                return_loss=False,  # No loss computation in test
            )
            
            # Store sparse features dict for detection pipeline
            pts_feat = dict()
            pts_feat['rescon_features'] = sparse_feat_dict
            
            return pts_feat, metrics
        
        # Phase 2: Normal flow with DA3 (generate pseudo points from images)
        # Extract images from mmdet3d data format
        multi_batch_ori_imgs = self._extract_images_from_data(img)
        B, N, C, H, W = multi_batch_ori_imgs.shape
        
        pts_feat = dict()
        losses = None

        # Run DA3 forward once for the whole batch (always frozen in test mode)
        # Extract intrinsics and extrinsics from img_metas if available
        multi_batch_intrinsics_gt = self._extract_intrinsics_from_meta(img_metas, device=device)
        # Use lidar2cam_rts directly from dataset (no conversion needed)
        multi_batch_extrinsics_gt = self._extract_lidar2cam_rts_from_meta(img_metas, device=device)
        
        # Prepare extrinsics and intrinsics for DA3 input processor
        # Since image is a torch.Tensor, input processor expects torch.Tensors for extrinsics/intrinsics too
        # Comment out extrinsics - let DA3 predict them
        # extrinsics_for_da3 = multi_batch_extrinsics_gt if multi_batch_extrinsics_gt is not None else None
        extrinsics_for_da3 = None  # Let DA3 predict extrinsics
        intrinsics_for_da3 = multi_batch_intrinsics_gt if multi_batch_intrinsics_gt is not None else None
        
        imgs_processed, extrinsics_processed, intrinsics_processed = self.input_processor(
            image=multi_batch_ori_imgs,  # (B, N, 3, H, W) torch.Tensor
            extrinsics=extrinsics_for_da3,  # (B, N, 4, 4) torch.Tensor or None
            intrinsics=intrinsics_for_da3,  # (B, N, 3, 3) torch.Tensor or None
            process_res=504,
            process_res_method="upper_bound_resize",
        )
        # Ensure images are in float32 (DA3's autocast will handle mixed precision internally)
        imgs_for_da3 = imgs_processed.to(device, non_blocking=True).float()
        
        # Convert extrinsics and intrinsics to float32 if they exist
        if extrinsics_processed is not None:
            extrinsics_processed = extrinsics_processed.to(device=device, dtype=torch.float32)
        if intrinsics_processed is not None:
            intrinsics_processed = intrinsics_processed.to(device=device, dtype=torch.float32)
        
        # Always use no_grad in test mode
        with torch.no_grad():
            da3_output = self.da3_model.forward(
                image=imgs_for_da3,
                extrinsics=extrinsics_processed,
                intrinsics=intrinsics_processed,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=self.use_ray_pose,
                ref_view_strategy=self.ref_view_strategy,
            )
        prediction = self._convert_to_prediction(da3_output, return_torch=True)
        
        # Skip alignment - DA3's predicted depth and extrinsics are already self-consistent and close to metric
        # When DA3 predicts extrinsics (extrinsics_for_da3=None), the depth is already in a scale that's close to metric
        # Aligning to GT extrinsics causes incorrect scaling and rotation issues (too small, sloped ground, wrong alignment)
        # Instead, we use DA3's predicted extrinsics and depth as-is for backprojection
        # The backprojection will use metric cam2lidar_rts from the dataset, which should work correctly
        # with DA3's learned depth scale (which is already close to metric, as observed when extrinsics=None)
        
        # GLB Export (for debugging) - comment/uncomment to enable/disable
        # glb_points_np, glb_colors_np = self._export_glb_debug(
        #     prediction=prediction,
        #     multi_batch_ori_imgs=multi_batch_ori_imgs,
        #     multi_batch_extrinsics_gt=multi_batch_extrinsics_gt,
        # )
        glb_points_np = None
        glb_colors_np = None
        
        if prediction is not None:
            # Back-project depth maps to point clouds (batched)
            # Convert depth to torch tensor if needed and ensure correct shape (B, N, H, W)
            if isinstance(prediction.depth, np.ndarray):
                depth_torch = torch.from_numpy(prediction.depth).to(device=device, dtype=torch.float32)
            elif isinstance(prediction.depth, torch.Tensor):
                depth_torch = prediction.depth.to(device=device, dtype=torch.float32)
            else:
                depth_torch = prediction.depth
            
            # Add batch dimension if needed: (N, H, W) -> (1, N, H, W)
            if depth_torch.ndim == 3:  # (N, H, W)
                depth_torch = depth_torch.unsqueeze(0)  # (1, N, H, W)
            
            multi_batch_depths = depth_torch
            B, N, H, W = multi_batch_depths.shape  # Get actual batch size
            
            # Convert intrinsics to torch tensor if needed and ensure correct shape (B, N, 3, 3)
            if isinstance(prediction.intrinsics, np.ndarray):
                intrinsics_torch = torch.from_numpy(prediction.intrinsics).to(device=device, dtype=torch.float32)
            elif isinstance(prediction.intrinsics, torch.Tensor):
                intrinsics_torch = prediction.intrinsics.to(device=device, dtype=torch.float32)
            else:
                intrinsics_torch = prediction.intrinsics
            
            # Add batch dimension if needed: (N, 3, 3) -> (B, N, 3, 3)
            if intrinsics_torch.ndim == 3:  # (N, 3, 3)
                intrinsics_torch = intrinsics_torch.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, 3, 3)
            elif intrinsics_torch.ndim == 4 and intrinsics_torch.shape[0] == 1:  # (1, N, 3, 3) but B > 1
                intrinsics_torch = intrinsics_torch.expand(B, -1, -1, -1)  # (B, N, 3, 3)
            
            multi_batch_intrinsics = intrinsics_torch
            multi_batch_cam2lidar_rts = self._extract_cam2lidar_rts_from_meta(img_metas, device=device)

            # Extract confidence maps and calculate threshold (same as GLB export)
            multi_batch_confs = None
            conf_thr = None
            if prediction.conf is not None and self.conf_thresh_percentile is not None:
                # Convert confidence to torch tensor if needed
                if isinstance(prediction.conf, np.ndarray):
                    conf_torch = torch.from_numpy(prediction.conf).to(device=device, dtype=torch.float32)
                elif isinstance(prediction.conf, torch.Tensor):
                    conf_torch = prediction.conf.to(device=device, dtype=torch.float32)
                else:
                    conf_torch = prediction.conf
                
                # Add batch dimension if needed: (N, H, W) -> (B, N, H, W)
                if conf_torch.ndim == 3:  # (N, H, W)
                    conf_torch = conf_torch.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, H, W)
                elif conf_torch.ndim == 4 and conf_torch.shape[0] == 1:  # (1, N, H, W) but B > 1
                    conf_torch = conf_torch.expand(B, -1, -1, -1)  # (B, N, H, W)
                
                multi_batch_confs = conf_torch
                
                # Calculate confidence threshold using percentile (same as GLB export)
                conf_flat = conf_torch.reshape(-1).cpu().numpy()
                lower = np.percentile(conf_flat, self.conf_thresh_percentile)
                upper = np.percentile(conf_flat, self.ensure_thresh_percentile)
                conf_thr = float(min(max(self.base_conf_thresh, lower), upper))
                # print(f"[DEBUG] Confidence threshold: {conf_thr:.4f} (percentile: {self.conf_thresh_percentile}%, ensure: {self.ensure_thresh_percentile}%, base: {self.base_conf_thresh})")

            # Back-project all batch items at once (returns lists of length B)
            all_points_batch, all_colors_batch = self._backproject_depth_to_points(
                multi_batch_depths,
                multi_batch_intrinsics,
                multi_batch_ori_imgs,
                multi_batch_cam2lidar_rts,
                multi_batch_confs,
                conf_thr,
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
            # Each element should be (N, 3), not (B, N, 3)
            gt_points_list = None
            if points is not None:
                if isinstance(points, list):
                    gt_points_list = [p.float().to(device) for p in points if p is not None]
                elif isinstance(points, torch.Tensor):
                    if points.dim() == 3:  # (B, N, 3)
                        gt_points_list = [points[i].float().to(device) for i in range(points.shape[0])]
                    elif points.dim() == 2:  # (N, 3) - single point cloud, expand to batch
                        # Expand to (B, N, 3) then split into list of (N, 3) tensors (one per batch item)
                        points_expanded = points.unsqueeze(0).expand(B, -1, -1).float().to(device)
                        gt_points_list = [points_expanded[i] for i in range(B)]
                    else:
                        raise ValueError(f"Unexpected points tensor dimension: {points.dim()}, shape: {points.shape}")

            # Colorize GT points if available using lidar2img (only if refinement uses colors)
            if gt_points_list is not None:
                # Check if refinement module uses colors
                use_color = getattr(self.refinement, 'use_color', False) if self.refinement is not None else False
                if use_color:
                    multi_batch_lidar2img = self._extract_lidar2img_from_meta(img_metas, device=device)
                    gt_points_list = self._get_gt_color_points(
                        gt_points_list=gt_points_list,
                        multi_batch_ori_imgs=multi_batch_ori_imgs,
                        multi_batch_lidar2img=multi_batch_lidar2img,
                    )
                # If not using colors, gt_points_list remains as XYZ only (B, N, 3)
            
            # Visualization for debugging (similar to forward_train)
            for b_idx in range(B):
                # print(img_metas[b_idx]['filename'])
                
                # Extract GT bboxes from img_metas for visualization (if available)
                # Note: gt_bboxes_3d should be in main data dict, not img_metas
                # This code is for backward compatibility if it's still in img_metas
                gt_bboxes_3d_batch = None
                if 'gt_bboxes_3d' in img_metas[b_idx]:
                    gt_bboxes_3d_batch = img_metas[b_idx]['gt_bboxes_3d']
                    # Handle DataContainer wrapper
                    if isinstance(gt_bboxes_3d_batch, DC):
                        gt_bboxes_3d_batch = gt_bboxes_3d_batch.data
                    # Handle list/tuple wrapping (DataContainer.data might be a list)
                    if isinstance(gt_bboxes_3d_batch, (list, tuple)) and len(gt_bboxes_3d_batch) > 0:
                        # If it's a list, take the first element (batch item)
                        gt_bboxes_3d_batch = gt_bboxes_3d_batch[0] if len(gt_bboxes_3d_batch) == 1 else gt_bboxes_3d_batch[b_idx]
                    
                    if gt_bboxes_3d_batch is not None:
                        # Convert to LiDARInstance3DBoxes and convert to original gravity
                        from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
                        
                        # Get tensor from various formats
                        if isinstance(gt_bboxes_3d_batch, LiDARInstance3DBoxes):
                            bbox_tensor = gt_bboxes_3d_batch.tensor
                        elif isinstance(gt_bboxes_3d_batch, torch.Tensor):
                            bbox_tensor = gt_bboxes_3d_batch
                        elif isinstance(gt_bboxes_3d_batch, np.ndarray):
                            bbox_tensor = torch.from_numpy(gt_bboxes_3d_batch)
                        elif hasattr(gt_bboxes_3d_batch, 'tensor'):
                            bbox_tensor = gt_bboxes_3d_batch.tensor
                        else:
                            print(f"[WARNING] Unknown bbox format: {type(gt_bboxes_3d_batch)}, skipping visualization")
                            gt_bboxes_3d_batch = None
                            continue
                        
                        # Create LiDARInstance3DBoxes with origin (0.5, 0.5, 0.5) and convert to LIDAR
                        gt_bboxes_3d_ori = LiDARInstance3DBoxes(
                            bbox_tensor,
                            box_dim=bbox_tensor.shape[-1],
                            origin=(0.5, 0.5, -0.5)
                        ).convert_to(Box3DMode.LIDAR)
                        
                        # Extract tensor and apply transformations
                        gt_boxes_3d_ori = gt_bboxes_3d_ori.tensor.cpu().numpy()
                        
                        # Apply transformations for visualization
                        # Flip yaw angle
                        gt_boxes_3d_ori[:, 6] = -gt_boxes_3d_ori[:, 6] - np.pi / 2
                        
                        # Swap width and length for visualization
                        gt_boxes_3d_ori[..., [3, 4]] = gt_boxes_3d_ori[..., [4, 3]]
                        
                        # Replace the original with transformed version
                        gt_bboxes_3d_batch = gt_boxes_3d_ori
                        
                        bbox_count = len(gt_bboxes_3d_batch)
                        # print(f"[DEBUG] Found {bbox_count} GT bounding boxes for batch {b_idx} (converted to original gravity)")
                
                # COMMENTED OUT: Disable visualization during testing
                # # Only visualize if GT points are available
                # if gt_points_list is not None and len(gt_points_list) > b_idx:
                #     gt_points_np = gt_points_list[b_idx].cpu().numpy()
                #     
                #     # Extract colors only if points have more than 3 columns
                #     if gt_points_np.shape[1] > 3:
                #         gt_colors = gt_points_np[:, 3:6]  # Extract RGB (assuming 6 columns: xyzrgb)
                #         # Ensure colors are in correct shape (N, 3)
                #         if gt_colors.shape[1] != 3:
                #             gt_colors = None
                #     else:
                #         gt_colors = None
                #     
                #     if gt_colors is not None:
                #         print(f"[DEBUG] GT point cloud: {len(gt_points_np)} points with colors shape {gt_colors.shape}")
                #     else:
                #         print(f"[DEBUG] GT point cloud: {len(gt_points_np)} points without colors (will use gray)")
                #     
                #     display_point_cloud(gt_points_np, colors=gt_colors, gt_bboxes_3d=gt_bboxes_3d_batch, window_name=f"GT Point Cloud ({len(gt_points_np):,} points)")
                # 
                # # Always visualize pseudo points (they should always be available)
                # if len(pseudo_points_list) > b_idx:
                #     pseudo_points_np = pseudo_points_list[b_idx].cpu().numpy()
                #     pseudo_colors = pseudo_points_np[:, 3:] if pseudo_points_np.shape[1] > 3 else None
                #     display_point_cloud(pseudo_points_np, colors=pseudo_colors, gt_bboxes_3d=None, window_name=f"Pseudo Point Cloud ({len(pseudo_points_np):,} points)")
                # 
                # # Display GLB point cloud if available (3rd visualization)
                # if glb_points_np is not None:
                #     if glb_colors_np is not None:
                #         print(f"[DEBUG] GLB point cloud: {len(glb_points_np)} points with colors shape {glb_colors_np.shape}, dtype={glb_colors_np.dtype}, range=[{glb_colors_np.min():.1f}, {glb_colors_np.max():.1f}]")
                #     else:
                #         print(f"[DEBUG] GLB point cloud: {len(glb_points_np)} points without colors (will use gray)")
                #     display_point_cloud(glb_points_np, colors=glb_colors_np, gt_bboxes_3d=gt_bboxes_3d_batch, window_name=f"GLB Point Cloud ({len(glb_points_np):,} points)")

            padded_pseudo, _ = self._padding_samples(pseudo_points_list, None)
            
            # Prepare GT points if available (for metrics computation)
            padded_gt = None
            if gt_points_list is not None:
                # Pad GT points separately (need to match pseudo padding)
                # Get target_num_points from padded_pseudo
                target_num_points = padded_pseudo.shape[1]  # (B, N, C) -> N
                padded_gt_list = self._pad_point_clouds(gt_points_list, target_num_points)
                padded_gt = torch.stack(padded_gt_list, dim=0)  # (B, N, C)
            
            # Refine entire batch at once (with GT if available for metrics)
            sparse_feat_dict, metrics = self.refinement(
                pseudo_points=padded_pseudo,  # (B, N, C) tensor
                gt_points=padded_gt,  # GT points if available
                return_loss=False,  # No loss computation in test
            )
            
            # Store sparse features dict for detection pipeline
            pts_feat['rescon_features'] = sparse_feat_dict
            

        
        return pts_feat, metrics
        



# add visualizatin function here for debugging purposes
def display_point_cloud(points, colors=None, gt_bboxes_3d=None, window_name="Point Cloud"):
    import open3d as o3d
    """Display point cloud using open3d.
    
    Args:
        points (np.ndarray): Point cloud as numpy array of shape (N, 3)
        colors (np.ndarray, optional): Colors as numpy array of shape (N, 3) in [0, 1]
        gt_bboxes_3d (list, optional): List of ground truth 3D bounding boxes
        window_name (str): Window title for the visualization. Default: "Point Cloud"
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
    if colors is not None and colors.size > 0:
        # Ensure colors have correct shape
        if len(colors.shape) == 1:
            # If 1D, reshape to (N, 3) assuming it's flattened RGB
            if colors.size % 3 == 0:
                colors = colors.reshape(-1, 3)
            else:
                colors = None
        elif len(colors.shape) == 2:
            if colors.shape[1] == 3:  # (N, 3) - correct shape
                pass
            elif colors.shape[1] > 3:
                # Take first 3 channels (RGB)
                colors = colors[:, :3]
            else:
                colors = None
        
        if colors is not None and colors.shape[1] == 3:
            # Normalize colors to [0, 1] range for Open3D
            # Open3D expects float colors in [0, 1] range
            if colors.dtype == np.uint8 or colors.max() > 1.0:
                # Colors are in [0, 255], normalize to [0, 1]
                colors = colors.astype(np.float32) / 255.0
            else:
                colors = colors.astype(np.float32)
            # Ensure colors are in valid range [0, 1]
            colors = np.clip(colors, 0.0, 1.0)
            # Verify color shape matches point count
            if colors.shape[0] != points.shape[0]:
                print(f"[WARNING] Color count {colors.shape[0]} doesn't match point count {points.shape[0]}, using gray")
                pcd.paint_uniform_color([0.5, 0.5, 0.5])
            else:
                pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Invalid color shape, use default
            # print(f"[DEBUG] Invalid color shape: {colors.shape if colors is not None else None}, using gray")
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
    else:
        # Default gray color (white on white background is invisible)
        # print(f"[DEBUG] No colors provided, using gray")
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1920, height=1080)
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
    if gt_bboxes_3d is not None:
        # Handle DataContainer wrapper
        if isinstance(gt_bboxes_3d, DC):
            gt_bboxes_3d = gt_bboxes_3d.data
        
        # Convert to numpy array if needed
        if isinstance(gt_bboxes_3d, np.ndarray) and len(gt_bboxes_3d.shape) == 2:
            bboxes_array = gt_bboxes_3d  # (N, 7)
        elif hasattr(gt_bboxes_3d, '__len__') and len(gt_bboxes_3d) > 0:
            # Try to extract as list of tensors/arrays
            bboxes_list = []
            for bbox in gt_bboxes_3d:
                if isinstance(bbox, torch.Tensor):
                    bboxes_list.append(bbox.cpu().numpy())
                elif isinstance(bbox, np.ndarray):
                    bboxes_list.append(bbox)
                elif hasattr(bbox, 'tensor'):
                    bboxes_list.append(bbox.tensor.cpu().numpy())
                else:
                    continue
            if bboxes_list:
                bboxes_array = np.array(bboxes_list)  # (N, 7)
            else:
                bboxes_array = None
        else:
            bboxes_array = None
        
        # Draw all boxes
        if bboxes_array is not None and len(bboxes_array) > 0:
            print(f"  Adding {len(bboxes_array)} bounding boxes to visualization")
            for bbox in bboxes_array:
                if len(bbox) < 7:
                    continue
                
                center = bbox[:3]  # x, y, z
                size = bbox[3:6]  # w, l, h
                yaw = bbox[6]  # yaw angle
                
                # Create rotation matrix from yaw
                cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                rotation_matrix = np.array([
                    [cos_yaw, -sin_yaw, 0],
                    [sin_yaw, cos_yaw, 0],
                    [0, 0, 1]
                ])
                
                # Create OrientedBoundingBox
                obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, size)
                obb.color = [1, 0, 0]
                vis.add_geometry(obb)
                
                # Color points inside box
                indices = obb.get_point_indices_within_bounding_box(pcd.points)
                if len(indices) > 0:
                    colors_array = np.asarray(pcd.colors)
                    colors_array[indices] = [1, 0, 0]
                    pcd.colors = o3d.utility.Vector3dVector(colors_array)
                    vis.update_geometry(pcd)
                
                # Draw heading direction
                heading_dir = rotation_matrix[:2, 0]
                yaw_vis = np.arctan2(heading_dir[1], heading_dir[0])
                front_center = center + size[0] * np.array([np.cos(yaw_vis), np.sin(yaw_vis), 0])
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector([center, front_center])
                line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0]])
                vis.add_geometry(line_set)
    
    # Block until window is closed
    vis.run()
    vis.destroy_window()