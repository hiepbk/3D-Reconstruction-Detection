# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

# import mmcv
# import torch
# from mmcv.parallel import DataContainer as DC
# from mmcv.runner import force_fp32
# from os import path as osp
# from torch import nn as nn
# from torch.nn import functional as F

# from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result, show_result)
# from mmdet3d.ops import Voxelization
# from mmdet.core import multi_apply

from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet.models import DETECTORS
# from projects.mmdet3d_plugin.models.utils.time_utils import T
# from projects.mmdet3d_plugin.core.post_processing.merge_augs import merge_aug_bboxes_3d

@DETECTORS.register_module()
class ResDet3D(MVXTwoStageDetector):
    def __init__(self,
                 reconstruction_backbone=None,
                 freeze_img=False,
                 freeze_img_level=None,
                 freeze_camlss=False,
                 freeze_pts=False,
                 trainneck_ms=False,
                 train_middle_encoder=False,
                 pts_pillar_layer=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 imgpts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_img=True,
                 use_grid_mask=False,
                 input_pts=True,
                 init_cfg=None):
        super(ResDet3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)
        # if pts_pillar_layer:
        #     self.pts_pillar_layer = Voxelization(**pts_pillar_layer)
        # self.freeze_img_level = freeze_img_level
        # self.freeze_camlss = freeze_camlss
        # self.imgpts_neck = builder.build_neck(imgpts_neck)
        
        # self.freeze_img = freeze_img
        # self.freeze_pts = freeze_pts
        # self.trainneck_ms = trainneck_ms
        # self.train_middle_encoder = train_middle_encoder

        # self.input_img = input_img
        # self.input_pts = input_pts

        # self.use_grid_mask = use_grid_mask
        # if self.use_grid_mask:
        #     from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
        #     self.grid_mask = GridMask(
        #         True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        
        # self.apply_dynamic_voxelize = 'Dynamic' in pts_voxel_encoder['type']
        
        
        # Build reconstruction backbone
        if reconstruction_backbone is not None:
            print(f"[DEBUG] ResDet3D: Building reconstruction backbone...")
            print(f"[DEBUG] ResDet3D: reconstruction_backbone config = {reconstruction_backbone}")
            self.reconstruction_backbone = builder.build_backbone(reconstruction_backbone)
            print(f"[DEBUG] ResDet3D: Reconstruction backbone built successfully")
        else:
            self.reconstruction_backbone = None
            print(f"[DEBUG] ResDet3D: No reconstruction backbone configured")
    
    def extract_feat(self, points=None, img=None, img_metas=None, return_loss=False):
        """Extract features using reconstruction backbone.
        
        This generates point clouds from multi-view images using DepthAnything3.
        The point cloud can then be used by the detection head.
        
        Args:
            points: Ground truth point cloud (optional, for training with refinement)
            img: Multi-view images (B, N, 3, H, W) or DataContainer
            img_metas: Image metadata list
            return_loss: Whether to return losses (for training)
        
        Returns:
            tuple: (img_feats, pts_feats) where pts_feats is the generated point cloud
            If return_loss=True and refinement enabled, also returns losses dict
        """
        if self.reconstruction_backbone is not None:
            # Forward through reconstruction backbone
            result = self.reconstruction_backbone(img, img_metas, return_loss=return_loss, points=points)
            
            # Handle new return format: (batch_point_clouds, losses)
            if isinstance(result, tuple) and len(result) == 2:
                pseudo_points, losses = result
                # Store losses for later use in loss computation
                if losses is not None:
                    self._reconstruction_losses = losses
            else:
                # Backward compatibility: just point clouds
                pseudo_points = result
                losses = None
            
            # Return format matching parent class: (img_feats, pts_feats)
            # For now, img_feats is None since we're only using reconstruction
            return (None, pseudo_points)
        else:
            # No reconstruction backbone, fall back to parent behavior
            return super().extract_feat(points, img, img_metas)
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward function for training.
        
        Override to handle refinement losses from reconstruction backbone.
        """
        losses = dict()
        
        if self.reconstruction_backbone is not None:
            # Extract features (generates point cloud with refinement)
            img_feats, pseudo_points = self.extract_feat(
                points=points,  # GT points for refinement loss
                img=img,
                img_metas=img_metas,
                return_loss=True
            )
            
            # Add refinement losses if available
            if hasattr(self, '_reconstruction_losses') and self._reconstruction_losses is not None:
                # Prefix losses with 'reconstruction_' to avoid conflicts
                for key, value in self._reconstruction_losses.items():
                    losses[f'reconstruction_{key}'] = value
                # Clear stored losses
                delattr(self, '_reconstruction_losses')
        
        # Call parent forward_train for detection losses (if head/neck exist)
        has_pts_bbox_head = hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None
        has_img_backbone = hasattr(self, 'img_backbone') and self.img_backbone is not None
        
        if has_pts_bbox_head or has_img_backbone:
            parent_losses = super().forward_train(
                points=pseudo_points if self.reconstruction_backbone is not None else points,
                img_metas=img_metas,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                img=img,
                proposals=proposals,
                gt_bboxes_ignore=gt_bboxes_ignore
            )
            losses.update(parent_losses)
        
        return losses
    
    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentation.
        
        Override to handle case where we don't have detection head/neck yet.
        Just pass through the point cloud for now.
        """
        if self.reconstruction_backbone is not None:
            # Extract features (generates point cloud)
            img_feats, pseudo_points = self.extract_feat(points, img=img, img_metas=img_metas)
            
            # Handle tuple return (backward compatibility)
            if isinstance(pseudo_points, tuple):
                pseudo_points, _ = pseudo_points
            
            # For now, return empty results since we don't have head/neck
            # Later, when head/neck are added, this will call simple_test_pts
            bbox_list = [dict() for i in range(len(img_metas))]
            
            # Store pseudo points and colors in result for potential use
            if pseudo_points is not None:
                # Handle list of point clouds (one per batch item)
                if isinstance(pseudo_points, list):
                    for i, result_dict in enumerate(bbox_list):
                        if i < len(pseudo_points):
                            result_dict['pseudo_points'] = pseudo_points[i]
                else:
                    # Single point cloud for all
                    for i, result_dict in enumerate(bbox_list):
                        result_dict['pseudo_points'] = pseudo_points
            
            return bbox_list
        else:
            # Fall back to parent behavior
            return super().simple_test(points, img_metas, img=img, rescale=rescale)
        
        