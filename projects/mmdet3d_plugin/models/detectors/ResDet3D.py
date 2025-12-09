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
    
    def extract_feat(self, points=None, img=None, img_metas=None):
        """Extract features using reconstruction backbone.
        
        This generates point clouds from multi-view images using DepthAnything3.
        The point cloud can then be used by the detection head.
        
        Args:
            points: Point cloud (optional, not used when reconstruction_backbone exists)
            img: Multi-view images (B, N, 3, H, W) or DataContainer
            img_metas: Image metadata list
        
        Returns:
            tuple: (img_feats, pts_feats) where pts_feats is the generated point cloud
        """
        if self.reconstruction_backbone is not None:
            # Generate point cloud from images
            generated_points = self.reconstruction_backbone(img, img_metas)
            # Return format matching parent class: (img_feats, pts_feats)
            # For now, img_feats is None since we're only using reconstruction
            return (None, generated_points)
        else:
            # No reconstruction backbone, fall back to parent behavior
            return super().extract_feat(points, img, img_metas)
    
    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentation.
        
        Override to handle case where we don't have detection head/neck yet.
        Just pass through the point cloud for now.
        """
        if self.reconstruction_backbone is not None:
            # Extract features (generates point cloud)
            img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
            
            # For now, return empty results since we don't have head/neck
            # Later, when head/neck are added, this will call simple_test_pts
            bbox_list = [dict() for i in range(len(img_metas))]
            
            # Store generated points in result for potential use
            if pts_feats is not None:
                for result_dict in bbox_list:
                    result_dict['generated_points'] = pts_feats
            
            return bbox_list
        else:
            # Fall back to parent behavior
            return super().simple_test(points, img_metas, img=img, rescale=rescale)
        
        