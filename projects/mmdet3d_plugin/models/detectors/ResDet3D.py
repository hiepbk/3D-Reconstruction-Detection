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
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result, show_result)
from mmdet.models import DETECTORS, build_loss
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
        # Follow CenterPoint architecture:
        # reconstruction_backbone (SparseEncoder) = pts_middle_encoder
        # Then: pts_backbone (SECOND) → pts_neck (SECONDFPN) → pts_bbox_head (CenterHead)
        super(ResDet3D, self).__init__(
            pts_voxel_layer=pts_voxel_layer,
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,  # Not used (we use reconstruction_backbone)
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,  # SECOND backbone (required, like CenterPoint)
            img_neck=img_neck,
            pts_neck=pts_neck,  # SECONDFPN neck
            pts_bbox_head=pts_bbox_head,  # CenterHead
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg
        )


        self.input_img = input_img
        self.input_pts = input_pts
        
        if reconstruction_backbone is not None:
            self.reconstruction_backbone = builder.build_backbone(reconstruction_backbone)
        else:
            self.reconstruction_backbone = None
            
        if self.with_reconstruction_backbone:
            self.reconstruction_backbone.init_cfg = dict(
                type='Pretrained', checkpoint=reconstruction_backbone['pretrained'])
        
    @property
    def with_reconstruction_backbone(self):
        """bool: Whether the detector has a reconstruction backbone."""
        return hasattr(self, 'reconstruction_backbone') and self.reconstruction_backbone is not None
        

     
            
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # if self.with_img_backbone and img is not None:
        #     input_shape = img.shape[-2:]
        #     # update real input shape of each single img
        #     for img_meta in img_metas:
        #         img_meta.update(input_shape=input_shape)

        #     if img.dim() == 5 and img.size(0) == 1:
        #         img = img.squeeze(0)
        #     elif img.dim() == 5 and img.size(0) > 1:
        #         B, N, C, H, W = img.size()
        #         img = img.view(B * N, C, H, W)
        #     if self.use_grid_mask and self.training:
        #         img = self.grid_mask(img)
        #     img_feats = self.img_backbone(img.float())
        # else:
        #     return None
        # if self.with_img_neck:
        #     img_feats = self.img_neck(img_feats)
        
        # For now, we don't use image features extraction
        
        raise NotImplementedError("Image features extraction is not implemented yet")
    
    def extract_pts_feat(self, points, img_feats, img_metas, return_loss=False):
        # extra can be losses or the metrics depending on the return_loss
        pts_con_feat_dict, extra = self.reconstruction_backbone(
            img=img_feats, 
            img_metas=img_metas, 
            return_loss=return_loss, 
            points=points
        )
    
        # Extract dense features for detection pipeline
        # Phase 1 (train teacher): use GT dense features (teacher learns from GT)
        # Phase 2 (train student): use pseudo dense features (student learns from pseudo)
        # Evaluation: route based on training phase (Phase 1 = evaluate teacher, Phase 2 = evaluate student)
        training_phase = getattr(self.reconstruction_backbone, 'training_phase', 2)
        
        if return_loss:
            # Training mode: Phase 1 uses GT, Phase 2 uses pseudo
            if training_phase == 1:
                # Phase 1: Train teacher with GT features
                x = pts_con_feat_dict['rescon_features']['gt']['dense_features']
            else:
                # Phase 2: Train student with pseudo features
                x = pts_con_feat_dict['rescon_features']['pseudo']['dense_features']
        else:
            # Evaluation/Test mode: route based on training phase
            if training_phase == 1:
                # Phase 1: Evaluate teacher performance (use GT features)
                x = pts_con_feat_dict['rescon_features']['gt']['dense_features']
            else:
                # Phase 2: Evaluate student performance (use pseudo features)
                x = pts_con_feat_dict['rescon_features']['pseudo']['dense_features']
            
        # Follow CenterPoint: dense_features → SECOND backbone → SECONDFPN neck
        if self.with_pts_backbone:
            x = self.pts_backbone(x)  # Returns tuple of tensors
            # Convert tuple to list for SECONDFPN
            if isinstance(x, tuple):
                x = list(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)  # Returns [out] (list with one tensor)
        
        # Return features from neck (not bbox head output)
        # bbox head will be called in forward_pts_train
        return x, extra
    
    def extract_feat(self, points, img, img_metas, return_loss=False):
        # For now, it will be not used
        if self.input_img:
            img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = None
        if self.input_pts:
            pts_feats, extra = self.extract_pts_feat(points, img, img_metas, return_loss=return_loss)
        else:
            pts_feats = None
            
        # we will implement this later
        # new_img_feat, new_pts_feat = self.imgpts_neck(img_feats[0], pts_feats[0], img_metas)
        return (img_feats, pts_feats, extra)
    
    
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

        # Extract features (generates point cloud with refinement)
        img_feats, pts_feats, extra = self.extract_feat(
            points=points,  # GT points for refinement loss
            img=img,
            img_metas=img_metas,
            return_loss=True
        )
        # Extract losses from extra dict (from reconstruction_backbone)
        if isinstance(extra, dict):
            losses.update(extra)

        losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                    gt_labels_3d, img_metas,
                                    gt_bboxes_ignore)
        # update the feature losses of detector here
        losses.update(losses_pts)
        return losses
    
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        # Ensure pts_feats is a list of tensors (not nested list or tuple)
        # SECONDFPN returns [out] which is correct format for CenterHead
        if isinstance(pts_feats, tuple):
            pts_feats = list(pts_feats)
        elif isinstance(pts_feats, list) and len(pts_feats) > 0:
            # Check if first element is also a list (double-wrapped)
            if isinstance(pts_feats[0], (list, tuple)):
                # Unwrap: [[tensor]] -> [tensor]
                pts_feats = list(pts_feats[0]) if isinstance(pts_feats[0], tuple) else pts_feats[0]
        
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None, **kwargs):
        """Test function of point cloud branch."""
        # Ensure x is a list of tensors (not nested list or tuple)
        # SECONDFPN returns [out] which is correct format for CenterHead
        if isinstance(x, tuple):
            x = list(x)
        elif isinstance(x, list) and len(x) > 0:
            # Check if first element is also a list (double-wrapped)
            if isinstance(x[0], (list, tuple)):
                # Unwrap: [[tensor]] -> [tensor]
                x = list(x[0]) if isinstance(x[0], tuple) else x[0]
        elif not isinstance(x, list):
            # If x is a single tensor, wrap it in a list
            x = [x]
        
        outs = self.pts_bbox_head(x)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    
    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentation.
        
        Override to handle case where we don't have detection head/neck yet.
        Just pass through the point cloud for now.
        
        Note: **kwargs may contain gt_bboxes_3d, gt_labels_3d from validation pipeline,
        but we ignore them during inference.
        """
        # Filter out GT data that shouldn't be passed to extract_feat
        # in the test mode, it will return the feature metrics of reconstruction backbone
        img_feats, pts_feats, feat_metrics = self.extract_feat(
            points=points,
            img=img,
            img_metas=img_metas,
            return_loss=False
        )
        
        bbox_results = self.simple_test_pts(pts_feats, img_feats, img_metas, rescale=rescale)
        return bbox_results


        
        