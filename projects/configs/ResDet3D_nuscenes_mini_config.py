"""
Inference configuration file for nuScenes inference script.
Modify these settings to control model behavior and post-processing.
"""

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8
evaluation = dict(interval=1,
                  show=False,
                  out_dir=f'work_dirs/ResDet3D_nuscenes_mini/vis_results',
                  vis_time=None,
                  score_3d_threshold=0.5,
                  )


load_dim = 5
use_dim = [0, 1, 2] # use x,y,z only, set use_color to False because the original point cloud has no color
use_color = False

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes_mini/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_scale = (800, 448)
num_views = 6
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

multistage_heatmap = 1
inter_channel = 128
extra_feat = True



# ============================================================================
# Post-processing pipeline (mimic mmdet3d style)
# Each step receives/returns a dict with at least: points, colors
# ============================================================================


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925 * 2, 0.3925 * 2],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[0.5, 0.5, 0.5]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='PointShuffle'),
    # dict(type='ScaleImageMultiViewImage', scales=img_scale),
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',

        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape',
            'lidar2img', 'cam2lidar_rts', 'lidar2cam_rts', 'cam_intrinsic',
            'pad_shape', 'scale_factor',
            'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
            'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
            'gt_bboxes_3d', 'gt_labels_3d',  # Add GT bboxes and labels to meta for visualization
        ),
    )
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            # dict(type='ScaleImageMultiViewImage', scales=img_scale),
            # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            # dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=True),
            dict(
                type='Collect3D',
                # keys=['points', 'img'],
                # meta_keys=(
                #     'filename', 'ori_shape', 'img_shape',
                #     'lidar2img', 'cam2lidar_rts', 'lidar2cam_rts', 'cam_intrinsic',
                #     'pad_shape', 'scale_factor',
                #     'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                #     'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                #     'gt_bboxes_3d', 'gt_labels_3d',  # Add GT bboxes and labels to meta for visualization
                keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
                meta_keys=(
                    'filename', 'ori_shape', 'img_shape',
                    'lidar2img', 'cam2lidar_rts', 'lidar2cam_rts', 'cam_intrinsic',
                    'pad_shape', 'scale_factor',
                    'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                    'gt_bboxes_3d', 'gt_labels_3d',  # Add GT bboxes and labels to meta for visualization
                
                ),
            )
        ])
]

rescon_pipeline = [
    dict(
        type='DepthAnything3Filter',
        
        transforms=[
    # Voxel downsample (always runs if voxel_size is not None)
            # dict(
            #     type='VoxelDownsample',
            #     voxel_size=0.1,
            #     point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            # )
            
            
    dict(
                type='FilterPointByRange', 
                point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 6.0]),
            
    # Density-aware ball query (optional)
    dict(
                type='BallQueryDownsample',
        enabled=True,
        min_radius=0.0,
        max_radius=0.5,
        sample_num=16,
        anchor_points=25000,
    ),
    # Uniform cap with FPS (optional)
    dict(
                type='FPSDownsample',
        enabled=True,
                num_points=40000,  # 40k points for convergence to real LiDAR point clouds
    ),
            
        ]
    )
]






data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_mini_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_mini_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_mini_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))



model = dict(
    type='ResDet3D',
    reconstruction_backbone=dict(
        type='ReconstructionBackbone',
        pretrained="depth-anything/DA3NESTED-GIANT-LARGE",
        cache_dir="ckpts",
        rescon_pipeline=rescon_pipeline,
        glb_config=dict(
            sky_depth_def=95.0,
            conf_thresh_percentile=30.0,
            filter_black_bg=False,
            filter_white_bg=False,
            max_depth=100.0,
        ),
        ref_view_strategy="saddle_balanced",
        use_ray_pose=True,
        max_points=1_000_000,
        filter_sky=False,
        max_depth=100.0,
        conf_thresh_percentile=15.0,  # Lower = more points kept (was 30.0)
        ensure_thresh_percentile=95.0,  # Upper bound for threshold
        base_conf_thresh=1.05,  # Lower = more points kept (was 1.05)
        freeze_da3=True,  # Freeze DepthAnything3 model (recommended)
        export_glb=True,  # Enable GLB export for debugging (set to True to test)
        glb_export_dir="output",  # Directory for GLB export
        refinement=dict(
            type='SparseRefinement',
            use_color=True,  # Set to False to disable color processing (only use XYZ)
            debug_viz=True,
            debug_viz_dir='work_dirs/resdet3d_nuscenes_mini/debug_viz',
            # Voxelization layer: converts point clouds to voxels
            pts_voxel_layer=dict(
                max_num_points=10,  # Maximum points per voxel
                voxel_size=voxel_size,  # [0.075, 0.075, 0.2]
                max_voxels=(120000, 160000),  # (training, testing) max voxels
                point_cloud_range=point_cloud_range,  # [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
            ),
            # Voxel encoder: encodes voxel features
            pts_voxel_encoder=dict(
                type='HardSimpleVFE',
                num_features=3,  # XYZ only (since use_color=False)
            ),
            # Sparse middle encoder: 3D sparse convolutions
            pts_middle_encoder=dict(
                type='SparseEncoderV2',
                in_channels=3,  # Should match num_features in voxel_encoder
                sparse_shape=[41, 1440, 1440],  # [Z, Y, X] calculated from point_cloud_range and voxel_size
                output_channels=128,
                order=('conv', 'norm', 'act'),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                encoder_strides=((1, 1, 2), (1, 1, 2), (1, 1, 2), (1, 1)),
                encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
                block_type='basicblock',
                return_type='sparse',
            ),
            loss_occupancy=dict(
                type='VoxelOccupancyAlignmentLoss',
                loss_type='dice',  # Dice loss recommended (robust to imbalance)
                reduction='mean',
                loss_weight=1.0,
                eps=1e-6,
            ),
            # Loss 2: Sparse Feature Alignment
            # Matches feature values at overlapping voxels (aligns semantics)
            # CRITICAL FIXES:
            # - normalize_features=True: Prevents trivial solution (always normalize)
            # - hard_mining_ratio=0.5: Use hardest 50% voxels to keep gradients alive
            # - loss_type='cosine': Better for normalized features (or use 'l2' with normalization)
            loss_feature=dict(
                type='SparseFeatureAlignmentLoss',
                loss_type='cosine',  # Cosine loss (works with normalized features)
                reduction='mean',
                loss_weight=1,  # Lower weight than occupancy (refinement, not structure)
                eps=1e-6,
                normalize_features=True,  # CRITICAL: Normalize to prevent trivial solution
                hard_mining_ratio=0.5,  # Use hardest 50% of voxels (keeps gradients alive)
            ),
            # Loss 3: Dense BEV Feature Alignment (Auxiliary Loss)
            # Aligns dense BEV features [B, C*D, H, W] using cosine similarity with foreground masking
            # This is a weak regularizer to complement sparse feature alignment
            loss_bev=dict(
                type='DenseBEVFeatureLoss',
                loss_weight=1.0,  # Weak auxiliary loss (much smaller than sparse losses)
                reduction='mean',
                eps=1e-6,
                use_foreground_mask=True,  # Only supervise foreground pixels (avoids background dominance)
                mask_threshold=0.01,  # Threshold for teacher energy mask
                mask_type='teacher_energy',  # 'teacher_energy' or 'topk'
                topk_ratio=0.1,  # Used if mask_type='topk'
            ),
            # Global weight multiplier for all losses
            loss_weight=1.0,
            # Individual loss weights (applied before global weight)
            loss_occupancy_weight=1.0,  # Occupancy loss weight (most important)
            loss_feature_weight=1.0,    # Feature loss weight (refinement)
            loss_bev_weight=0.1,        # Dense BEV loss weight (auxiliary, weak)
        ),
        # refinement=None
    ),
    
    freeze_img=True,
    freeze_pts=True,
    input_img=False,
    # img_backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch'),
    # img_neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     num_outs=5),
    pts_voxel_layer=None,
    pts_voxel_encoder=None,
    pts_middle_encoder=None,  # Not used - we use reconstruction_backbone's SparseEncoder
    # SECOND backbone: processes dense BEV features from SparseEncoder
    # Following CenterPoint exactly:
    # - CenterPoint: SparseEncoder (output_channels=128) → dense [B, 128*2, H, W] = [B, 256, H, W] → SECOND in_channels=256
    # - Ours: SparseEncoder (output_channels=128) → dense [B, 128*D, H, W]
    # Calculation of D from encoder_strides and conv_out (verified from debug output):
    #   encoder_strides: ((1,1,2), (1,1,2), (1,1,2), (1,1)) - 4 stages
    #   - Stage 0: last block stride=2 in Z → 41/2 = 21
    #   - Stage 1: last block stride=2 in Z → 21/2 = 11
    #   - Stage 2: last block stride=2 in Z → 11/2 = 5
    #   - Stage 3: final stage, no stride change → 5
    #   - conv_out: stride=(2,1,1) → 5/2 = 2
    # So D = 2, and in_channels = 128 * 2 = 256 (matches CenterPoint exactly!)
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,  # C*D: 128 (output_channels) * 2 (D after encoder + conv_out) = 256
        out_channels=[128, 256],  # Same as CenterPoint
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    # SECONDFPN neck: processes SECOND backbone outputs
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],  # Output channels from SECOND backbone
        out_channels=[256, 256],  # Same as CenterPoint
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    # Detection head: CenterHead (same as CenterPoint)
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),  # Sum of SECONDFPN out_channels
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],  # Required: [x_min, y_min] = [-54.0, -54.0]
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    imgpts_neck=None,
    train_cfg=dict(
        pts=dict(
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            point_cloud_range=point_cloud_range,  # Required by CenterHead.get_targets_single
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2))
    
    )
    
optimizer = dict(
    type='AdamW', 
    lr=0.001,  # Increased base learning rate (10x from 0.0001) - refinement network will benefit
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=100.0, norm_type=2))  # Increased to 10.0 to allow larger gradients
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 8

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),  # console logging of iter/loss
        dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook',
        #      init_kwargs=dict(
        #          project='ResDet3D',
        #          name=f'ResDet3D_nuscenes_mini',
        #      ))
    ])

custom_hooks = []




dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 4)
find_unused_parameters = True

