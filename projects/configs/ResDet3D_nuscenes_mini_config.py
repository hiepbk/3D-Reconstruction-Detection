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
evaluation = dict(interval=1)


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
            'lidar2img', 'cam2lidar_rts',
            'pad_shape', 'scale_factor',
            'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
            'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
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
                with_label=False),
            dict(
                type='Collect3D',
                keys=['points', 'img'],
                meta_keys=(
                    'filename', 'ori_shape', 'img_shape',
                    'lidar2img', 'cam2lidar_rts',
                    'pad_shape', 'scale_factor',
                    'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
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
    samples_per_gpu=1,
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
        test_mode=True,
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
            sky_depth_def=98.0,
            conf_thresh_percentile=30.0,
            filter_black_bg=False,
            filter_white_bg=False,
            max_depth=100.0,
        ),
        ref_view_strategy="saddle_balanced",
        use_ray_pose=False,
        max_points=1_000_000,
        filter_sky=True,
        max_depth=100.0,
        conf_thresh_percentile=30.0,
        freeze_da3=True,  # Freeze DepthAnything3 model (recommended)
        refinement=dict(
            type='SparseRefinement',
            use_color=False,  # Set to False to disable color processing (only use XYZ)
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
                type='SparseEncoder',
                in_channels=3,  # Should match num_features in voxel_encoder
                sparse_shape=[41, 1440, 1440],  # [Z, Y, X] calculated from point_cloud_range and voxel_size
                output_channels=128,
                order=('conv', 'norm', 'act'),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
                block_type='basicblock',
            ),
            # Feature loss: compares sparse features between pseudo and GT
            loss_feature=dict(
                type='SimpleL2Loss',
                reduction='mean',
                loss_weight=1.0,
            ),
            loss_weight=1.0,  # Weight for feature loss
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
    pts_middle_encoder=None,
    pts_backbone=None,
    pts_neck=None,
    imgpts_neck=None,
    pts_bbox_head=None,
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            # code weights related to feature of bbox, 10, not related to number of classes
            # x,y,z,w,l,h,rot,velx,vely,velz
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        ))
    
    )

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
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
    interval=1,
    hooks=[dict(type='ComponentMemoryLoggerHook'),  # Custom hook with component memory breakdown (extends TextLoggerHook)
           dict(type='TensorboardLoggerHook'),
        #    dict(type='WandbLoggerHook',
        #         init_kwargs=dict(
        #             project='ResDet3D',
        #             name=f'ResDet3D_nuscenes_mini',
        #         ))
           ])

custom_hooks = [
]




dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 4)
find_unused_parameters = True

