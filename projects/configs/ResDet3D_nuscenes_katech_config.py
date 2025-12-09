"""
Inference configuration file for nuScenes inference script.
Modify these settings to control model behavior and post-processing.
"""

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
# Device to use for inference
# Options: "cuda" or "cpu"
DEVICE = "cuda"  # Will auto-fallback to "cpu" if CUDA not available


# ============================================================================
# Post-processing pipeline (mimic mmdet3d style)
# Each step receives/returns a dict with at least: points, colors, polygon_mask
# ============================================================================
test_pipeline = [
    
    # Voxel downsample (always runs if voxel_size is not None)
    dict(
        type="VoxelDownsample",
        voxel_size=0.1,
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    ),
    # Density-aware ball query (optional)
    dict(
        type="BallQueryDownsample",
        enabled=True,
        min_radius=0.0,
        max_radius=0.5,
        sample_num=16,
        anchor_points=25000,
    ),
    # Uniform cap with FPS (optional)
    dict(
        type="FPSDownsample",
        enabled=True,
        num_points=40000,
    ),
]




data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=f'{data_root}BATCH1/',
                    ann_file=data_root + f'BATCH1/{data_name}_infos_train.pkl',
                    custom_eval_set=data_name if data_name != 'nuscenes' else None,
                    load_interval=1,
                    pipeline=train_pipeline,
                    classes=class_names,
                    modality=input_modality,
                    test_mode=False,
                    box_type_3d='LiDAR'),
                dict(
                    type=dataset_type,
                    data_root=f'{data_root}BATCH2/',
                    ann_file=data_root + f'BATCH2/{data_name}_infos_train.pkl',
                    custom_eval_set=data_name if data_name != 'nuscenes' else None,
                    load_interval=1,
                    pipeline=train_pipeline,
                    classes=class_names,
                    modality=input_modality,
                    test_mode=False,
                    box_type_3d='LiDAR'),
            ])),

    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=f'{data_root}BATCH1/',
                ann_file=data_root + f'BATCH1/{data_name}_infos_val.pkl',
                custom_eval_set=data_name if data_name != 'nuscenes' else None,
                load_interval=1,
                pipeline=test_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=True,
                box_type_3d='LiDAR'),
            dict(
                type=dataset_type,
                data_root=f'{data_root}BATCH2/',
                ann_file=data_root + f'BATCH2/{data_name}_infos_val.pkl',
                custom_eval_set=data_name if data_name != 'nuscenes' else None,
                load_interval=1,
                pipeline=test_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=True,
                box_type_3d='LiDAR'),
        ]),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=f'{data_root}BATCH1/',
                ann_file=data_root + f'BATCH1/{data_name}_infos_val.pkl',
                custom_eval_set=data_name if data_name != 'nuscenes' else None,
                load_interval=1,
                pipeline=test_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=True,
                box_type_3d='LiDAR'),
            dict(
                type=dataset_type,
                data_root=f'{data_root}BATCH2/',
                ann_file=data_root + f'BATCH2/{data_name}_infos_val.pkl',
                custom_eval_set=data_name if data_name != 'nuscenes' else None,
                load_interval=1,
                pipeline=test_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=True,
                box_type_3d='LiDAR'),
        ]),
    )




model = dict(
    type='ResDet3D',
    reconstruction_model=dict(
        type='DepthAnything3',
        pretrained = "depth-anything/DA3NESTED-GIANT-LARGE",
        cam_types=['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        export_format = "glb",
        ref_view_strategy = "saddle_balanced",
        use_ray_pose = False,
        max_points = 1_000_000,
        glb_config = dict(
            sky_depth_def = 98.0,
            conf_thresh_percentile = 30.0,
            filter_black_bg = False,
            filter_white_bg = False,
            max_depth = 100.0,
        ),
        
    )

)



