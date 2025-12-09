#!/usr/bin/env python3
"""
Inference script for nuScenes using mmdet3d infrastructure.
Uses mmdet3d's data loading, model building, and inference pipeline.

The reconstruction backbone is integrated into ResDet3D and will be called
automatically during model forward pass.

Usage:
    python tools/inference_mmdet3d.py --config projects/configs/ResDet3D_nuscenes_mini_config.py
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings

from mmdet3d.apis import init_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def display_point_cloud(points, sample_token, colors=None, gt_bboxes_3d=None):
    """Display point cloud using open3d.
    
    Args:
        points (np.ndarray): Point cloud as numpy array of shape (N, 3)
        sample_token (str): Sample token for window title
        colors (np.ndarray, optional): Colors as numpy array of shape (N, 3) in [0, 1]
        gt_bboxes_3d (list, optional): List of ground truth 3D bounding boxes
    """
    if points is None or len(points) == 0:
        print(f"  Warning: No point cloud to display for sample {sample_token}")
        return
    
    print(f"  Displaying point cloud with {len(points)} points...")
    print(f"  Press 'Q' or close the window to continue")
    
    # Convert numpy to open3d PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Set colors if provided
    if colors is not None:
        if colors.max() > 1.0:
            # Assume colors are in [0, 255], normalize to [0, 1]
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default color (white)
        pcd.paint_uniform_color([1, 1, 1])
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud - Sample {sample_token[:8]}", width=1920, height=1080)
    vis.add_geometry(pcd)
    
    # Set up view
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, -1])
    view_ctl.set_lookat([0, 0, 0])
    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.7)
    
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


def main():
    parser = argparse.ArgumentParser(
        description="mmdet3d-style inference for ResDet3D with DepthAnything3 reconstruction"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (mmdet3d-style)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (optional, for loading trained weights)",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, key=value format",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Process only a specific sample by index (0-based, optional)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (optional)",
    )
    parser.add_argument(
        "--launcher",
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help="Job launcher for distributed training (default: none = single GPU)",
    )

    args = parser.parse_args()

    # Load config (mimic mmdet3d behavior)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Handle custom imports/plugins (mimic mmdet3d)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            # Remove trailing slash if present
            plugin_dir = plugin_dir.rstrip('/')
            # Convert path to module path (replace / with .)
            _module_path = plugin_dir.replace('/', '.')
            print(f"Importing plugin from: {_module_path}")
            importlib.import_module(_module_path)
        else:
            _module_dir = os.path.dirname(args.config)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(f"Importing plugin from: {_module_path}")
            importlib.import_module(_module_path)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect device (use CUDA if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("ResDet3D - mmdet3d Inference")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Detect if distributed (like test.py)
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        from mmcv.runner import init_dist
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    # Build dataset using mmdet3d infrastructure
    print("Building dataset...")
    dataset = build_dataset(cfg.data.test)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Build data loader
    print("Building data loader...")
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
        dist=distributed,
        shuffle=False,
    )
    print("Data loader built")

    # Build model using mmdet3d infrastructure
    print("Building model...")
    print(f"[DEBUG] Model config keys: {list(cfg.model.keys())}")
    if 'reconstruction_backbone' in cfg.model:
        print(f"[DEBUG] Reconstruction backbone config present: {cfg.model['reconstruction_backbone']}")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    print("[DEBUG] Model built, moving to device...")
    
    # Load checkpoint if provided
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, map_location=device)
    
    print(f"[DEBUG] Moving model to device: {device}")
    model.to(device)
    print(f"[DEBUG] Setting model to eval mode")
    model.eval()
    
    # Wrap model based on distributed mode (like test.py)
    if not distributed:
        # Single GPU: use MMDataParallel
        model = MMDataParallel(model, device_ids=[device.index if device.type == 'cuda' else 0])
        print("Model wrapped with MMDataParallel (single GPU)")
    else:
        # Multi GPU: use MMDistributedDataParallel (for future implementation)
        from mmcv.parallel import MMDistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        print("Model wrapped with MMDistributedDataParallel (multi GPU)")
    print("Model built and ready")

    # Process samples
    print(f"\nProcessing samples...")
    successful = 0
    failed = 0
    
    # Filter samples if needed
    indices = list(range(len(dataset)))
    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(dataset):
            print(f"Error: Sample index {args.sample_index} is out of range (0-{len(dataset)-1})")
            sys.exit(1)
        indices = [args.sample_index]
        print(f"Processing sample at index {args.sample_index}")
    
    if args.max_samples:
        indices = indices[:args.max_samples]
        print(f"Processing first {len(indices)} samples")

    # Process each sample
    for i, idx in enumerate(indices, 1):
        print(f"\n[{i}/{len(indices)}] Processing sample index: {idx}")
        
        # Get data from dataset
        data = dataset[idx]
        
        # Collate data (convert to batch format)
        data = collate([data], samples_per_gpu=1)
        
        # MMDataParallel handles device placement automatically
        # No need to manually move data - just pass it to the model
        
        # Get sample token from data
        sample_token = None
        if 'img_metas' in data:
            img_metas = data['img_metas']
            if isinstance(img_metas, list) and len(img_metas) > 0:
                if isinstance(img_metas[0], dict):
                    sample_token = img_metas[0].get('sample_idx', None)
        
        sample_token = sample_token or f"sample_{idx}"
        
        try:
            # Run model forward - reconstruction backbone will be called internally
            with torch.no_grad():
                # Call extract_feat to get point cloud from reconstruction backbone
                points = model.extract_feat(data['img'], data['img_metas'])
                # Convert to numpy
                if isinstance(points, torch.Tensor):
                    points = points.cpu().numpy()
                print(f"✓ Generated point cloud with {len(points)} points for sample {sample_token}")
                
                # Optionally save point cloud
                if args.output_dir:
                    output_path = os.path.join(args.output_dir, f"{sample_token}_points.npy")
                    np.save(output_path, points)
                    print(f"  Saved point cloud to {output_path}")
                
                # Get ground truth boxes if available (for visualization)
                gt_bboxes_3d = None
                if 'gt_bboxes_3d' in data:
                    gt_bboxes_3d = data['gt_bboxes_3d']
                    if hasattr(gt_bboxes_3d, 'data'):
                        gt_bboxes_3d = gt_bboxes_3d.data
                    if isinstance(gt_bboxes_3d, list) and len(gt_bboxes_3d) > 0:
                        gt_bboxes_3d = gt_bboxes_3d[0]  # Take first batch item
                
                # Display point cloud if available
                if points is not None and len(points) > 0:
                    display_point_cloud(points, sample_token, colors=None, gt_bboxes_3d=gt_bboxes_3d)
                
                # Run full model forward for detection (if detection head exists)
                result = model(return_loss=False, rescale=True, **data)
                print(f"✓ Successfully processed sample {sample_token}")
            
            successful += 1
        
        except Exception as e:
            print(f"✗ Error processing sample {sample_token}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print("Processing Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(indices)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
