#!/usr/bin/env python3
"""
Inference script for nuScenes-format datasets
Iterates through nusc.sample to process synchronized camera images and generate point clouds.

Usage:
    python scripts/pseudo_nuscenes.py --data_dir /path/to/nuscenes --output_dir /path/to/output
"""

import argparse
import os
import sys
import threading
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from depth_anything_3.api import DepthAnything3
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def read_bin_file(bin_file_path, feature_dim=5):
    points = np.fromfile(bin_file_path, dtype=np.float32)
    points = points.reshape(-1, feature_dim)
    return points


def get_nusc_info(nusc, sample):
    """
    Get nuScenes information for a sample.
    
    Args:
        nusc: NuScenes instance
        sample: Sample dictionary from nusc.sample
    
    Returns:
        nusc_info: Dictionary with cam2lidar transforms and gt_lidar_boxes
    """
    nusc_info = {
        'cam2lidar_rotation': None,
        'cam2lidar_translation': None,
        'gt_lidar_boxes': None,
    }
    
    # Get LIDAR_TOP token
    lidar_token = sample['data']['LIDAR_TOP']
    
    # Get lidar sample_data
    lidar_sd = nusc.get('sample_data', lidar_token)
    
    # Get lidar calibrated sensor and ego pose
    lidar_cs_record = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_pose_record = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    
    # Get boxes in lidar coordinates
    lidar_path, gt_lidar_boxes, _ = nusc.get_sample_data(lidar_token)
    points = read_bin_file(lidar_path)
    
    # Extract transformation matrices
    l2e_r = lidar_cs_record['rotation']
    l2e_t = lidar_cs_record['translation']
    e2g_r = lidar_pose_record['rotation']
    e2g_t = lidar_pose_record['translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # Get CAM_FRONT token and compute transformation
    camera_types = 'CAM_FRONT'
    cam_token = sample['data'][camera_types]

    cam_front_to_lidar = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                    e2g_t, e2g_r_mat, camera_types)

    nusc_info['cam2lidar_rotation'] = cam_front_to_lidar['sensor2lidar_rotation']
    nusc_info['cam2lidar_translation'] = cam_front_to_lidar['sensor2lidar_translation']
    nusc_info['gt_lidar_boxes'] = gt_lidar_boxes
    nusc_info['points'] = points
    return nusc_info


def get_camera_images_from_sample(nusc, sample, data_dir):
    """
    Get camera image paths from a nuScenes sample.
    
    Args:
        nusc: NuScenes instance
        sample: Sample dictionary from nusc.sample
        data_dir: Root directory of nuScenes dataset
    
    Returns:
        List of (camera_name, image_path) tuples
    """
    camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    image_paths = []
    
    for cam_type in camera_types:
        if cam_type in sample['data']:
            cam_token = sample['data'][cam_type]
            cam_sd = nusc.get('sample_data', cam_token)
            
            # Get the filename from sample_data
            filename = cam_sd['filename']
            image_path = os.path.join(data_dir, filename)
            
            if os.path.exists(image_path):
                # Extract camera name from cam_type (e.g., 'CAM_FRONT' -> 'CAM_FRONT')
                camera_name = cam_type
                image_paths.append((camera_name, image_path))
            else:
                print(f"  Warning: Image not found: {image_path}")
    
    return image_paths


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    sweep = {
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # Original transformation: sensor->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T

    # Simplified transformation: sensor->ego->lidar (skip global)
    # Since camera and lidar are from the same sample, they share the same ego frame
    # cam2lidar = ego2lidar @ cam2ego
    # where ego2lidar = (lidar2ego)^-1
    # l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix  # sensor to ego rotation
    # l2e_r_mat_inv = np.linalg.inv(l2e_r_mat)  # ego to lidar rotation (inverse of lidar to ego)
    
    # # Rotation: R_cam2lidar = R_ego2lidar @ R_cam2ego = R_lidar2ego^-1 @ R_cam2ego
    # R = l2e_r_mat_inv @ l2e_r_s_mat
    
    # # Translation: T_cam2lidar = R_ego2lidar @ (T_cam2ego - T_lidar2ego)
    # #              = R_lidar2ego^-1 @ (T_cam2ego - T_lidar2ego)
    # # Convert translations to numpy arrays and ensure correct shape
    # l2e_t_s_arr = np.array(l2e_t_s, dtype=np.float64)
    # l2e_t_arr = np.array(l2e_t, dtype=np.float64)
    # # Handle both row and column vectors
    # if l2e_t_s_arr.ndim == 1:
    #     l2e_t_s_arr = l2e_t_s_arr.reshape(1, 3)  # Make it a row vector
    # if l2e_t_arr.ndim == 1:
    #     l2e_t_arr = l2e_t_arr.reshape(1, 3)  # Make it a row vector
    
    # # T = (T_cam2ego - T_lidar2ego) @ R_ego2lidar^T = (T_cam2ego - T_lidar2ego) @ R_lidar2ego^-T
    # T = (l2e_t_s_arr - l2e_t_arr) @ l2e_r_mat_inv.T
    # T = T.flatten()  # Convert to 1D array
    
    # sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    # sweep['sensor2lidar_translation'] = T
    return sweep





def convert_points_cam_to_ego(points):
    """
    Convert points from camera coordinate system to ego coordinate system.
    
    Args:
        points: numpy array of shape (N, 3) in camera coordinates
    
    Returns:
        numpy array of shape (N, 3) in LiDAR coordinates
    """
    if points.shape[0] == 0:
        return points
    
    # Rotation matrix from CAM to LIDAR (from mmdet3d)
    rt_mat = np.array([[0, 0, 1],   # new x = old z (front)
                       [-1, 0, 0],  # new y = -old x (left)
                       [0, -1, 0]]) # new z = -old y (up)
    
    # Apply rotation: points_lidar = points_cam @ rt_mat.T
    points_lidar = points @ rt_mat.T
    
    return points_lidar


def load_point_cloud_from_prediction(prediction):
    """
    Load point cloud from prediction results.
    Returns: open3d PointCloud object
    """
    
    # Generate point cloud from depth maps and camera parameters
    if hasattr(prediction, 'depth') and hasattr(prediction, 'extrinsics') and hasattr(prediction, 'intrinsics'):
        points_list = []
        colors_list = []
        
        for i in range(len(prediction.depth)):
            depth = prediction.depth[i]
            K = prediction.intrinsics[i]
            ext_w2c = prediction.extrinsics[i]
            
            # Convert to c2w
            c2w = np.linalg.inv(np.vstack([ext_w2c, [0, 0, 0, 1]]))[:3, :]
            
            H, W = depth.shape
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            
            # Back-project to 3D
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth
            
            # Transform to world coordinates
            points_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
            points_world = (c2w @ np.hstack([points_cam, np.ones((points_cam.shape[0], 1))]).T).T[:, :3]
            
            # Filter valid points
            valid = depth.flatten() > 0
            points_list.append(points_world[valid])
            
            # Get colors from processed images
            if hasattr(prediction, 'processed_images'):
                img = prediction.processed_images[i]
                colors = img.reshape(-1, 3)[valid] / 255.0
                colors_list.append(colors)
        
        if points_list:
            all_points = np.concatenate(points_list, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            
            if colors_list:
                all_colors = np.concatenate(colors_list, axis=0)
                pcd.colors = o3d.utility.Vector3dVector(all_colors)
            
            return pcd
    
    return None


def create_point_cloud_from_lidar(points):
    """
    Create Open3D point cloud from LIDAR points.
    
    Args:
        points: numpy array of shape (N, 4) or (N, 5) - [x, y, z, intensity, ...]
    
    Returns:
        open3d.geometry.PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    
    # Extract x, y, z coordinates (first 3 columns)
    points_xyz = points[:, :3]
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    
    # If intensity is available (4th column), use it for coloring
    if points.shape[1] >= 4:
        intensity = points[:, 3]
        # Normalize intensity to [0, 1] for visualization
        intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        # Create grayscale colors
        colors = np.stack([intensity_normalized, intensity_normalized, intensity_normalized], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def display_point_cloud(pcd, window_title, gt_lidar_boxes=None):
    """
    Display point cloud using open3d.
    
    Args:
        pcd: open3d.geometry.PointCloud object
        window_title: Title for the visualization window
        gt_lidar_boxes: List of nuScenes Box objects for bounding boxes
    """
    if pcd is None or len(pcd.points) == 0:
        print(f"  Warning: No point cloud to display: {window_title}")
        return
    
    print(f"  Displaying {window_title} with {len(pcd.points)} points...")
    print(f"  Press 'Q' or close the window to continue")
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1920, height=1080)
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
    
    # Draw the gt_lidar_boxes on the point cloud
    if gt_lidar_boxes is not None and len(gt_lidar_boxes) > 0:
        print(f"  Adding {len(gt_lidar_boxes)} bounding boxes to visualization")
        for gt_lidar_box in gt_lidar_boxes:
            # Extract from nuScenes Box object
            center = np.array(gt_lidar_box.center, dtype=np.float64)  # x, y, z
            size = np.array(gt_lidar_box.wlh, dtype=np.float64)      # width, length, height
            # swap the width and length
            size[0], size[1] = size[1], size[0]
            
            # Convert Quaternion to rotation matrix
            # gt_lidar_box.orientation is a Quaternion object
            rotation_matrix = gt_lidar_box.orientation.rotation_matrix
            
            # Create OrientedBoundingBox
            obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, size)
            obb.color = [1, 0, 0]  # Red color for boxes
            vis.add_geometry(obb)
    
    vis.run()
    vis.destroy_window()


def display_point_cloud_threaded(pcd, window_title, gt_lidar_boxes=None):
    """
    Display point cloud in a separate thread (for simultaneous windows).
    
    Args:
        pcd: open3d.geometry.PointCloud object
        window_title: Title for the visualization window
        gt_lidar_boxes: List of nuScenes Box objects for bounding boxes
    """
    def _display():
        display_point_cloud(pcd, window_title, gt_lidar_boxes)
    
    thread = threading.Thread(target=_display, daemon=False)
    thread.start()
    return thread


def run_inference_for_frame(
    model,
    sample_token,
    image_paths,
    output_dir,
    device,
    export_format="glb",
    ref_view_strategy="saddle_balanced",
    use_ray_pose=False,
    max_points=1_000_000,
    display=True,
    nusc_info=None,
):
    """Run inference for a single frame and optionally display point cloud."""
    print(f"\n{'='*60}")
    print(f"Processing sample token: {sample_token}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Cameras: {[name for name, _ in image_paths]}")
    print(f"{'='*60}")
    
    # Extract just the image paths
    image_files = [path for _, path in image_paths]
    
    # Create output directory for this frame (use sample token)
    frame_output_dir = os.path.join(output_dir, sample_token)
    os.makedirs(frame_output_dir, exist_ok=True)
    
    
    try:
        # Run inference (no extrinsics/intrinsics - model estimates them!)
        prediction = model.inference(
            image=image_files,
            export_dir=frame_output_dir,
            export_format=export_format,
            ref_view_strategy=ref_view_strategy,
            use_ray_pose=use_ray_pose,
            num_max_points=max_points,
        )
        
        print(f"✓ Successfully processed sample {sample_token}")
        print(f"  Output directory: {frame_output_dir}")
        
        # Load and display point clouds
        if display:
            threads = []
            
            # Display raw LIDAR point cloud with bounding boxes in a separate thread
            if nusc_info.get('points') is not None:
                raw_pcd = create_point_cloud_from_lidar(nusc_info['points'])
                thread = display_point_cloud_threaded(
                    raw_pcd, 
                    f"Raw LIDAR Point Cloud - {sample_token}",
                    nusc_info['gt_lidar_boxes']
                )
                threads.append(thread)
            
            # Display pseudo (depth-estimated) point cloud with bounding boxes in a separate thread
            pcd = load_point_cloud_from_prediction(prediction)
            if pcd is not None:
                # Convert point cloud from camera coordinates to LiDAR coordinates
                # (to match nuScenes bounding box coordinate system)
                points_cam = np.asarray(pcd.points)
                
                # convert the points from camera coordinates to LiDAR coordinates
                points_lidar = points_cam @ nusc_info['cam2lidar_rotation'].T + nusc_info['cam2lidar_translation']

                pcd.points = o3d.utility.Vector3dVector(points_lidar)
                print(f"  Converted pseudo point cloud from camera to LiDAR coordinates")
                
                thread = display_point_cloud_threaded(
                    pcd, 
                    f"Pseudo Point Cloud (Depth-Estimated) - {sample_token}",
                    nusc_info['gt_lidar_boxes']
                )
                threads.append(thread)
            else:
                print(f"  Warning: Could not load pseudo point cloud for display")
            
            # Wait for all visualization windows to close
            print(f"  Both windows are open. Close both windows to continue...")
            for thread in threads:
                thread.join()
        
        return True, prediction
        
    except Exception as e:
        print(f"✗ Error processing sample {sample_token}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 Inference for nuScenes-format datasets (Sample-based iteration)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to nuScenes dataset root (should contain 'samples' folder)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Model name or path (default: depth-anything/DA3NESTED-GIANT-LARGE)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--export_format",
        type=str,
        default="glb",
        help="Export format: glb, npz, mini_npz, or combinations like 'glb-npz' (default: glb)",
    )
    parser.add_argument(
        "--ref_view_strategy",
        type=str,
        default="saddle_balanced",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        help="Reference view selection strategy (default: saddle_balanced)",
    )
    parser.add_argument(
        "--use_ray_pose",
        action="store_true",
        help="Use ray-based pose estimation (more accurate but slower)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=1_000_000,
        help="Maximum number of points in point cloud (default: 1000000)",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Process only a specific sample by index (0-based, e.g., 0 for first sample, optional)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (optional, processes all if not specified)",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Skip point cloud visualization (default: display point clouds)",
    )
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="NuScenes version to use (default: v1.0-trainval)")

    args = parser.parse_args()

    # Check data directory
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        args.device = "cpu"

    print(f"\n{'='*60}")
    print("Depth Anything 3 - nuScenes Inference (Sample-based)")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Export format: {args.export_format}")
    print(f"Display point clouds: {not args.no_display}")
    print(f"{'='*60}\n")

    # Initialize nuScenes
    print("Initializing nuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.data_dir, verbose=True)
    print(f"Loaded nuScenes dataset with {len(nusc.sample)} samples")

    # Get all samples
    all_samples = list(nusc.sample)
    
    # Filter by specific sample index if requested
    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(all_samples):
            print(f"Error: Sample index {args.sample_index} is out of range (0-{len(all_samples)-1})")
            sys.exit(1)
        all_samples = [all_samples[args.sample_index]]
        print(f"Processing sample at index {args.sample_index}: {all_samples[0]['token']}")

    # Limit number of samples if requested
    if args.max_samples:
        all_samples = all_samples[:args.max_samples]
        print(f"Processing first {len(all_samples)} samples")

    # Determine cache directory (ckpts folder in repository root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    cache_dir = os.path.join(repo_root, "ckpts")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"\nModel cache directory: {cache_dir}")

    # Load model
    print(f"Loading model: {args.model_name}...")
    try:
        model = DepthAnything3.from_pretrained(args.model_name, cache_dir=cache_dir)
        model = model.to(device=args.device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Process each sample
    print(f"\nProcessing {len(all_samples)} sample(s)...")
    successful = 0
    failed = 0
    
    for i, sample in enumerate(all_samples, 1):
        sample_token = sample['token']
        print(f"\n[{i}/{len(all_samples)}] Processing sample: {sample_token}")
        
        # Get camera images for this sample
        image_paths = get_camera_images_from_sample(nusc, sample, args.data_dir)
        
        print(f"image_paths: {image_paths}")
        
        if not image_paths:
            print(f"  Warning: No camera images found for sample {sample_token}, skipping...")
            failed += 1
            continue
        
        print(f"  Found {len(image_paths)} camera images")
        
        # Get nuScenes info (transforms and bounding boxes)
        nusc_info = get_nusc_info(nusc, sample)
    
        success, _ = run_inference_for_frame(
            model=model,
            sample_token=sample_token,
            image_paths=image_paths,
            output_dir=args.output_dir,
            device=args.device,
            export_format=args.export_format,
            ref_view_strategy=args.ref_view_strategy,
            use_ray_pose=args.use_ray_pose,
            max_points=args.max_points,
            display=not args.no_display,
            nusc_info=nusc_info,
        )
        
        if success:
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print("Processing Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
