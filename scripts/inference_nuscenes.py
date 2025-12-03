#!/usr/bin/env python3
"""
Inference script for nuScenes-format datasets
Uses LIDAR_TOP as reference to find closest camera timestamps and generates point clouds.

Usage:
    python scripts/inference_nuscenes.py --data_dir /path/to/nuscenes --output_dir /path/to/output
"""

import argparse
import glob
import os
import sys
import threading
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import cv2
import matplotlib.pyplot as plt
from depth_anything_3.api import DepthAnything3
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


CAM_TYPES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def get_nusc_info(nusc, timestamp_int):
    """
    Get nuScenes information for all cameras.
    Extracts camera-to-LiDAR transformations for all cameras in the sample.
    
    Returns:
        nusc_info: Dictionary with transformations for each camera and gt_lidar_boxes
    """
    nusc_info = {
        'gt_lidar_boxes': None,
    }
    
    # Find sample_data tokens with matching timestamp
    sample_data_tokens = nusc.field2token('sample_data', 'timestamp', timestamp_int)
    
    print(f"sample_data_tokens: {sample_data_tokens}")
    
    # Get first sample_data and find sample
    sample_data = nusc.get('sample_data', sample_data_tokens[0])
    sample = nusc.get('sample', sample_data['sample_token'])
    lidar_token = sample['data']['LIDAR_TOP']
    
    # Get lidar calibrated sensor and ego pose
    lidar_cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    lidar_pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    
    # Get boxes in lidar coordinates
    _, gt_lidar_boxes, _ = nusc.get_sample_data(lidar_token)
    
    # Extract transformation matrices for LiDAR
    l2e_r = lidar_cs_record['rotation']
    l2e_t = lidar_cs_record['translation']
    e2g_r = lidar_pose_record['rotation']
    e2g_t = lidar_pose_record['translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # Extract transformations for ALL cameras
    for camera_type in CAM_TYPES:
        if camera_type in sample['data']:
            cam_token = sample['data'][camera_type]
            cam_to_lidar = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                            e2g_t, e2g_r_mat, camera_type)
            
            # Store transformation for this camera
            nusc_info[camera_type] = {
                'cam2lidar_rotation': cam_to_lidar['sensor2lidar_rotation'],
                'cam2lidar_translation': cam_to_lidar['sensor2lidar_translation'],
            }
            print(f"  Extracted transformation for {camera_type}")
    
    nusc_info['gt_lidar_boxes'] = gt_lidar_boxes
    return nusc_info


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

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
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
    return sweep



def find_lidar_timestamps(data_dir):
    """Extract timestamps from LIDAR_TOP folder."""
    lidar_dir = os.path.join(data_dir, "samples", "LIDAR_TOP")
    if not os.path.exists(lidar_dir):
        raise ValueError(f"LIDAR_TOP directory not found: {lidar_dir}")
    
    lidar_files = glob.glob(os.path.join(lidar_dir, "*.bin"))
    timestamps = []
    for lidar_file in sorted(lidar_files):
        timestamp = os.path.splitext(os.path.basename(lidar_file))[0]
        timestamps.append(timestamp)
    
    return sorted(timestamps)


def find_camera_folders(data_dir):
    """Find all camera folders in the samples directory."""
    samples_dir = os.path.join(data_dir, "samples")
    if not os.path.exists(samples_dir):
        raise ValueError(f"Samples directory not found: {samples_dir}")
    
    camera_folders = []
    # for item in os.listdir(samples_dir):
    for item in CAM_TYPES:
        item_path = os.path.join(samples_dir, item)
        # if os.path.isdir(item_path) and item.startswith("CAM_"):
        if os.path.isdir(item_path):
            camera_folders.append((item, item_path))
    return camera_folders
            
    


def extract_timestamp_from_filename(filename, camera_name):
    """
    Extract timestamp from filename.
    Handles two formats:
    1. timestamp.jpg (just timestamp)
    2. CAM_FRONT_timestamp.jpg (camera_name_timestamp)
    
    Returns: timestamp as string or None if not found
    """
    base_name = os.path.splitext(filename)[0]
    
    # Try format: camera_name_timestamp
    prefix = f"{camera_name}_"
    if base_name.startswith(prefix):
        timestamp = base_name[len(prefix):]
        # Verify it's a valid timestamp (numeric)
        if timestamp.isdigit():
            return timestamp
    
    # Try format: just timestamp (numeric only)
    if base_name.isdigit():
        return base_name
    
    return None


def find_closest_timestamp(target_timestamp, camera_path, camera_name, image_extensions=("*.jpg", "*.jpeg", "*.png")):
    """
    Find the closest image timestamp to the target timestamp in a camera folder.
    Handles filenames in format: timestamp.jpg or CAM_NAME_timestamp.jpg
    
    Returns: (closest_timestamp, image_path) or (None, None) if no images found
    """
    # Find all images in this camera folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(camera_path, ext)))
    
    if not image_files:
        return None, None
    
    # Extract timestamps and find closest
    timestamps = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        timestamp_str = extract_timestamp_from_filename(filename, camera_name)
        
        if timestamp_str:
            try:
                timestamp_int = int(timestamp_str)
                timestamps.append((timestamp_int, img_path))
            except ValueError:
                continue
    
    if not timestamps:
        return None, None
    
    # Find closest timestamp
    target_ts = int(target_timestamp)
    closest = min(timestamps, key=lambda x: abs(x[0] - target_ts))
    
    return str(closest[0]), closest[1]


def group_images_by_lidar_frame(data_dir, lidar_timestamps, camera_folders):
    """
    For each lidar timestamp, find closest camera images.
    Returns: list of (frame_timestamp, list of (camera_name, image_path))
    Ensures consistent camera ordering based on CAM_TYPES.
    """
    frame_groups = []
    
    # Create a mapping from camera name to path for quick lookup
    camera_path_map = {name: path for name, path in camera_folders}
    
    for frame_timestamp in lidar_timestamps:
        frame_images = []
        
        # Process cameras in a consistent order (CAM_TYPES order)
        for camera_name in CAM_TYPES:
            if camera_name in camera_path_map:
                camera_path = camera_path_map[camera_name]
                closest_ts, image_path = find_closest_timestamp(frame_timestamp, camera_path, camera_name)
                if image_path:
                    frame_images.append((camera_name, image_path))
        
        if frame_images:  # Only add if we found at least one camera image
            frame_groups.append((frame_timestamp, frame_images))
    
    return frame_groups




def load_point_cloud_from_prediction(prediction, image_paths=None):
    """
    Load point cloud from prediction results.
    Keeps points in their own camera coordinate systems (doesn't transform to world).
    Tracks which points belong to which camera.
    
    IMPORTANT: prediction.depth[i] corresponds to image[i] in the ORIGINAL input order.
    The model may reorder views internally for reference view selection, but the final
    output is restored to match the original input order (see restore_original_order in
    vision_transformer.py and InputProcessor documentation).
    
    Args:
        prediction: Model prediction object
        image_paths: List of (camera_name, image_path) tuples in the SAME ORDER as passed to model.inference()
    
    Returns: 
        Dictionary with:
        - 'points_by_camera': dict mapping camera_name -> points in camera coordinates
        - 'colors_by_camera': dict mapping camera_name -> colors
        - 'camera_order': list of camera names in order
    """
    
    result = {
        'points_by_camera': {},
        'colors_by_camera': {},
        'camera_order': []
    }
    
    # Generate point cloud from depth maps and camera parameters
    if hasattr(prediction, 'depth') and hasattr(prediction, 'intrinsics'):
        # Verify: prediction.depth[i] should correspond to image_paths[i] in original input order
        if image_paths:
            print(f"  Input camera order (passed to model): {[name for name, _ in image_paths]}")
            print(f"  Number of depth maps: {len(prediction.depth)}")
            if len(prediction.depth) != len(image_paths):
                print(f"  WARNING: Mismatch! Depth maps ({len(prediction.depth)}) != Input images ({len(image_paths)})")
        
        for i in range(len(prediction.depth)):
            depth = prediction.depth[i]
            K = prediction.intrinsics[i]
            
            # Get camera name for this depth map
            # prediction.depth[i] corresponds to image_paths[i] in original input order
            cam_name = None
            if image_paths and i < len(image_paths):
                cam_name = image_paths[i][0]
                print(f"  Depth map [{i}] -> Camera: {cam_name} (matches input order)")
            else:
                cam_name = f"CAM_{i}"
                print(f"  Depth map [{i}] -> Camera: {cam_name} (no input mapping)")
            
            result['camera_order'].append(cam_name)
            
            H, W = depth.shape
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            
            # Back-project to 3D in camera coordinates (keep in camera frame!)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth
            
            # Points are in camera coordinate system (NOT transformed to world)
            points_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
            
            # Filter valid points
            valid = depth.flatten() > 0
            points_cam_valid = points_cam[valid]
            
            # Store points for this camera (in camera coordinates)
            result['points_by_camera'][cam_name] = points_cam_valid
            
            # Get colors from processed images
            if hasattr(prediction, 'processed_images'):
                img = prediction.processed_images[i]
                colors = img.reshape(-1, 3)[valid] / 255.0
                result['colors_by_camera'][cam_name] = colors
    
    return result


def display_camera_images(image_paths, frame_timestamp):
    """
    Display CAM_FRONT and CAM_BACK images side by side using OpenCV (thread-safe).
    
    Args:
        image_paths: List of (camera_name, image_path) tuples
        frame_timestamp: Frame timestamp for window title
    """
    # Find CAM_FRONT and CAM_BACK images (in order)
    cam_front_path = None
    cam_back_path = None
    
    # Process in order to maintain consistency
    for camera_name, image_path in image_paths:
        if camera_name == 'CAM_FRONT' and cam_front_path is None:
            cam_front_path = image_path
        elif camera_name == 'CAM_BACK' and cam_back_path is None:
            cam_back_path = image_path
    
    # Load images
    img_front = None
    img_back = None
    
    if cam_front_path and os.path.exists(cam_front_path):
        img_front = cv2.imread(cam_front_path)
    
    if cam_back_path and os.path.exists(cam_back_path):
        img_back = cv2.imread(cam_back_path)
    
    if img_front is None and img_back is None:
        print(f"  Warning: No CAM_FRONT or CAM_BACK images found to display")
        return
    
    # Combine images side by side
    if img_front is not None and img_back is not None:
        # Resize images to same height if needed
        h1, w1 = img_front.shape[:2]
        h2, w2 = img_back.shape[:2]
        target_height = max(h1, h2)
        
        if h1 != target_height:
            scale = target_height / h1
            new_w = int(w1 * scale)
            img_front = cv2.resize(img_front, (new_w, target_height))
        if h2 != target_height:
            scale = target_height / h2
            new_w = int(w2 * scale)
            img_back = cv2.resize(img_back, (new_w, target_height))
        
        combined = np.hstack([img_front, img_back])
        
        # Add text labels
        cv2.putText(combined, 'CAM_FRONT', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, 'CAM_BACK', (img_front.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif img_front is not None:
        combined = img_front.copy()
        cv2.putText(combined, 'CAM_FRONT', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        combined = img_back.copy()
        cv2.putText(combined, 'CAM_BACK', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display window
    window_name = f'Camera Images - Frame {frame_timestamp}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 800)
    cv2.imshow(window_name, combined)
    print(f"  Displaying camera images. Press any key or close the window to continue...")
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyWindow(window_name)


def display_camera_images_threaded(image_paths, frame_timestamp):
    """
    Display camera images in a separate thread (for simultaneous windows).
    
    Args:
        image_paths: List of (camera_name, image_path) tuples
        frame_timestamp: Frame timestamp for window title
    """
    def _display():
        display_camera_images(image_paths, frame_timestamp)
    
    thread = threading.Thread(target=_display, daemon=False)
    thread.start()
    return thread


def display_point_cloud(pcd, frame_timestamp, gt_lidar_boxes=None):
    """Display point cloud using open3d."""
    if pcd is None or len(pcd.points) == 0:
        print(f"  Warning: No point cloud to display for frame {frame_timestamp}")
        return
    
    print(f"  Displaying point cloud with {len(pcd.points)} points...")
    print(f"  Press 'Q' or close the window to continue")
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud - Frame {frame_timestamp}", width=1920, height=1080)
    vis.add_geometry(pcd)
    
    # Set up view
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, -1])
    view_ctl.set_lookat([0, 0, 0])
    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.7)
    
    # draw the axis of the point cloud
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(axis)
    
    # draw the gt_lidar_boxes on the point cloud
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


def display_point_cloud_threaded(pcd, frame_timestamp, gt_lidar_boxes=None):
    """
    Display point cloud in a separate thread (for simultaneous windows).
    
    Args:
        pcd: open3d.geometry.PointCloud object
        frame_timestamp: Frame timestamp for window title
        gt_lidar_boxes: List of nuScenes Box objects for bounding boxes
    """
    def _display():
        display_point_cloud(pcd, frame_timestamp, gt_lidar_boxes)
    
    thread = threading.Thread(target=_display, daemon=False)
    thread.start()
    return thread


def run_inference_for_frame(
    model,
    frame_timestamp,
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
    print(f"Processing frame timestamp: {frame_timestamp}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Cameras: {[name for name, _ in image_paths]}")
    print(f"{'='*60}")
    
    # Sort to ensure consistent order based on CAM_TYPES
    # This ensures cameras are always in the same order
    camera_path_map = {name: path for name, path in image_paths}
    sorted_paths = []
    for cam_type in CAM_TYPES:
        if cam_type in camera_path_map:
            sorted_paths.append((cam_type, camera_path_map[cam_type]))
    
    # Extract just the image paths in consistent order
    image_files = [path for _, path in sorted_paths]
    camera_order = [name for name, _ in sorted_paths]
    
    print(f"  Camera order: {camera_order}")
    print(f"  Number of images for inference: {len(image_files)}")
    
    # Create output directory for this frame
    frame_output_dir = os.path.join(output_dir, frame_timestamp)
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
        
        print(f"✓ Successfully processed frame {frame_timestamp}")
        print(f"  Output directory: {frame_output_dir}")
        
        # Display camera images and point cloud simultaneously
        if display:
            # Display camera images (CAM_FRONT and CAM_BACK) in a separate thread (OpenCV is thread-safe)
            camera_thread = display_camera_images_threaded(image_paths, frame_timestamp)
            
            # Load point clouds from prediction (points stay in camera coordinates)
            pcd_data = load_point_cloud_from_prediction(prediction, sorted_paths)
            
            if pcd_data and pcd_data['points_by_camera']:
                # Transform each camera's points from camera coordinates to LiDAR coordinates
                # using ground truth extrinsics from nuScenes (not model's predicted extrinsics)
                all_points_lidar = []
                all_colors = []
                
                print(f"  Transforming points from camera coordinates to LiDAR coordinates...")
                for cam_name in pcd_data['camera_order']:
                    if cam_name in pcd_data['points_by_camera']:
                        points_cam = pcd_data['points_by_camera'][cam_name]
                        
                        # Check if we have transformation for this camera
                        if cam_name in nusc_info:
                            # Transform points from camera coordinates to LiDAR coordinates
                            # using ground truth extrinsics from nuScenes
                            R = nusc_info[cam_name]['cam2lidar_rotation']
                            T = nusc_info[cam_name]['cam2lidar_translation']
                            
                            # Transform: points_lidar = points_cam @ R.T + T
                            points_lidar = points_cam @ R.T + T
                            all_points_lidar.append(points_lidar)
                            
                            # Get colors if available
                            if cam_name in pcd_data['colors_by_camera']:
                                all_colors.append(pcd_data['colors_by_camera'][cam_name])
                            
                            print(f"    {cam_name}: {len(points_cam)} points -> {len(points_lidar)} points in LiDAR frame")
                        else:
                            print(f"    WARNING: No transformation found for {cam_name}, skipping...")
                
                if all_points_lidar:
                    # Concatenate all points
                    combined_points = np.concatenate(all_points_lidar, axis=0)
                    
                    # Create point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(combined_points)
                    
                    if all_colors:
                        combined_colors = np.concatenate(all_colors, axis=0)
                        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
                    
                    print(f"  Total points in LiDAR frame: {len(combined_points)}")
                    
                    # Display point cloud in MAIN thread (Open3D/Qt requires main thread)
                    print(f"  Both windows are open. Close both windows to continue...")
                    display_point_cloud(pcd, frame_timestamp, nusc_info['gt_lidar_boxes'])
                else:
                    print(f"  Warning: No valid points to display")
            else:
                print(f"  Warning: Could not load point cloud for display")
            
            # Wait for camera images window to close
            camera_thread.join()
        
        return True, prediction
        
    except Exception as e:
        print(f"✗ Error processing frame {frame_timestamp}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 Inference for nuScenes-format datasets (LIDAR-based frame grouping)"
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
        "--frame_index",
        type=int,
        default=None,
        help="Process only a specific frame by index (0-based, e.g., 0 for first frame, optional)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (optional, processes all if not specified)",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Skip point cloud visualization (default: display point clouds)",
    )

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
    print("Depth Anything 3 - nuScenes Inference (LIDAR-based)")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Export format: {args.export_format}")
    print(f"Display point clouds: {not args.no_display}")
    print(f"{'='*60}\n")

    # Find LIDAR timestamps (frame references)
    print("Extracting frame timestamps from LIDAR_TOP...")
    lidar_timestamps = find_lidar_timestamps(args.data_dir)
    print(f"Found {len(lidar_timestamps)} lidar frames")

    # Filter by specific frame index if requested
    if args.frame_index is not None:
        if args.frame_index < 0 or args.frame_index >= len(lidar_timestamps):
            print(f"Error: Frame index {args.frame_index} is out of range (0-{len(lidar_timestamps)-1})")
            sys.exit(1)
        lidar_timestamps = [lidar_timestamps[args.frame_index]]
        print(f"Processing frame at index {args.frame_index}: {lidar_timestamps[0]}")

    # Limit number of frames if requested
    if args.max_frames:
        lidar_timestamps = lidar_timestamps[:args.max_frames]
        print(f"Processing first {len(lidar_timestamps)} frames")

    # Find camera folders
    print("\nScanning camera folders...")
    camera_folders = find_camera_folders(args.data_dir)
    print(f"Found {len(camera_folders)} camera folders:")
    for name, path in camera_folders:
        print(f"  - {name}")

    # Group images by lidar frames
    print(f"\nGrouping camera images for {len(lidar_timestamps)} frames...")
    frame_groups = group_images_by_lidar_frame(args.data_dir, lidar_timestamps, camera_folders)
    print(f"Successfully grouped {len(frame_groups)} frames with camera images")

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

    # Process each frame
    print(f"\nProcessing {len(frame_groups)} frame(s)...")
    successful = 0
    failed = 0
    
    # using Nuscenes to utilize the bbox, we will visualize the bbox on the point cloud to verify the accuracy of the point cloud
    
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.data_dir, verbose=True)
    

    
    for i, (frame_timestamp, image_paths) in enumerate(frame_groups, 1):
        print(f"\n[{i}/{len(frame_groups)}] Processing frame: {frame_timestamp}")
        
        print(f"image_paths: {image_paths}")
        
        timestamp_int = int(frame_timestamp)
        
        nusc_info = get_nusc_info(nusc, timestamp_int)

    
        success, _ = run_inference_for_frame(
            model=model,
            frame_timestamp=frame_timestamp,
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
    print(f"Total frames: {len(frame_groups)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
