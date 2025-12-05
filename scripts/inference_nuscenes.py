#!/usr/bin/env python3
"""
Inference script for nuScenes-format datasets
Iterates through nusc.sample to process each sample and generates point clouds.

Usage:
    python scripts/inference_nuscenes.py --data_dir /path/to/nuscenes --output_dir /path/to/output
"""

import argparse
import csv
import os
import sys
import threading

import numpy as np
import open3d as o3d
import torch
import cv2
from scipy.spatial import cKDTree
from depth_anything_3.api import DepthAnything3
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# Import inference configuration
# Add scripts directory to path to allow importing infer_config
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import infer_config as cfg


def get_nusc_info(nusc, sample):
    """
    Get nuScenes information for all cameras from a sample.
    Extracts camera-to-LiDAR transformations for all cameras in the sample.
    
    Args:
        nusc: NuScenes dataset object
        sample: Sample dictionary from nusc.sample
    
    Returns:
        nusc_info: Dictionary with transformations for each camera and gt_lidar_boxes
    """
    nusc_info = {
        'gt_lidar_boxes': None,
    }
    
    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    # Get boxes in lidar coordinates
    _, gt_lidar_boxes, _ = nusc.get_sample_data(lidar_token)
    
    # Extract transformation matrices for LiDAR
    l2e_r = cs_record['rotation']
    l2e_t = cs_record['translation']
    e2g_r = pose_record['rotation']
    e2g_t = pose_record['translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # Extract transformations for ALL cameras
    for camera_type in cfg.CAM_TYPES:
        if camera_type in sample['data']:
            cam_token = sample['data'][camera_type]
            cam_to_lidar = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                            e2g_t, e2g_r_mat, camera_type)
            
            # Get calibrated sensor record to extract camera intrinsic
            sd_rec = nusc.get('sample_data', cam_token)
            cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
            camera_intrinsic = np.array(cs_record['camera_intrinsic'])  # 3x3 matrix
            
            # Store transformation and camera info for this camera
            nusc_info[camera_type] = {
                'cam_token': cam_token,
                'cam2lidar_rotation': cam_to_lidar['sensor2lidar_rotation'],
                'cam2lidar_translation': cam_to_lidar['sensor2lidar_translation'],
                'camera_intrinsic': camera_intrinsic,
            }
            print(f"  Extracted transformation for {camera_type}")
    
    if cfg.SHOW_GT_BOXES:
        nusc_info['gt_lidar_boxes'] = gt_lidar_boxes
    else:
        nusc_info['gt_lidar_boxes'] = None
        
    # Write camera info to CSV file
    # cam_name, cam_token, camera_intrinsic (as flattened string: fx fy cx cy)
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'camera_info.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cam_name', 'cam_token', 'fx', 'fy', 'cx', 'cy', 'intrinsic_matrix'])
        for camera_type in cfg.CAM_TYPES:
            if camera_type in nusc_info:
                cam_token = nusc_info[camera_type]['cam_token']
                camera_intrinsic = nusc_info[camera_type]['camera_intrinsic']
                # Extract fx, fy, cx, cy from intrinsic matrix
                fx = camera_intrinsic[0, 0]
                fy = camera_intrinsic[1, 1]
                cx = camera_intrinsic[0, 2]
                cy = camera_intrinsic[1, 2]
                # Format intrinsic matrix as string (flattened: row1,row2,row3)
                intrinsic_str = ';'.join([','.join(map(str, row)) for row in camera_intrinsic])
                writer.writerow([camera_type, cam_token, fx, fy, cx, cy, intrinsic_str])
        
    
    return nusc_info


def get_camera_images_from_sample(nusc, sample, data_dir):
    """
    Get camera image paths from a sample in CAM_TYPES order.
    
    Args:
        nusc: NuScenes dataset object
        sample: Sample dictionary from nusc.sample
        data_dir: Root directory of nuScenes dataset
    
    Returns:
        List of (camera_name, image_path) tuples in CAM_TYPES order
    """
    image_paths = []
    
    # Normalize data_dir
    data_dir = os.path.normpath(data_dir)
    data_dir_basename = os.path.basename(data_dir)
    
    for camera_type in cfg.CAM_TYPES:
        if camera_type in sample['data']:
            cam_token = sample['data'][camera_type]
            cam_path, _, _ = nusc.get_sample_data(cam_token)
            cam_path = os.path.normpath(cam_path)
            
            # Check if path is absolute
            if os.path.isabs(cam_path):
                final_path = cam_path
            else:
                # Relative path - check if it already starts with data_dir or data_dir basename
                # to avoid duplication (e.g., "data/nuscenes_mini/..." when data_dir is "data/nuscenes_mini")
                if cam_path.startswith(data_dir + os.sep) or cam_path == data_dir:
                    # Path already includes full data_dir, use as-is
                    final_path = os.path.abspath(cam_path)
                elif cam_path.startswith(data_dir_basename + os.sep):
                    # Path starts with data_dir basename (e.g., "data/nuscenes_mini/...")
                    # Use as-is relative to current directory
                    final_path = os.path.abspath(cam_path)
                else:
                    # True relative path, join with data_dir
                    final_path = os.path.abspath(os.path.join(data_dir, cam_path))
            
            image_paths.append((camera_type, final_path))
    
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



def load_polygon_from_txt(txt_path):
    """
    Load polygon from txt file.
    
    File format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates 0-1)
    Example: "0 0.127344 0.890278 0.124219 0.890278 ..."
    
    Args:
        txt_path: Path to polygon file
    
    Returns:
        numpy array of shape (N, 2) with normalized coordinates [0, 1]
        Returns None if file doesn't exist or is invalid
    """
    if not os.path.exists(txt_path):
        return None
    
    try:
        with open(txt_path, 'r') as f:
            content = f.read().strip()
            if not content:
                return None
            
            # Parse the file: first number is class_id, rest are x y pairs
            values = list(map(float, content.split()))
            if len(values) < 3:  # Need at least class_id + one x,y pair
                return None
            
            # Extract class_id (first value) and polygon vertices (rest)
            class_id = int(values[0])
            coords = np.array(values[1:], dtype=np.float32)
            
            # Reshape to (N, 2) - pairs of x, y coordinates
            if len(coords) % 2 != 0:
                print(f"  Warning: Odd number of coordinates in {txt_path}, ignoring last value")
                coords = coords[:-1]
            
            polygon = coords.reshape(-1, 2)  # Shape: (N, 2)
            
            # Validate: coordinates should be in [0, 1] range
            if np.any(polygon < 0) or np.any(polygon > 1):
                print(f"  Warning: Polygon coordinates out of [0, 1] range in {txt_path}")
            
            return polygon
    
    except Exception as e:
        print(f"  Error loading polygon from {txt_path}: {e}")
        return None


def create_polygon_mask(polygon_normalized, height, width):
    """
    Create a binary mask from normalized polygon coordinates.
    
    Args:
        polygon_normalized: Polygon as numpy array (N, 2) with normalized coordinates [0, 1]
        height: Image height in pixels
        width: Image width in pixels
    
    Returns:
        Binary mask of shape (height, width), True inside polygon, False outside
    """
    if polygon_normalized is None:
        return None
    
    # Convert normalized coordinates to pixel coordinates
    polygon_pixels = polygon_normalized.copy()
    polygon_pixels[:, 0] *= width   # x coordinates
    polygon_pixels[:, 1] *= height  # y coordinates
    polygon_pixels = polygon_pixels.astype(np.int32)
    
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fill polygon using cv2.fillPoly
    cv2.fillPoly(mask, [polygon_pixels], 255)
    
    # Convert to boolean
    return mask > 0


def draw_polygon_on_image(img, polygon_normalized, color=(0, 0, 255), thickness=2):
    """
    Draw polygon on image.
    
    Args:
        img: Image as numpy array (H, W, 3) in BGR format
        polygon_normalized: Polygon as numpy array (N, 2) with normalized coordinates [0, 1]
        color: BGR color tuple (default: red)
        thickness: Line thickness (default: 2)
    
    Returns:
        Image with polygon drawn on it
    """
    if polygon_normalized is None:
        return img
    
    H, W = img.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    polygon_pixels = polygon_normalized.copy()
    polygon_pixels[:, 0] *= W   # x coordinates
    polygon_pixels[:, 1] *= H  # y coordinates
    polygon_pixels = polygon_pixels.astype(np.int32)
    
    # Draw filled polygon
    overlay = img.copy()
    cv2.fillPoly(overlay, [polygon_pixels], color)
    
    # Blend with original image (70% original, 30% red overlay)
    img_with_polygon = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    # Also draw polygon outline
    cv2.polylines(img_with_polygon, [polygon_pixels], isClosed=True, color=color, thickness=thickness)
    
    return img_with_polygon



def load_point_cloud_from_prediction(
    prediction, 
    image_paths=None,
    max_depth=None,
    conf_thresh_percentile=None,
    filter_sky=True,
    polygon_by_camera=None,
):
    """
    Load point cloud from prediction results.
    Keeps points in their own camera coordinate systems (doesn't transform to world).
    Tracks which points belong to which camera.
    Applies filtering for sky, far objects, and low confidence.
    
    IMPORTANT: prediction.depth[i] corresponds to image[i] in the ORIGINAL input order.
    The model may reorder views internally for reference view selection, but the final
    output is restored to match the original input order (see restore_original_order in
    vision_transformer.py and InputProcessor documentation).
    
    Available Prediction attributes:
        - depth: np.ndarray (N, H, W) - Main depth output
        - sky: np.ndarray | None (N, H, W) - Sky mask (boolean, True=sky, False=non-sky)
        - conf: np.ndarray | None (N, H, W) - Confidence map (higher = more reliable)
        - extrinsics: np.ndarray | None (N, 4, 4) - Camera extrinsics
        - intrinsics: np.ndarray | None (N, 3, 3) - Camera intrinsics
        - processed_images: np.ndarray | None (N, H, W, 3) - Processed images
        - gaussians: Gaussians | None - 3D Gaussian Splats (if infer_gs=True)
        - aux: dict[str, Any] - Auxiliary features (feat_layer_X if export_feat_layers set)
        - is_metric: int - Whether depth is metric
        - scale_factor: Optional[float] - Metric scale factor
    
    Args:
        prediction: Model prediction object
        image_paths: List of (camera_name, image_path) tuples in the SAME ORDER as passed to model.inference()
        max_depth: Maximum depth threshold to filter far objects/infinity (in meters, None = no limit)
        conf_thresh_percentile: Lower percentile for confidence threshold (None = no filtering)
        filter_sky: Whether to filter out sky regions (True = remove sky, False = keep sky)
        polygon_by_camera: Dict mapping camera_name -> polygon (numpy array (N, 2) with normalized coordinates)
    
    Returns: 
        Dictionary with:
        - 'points_by_camera': dict mapping camera_name -> points in camera coordinates
        - 'colors_by_camera': dict mapping camera_name -> colors
        - 'polygon_mask_by_camera': dict mapping camera_name -> boolean mask (True = polygon point)
        - 'camera_order': list of camera names in order
    """
    
    result = {
        'points_by_camera': {},
        'colors_by_camera': {},
        'polygon_mask_by_camera': {},
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
        
        # Compute confidence threshold if needed
        conf_thresh = None
        if conf_thresh_percentile is not None and hasattr(prediction, 'conf') and prediction.conf is not None:
            # Compute confidence threshold similar to GLB export
            # Note: prediction.sky is the sky mask from the model (boolean array, True = sky)
            sky = getattr(prediction, 'sky', None)
            if sky is not None and (~sky).sum() > 10:
                conf_pixels = prediction.conf[~sky]
            else:
                conf_pixels = prediction.conf.flatten()
            conf_thresh = np.percentile(conf_pixels, conf_thresh_percentile)
            print(f"  Confidence threshold (percentile {conf_thresh_percentile}): {conf_thresh:.4f}")
        
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
            
            # debug: use the zero depth to make all point at the same plane
            # z = np.zeros_like(depth)
            
            
            # Points are in camera coordinate system (NOT transformed to world)
            points_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
            
            # Filter valid points
            valid = depth.flatten() > 0
            valid = valid & np.isfinite(depth.flatten())
            
            # Filter by maximum depth (far objects/infinity)
            if max_depth is not None:
                valid = valid & (depth.flatten() <= max_depth)
                print(f"    Filtered by max_depth ({max_depth}m): {valid.sum()} valid points")
            
            # Filter by confidence threshold
            if conf_thresh is not None and hasattr(prediction, 'conf') and prediction.conf is not None:
                conf_i = prediction.conf[i].flatten()
                valid = valid & (conf_i >= conf_thresh)
                print(f"    Filtered by confidence (>= {conf_thresh:.4f}): {valid.sum()} valid points")
            
            # Filter sky regions if sky prediction is available
            # Note: prediction.sky is a boolean mask from the model (True = sky, False = non-sky)
            # The model outputs sky predictions which are converted to boolean in OutputProcessor
            if filter_sky:
                sky = getattr(prediction, 'sky', None)
                if sky is not None:
                    sky_i = sky[i].flatten()
                    # Keep non-sky regions (sky is boolean: False = non-sky, True = sky)
                    non_sky = ~sky_i
                    valid = valid & non_sky
                    print(f"    Filtered sky regions: {valid.sum()} valid points (removed {sky_i.sum()} sky pixels)")
            
            points_cam_valid = points_cam[valid]
            
            # Store points for this camera (in camera coordinates)
            result['points_by_camera'][cam_name] = points_cam_valid
            
            # Get colors from processed images
            if hasattr(prediction, 'processed_images'):
                img = prediction.processed_images[i]
                colors = img.reshape(-1, 3)[valid] / 255.0
                result['colors_by_camera'][cam_name] = colors
            
            # Check if points belong to polygon region
            polygon_mask_flat = None
            if polygon_by_camera and cam_name in polygon_by_camera:
                polygon = polygon_by_camera[cam_name]
                if polygon is not None:
                    # Create polygon mask for this camera's image size
                    polygon_mask = create_polygon_mask(polygon, H, W)
                    if polygon_mask is not None:
                        # Flatten polygon mask and apply same valid filter
                        polygon_mask_flat = polygon_mask.flatten()[valid]
                        result['polygon_mask_by_camera'][cam_name] = polygon_mask_flat
                        print(f"    Polygon points: {polygon_mask_flat.sum()} / {len(polygon_mask_flat)}")
                    else:
                        # No valid polygon mask, create all-False mask
                        result['polygon_mask_by_camera'][cam_name] = np.zeros(len(points_cam_valid), dtype=bool)
                else:
                    # No polygon for this camera, create all-False mask
                    result['polygon_mask_by_camera'][cam_name] = np.zeros(len(points_cam_valid), dtype=bool)
            else:
                # No polygon information provided, create all-False mask
                result['polygon_mask_by_camera'][cam_name] = np.zeros(len(points_cam_valid), dtype=bool)
    
    return result


def display_camera_images(image_paths, sample_token, polygon_by_camera=None):
    """
    Display all 6 camera images in a 2x3 grid layout using OpenCV (thread-safe).
    
    Grid layout:
        Row 1: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT
        Row 2: CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
    
    Args:
        image_paths: List of (camera_name, image_path) tuples
        sample_token: Sample token for window title
        polygon_by_camera: Dict mapping camera_name -> polygon (numpy array (N, 2) with normalized coordinates)
    """
    # Create a dictionary to store images by camera name
    camera_images = {}
    
    # Load all available camera images
    for camera_name, image_path in image_paths:
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                camera_images[camera_name] = img
    
    if not camera_images:
        print(f"  Warning: No camera images found to display")
        return
    
    # Get images in CAM_TYPES order
    images_to_display = []
    labels = []
    for cam_type in cfg.CAM_TYPES:
        if cam_type in camera_images:
            images_to_display.append(camera_images[cam_type])
            labels.append(cam_type)
            
            if polygon_by_camera and cam_type in polygon_by_camera:
                polygon = polygon_by_camera[cam_type]
                if polygon is not None:
                    # Draw polygon on image in red
                    images_to_display[-1] = draw_polygon_on_image(images_to_display[-1], polygon, color=(0, 0, 255), thickness=2)
                    print(f"  Drawn polygon on {cam_type} image")
    
    if not images_to_display:
        print(f"  Warning: No valid camera images to display")
        return
    
    # Resize all images to the same size
    if images_to_display:
        # Get target size (use median dimensions to balance)
        heights = [img.shape[0] for img in images_to_display]
        widths = [img.shape[1] for img in images_to_display]
        target_h = int(np.median(heights))
        target_w = int(np.median(widths))
        
        # Resize all images to target size
        resized_images = []
        for img in images_to_display:
            resized = cv2.resize(img, (target_w, target_h))
            resized_images.append(resized)
        
        # Arrange in 2x3 grid (2 rows, 3 columns)
        # Row 1: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT (indices 0, 1, 2)
        # Row 2: CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT (indices 3, 4, 5)
        
        # Create grid with 6 slots (always 2x3)
        grid_images = []
        for i in range(6):
            if i < len(resized_images):
                grid_images.append(resized_images[i])
            else:
                # Create black placeholder for missing cameras
                grid_images.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
        
        # Create rows
        row1 = np.hstack([grid_images[0], grid_images[1], grid_images[2]])  # Front cameras
        row2 = np.hstack([grid_images[3], grid_images[4], grid_images[5]])  # Back cameras
        combined = np.vstack([row1, row2])
        grid_h, grid_w = combined.shape[:2]
        
        # Add text labels to each image
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0)  # Green
        
        for i, label in enumerate(labels):
            if i < 3:
                # First row
                x = (i % 3) * target_w + 10
                y = 30
            else:
                # Second row
                x = ((i - 3) % 3) * target_w + 10
                y = target_h + 30
            
            cv2.putText(combined, label, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    # Display window
    window_name = f'Camera Images - Sample {sample_token[:8]}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set window size based on grid
    if 'combined' in locals():
        display_w = min(grid_w, 2400)  # Max width
        display_h = min(grid_h, 1600)  # Max height
        cv2.resizeWindow(window_name, display_w, display_h)
        cv2.imshow(window_name, combined)
        print(f"  Displaying {len(images_to_display)} camera images in grid layout. Press any key or close the window to continue...")
        cv2.waitKey(0)  # Wait for key press
        cv2.destroyWindow(window_name)


def display_camera_images_threaded(image_paths, sample_token, polygon_by_camera=None):
    """
    Display camera images in a separate thread (for simultaneous windows).
    
    Args:
        image_paths: List of (camera_name, image_path) tuples
        sample_token: Sample token for window title
    """
    def _display():
        display_camera_images(image_paths, sample_token, polygon_by_camera)
    
    thread = threading.Thread(target=_display, daemon=False)
    thread.start()
    return thread


def display_point_cloud(pcd, sample_token, gt_lidar_boxes=None):
    """Display point cloud using open3d."""
    if pcd is None or len(pcd.points) == 0:
        print(f"  Warning: No point cloud to display for sample {sample_token}")
        return
    
    print(f"  Displaying point cloud with {len(pcd.points)} points...")
    print(f"  Press 'Q' or close the window to continue")
    
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
            
            # Change the color of points which are in this box
            indices = obb.get_point_indices_within_bounding_box(pcd.points)
            if len(indices) > 0:
                # Convert colors to numpy array, modify, then assign back
                colors_array = np.asarray(pcd.colors)
                colors_array[indices] = [1, 0, 0]  # Red color for points in box
                pcd.colors = o3d.utility.Vector3dVector(colors_array)
                vis.update_geometry(pcd)
            
            # Find the center of front face (heading direction)
            # Extract yaw from rotation matrix (rotation around z-axis)
            # The rotation matrix's first column gives the heading direction
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


def downsample_point_cloud_mmcv(
    points,
    colors=None,
    polygon_mask=None,
    voxel_size=0.1,
    use_fps=False,
    fps_num_points=None,
    point_cloud_range=None,
):
    """
    Downsample point cloud using mmdet3d.ops operations (voxelization + optional FPS).
    This helps reduce overlapping points and manage large point clouds.
    
    Args:
        points: numpy array of shape (N, 3) - point coordinates
        colors: numpy array of shape (N, 3) - point colors (optional)
        polygon_mask: numpy array of shape (N,) - boolean mask for polygon points (optional)
        voxel_size: float or list of 3 floats - size of voxel for downsampling (default: 0.1m)
        use_fps: bool - whether to apply FPS after voxelization (default: False)
        fps_num_points: int - number of points to sample with FPS (required if use_fps=True)
        point_cloud_range: list of 6 floats [x_min, y_min, z_min, x_max, y_max, z_max] - 
                          point cloud range for voxelization (optional, auto-computed if None)
    
    Returns:
        Dictionary with:
        - 'points': downsampled points (M, 3)
        - 'colors': downsampled colors (M, 3) if colors provided
        - 'polygon_mask': downsampled polygon mask (M,) if polygon_mask provided
        - 'indices': indices of selected points in original point cloud (M,)
    """
    # Import mmdet3d ops at the start
    try:
        from mmdet3d.ops import Voxelization, furthest_point_sample
    except ImportError as e:
        raise ImportError(
            "mmdet3d.ops is required for point cloud downsampling but could not be imported.\n"
            f"Error: {e}\n"
            "Please ensure mmdet3d ops are properly compiled. "
            "You may need to rebuild the ops extensions for your Python version."
        )
    
    if len(points) == 0:
        return {
            'points': points,
            'colors': colors if colors is not None else None,
            'polygon_mask': polygon_mask if polygon_mask is not None else None,
            'indices': np.array([], dtype=np.int64)
        }
    
    # Convert to torch tensor
    points_tensor = torch.from_numpy(points).float()
    
    # Detect device for GPU operations (FPS needs GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle FPS-only mode (no voxelization)
    if voxel_size is None:
        if not use_fps or fps_num_points is None:
            # No downsampling requested, return original
            return {
                'points': points,
                'colors': colors if colors is not None else None,
                'polygon_mask': polygon_mask if polygon_mask is not None else None,
                'indices': np.arange(len(points))
            }
        
        # FPS-only mode: skip voxelization, apply FPS directly
        if len(points) > fps_num_points:
            print(f"  Applying FPS-only to downsample from {len(points)} to {fps_num_points} points...")
            # Move tensor to GPU for FPS operation
            points_for_fps = points_tensor.to(device).unsqueeze(0)  # (1, N, 3)
            fps_indices = furthest_point_sample(points_for_fps, fps_num_points)
            fps_indices = fps_indices.squeeze(0).cpu().numpy()  # (fps_num_points,)
            
            downsampled_points = points[fps_indices]
            downsampled_colors = colors[fps_indices] if colors is not None else None
            downsampled_polygon_mask = polygon_mask[fps_indices] if polygon_mask is not None else None
            
            return {
                'points': downsampled_points,
                'colors': downsampled_colors,
                'polygon_mask': downsampled_polygon_mask,
                'indices': fps_indices
            }
        else:
            # Already fewer points than requested, return original
            return {
                'points': points,
                'colors': colors if colors is not None else None,
                'polygon_mask': polygon_mask if polygon_mask is not None else None,
                'indices': np.arange(len(points))
            }
    
    # Voxelization mode (with optional FPS after)
    # Auto-compute point cloud range if not provided
    if point_cloud_range is None:
        x_min, y_min, z_min = points.min(axis=0) - 1.0
        x_max, y_max, z_max = points.max(axis=0) + 1.0
        point_cloud_range = [x_min, y_min, z_min, x_max, y_max, z_max]
    
    # Convert voxel_size to list if single float
    if isinstance(voxel_size, (int, float)):
        voxel_size = [voxel_size, voxel_size, voxel_size]
    
    # Initialize voxelization
    # Note: mmdet3d Voxelization takes input directly (no batch dimension needed in forward)
    max_num_points = 100  # Max points per voxel (we'll average them)
    max_voxels = 200000  # Max number of voxels
    
    try:
        voxel_layer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels
        )
        # Set to eval mode to use max_voxels[1] (testing mode)
        voxel_layer.eval()
        
        # mmdet3d Voxelization.forward() takes input directly (NC points)
        # Input shape: (N, C) where N is number of points, C is number of channels (3 for xyz)
        points_input = points_tensor  # (N, 3)
        
        # Apply voxelization
        with torch.no_grad():
            voxels, coors, num_points_per_voxel = voxel_layer(points_input)
        
        # Extract voxel centers (mean of points in each voxel)
        # voxels shape: (num_voxels, max_num_points, 3)
        # num_points_per_voxel shape: (num_voxels,)
        
        # Compute mean of points in each voxel
        num_voxels = voxels.shape[0]
        voxel_centers = []
        
        # For each voxel, compute mean position (voxel center)
        for i in range(num_voxels):
            n_points = num_points_per_voxel[i].item()
            if n_points > 0:
                voxel_points = voxels[i, :n_points, :]  # (n_points, 3)
                center = voxel_points.mean(dim=0).numpy()  # (3,)
                voxel_centers.append(center)
        
        if len(voxel_centers) == 0:
            print("  Warning: Voxelization produced no voxels, using original points")
            return {
                'points': points,
                'colors': colors,
                'polygon_mask': polygon_mask,
                'indices': np.arange(len(points))
            }
        
        voxel_centers = np.array(voxel_centers)  # (M, 3)
        
        # If colors or polygon_mask provided, we need to map voxels back to original points
        # This is complex with mmdet3d Voxelization, so we'll use a simpler approach:
        # Find closest original point to each voxel center using KDTree for memory efficiency
        if colors is not None or polygon_mask is not None:
            # Use KDTree for efficient nearest neighbor search (memory efficient)
            # This avoids creating a huge (M, N, 3) array
            tree = cKDTree(points)
            _, closest_indices = tree.query(voxel_centers, k=1)  # (M,)
            
            if colors is not None:
                voxel_colors = colors[closest_indices]
            else:
                voxel_colors = None
            
            if polygon_mask is not None:
                voxel_polygon_mask = polygon_mask[closest_indices]
            else:
                voxel_polygon_mask = None
        else:
            voxel_colors = None
            voxel_polygon_mask = None
            closest_indices = np.arange(len(voxel_centers))
        
        downsampled_points = voxel_centers
        
        # Apply FPS if requested
        if use_fps and fps_num_points is not None and len(downsampled_points) > fps_num_points:
            print(f"  Applying FPS to downsample from {len(downsampled_points)} to {fps_num_points} points...")
            # FPS expects batch dimension: (1, N, 3)
            # Move tensor to GPU for FPS operation
            points_for_fps = torch.from_numpy(downsampled_points).float().to(device).unsqueeze(0)
            fps_indices = furthest_point_sample(points_for_fps, fps_num_points)
            fps_indices = fps_indices.squeeze(0).cpu().numpy()  # (fps_num_points,)
            
            downsampled_points = downsampled_points[fps_indices]
            if voxel_colors is not None:
                voxel_colors = voxel_colors[fps_indices]
            if voxel_polygon_mask is not None:
                voxel_polygon_mask = voxel_polygon_mask[fps_indices]
            closest_indices = closest_indices[fps_indices]
        
        return {
            'points': downsampled_points,
            'colors': voxel_colors,
            'polygon_mask': voxel_polygon_mask,
            'indices': closest_indices
        }
        
    except Exception as e:
        raise RuntimeError(f"mmdet3d voxelization failed: {e}")


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
    """Run inference for a single sample and optionally display point cloud."""
    print(f"\n{'='*60}")
    print(f"Processing sample token: {sample_token}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Cameras: {[name for name, _ in image_paths]}")
    print(f"{'='*60}")
    
    # image_paths is already in cfg.CAM_TYPES order from get_camera_images_from_sample
    # Extract just the image paths in consistent order
    image_files = [path for _, path in image_paths]
    camera_order = [name for name, _ in image_paths]
    
    print(f"  Camera order: {camera_order}")
    print(f"  Number of images for inference: {len(image_files)}")
    
    # Create output directory for this sample
    frame_output_dir = os.path.join(output_dir, sample_token)
    os.makedirs(frame_output_dir, exist_ok=True)
    
    
    
    
    try:
        # Prepare export_kwargs for GLB export filtering options
        export_kwargs = {}
        if "glb" in export_format:
            export_kwargs["glb"] = {
                "sky_depth_def": cfg.GLB_CONFIG["sky_depth_def"],
                "conf_thresh_percentile": cfg.GLB_CONFIG["conf_thresh_percentile"],
                "filter_black_bg": cfg.GLB_CONFIG["filter_black_bg"],
                "filter_white_bg": cfg.GLB_CONFIG["filter_white_bg"],
            }
        
        # Run inference (no extrinsics/intrinsics - model estimates them!)
        prediction = model.inference(
            image=image_files,
            export_dir=frame_output_dir,
            export_format=export_format,
            ref_view_strategy=ref_view_strategy,
            use_ray_pose=use_ray_pose,
            num_max_points=max_points,
            export_kwargs=export_kwargs if export_kwargs else None,
        )
        
        print(f"✓ Successfully processed sample {sample_token}")
        print(f"  Output directory: {frame_output_dir}")
        
        # Display camera images and point cloud simultaneously
        if display:
            # Display camera images (all 6 cameras) in a separate thread (OpenCV is thread-safe)
            camera_thread = display_camera_images_threaded(image_paths, sample_token, nusc_info['polygon_by_camera'])
            
            
            # Load point clouds from prediction (points stay in camera coordinates)
            # Apply filtering for sky, confidence, and max depth
            pcd_data = load_point_cloud_from_prediction(
                prediction, 
                image_paths,
                max_depth=cfg.GLB_CONFIG["max_depth"],
                conf_thresh_percentile=cfg.GLB_CONFIG["conf_thresh_percentile"],
                polygon_by_camera=nusc_info['polygon_by_camera'],
            )
            
            if pcd_data and pcd_data['points_by_camera']:
                # Transform each camera's points from camera coordinates to LiDAR coordinates
                # using ground truth extrinsics from nuScenes (not model's predicted extrinsics)
                all_points_lidar = []
                all_colors = []
                all_polygon_mask = []
                
                print(f"  Transforming points from camera coordinates to LiDAR coordinates...")
                total_points_before_downsample = 0
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
                            
                            # Get colors if available
                            cam_colors = None
                            if cam_name in pcd_data['colors_by_camera']:
                                cam_colors = pcd_data['colors_by_camera'][cam_name]
                            
                            # Get polygon mask if available
                            cam_polygon_mask = None
                            if cam_name in pcd_data['polygon_mask_by_camera']:
                                cam_polygon_mask = pcd_data['polygon_mask_by_camera'][cam_name]
                            
                            # DOWNSAMPLE PER CAMERA BEFORE COMBINING (much more memory efficient!)
                            # This avoids building huge KDTree on 600k+ points
                            if nusc_info.get('downsample_voxel_size', None) is not None:
                                num_points_before = len(points_lidar)
                                total_points_before_downsample += num_points_before
                                
                                downsample_result = downsample_point_cloud_mmcv(
                                    points_lidar,
                                    colors=cam_colors,
                                    polygon_mask=cam_polygon_mask,
                                    voxel_size=nusc_info.get('downsample_voxel_size', 0.1),
                                    use_fps=False,  # Don't use FPS per camera, only on final combined if needed
                                    fps_num_points=None,
                                    point_cloud_range=nusc_info.get('downsample_point_cloud_range', None),
                                )
                                
                                points_lidar = downsample_result['points']
                                cam_colors = downsample_result['colors']
                                cam_polygon_mask = downsample_result['polygon_mask']
                                
                                num_points_after = len(points_lidar)
                                reduction = num_points_before - num_points_after
                                reduction_pct = (reduction / num_points_before * 100) if num_points_before > 0 else 0
                                print(f"    {cam_name}: {len(points_cam)} points -> {num_points_before} points in LiDAR -> {num_points_after} after downsampling ({reduction_pct:.1f}% reduction)")
                            else:
                                total_points_before_downsample += len(points_lidar)
                                print(f"    {cam_name}: {len(points_cam)} points -> {len(points_lidar)} points in LiDAR frame")
                            
                            all_points_lidar.append(points_lidar)
                            
                            if cam_colors is not None:
                                all_colors.append(cam_colors)
                            
                            if cam_polygon_mask is not None:
                                all_polygon_mask.append(cam_polygon_mask)
                        else:
                            print(f"    WARNING: No transformation found for {cam_name}, skipping...")
                
                if all_points_lidar:
                    # Concatenate all points (already downsampled per camera)
                    combined_points = np.concatenate(all_points_lidar, axis=0)
                    combined_colors = None
                    combined_polygon_mask = None
                    
                    if all_colors:
                        combined_colors = np.concatenate(all_colors, axis=0)
                    
                    if all_polygon_mask:
                        combined_polygon_mask = np.concatenate(all_polygon_mask, axis=0)
                    
                    num_points_after_camera_downsample = len(combined_points)
                    print(f"  Total points in LiDAR frame (after per-camera downsampling): {num_points_after_camera_downsample} (was {total_points_before_downsample} before)")
                    
                    # Apply FPS on combined point cloud if requested (optional final step)
                    if nusc_info.get('downsample_use_fps', False) and nusc_info.get('downsample_fps_num_points', None) is not None:
                        if len(combined_points) > nusc_info.get('downsample_fps_num_points', None):
                            print(f"  Applying FPS to final combined point cloud...")
                            downsample_result = downsample_point_cloud_mmcv(
                                combined_points,
                                colors=combined_colors,
                                polygon_mask=combined_polygon_mask,
                                voxel_size=None,  # Skip voxelization, only FPS
                                use_fps=True,
                                fps_num_points=nusc_info.get('downsample_fps_num_points', None),
                                point_cloud_range=None,
                            )
                            
                            combined_points = downsample_result['points']
                            combined_colors = downsample_result['colors']
                            combined_polygon_mask = downsample_result['polygon_mask']
                            
                            num_points_after_fps = len(combined_points)
                            print(f"  Points after FPS: {num_points_after_fps}")
                    
                    # Create point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(combined_points)
                    
                    if combined_colors is not None:
                        # Override colors for polygon points (red)
                        if combined_polygon_mask is not None:
                            # Set polygon points to red [1, 0, 0]
                            combined_colors[combined_polygon_mask] = [1.0, 0.0, 0.0]
                            print(f"  Colored {combined_polygon_mask.sum()} polygon points in red")
                        
                        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
                    
                    print(f"  Total points in LiDAR frame (final): {len(combined_points)}")
                    
                    # Display point cloud in MAIN thread (Open3D/Qt requires main thread)
                    print(f"  Both windows are open. Close both windows to continue...")
                    display_point_cloud(pcd, sample_token, nusc_info['gt_lidar_boxes'])
                else:
                    print(f"  Warning: No valid points to display")
            else:
                print(f"  Warning: Could not load point cloud for display")
            
            # Wait for camera images window to close
            camera_thread.join()
        
        return True, prediction
        
    except Exception as e:
        print(f"✗ Error processing sample {sample_token}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 Inference for nuScenes-format datasets (sample-based iteration)"
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
        "--version",
        type=str,
        default="v1.0-trainval",
        help="NuScenes dataset version (default: v1.0-trainval)",
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

    # Get device from config (with auto-fallback)
    device = cfg.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        device = "cpu"

    print(f"\n{'='*60}")
    print("Depth Anything 3 - nuScenes Inference (Sample-based)")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {cfg.MODEL_NAME} (from infer_config.py)")
    print(f"Device: {device}")
    print(f"Export format: {cfg.EXPORT_FORMAT} (from infer_config.py)")
    print(f"Display point clouds: {not args.no_display}")
    print(f"{'='*60}\n")

    # Initialize NuScenes dataset
    print(f"Loading NuScenes dataset (version: {args.version})...")
    try:
        nusc = NuScenes(version=args.version, dataroot=args.data_dir, verbose=True)
        print(f"Loaded {len(nusc.sample)} samples")
    except Exception as e:
        print(f"Error loading NuScenes dataset: {e}")
        sys.exit(1)

    # Get list of samples
    samples = list(nusc.sample)
    
    # Filter by specific sample index if requested
    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(samples):
            print(f"Error: Sample index {args.sample_index} is out of range (0-{len(samples)-1})")
            sys.exit(1)
        samples = [samples[args.sample_index]]
        print(f"Processing sample at index {args.sample_index}: {samples[0]['token']}")

    # Limit number of samples if requested
    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Processing first {len(samples)} samples")

    # Determine cache directory (ckpts folder in repository root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    cache_dir = os.path.join(repo_root, "ckpts")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"\nModel cache directory: {cache_dir}")

    # Load model
    print(f"Loading model: {cfg.MODEL_NAME}...")
    try:
        model = DepthAnything3.from_pretrained(cfg.MODEL_NAME, cache_dir=cache_dir)
        model = model.to(device=device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Process each sample
    print(f"\nProcessing {len(samples)} sample(s)...")
    successful = 0
    failed = 0
    
    
    
    
    for i, sample in enumerate(samples, 1):
        sample_token = sample['token']
        print(f"\n[{i}/{len(samples)}] Processing sample: {sample_token}")
        
        # Get camera images from sample in cfg.CAM_TYPES order
        image_paths = get_camera_images_from_sample(nusc, sample, args.data_dir)
        
        if not image_paths:
            print(f"  Warning: No camera images found for sample {sample_token}, skipping...")
            failed += 1
            continue
        
        print(f"  Found {len(image_paths)} camera images")
        
        
        
            
        # For debug: Load and draw polygon on CAM_FRONT_LEFT
        # Hard-coded path for now
        polygon_path = 'data/segmentation_polygon/CAM_FRONT_LEFT_1747104154682264.txt'
        polygon_by_camera = {}
        polygon_by_camera['CAM_FRONT_LEFT'] = load_polygon_from_txt(polygon_path)
        
        
        

        polygon_by_camera = None
        
        
        # Get nuScenes info (transformations, boxes, etc.)
        nusc_info = get_nusc_info(nusc, sample)
        if polygon_by_camera is not None:
            nusc_info['polygon_by_camera'] = polygon_by_camera
        else:
            nusc_info['polygon_by_camera'] = {}
        
        # Add downsampling parameters to nusc_info (from config)
        nusc_info['downsample_voxel_size'] = cfg.DOWNSAMPLE_VOXEL_SIZE
        nusc_info['downsample_use_fps'] = cfg.DOWNSAMPLE_USE_FPS
        nusc_info['downsample_fps_num_points'] = cfg.DOWNSAMPLE_FPS_NUM_POINTS
        nusc_info['downsample_point_cloud_range'] = cfg.DOWNSAMPLE_POINT_CLOUD_RANGE
    
        success, _ = run_inference_for_frame(
            model=model,
            sample_token=sample_token,
            image_paths=image_paths,
            output_dir=args.output_dir,
            device=device,
            export_format=cfg.EXPORT_FORMAT,
            ref_view_strategy=cfg.REF_VIEW_STRATEGY,
            use_ray_pose=cfg.USE_RAY_POSE,
            max_points=cfg.MAX_POINTS,
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
    print(f"Total samples: {len(samples)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
