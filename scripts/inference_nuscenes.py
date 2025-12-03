#!/usr/bin/env python3
"""
Inference script for nuScenes-format datasets
Iterates through nusc.sample to process each sample and generates point clouds.

Usage:
    python scripts/inference_nuscenes.py --data_dir /path/to/nuscenes --output_dir /path/to/output
"""

import argparse
import os
import sys
import threading

import numpy as np
import open3d as o3d
import torch
import cv2
from depth_anything_3.api import DepthAnything3
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


CAM_TYPES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


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
    
    for camera_type in CAM_TYPES:
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







def load_point_cloud_from_prediction(
    prediction, 
    image_paths=None,
    max_depth=None,
    conf_thresh_percentile=None,
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
    
    return result


def display_camera_images(image_paths, sample_token):
    """
    Display CAM_FRONT and CAM_BACK images side by side using OpenCV (thread-safe).
    
    Args:
        image_paths: List of (camera_name, image_path) tuples
        sample_token: Sample token for window title
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
    window_name = f'Camera Images - Sample {sample_token[:8]}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 800)
    cv2.imshow(window_name, combined)
    print(f"  Displaying camera images. Press any key or close the window to continue...")
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyWindow(window_name)


def display_camera_images_threaded(image_paths, sample_token):
    """
    Display camera images in a separate thread (for simultaneous windows).
    
    Args:
        image_paths: List of (camera_name, image_path) tuples
        sample_token: Sample token for window title
    """
    def _display():
        display_camera_images(image_paths, sample_token)
    
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
    sky_depth_def=98.0,
    conf_thresh_percentile=40.0,
    filter_black_bg=False,
    filter_white_bg=False,
    max_depth=None,
):
    """Run inference for a single sample and optionally display point cloud."""
    print(f"\n{'='*60}")
    print(f"Processing sample token: {sample_token}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Cameras: {[name for name, _ in image_paths]}")
    print(f"{'='*60}")
    
    # image_paths is already in CAM_TYPES order from get_camera_images_from_sample
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
                "sky_depth_def": sky_depth_def,
                "conf_thresh_percentile": conf_thresh_percentile,
                "filter_black_bg": filter_black_bg,
                "filter_white_bg": filter_white_bg,
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
            # Display camera images (CAM_FRONT and CAM_BACK) in a separate thread (OpenCV is thread-safe)
            camera_thread = display_camera_images_threaded(image_paths, sample_token)
            
            # Load point clouds from prediction (points stay in camera coordinates)
            # Apply filtering for sky, confidence, and max depth
            pcd_data = load_point_cloud_from_prediction(
                prediction, 
                image_paths,
                max_depth=max_depth,
                conf_thresh_percentile=conf_thresh_percentile,
            )
            
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
    # Depth filtering options
    parser.add_argument(
        "--sky_depth_def",
        type=float,
        default=98.0,
        help="[GLB] Percentile used to fill sky pixels with plausible depth values (default: 98.0)",
    )
    parser.add_argument(
        "--conf_thresh_percentile",
        type=float,
        default=40.0,
        help="[GLB] Lower percentile for adaptive confidence threshold (default: 40.0)",
    )
    parser.add_argument(
        "--filter_black_bg",
        action="store_true",
        help="[GLB] Filter near-black background pixels (default: False)",
    )
    parser.add_argument(
        "--filter_white_bg",
        action="store_true",
        help="[GLB] Filter near-white background pixels (default: False)",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=None,
        help="Maximum depth threshold to filter far objects/infinity (in meters, default: None = no limit)",
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
    print("Depth Anything 3 - nuScenes Inference (Sample-based)")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Export format: {args.export_format}")
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
    print(f"Loading model: {args.model_name}...")
    try:
        model = DepthAnything3.from_pretrained(args.model_name, cache_dir=cache_dir)
        model = model.to(device=args.device)
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
        
        # Get camera images from sample in CAM_TYPES order
        image_paths = get_camera_images_from_sample(nusc, sample, args.data_dir)
        
        if not image_paths:
            print(f"  Warning: No camera images found for sample {sample_token}, skipping...")
            failed += 1
            continue
        
        print(f"  Found {len(image_paths)} camera images")
        
        # Get nuScenes info (transformations, boxes, etc.)
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
            sky_depth_def=args.sky_depth_def,
            conf_thresh_percentile=args.conf_thresh_percentile,
            filter_black_bg=args.filter_black_bg,
            filter_white_bg=args.filter_white_bg,
            max_depth=args.max_depth,
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
