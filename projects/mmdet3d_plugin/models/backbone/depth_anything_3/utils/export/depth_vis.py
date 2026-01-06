# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import imageio
import numpy as np
import matplotlib
import matplotlib.cm as cm
import cv2

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.visualize import visualize_depth


def visualize_ray_channel(ray_channel: np.ndarray, cmap: str = "RdBu_r", percentile: float = 2.0):
    """
    Visualize a single channel of the ray map using a colormap.
    
    Args:
        ray_channel: Single channel ray map (H, W)
        cmap: Matplotlib colormap name
        percentile: Percentile for min/max normalization
        
    Returns:
        Colored visualization as uint8 numpy array (H, W, 3)
    """
    ray_channel = ray_channel.copy()
    valid_mask = np.isfinite(ray_channel)
    
    if valid_mask.sum() <= 10:
        # Return gray image if no valid values
        h, w = ray_channel.shape
        return np.full((h, w, 3), 128, dtype=np.uint8)
    
    # Compute percentiles for normalization
    ray_min = np.percentile(ray_channel[valid_mask], percentile)
    ray_max = np.percentile(ray_channel[valid_mask], 100 - percentile)
    
    if ray_min == ray_max:
        ray_min = ray_min - 1e-6
        ray_max = ray_max + 1e-6
    
    # Normalize to [0, 1]
    ray_normalized = ((ray_channel - ray_min) / (ray_max - ray_min)).clip(0, 1)
    
    # Apply colormap
    colormap = matplotlib.colormaps[cmap]
    ray_colored = colormap(ray_normalized, bytes=False)[:, :, :3]  # (H, W, 3), values 0-1
    
    # Convert to uint8
    ray_colored = (ray_colored * 255.0).astype(np.uint8)
    
    return ray_colored


def export_to_depth_vis(
    prediction: Prediction,
    export_dir: str,
):
    # Use prediction.processed_images, which is already processed image data
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    os.makedirs(os.path.join(export_dir, "depth_vis"), exist_ok=True)
    
    # Check if ray is available in aux dict
    ray = None
    if prediction.aux is not None:
        ray = prediction.aux.get("ray", None)
    
    # If ray is not in aux, try to get it from the prediction object directly (if it exists)
    if ray is None and hasattr(prediction, 'ray'):
        ray = prediction.ray
    
    # Convert ray to numpy if it's a torch tensor
    if ray is not None:
        if hasattr(ray, 'cpu'):
            ray = ray.cpu().numpy()
        elif hasattr(ray, 'numpy'):
            ray = ray.numpy()
        # Handle different possible shapes: (N, C, H, W) or (N, H, W, C) or (B, S, H, W, C)
        if ray.ndim == 5:
            # (B, S, H, W, C) -> (N, H, W, C) by removing batch dimension
            if ray.shape[0] == 1:
                ray = ray[0]  # (S, H, W, C)
            else:
                ray = ray.reshape(-1, *ray.shape[2:])  # (B*S, H, W, C)
        elif ray.ndim == 4:
            # Could be (N, C, H, W) or (N, H, W, C)
            if ray.shape[1] < ray.shape[3]:  # (N, C, H, W) where C < W
                ray = ray.transpose(0, 2, 3, 1)  # Convert to (N, H, W, C)
        # Now ray should be (N, H, W, C)
    
    for idx in range(prediction.depth.shape[0]):
        # Export depth visualization
        depth_vis = visualize_depth(prediction.depth[idx])
        image_vis = images_u8[idx]
        depth_vis = depth_vis.astype(np.uint8)
        image_vis = image_vis.astype(np.uint8)
        vis_image = np.concatenate([image_vis, depth_vis], axis=1)
        save_path = os.path.join(export_dir, f"depth_vis/{idx:04d}.jpg")
        imageio.imwrite(save_path, vis_image, quality=95)
        
        # Export ray channel visualizations if available
        if ray is not None and idx < ray.shape[0]:
            ray_map = ray[idx]  # (H, W, C)
            num_channels = ray_map.shape[2] if ray_map.ndim == 3 else 1
            
            # Visualize each channel (up to 6 channels as requested)
            num_channels_to_vis = min(num_channels, 6)
            for ch_idx in range(num_channels_to_vis):
                if ray_map.ndim == 3:
                    ray_channel = ray_map[:, :, ch_idx]  # (H, W)
                else:
                    ray_channel = ray_map  # (H, W) - single channel
                
                # Visualize this channel
                ray_vis = visualize_ray_channel(ray_channel, cmap="RdBu_r", percentile=2.0)
                
                # Resize ray_vis to match image_vis dimensions if they don't match
                img_h, img_w = image_vis.shape[:2]
                ray_h, ray_w = ray_vis.shape[:2]
                if img_h != ray_h or img_w != ray_w:
                    ray_vis = cv2.resize(ray_vis, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                
                # Concatenate with original image
                ray_vis_image = np.concatenate([image_vis, ray_vis], axis=1)
                
                # Save as {idx:04d}_ray_{ch_idx}.jpg
                ray_save_path = os.path.join(export_dir, f"depth_vis/{idx:04d}_ray_{ch_idx}.jpg")
                imageio.imwrite(ray_save_path, ray_vis_image, quality=95)
