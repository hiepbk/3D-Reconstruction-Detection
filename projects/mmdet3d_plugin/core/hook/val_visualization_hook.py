"""
Custom hook for validation visualization with WandB integration.

This hook runs after validation and generates visualization images for
deterministic samples, uploading them to WandB with slider support.
"""

import numpy as np
import cv2
import os
from os import path as osp
from typing import List, Dict, Optional
import torch

from mmcv.runner import HOOKS, Hook
from mmdet3d.core.bbox import Box3DMode, Coord3DMode
from mmdet3d.core import show_result
from projects.mmdet3d_plugin.datasets.utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)


@HOOKS.register_module()
class ValVisualizationHook(Hook):
    """Hook to visualize validation results and upload to WandB.
    
    This hook:
    1. Runs after validation evaluation
    2. Generates visualization images for 10 deterministic samples
    3. Uploads images to WandB with slider support (one image per epoch)
    
    Args:
        num_samples (int): Number of deterministic samples to visualize. Default: 10.
        score_threshold (float): Score threshold for filtering predictions. Default: 0.1.
        interval (int): Evaluation interval. Default: 1.
    """
    
    def __init__(
        self,
        num_samples: int = 10,
        score_threshold: float = 0.1,
        interval: int = 1,
    ):
        super(ValVisualizationHook, self).__init__()
        self.num_samples = num_samples
        self.score_threshold = score_threshold
        self.interval = interval
        
        # Store deterministic sample indices (computed once, by timestamp)
        self.sample_indices = None
        self.sample_tokens = None  # Token identifiers for logging
        self.sample_timestamps = None  # Timestamps for naming
    
    def after_train_epoch(self, runner):
        """Generate visualizations after validation (called after EvalHook)."""
        # Only run if evaluation interval matches
        if runner.epoch % self.interval != 0:
            return
        
        # Get validation dataset
        # Try multiple ways to get validation dataset
        dataset = None
        if hasattr(runner, 'val_data_loader') and runner.val_data_loader is not None:
            dataset = runner.val_data_loader.dataset
        elif hasattr(runner, 'data_loader'):
            if isinstance(runner.data_loader, list) and len(runner.data_loader) >= 2:
                dataset = runner.data_loader[1].dataset
            elif hasattr(runner.data_loader, 'dataset'):
                dataset = runner.data_loader.dataset
        
        if dataset is None:
            runner.logger.warning("[ValVisualizationHook] No validation dataset found")
            return
        
        # Unwrap dataset if it's a wrapper (e.g., CBGSDataset)
        # CBGSDataset wraps the actual dataset in .dataset attribute
        while hasattr(dataset, 'dataset') and not hasattr(dataset, '_get_pipeline'):
            dataset = dataset.dataset
            runner.logger.debug(f"[ValVisualizationHook] Unwrapped dataset to {type(dataset).__name__}")
        
        # Get results first to know the valid range
        # Find EvalHook and get results to know the valid index range
        results = None
        for hook in runner.hooks:
            hook_type = type(hook).__name__
            if 'EvalHook' in hook_type and hasattr(hook, 'latest_results'):
                if hook.latest_results is not None:
                    results = hook.latest_results
                    break
        
        if results is None:
            runner.logger.debug(
                "[ValVisualizationHook] No results from EvalHook yet. "
                "Evaluation may not have run for this epoch."
            )
            return
        
        results_len = len(results)
        
        # Initialize sample indices if not done
        # Select samples deterministically by timestamp (not by dataloader index)
        # IMPORTANT: Results are indexed sequentially [0, results_len), so we must select
        # indices that are valid for the results, not the full dataset
        if self.sample_indices is None:
            # Get all timestamps from the validation dataset
            timestamps_with_indices = []
            dataset_len = min(len(dataset), results_len)  # Use min to avoid out-of-bounds
            
            for idx in range(dataset_len):
                if hasattr(dataset, 'data_infos') and idx < len(dataset.data_infos):
                    info = dataset.data_infos[idx]
                    # Use timestamp as unique identifier (convert to float for sorting)
                    timestamp = float(info.get('timestamp', idx))
                    # Also store token as backup identifier
                    token = info.get('token', f'idx_{idx}')
                    timestamps_with_indices.append((timestamp, token, idx))
            
            if len(timestamps_with_indices) == 0:
                runner.logger.warning("[ValVisualizationHook] Could not get timestamps, falling back to index-based selection")
                step = max(1, results_len // self.num_samples)
                self.sample_indices = list(range(0, results_len, step))[:self.num_samples]
                self.sample_tokens = [f"idx_{idx}" for idx in self.sample_indices]
                self.sample_timestamps = [None] * len(self.sample_indices)
            else:
                # Sort by timestamp
                timestamps_with_indices.sort(key=lambda x: x[0])
                
                # Select evenly spaced samples by timestamp
                total_samples = len(timestamps_with_indices)
                step = max(1, total_samples // self.num_samples)
                selected = timestamps_with_indices[::step][:self.num_samples]
                
                # Extract indices, tokens, and timestamps
                # Ensure indices are within results range
                self.sample_indices = [item[2] for item in selected if item[2] < results_len]  # indices
                self.sample_tokens = [item[1] for item in selected if item[2] < results_len]  # tokens for logging
                self.sample_timestamps = [item[0] for item in selected if item[2] < results_len]  # timestamps for naming
                
                # If we filtered out some samples, log a warning
                if len(self.sample_indices) < len(selected):
                    runner.logger.warning(
                        f"[ValVisualizationHook] Filtered out {len(selected) - len(self.sample_indices)} "
                        f"samples that were out of results range (results_len={results_len})"
                    )
                
                runner.logger.info(
                    f"[ValVisualizationHook] Selected {len(self.sample_indices)} "
                    f"deterministic samples by timestamp (indices: {self.sample_indices}): {self.sample_tokens}"
                )
        
        # Get results from EvalHook (already retrieved above for index range check)
        # Since our hook has LOW priority and EvalHook has NORMAL priority,
        # our hook runs AFTER EvalHook's after_train_epoch completes,
        # which means _do_evaluate has finished and results are stored
        from mmcv.runner import get_dist_info
        
        rank, _ = get_dist_info()
        # Only run visualization on rank 0 (where results are gathered in distributed training)
        if rank != 0:
            return
        
        # Results were already retrieved above, but log here
        if results is not None:
            runner.logger.info(
                f"[ValVisualizationHook] Found {len(results)} results from EvalHook"
            )
        
        if len(results) == 0:
            runner.logger.warning("[ValVisualizationHook] Empty results from EvalHook")
            return
        
        # Generate visualization images for selected samples
        vis_images = []
        vis_tokens = []  # Track which tokens were successfully visualized
        vis_timestamps = []  # Track timestamps for naming
        # Use _get_pipeline(None) to get the dataset's default pipeline (same as show method)
        pipeline = dataset._get_pipeline(None)
        
        for i, idx in enumerate(self.sample_indices):
            if idx >= len(results):
                runner.logger.warning(
                    f"[ValVisualizationHook] Sample index {idx} >= results length {len(results)}, skipping"
                )
                continue
            
            # Get token and timestamp for this sample (for logging and naming)
            token = self.sample_tokens[i] if (hasattr(self, 'sample_tokens') and self.sample_tokens and i < len(self.sample_tokens)) else f"idx_{idx}"
            timestamp = self.sample_timestamps[i] if (hasattr(self, 'sample_timestamps') and self.sample_timestamps and i < len(self.sample_timestamps)) else None
            
            # Get result for this sample
            result = results[idx]
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            
            # Generate visualization image
            try:
                vis_image = self._generate_visualization(
                    dataset=dataset,
                    result=result,
                    sample_idx=idx,
                    pipeline=pipeline,
                )
                
                if vis_image is not None:
                    vis_images.append(vis_image)
                    vis_tokens.append(token)
                    vis_timestamps.append(timestamp)
                    runner.logger.debug(f"[ValVisualizationHook] Generated visualization for sample {token} (idx {idx}, timestamp {timestamp})")
                else:
                    runner.logger.warning(f"[ValVisualizationHook] Failed to generate visualization for sample {token} (idx {idx}): returned None")
            except Exception as e:
                runner.logger.error(
                    f"[ValVisualizationHook] Error generating visualization for sample {token} (idx {idx}): {e}"
                )
                import traceback
                runner.logger.debug(traceback.format_exc())
        
        if len(vis_images) == 0:
            runner.logger.warning("[ValVisualizationHook] No visualization images generated")
            return
        
        # Upload to WandB with slider support
        try:
            import wandb
            
            # Create wandb media list for slider (one image per epoch)
            # Format: list of dicts with 'epoch' and 'image' keys
            wandb_images = []
            for i, img in enumerate(vis_images):
                # Convert BGR to RGB for wandb
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Use token and timestamp as caption
                token = vis_tokens[i] if i < len(vis_tokens) else f"idx_{self.sample_indices[i]}"
                timestamp = vis_timestamps[i] if i < len(vis_timestamps) and vis_timestamps[i] is not None else None
                
                if timestamp is not None:
                    # Format timestamp (convert from microseconds to readable format)
                    timestamp_str = f"{timestamp:.6f}" if timestamp < 1e10 else f"{timestamp/1e6:.3f}s"
                    caption = f"Sample {token} (t={timestamp_str})"
                else:
                    caption = f"Sample {token}"
                
                wandb_images.append(wandb.Image(img_rgb, caption=caption))
            
            # Log to wandb with slider support
            # For slider: log each sample with epoch info in the key
            # WandB will create a slider when the same key pattern is logged over time
            log_dict = {}
            
            # Log each sample with consistent key pattern for slider support
            # WandB creates sliders when the same key is logged multiple times across steps
            # Use timestamp as key identifier for better determinism (unique per frame)
            for i, img in enumerate(wandb_images):
                token = vis_tokens[i] if i < len(vis_tokens) else f"idx_{self.sample_indices[i]}"
                timestamp = vis_timestamps[i] if i < len(vis_timestamps) and vis_timestamps[i] is not None else None
                
                if timestamp is not None:
                    # Use timestamp as key for unique identification
                    key = f'val_visualization/sample_t{timestamp:.6f}'
                else:
                    # Fallback to token if timestamp not available
                    key = f'val_visualization/sample_{token}'
                
                # Each epoch, this same key gets a new image, creating a slider
                log_dict[key] = img
            
            # Gallery view (commented out - redundant since individual samples have sliders)
            # The gallery would show all samples in a grid at current epoch only
            # Individual samples above already have sliders to track changes over epochs
            # Uncomment if you want a quick grid overview:
            # try:
            #     log_dict['val_visualization/gallery'] = wandb.Images(wandb_images)
            # except:
            #     log_dict['val_visualization/gallery'] = wandb_images
            
            wandb.log(log_dict, step=runner.iter)
            
            runner.logger.info(
                f"[ValVisualizationHook] Uploaded {len(vis_images)} visualization images "
                f"to WandB (epoch {runner.epoch}, iter {runner.iter})"
            )
        except ImportError:
            runner.logger.warning("[ValVisualizationHook] WandB not available, skipping upload")
        except Exception as e:
            runner.logger.error(f"[ValVisualizationHook] Error uploading to WandB: {e}")
    
    def _generate_visualization(
        self,
        dataset,
        result: Dict,
        sample_idx: int,
        pipeline,
    ) -> Optional[np.ndarray]:
        """Generate visualization image for a single sample.
        
        Args:
            dataset: Dataset instance
            result: Detection result dict
            sample_idx: Index of the sample
            pipeline: Data pipeline for loading raw data
            
        Returns:
            Visualization image as numpy array, or None if failed
        """
        try:
            # Get data info
            data_info = dataset.data_infos[sample_idx]
            
            # Load points
            points = dataset._extract_data(sample_idx, pipeline, 'points')
            if isinstance(points, torch.Tensor):
                points = points.numpy()
            # Convert points to depth mode for visualization
            points = Coord3DMode.convert_point(
                points, Coord3DMode.LIDAR, Coord3DMode.DEPTH
            )
            
            # Filter predictions by score threshold
            inds = result['scores_3d'] > self.score_threshold
            
            # Get GT boxes
            gt_bboxes = dataset.get_ann_info(sample_idx)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
            
            # Get prediction boxes
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
            
            # Load images
            raw_imgs = dataset._extract_data(sample_idx, pipeline, 'img')
            lidar2img = dataset.get_data_info(sample_idx)["lidar2img"]
            
            # Prepare prediction dict
            pred_bboxes_3d = pred_bboxes.copy()
            pred_bboxes_3d[:, 2] += pred_bboxes_3d[:, 5] / 2  # Convert to gravity center
            
            pred_labels_3d = result["labels_3d"][inds].tolist()
            pred_cat_names = [dataset.CLASSES[label] for label in pred_labels_3d]
            pred_scores_3d = result["scores_3d"][inds].tolist()
            
            # Color mapping
            pred_color = []
            for label_id in pred_labels_3d:
                if label_id < len(dataset.ID_COLOR_MAP):
                    pred_color.append(dataset.ID_COLOR_MAP[label_id])
                else:
                    pred_color.append((255, 0, 0))  # Default red
            
            # Prepare GT dict
            gt_bboxes_3d = gt_bboxes.copy()
            gt_bboxes_3d[:, 2] += gt_bboxes_3d[:, 5] / 2  # Convert to gravity center
            
            gt_labels_3d = dataset.get_ann_info(sample_idx)["gt_labels_3d"]
            gt_cat_names = [dataset.CLASSES[label] for label in gt_labels_3d]
            gt_color = [(180, 180, 180) for _ in range(len(gt_bboxes_3d))]
            
            pred_dict = {
                "bboxes_3d": pred_bboxes_3d,
                "labels_3d": pred_labels_3d,
                "cat_names": pred_cat_names,
                "scores_3d": pred_scores_3d,
                "colors": pred_color,
            }
            
            gt_dict = {
                "bboxes_3d": gt_bboxes_3d,
                "labels_3d": gt_labels_3d,
                "cat_names": gt_cat_names,
                "colors": gt_color,
                "scores_3d": [1.0] * len(gt_bboxes_3d),
            }
            
            # Draw boxes on images
            imgs = []
            for j, img_origin in enumerate(raw_imgs):
                if isinstance(img_origin, torch.Tensor):
                    img = img_origin.permute(1, 2, 0).numpy().astype(np.uint8).copy()
                else:
                    img = img_origin.copy()
                
                img = draw_lidar_bbox3d_on_img(
                    img,
                    pred_dict,
                    gt_dict,
                    lidar2img[j],
                    img_metas=None,
                    thickness=3,
                )
                imgs.append(img)
            
            # Draw BEV
            bev = draw_lidar_bbox3d_on_bev(
                pred_dict,
                gt_dict,
                bev_size=imgs[0].shape[0] * 2,
            )
            
            # Add text labels to images
            camera_names = [
                "front",
                "front right",
                "front left",
                "rear",
                "rear left",
                "rear right",
            ]
            for j, name in enumerate(camera_names):
                if j < len(imgs):
                    imgs[j] = cv2.rectangle(
                        imgs[j],
                        (0, 0),
                        (440, 80),
                        color=(255, 255, 255),
                        thickness=-1,
                    )
                    w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                    text_x = int(220 - w / 2)
                    text_y = int(40 + h / 2)
                    
                    imgs[j] = cv2.putText(
                        imgs[j],
                        name,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
            
            # Concatenate images
            if len(imgs) >= 6:
                image = np.concatenate(
                    [
                        np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                        np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                    ],
                    axis=0,
                )
            else:
                # Fallback if not enough images
                image = np.concatenate(imgs, axis=1)
            
            # Add BEV
            image = np.concatenate([image, bev], axis=1)
            
            return image
            
        except Exception as e:
            import traceback
            print(f"[ValVisualizationHook] Error generating visualization for sample {sample_idx}: {e}")
            print(traceback.format_exc())
            return None

