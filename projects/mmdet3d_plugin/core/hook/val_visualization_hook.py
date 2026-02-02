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
        
        # Get validation dataset (single path for consistency)
        if not hasattr(runner, 'val_data_loader') or runner.val_data_loader is None:
            runner.logger.warning("[ValVisualizationHook] No val_data_loader; skip visualization")
            return
        dataset = runner.val_data_loader.dataset
        
        # Unwrap dataset if it's a wrapper (e.g., CBGSDataset)
        # CBGSDataset wraps the actual dataset in .dataset attribute
        while hasattr(dataset, 'dataset') and not hasattr(dataset, '_get_pipeline'):
            dataset = dataset.dataset
            runner.logger.debug(f"[ValVisualizationHook] Unwrapped dataset to {type(dataset).__name__}")

        if not hasattr(dataset, 'generate_vis_image_for_sample'):
            runner.logger.warning(
                "[ValVisualizationHook] Dataset has no generate_vis_image_for_sample; skip visualization"
            )
            return

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
                runner.logger.warning(
                    "[ValVisualizationHook] No timestamps from dataset; skip visualization (single path only)"
                )
                return
            # Sort by timestamp and select evenly spaced samples
            timestamps_with_indices.sort(key=lambda x: x[0])
            total_samples = len(timestamps_with_indices)
            step = max(1, total_samples // self.num_samples)
            selected = timestamps_with_indices[::step][:self.num_samples]
            self.sample_indices = [item[2] for item in selected if item[2] < results_len]
            self.sample_tokens = [item[1] for item in selected if item[2] < results_len]
            self.sample_timestamps = [item[0] for item in selected if item[2] < results_len]
            if len(self.sample_indices) < len(selected):
                runner.logger.warning(
                    f"[ValVisualizationHook] Filtered out {len(selected) - len(self.sample_indices)} "
                    f"samples out of results range (results_len={results_len})"
                )
            runner.logger.info(
                f"[ValVisualizationHook] Selected {len(self.sample_indices)} samples by timestamp: {self.sample_tokens}"
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
            
            # Use dataset's method only (same projection/drawing as show() for consistency)
            try:
                vis_image = dataset.generate_vis_image_for_sample(
                    idx, result, pipeline, self.score_threshold
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
            
            # Log each sample with consistent key pattern for slider support (token only)
            for i, img in enumerate(wandb_images):
                token = vis_tokens[i] if i < len(vis_tokens) else f"idx_{self.sample_indices[i]}"
                key = f'val_visualization/sample_{token}'
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
    