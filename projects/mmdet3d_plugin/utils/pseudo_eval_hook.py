import time
import copy
import torch
from mmcv.runner import HOOKS, Hook
from mmdet3d.datasets import build_dataloader, build_dataset


@HOOKS.register_module()
class PseudoEvalHook(Hook):
    """
    Periodically run a light eval on the val set (inference only, no grads).
    Useful for monitoring refinement quality without running full validation epochs.
    """

    def __init__(self, eval_interval=20, eval_batches=1, save_ckpt=False, ckpt_interval=None):
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.save_ckpt = save_ckpt
        # If ckpt_interval is None, save checkpoint every time eval runs
        # Otherwise, save checkpoint every ckpt_interval iterations
        self.ckpt_interval = ckpt_interval if ckpt_interval is not None else eval_interval
        self.val_loader = None

    def before_run(self, runner):
        # Build val dataloader once. cfg is attached to model in train_mmdet3d.
        # Handle wrapped models (DataParallel, etc.)
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        cfg = getattr(model, 'cfg', None)
        if cfg is None:
            runner.logger.warning("[pseudo-eval] No cfg found on model, skipping val loader build")
            return
        
        # Check if val dataset config exists
        if not hasattr(cfg, 'data') or not hasattr(cfg.data, 'val') or cfg.data.val is None:
            runner.logger.warning("[pseudo-eval] No val dataset config found, skipping val loader build")
            return
        
        try:
            # Deep copy to avoid altering train cfg
            val_cfg = copy.deepcopy(cfg.data.val)
            # Ensure test_mode=True for val dataset
            val_cfg.test_mode = True
            dataset = build_dataset(val_cfg)
            dist = getattr(runner, 'distributed', False)
            self.val_loader = build_dataloader(
                dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.get("workers_per_gpu", 2),
                dist=dist,
                shuffle=False,
            )
            runner.logger.info(f"[pseudo-eval] Val loader built successfully with {len(dataset)} samples")
        except Exception as e:
            import traceback
            runner.logger.error(f"[pseudo-eval] Failed to build val loader: {e}")
            runner.logger.error(f"[pseudo-eval] Traceback: {traceback.format_exc()}")
            self.val_loader = None

    def after_train_iter(self, runner):
        # Skip iteration 0 - wait for at least one training iteration to complete
        if runner.iter == 0:
            return
        
        if not self.every_n_iters(runner, self.eval_interval):
            return
        
        # Check if val_loader is available
        if self.val_loader is None:
            return
        
        model = runner.model
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        
        # Handle wrapped models (DataParallel, etc.)
        actual_model = model.module if hasattr(model, 'module') else model
        
        runner.logger.info(f"[pseudo-eval] Running eval on {self.eval_batches} batch(es)...")
        runner.logger.info(f"{'batch':<6} | {'infer_time':<10} | {'stats_time':<10} | {'total_time':<10} | "
              f"{'refined':<8} | {'gt':<6} | {'pseudo':<8} | "
              f"{'count_diff':<10} | {'gen_ratio':<10} | {'feat_dist':<10} | {'chamfer':<10}")
        runner.logger.info("-" * 120)
        
        # Aggregate metrics across batches
        all_metrics = []
        total_infer_time = 0.0
        total_stats_time = 0.0
        
        with torch.no_grad():
            for bidx, data in enumerate(self.val_loader):
                    if bidx >= self.eval_batches:
                        break
                    start = time.time()
                    # Clear any previous metrics
                    if hasattr(actual_model, '_reconstruction_metrics'):
                        actual_model._reconstruction_metrics = None
                    # scatter to device is handled by model parallel wrappers
                    outputs = model(return_loss=False, **data)
                    infer_time = time.time() - start

                    # Extract metrics from model (stored in _reconstruction_metrics)
                    stats_start = time.time()
                    metrics = {}
                    if hasattr(actual_model, '_reconstruction_metrics') and actual_model._reconstruction_metrics is not None:
                        metrics = actual_model._reconstruction_metrics
                    stats_time = time.time() - stats_start

                    total_time = infer_time + stats_time
                    total_infer_time += infer_time
                    total_stats_time += stats_time
                    
                    # Store metrics for aggregation
                    all_metrics.append(metrics)

                    # Print per-batch metrics
                    refined = metrics.get("refined_count", "-")
                    gt = metrics.get("gt_count", "-")
                    pseudo = metrics.get("pseudo_count", "-")
                    count_diff = metrics.get("count_diff", "-")
                    gen_ratio = metrics.get("gen_ratio", "-")
                    feat_dist = metrics.get("feat_dist", "-")
                    chamfer = metrics.get("chamfer_like_dist", "-")

                    # Format values
                    if isinstance(refined, (int, float)):
                        refined = f"{int(refined)}"
                    if isinstance(gt, (int, float)):
                        gt = f"{int(gt)}"
                    if isinstance(pseudo, (int, float)):
                        pseudo = f"{int(pseudo)}"
                    if isinstance(count_diff, float):
                        count_diff = f"{count_diff:.4f}"
                    if isinstance(gen_ratio, float):
                        gen_ratio = f"{gen_ratio:.4f}"
                    if isinstance(feat_dist, float):
                        feat_dist = f"{feat_dist:.4f}"
                    if isinstance(chamfer, float):
                        chamfer = f"{chamfer:.4f}"

                    runner.logger.info(
                        f"{bidx:<6} | {infer_time:<10.4f} | {stats_time:<10.4f} | {total_time:<10.4f} | "
                        f"{refined:<8} | {gt:<6} | {pseudo:<8} | "
                        f"{count_diff:<10} | {gen_ratio:<10} | {feat_dist:<10} | {chamfer:<10}"
                    )
        
        # Print aggregated metrics if multiple batches
        if len(all_metrics) > 1:
            # Aggregate: sum for counts, average for ratios/distances
            agg_refined = sum(m.get("refined_count", 0) for m in all_metrics if isinstance(m.get("refined_count"), (int, float)))
            agg_gt = sum(m.get("gt_count", 0) for m in all_metrics if isinstance(m.get("gt_count"), (int, float)))
            agg_pseudo = sum(m.get("pseudo_count", 0) for m in all_metrics if isinstance(m.get("pseudo_count"), (int, float)))
            agg_count_diff = sum(m.get("count_diff", 0.0) for m in all_metrics if isinstance(m.get("count_diff"), (int, float))) / len(all_metrics)
            agg_gen_ratio = sum(m.get("gen_ratio", 0.0) for m in all_metrics if isinstance(m.get("gen_ratio"), (int, float))) / len(all_metrics)
            agg_feat_dist = sum(m.get("feat_dist", 0.0) for m in all_metrics if isinstance(m.get("feat_dist"), (int, float))) / len(all_metrics)
            agg_chamfer = sum(m.get("chamfer_like_dist", 0.0) for m in all_metrics if isinstance(m.get("chamfer_like_dist"), (int, float))) / len(all_metrics)
            
            runner.logger.info("-" * 120)
            runner.logger.info(f"{'AVG':<6} | {total_infer_time/len(all_metrics):<10.4f} | {total_stats_time/len(all_metrics):<10.4f} | {(total_infer_time+total_stats_time)/len(all_metrics):<10.4f} | "
                  f"{int(agg_refined):<8} | {int(agg_gt):<6} | {int(agg_pseudo):<8} | "
                  f"{agg_count_diff:<10.4f} | {agg_gen_ratio:<10.4f} | {agg_feat_dist:<10.4f} | {agg_chamfer:<10.4f}")
        
        # Save checkpoint if enabled and interval is met
        if self.save_ckpt and self.every_n_iters(runner, self.ckpt_interval):
            try:
                # Always use mmcv's save_checkpoint directly to ensure iteration-based filenames
                from mmcv.runner import save_checkpoint
                import os
                os.makedirs(runner.work_dir, exist_ok=True)
                
                # Get optimizer if available
                optimizer = None
                if hasattr(runner, 'optimizer'):
                    optimizer = runner.optimizer
                
                # Get meta information
                meta = dict(iter=runner.iter)
                if hasattr(runner, 'epoch'):
                    meta['epoch'] = runner.epoch
                if hasattr(runner, 'meta') and runner.meta:
                    meta.update(runner.meta)
                
                # Use iteration-based filename
                filename = os.path.join(runner.work_dir, f'iter_{runner.iter}.pth')
                save_checkpoint(
                    model,
                    filename=filename,
                    optimizer=optimizer,
                    meta=meta
                )
                runner.logger.info(f"[pseudo-eval] Checkpoint saved: {filename}")
            except Exception as e:
                import traceback
                runner.logger.warning(f"[pseudo-eval] Failed to save checkpoint: {e}")
                runner.logger.error(f"[pseudo-eval] Traceback: {traceback.format_exc()}")
        
        if was_training:
            model.train()

