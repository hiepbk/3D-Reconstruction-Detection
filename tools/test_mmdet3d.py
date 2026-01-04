#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Test script that mimics the evaluation process during training.

This script replicates the exact evaluation process that happens after each epoch
during training, including:
- Loading model from checkpoint
- Building validation dataset
- Running inference
- Computing evaluation metrics
"""

import argparse
import os
import time
import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint, get_dist_info, init_dist, set_random_seed
from mmcv.utils import import_modules_from_strings


# Patch mmcv Scatter.forward to handle device indices correctly
# This fixes a bug where target_gpus contains integers but _get_stream expects device objects
def _patch_scatter_forward():
    """Patch mmcv Scatter.forward to convert device indices to device objects."""
    from mmcv.parallel._functions import Scatter
    from torch.nn.parallel._functions import _get_stream
    from typing import List, Union
    from torch import Tensor
    
    original_forward = Scatter.forward
    
    @staticmethod
    def patched_forward(target_gpus: List[int], input: Union[List, Tensor]) -> tuple:
        from mmcv.parallel._functions import get_input_device, scatter, synchronize_stream
        
        input_device = get_input_device(input)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            # Convert device indices to device objects
            device_objects = [torch.device(f'cuda:{d}') if isinstance(d, int) else d for d in target_gpus]
            streams = [_get_stream(device) for device in device_objects]
        
        outputs = scatter(input, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)
        
        return tuple(outputs) if isinstance(outputs, list) else (outputs, )
    
    Scatter.forward = patched_forward


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test ResDet3D model (mimics training evaluation)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save evaluation results')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--show',
        action='store_true',
        help='show results')
    parser.add_argument(
        '--show-dir',
        help='directory where results will be saved')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple workers')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    
    # Import custom modules if specified (same as train_mmdet3d.py)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])

    # Handle plugin imports (same as train_mmdet3d.py)
    # This must be done BEFORE importing mmdet3d modules that use projects.mmdet3d_plugin
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
    
    # Now import mmdet3d modules AFTER plugin is loaded (same pattern as train_mmdet3d.py)
    from mmdet3d.apis import single_gpu_test
    from mmdet.apis import multi_gpu_test
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from mmdet3d.utils import collect_env, get_root_logger
    
    cfg.launcher = args.launcher
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = os.path.join('./work_dirs',
                                    os.path.splitext(os.path.basename(args.config))[0])
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpu_ids is None else args.gpu_ids

    # Patch mmcv Scatter.forward before building model/data (same as train_mmdet3d.py)
    _patch_scatter_forward()
    
    # Patch MMDistributedDataParallel (same as in train_mmdet3d.py)
    from tools.train_mmdet3d import _patch_mm_distributed_data_parallel
    _patch_mm_distributed_data_parallel()

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if cfg.get('seed', None) is not None:
        logger.info(f'Set random seed to {cfg.seed}, '
                    f'deterministic: {cfg.get("deterministic", False)}')
        set_random_seed(cfg.seed, deterministic=cfg.get('deterministic', False))

    # Build validation dataset (same as training evaluation)
    # build_dataset automatically builds the pipeline as a Compose object
    if 'val' in cfg.data:
        # Ensure test_mode is set (build_dataset will handle pipeline construction)
        if isinstance(cfg.data.val, dict):
            cfg.data.val.test_mode = True
        val_dataset = build_dataset(cfg.data.val)
    else:
        raise ValueError('No validation dataset found in config')

    # Build dataloader
    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
        dist=distributed,
        shuffle=False)

    # Build model
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.cfg = cfg

    # Load checkpoint
    logger.info(f'Loading checkpoint from: {args.checkpoint}')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # Check for state dict mismatches (load_checkpoint may warn, but we can get details)
    if 'state_dict' in checkpoint:
        checkpoint_keys = set(checkpoint['state_dict'].keys())
        model_keys = set(model.state_dict().keys())
        unexpected_keys = checkpoint_keys - model_keys
        missing_keys = model_keys - checkpoint_keys
        
        if unexpected_keys:
            logger.warning(f'Checkpoint contains {len(unexpected_keys)} unexpected keys (will be ignored):')
            # Show first few unexpected keys
            for key in list(unexpected_keys)[:5]:
                logger.warning(f'  - {key}')
            if len(unexpected_keys) > 5:
                logger.warning(f'  ... and {len(unexpected_keys) - 5} more')
        
        if missing_keys:
            logger.warning(f'Model contains {len(missing_keys)} keys not in checkpoint:')
            # Show first few missing keys
            for key in list(missing_keys)[:5]:
                logger.warning(f'  - {key}')
            if len(missing_keys) > 5:
                logger.warning(f'  ... and {len(missing_keys) - 5} more')
        
        if not unexpected_keys and not missing_keys:
            logger.info('Checkpoint loaded successfully with no mismatches')
    
    if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = val_dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(
            model=model,
            data_loader=val_dataloader,
            show=args.show,
            out_dir=args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(
            model, val_dataloader, args.tmpdir if args.tmpdir else None, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        # Get evaluation config (same as training)
        eval_cfg = cfg.get('evaluation', {})
        if args.eval:
            eval_cfg['metric'] = args.eval
        if args.eval_options is not None:
            eval_cfg.update(args.eval_options)

        # Remove EvalHook-specific keys
        eval_kwargs = eval_cfg.copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
            eval_kwargs.pop(key, None)

        logger.info('Starting evaluation...')
        logger.info(f'Evaluation config: {eval_kwargs}')

        # Run evaluation (same as dataset.evaluate() in training)
        results = val_dataset.evaluate(outputs, **eval_kwargs)
        
        logger.info('Evaluation results:')
        for key, value in results.items():
            logger.info(f'{key}: {value}')


if __name__ == '__main__':
    main()

