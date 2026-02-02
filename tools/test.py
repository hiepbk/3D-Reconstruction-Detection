import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor


def _patch_scatter_forward():
    """Patch mmcv Scatter.forward so target_gpus (int device ids) are converted to device objects.
    Fixes: AttributeError: 'int' object has no attribute 'type' in _get_stream(device)."""
    try:
        from mmcv.parallel import scatter_gather
        from mmcv.parallel._functions import get_input_device, scatter, synchronize_stream
        from torch.nn.parallel._functions import _get_stream
        from typing import List, Union
        from torch import Tensor

        Scatter = scatter_gather.Scatter  # Patch the same Scatter that scatter_kwargs uses

        @staticmethod
        def patched_forward(target_gpus: List[int], input: Union[List, Tensor]) -> tuple:
            try:
                input_device = get_input_device(input)
            except Exception:
                input_device = -1
            streams = None
            if input_device == -1 and target_gpus != [-1]:
                # Convert int device ids to device objects (PyTorch _get_stream expects device)
                device_objects = [
                    torch.device(f'cuda:{d}') if isinstance(d, int) else d
                    for d in target_gpus
                ]
                streams = [_get_stream(device) for device in device_objects]

            outputs = scatter(input, target_gpus, streams)
            if streams is not None:
                synchronize_stream(outputs, target_gpus, streams)

            return tuple(outputs) if isinstance(outputs, list) else (outputs,)

        Scatter.forward = patched_forward
        # Also patch the class in _functions so all references use the patched forward
        import mmcv.parallel._functions as _funcs
        _funcs.Scatter.forward = patched_forward
    except Exception as e:
        warnings.warn(f"Could not patch Scatter.forward: {e}. Multi-GPU test may fail.")


def _patch_mm_distributed_data_parallel():
    """Patch MMDistributedDataParallel to handle missing _use_replicated_tensor_module attribute.
    Fixes version compatibility between mmcv and PyTorch DDP."""
    try:
        from mmcv.parallel import MMDistributedDataParallel

        original_run_ddp_forward = MMDistributedDataParallel._run_ddp_forward

        def patched_run_ddp_forward(self, *inputs, **kwargs):
            use_replicated = getattr(self, '_use_replicated_tensor_module', False)
            if use_replicated:
                module_to_run = getattr(self, '_replicated_tensor_module', self.module)
            else:
                module_to_run = self.module

            if self.device_ids:
                inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
                return module_to_run(*inputs[0], **kwargs[0])
            else:
                return module_to_run(*inputs, **kwargs)

        MMDistributedDataParallel._run_ddp_forward = patched_run_ddp_forward
    except Exception as e:
        warnings.warn(f"Could not patch MMDistributedDataParallel: {e}. Multi-GPU test may fail.")


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['bbox'],
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--jsonfile_prefix', type=str, default=None, help='load json prediction file from previous test')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--UI_result', action='store_true', help='generate UI_result')
    args = parser.parse_args()
    
    # Handle local_rank from environment variable (recommended for newer PyTorch)
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.format_only:
        print('Only for submission ...')
        args.eval = None 

    if 'waymo' in args.config.lower():
        args.eval = ['waymo']

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # Patches for multi-GPU test (mmcv/PyTorch version compatibility)
    _patch_scatter_forward()  # int device ids -> device objects for Scatter
    _patch_mm_distributed_data_parallel()

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # Use 0 workers for test to avoid "cannot pickle 'dict_keys'" when DataLoader spawns workers
    workers_per_gpu_test = 0
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu_test,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint != 'none':
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if args.checkpoint != 'none':
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model=model, 
                                  data_loader=data_loader, 
                                  show=args.show, 
                                  out_dir=args.show_dir, 
                                  UI_result=args.UI_result)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                args.gpu_collect)


    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            kwargs['jsonfile_prefix'] = './work_dirs/submissions/'
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            print('args.eval', args.eval)
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'show', 'out_dir', 'max_vis_samples'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, jsonfile_prefix=args.jsonfile_prefix, **kwargs))
            print('eval_kwargs', eval_kwargs)
            print(dataset.evaluate(outputs, **eval_kwargs))
        # Show dir: from --show-dir or from config (evaluation.show + out_dir)
        eval_cfg = cfg.get('evaluation', {})
        show_dir = args.show_dir or (eval_cfg.get('show') and eval_cfg.get('out_dir'))
        if show_dir:
            score_3d_thr = eval_cfg.get('score_3d_threshold', 0.5)
            max_vis = eval_cfg.get('max_vis_samples', 100)
            if getattr(dataset, 'show', None) is not None:
                dataset.show(
                    outputs,
                    show_dir,
                    show=False,
                    snapshot=True,
                    score_3d_threshold=score_3d_thr,
                    max_vis_samples=max_vis,
                )
                print(f'\nSaved up to {max_vis} visualization images to {show_dir}')
            else:
                print(f'Warning: dataset has no show() method, skipping visualization to {show_dir}')


if __name__ == '__main__':
    main()
