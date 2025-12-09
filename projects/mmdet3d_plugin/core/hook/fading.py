import enum
import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class Fading(Hook):
    def __init__(self, fade_epoch = 100000):
        self.fade_epoch = fade_epoch

    def before_train_epoch(self, runner):
        if runner.epoch >= self.fade_epoch:
            # Get the inner dataset (after CBGSDataset wrapper)
            inner_dataset = runner.data_loader.dataset.dataset
            
            # Check if it's a ConcatDataset (has 'datasets' attribute)
            if hasattr(inner_dataset, 'datasets'):
                # Handle ConcatDataset: iterate through all datasets and remove ObjectSample from each
                for dataset in inner_dataset.datasets:
                    if hasattr(dataset, 'pipeline') and hasattr(dataset.pipeline, 'transforms'):
                        for i, transform in enumerate(dataset.pipeline.transforms):
                            if type(transform).__name__ == 'ObjectSample':
                                dataset.pipeline.transforms.pop(i)
                                break
            else:
                # Handle single dataset (original behavior)
                if hasattr(inner_dataset, 'pipeline') and hasattr(inner_dataset.pipeline, 'transforms'):
                    for i, transform in enumerate(inner_dataset.pipeline.transforms):
                        if type(transform).__name__ == 'ObjectSample':
                            inner_dataset.pipeline.transforms.pop(i)
                            break