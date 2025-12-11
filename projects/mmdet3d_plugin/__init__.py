# Marker file for mmdet3d plugin package
# Explicitly import classes to trigger registration decorators

# Import detectors to register ResDet3D
from .models.detectors import ResDet3D

# Import backbones to register ReconstructionBackbone
from .models.backbone import ReconstructionBackbone

# Import pipelines to register DepthAnything3Filter, VoxelDownsample, BallQueryDownsample, FPSDownsample
from .datasets.pipelines import (
    DepthAnything3Filter,
    VoxelDownsample,
    BallQueryDownsample,
    FPSDownsample,
)

# Import losses to register EMDLoss, SmoothnessLoss, ColorLoss
from .models.losses import EMDLoss, SmoothnessLoss, ColorLoss

# Import hooks to register ComponentMemoryLoggerHook
from .utils.memory_logger_hook import ComponentMemoryLoggerHook

__all__ = [
    'ResDet3D',
    'ReconstructionBackbone',
    'DepthAnything3Filter',
    'VoxelDownsample',
    'BallQueryDownsample',
    'FPSDownsample',
    'EMDLoss',
    'SmoothnessLoss',
    'ColorLoss',
    'ComponentMemoryLoggerHook',
]

