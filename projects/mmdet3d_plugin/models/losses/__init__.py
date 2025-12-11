# Copyright (c) OpenMMLab. All rights reserved.
from .emd_loss import EMDLoss, emd_loss
from .smoothness_loss import SmoothnessLoss, smoothness_loss
from .color_loss import ColorLoss, color_loss

__all__ = [
    'EMDLoss', 'emd_loss',
    'SmoothnessLoss', 'smoothness_loss',
    'ColorLoss', 'color_loss',
]

