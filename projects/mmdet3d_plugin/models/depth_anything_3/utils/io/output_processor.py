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

"""
Output processor for Depth Anything 3.

This module handles model output processing, including tensor-to-numpy conversion,
batch dimension removal, and Prediction object creation.
"""

from __future__ import annotations

import numpy as np
import torch
from addict import Dict as AddictDict

from depth_anything_3.specs import Prediction


class OutputProcessor:
    """
    Output processor for converting model outputs to Prediction objects.

    Handles tensor-to-numpy conversion, batch dimension removal,
    and creates structured Prediction objects with proper data types.
    """

    def __init__(self) -> None:
        """Initialize the output processor."""

    def __call__(self, model_output: dict[str, torch.Tensor], return_torch: bool = False) -> Prediction:
        """
        Convert model output to Prediction object.

        Args:
            model_output: Model output dictionary containing depth, conf, extrinsics, intrinsics
                         Expected shapes: depth (B, N, 1, H, W), conf (B, N, 1, H, W),
                         extrinsics (B, N, 4, 4), intrinsics (B, N, 3, 3)

        Returns:
            Prediction: Object containing depth estimation results with shapes:
                       depth (N, H, W), conf (N, H, W), extrinsics (N, 4, 4), intrinsics (N, 3, 3)
        """
        device = self._infer_device(model_output)

        # Extract data from batch dimension (B=1, N=number of images)
        depth = self._extract_depth(model_output, return_torch, device)
        conf = self._extract_conf(model_output, return_torch, device)
        extrinsics = self._extract_extrinsics(model_output, return_torch, device)
        intrinsics = self._extract_intrinsics(model_output, return_torch, device)
        sky = self._extract_sky(model_output, return_torch, device)
        aux = self._extract_aux(model_output, return_torch, device)
        gaussians = model_output.get("gaussians", None)
        scale_factor = model_output.get("scale_factor", None)

        return Prediction(
            depth=depth,
            sky=sky,
            conf=conf,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            is_metric=getattr(model_output, "is_metric", 0),
            gaussians=gaussians,
            aux=aux,
            scale_factor=scale_factor,
        )

    def _extract_depth(self, model_output: dict[str, torch.Tensor], return_torch: bool, device: torch.device):
        """
        Extract depth tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Depth array/tensor with shape (N, H, W)
        """
        depth_t = model_output["depth"].squeeze(0).squeeze(-1)  # (N, H, W)
        if return_torch:
            return depth_t.to(device)
        return depth_t.cpu().numpy()

    def _extract_conf(self, model_output: dict[str, torch.Tensor], return_torch: bool, device: torch.device):
        """
        Extract confidence tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Confidence array/tensor with shape (N, H, W) or None
        """
        conf = model_output.get("depth_conf", None)
        if conf is not None:
            conf = conf.squeeze(0)  # (N, H, W)
            if return_torch:
                conf = conf.to(device)
            else:
                conf = conf.cpu().numpy()
        return conf

    def _extract_extrinsics(self, model_output: dict[str, torch.Tensor], return_torch: bool, device: torch.device):
        """
        Extract extrinsics tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Extrinsics array/tensor with shape (N, 4, 4) or None
        """
        extrinsics = model_output.get("extrinsics", None)
        if extrinsics is not None:
            extrinsics = extrinsics.squeeze(0)  # (N, 4, 4)
            if return_torch:
                extrinsics = extrinsics.to(device)
            else:
                extrinsics = extrinsics.cpu().numpy()
        return extrinsics

    def _extract_intrinsics(self, model_output: dict[str, torch.Tensor], return_torch: bool, device: torch.device):
        """
        Extract intrinsics tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Intrinsics array/tensor with shape (N, 3, 3) or None
        """
        intrinsics = model_output.get("intrinsics", None)
        if intrinsics is not None:
            intrinsics = intrinsics.squeeze(0)  # (N, 3, 3)
            if return_torch:
                intrinsics = intrinsics.to(device)
            else:
                intrinsics = intrinsics.cpu().numpy()
        return intrinsics

    def _extract_sky(self, model_output: dict[str, torch.Tensor], return_torch: bool, device: torch.device):
        """
        Extract sky tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Sky mask array/tensor with shape (N, H, W) or None
        """
        sky = model_output.get("sky", None)
        if sky is not None:
            sky = sky.squeeze(0)  # (N, H, W)
            if return_torch:
                sky = (sky >= 0.5).to(device)
            else:
                sky = (sky.cpu().numpy() >= 0.5)
        return sky

    def _extract_aux(self, model_output: dict[str, torch.Tensor], return_torch: bool, device: torch.device) -> AddictDict:
        """
        Extract auxiliary data from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Dictionary containing auxiliary data
        """
        aux = model_output.get("aux", None)
        ret = AddictDict()
        if aux is not None:
            for k in aux.keys():
                if isinstance(aux[k], torch.Tensor):
                    tensor_k = aux[k].squeeze(0)
                    ret[k] = tensor_k.to(device) if return_torch else tensor_k.cpu().numpy()
                else:
                    ret[k] = aux[k]
        return ret

    def _infer_device(self, model_output: dict[str, torch.Tensor]) -> torch.device:
        """Infer device from any tensor in model_output; default to CPU."""
        for v in model_output.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")


# Backward compatibility alias
OutputAdapter = OutputProcessor
