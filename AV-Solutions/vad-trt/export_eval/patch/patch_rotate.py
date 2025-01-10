# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from https://github.com/DerryHub/BEVFormer_tensorrt/blob/main/det2trt/models/functions/rotate.py
# Licensed under https://github.com/DerryHub/BEVFormer_tensorrt/blob/main/LICENCE

import numpy as np
import torch
from torch.autograd import Function
from torchvision.transforms.functional import rotate as rotate_func, InterpolationMode

class _Rotate(Function):
    @staticmethod
    def symbolic(g, img, angle, center, interpolation):
        return g.op("custom_op::RotatePlugin", img, angle, center, interpolation_i=interpolation)

    @staticmethod
    def forward(ctx, img, angle, center, interpolation):
        assert img.ndim == 3
        oh, ow = img.shape[-2:]
        if isinstance(center[0], int):
            center = torch.FloatTensor(center).to(img.device)
        cx = center[0] - center[0].new_tensor(ow * 0.5)
        cy = center[1] - center[1].new_tensor(oh * 0.5)
        if isinstance(angle, float):
            angle_ = torch.FloatTensor(angle).to(img.device)
        else:
            angle_ = angle.to(img.device).float()

        angle_ = -angle_ * np.pi / 180
        theta = torch.stack(
            [
                torch.cos(angle_),
                torch.sin(angle_),
                -cx * torch.cos(angle_) - cy * torch.sin(angle_) + cx,
                -torch.sin(angle_),
                torch.cos(angle_),
                cx * torch.sin(angle_) - cy * torch.cos(angle_) + cy,
            ]
        ).view(1, 2, 3)

        # grid will be generated on the same device as theta and img
        d = 0.5
        base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
        x_grid = torch.linspace(
            -ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device
        )
        base_grid[..., 0] = x_grid.expand(1, oh, ow)
        y_grid = torch.linspace(
            -oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device
        ).unsqueeze_(-1)
        base_grid[..., 1] = y_grid.expand(1, oh, ow)
        base_grid[..., 2].fill_(1)

        rescaled_theta = 2 * theta.transpose(1, 2)
        rescaled_theta[..., 0] /= ow
        rescaled_theta[..., 1] /= oh

        output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
        grid = output_grid.view(1, oh, ow, 2)

        req_dtypes = [
            grid.dtype,
        ]
        # make image NCHW
        img = img.unsqueeze(dim=0)

        out_dtype = img.dtype
        need_cast = False
        if out_dtype not in req_dtypes:
            need_cast = True
            req_dtype = req_dtypes[0]
            img = img.to(req_dtype)

        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])
        img = torch.grid_sampler(img, grid, interpolation, 0, False)

        img = img.squeeze(dim=0)

        if need_cast:
            if out_dtype in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                # it is better to round before cast
                img = torch.round(img)
            img = img.to(out_dtype)

        return img

_rotate = _Rotate.apply

_MODE = {"bilinear": 0, "nearest": 1}

def rotate(img, angle, center, interpolation="nearest"):
    if torch.onnx.is_in_onnx_export():
        # angle = torch.cuda.FloatTensor([angle]).to(img.device)
        center = torch.cuda.FloatTensor(center).to(img.device)
        return _rotate(img, angle, center, _MODE[interpolation])
    # NOTE: mode is hard-coded
    return rotate_func(img, angle.item(), center=center, interpolation=InterpolationMode.NEAREST)

from bev_deploy.infer_shape.infer import G_HANDLERS

def infer_rotate(node):
    input_shape = node.inputs[0].shape
    node.outputs[0].shape = input_shape
    node.outputs[0].dtype = node.inputs[0].dtype
    return True

G_HANDLERS["RotatePlugin"] = infer_rotate

def setup_rotate(ch):
    # from torchvision.transforms.functional import rotate
    from bev_deploy.capture.hook import _hook_function
    import sys
    _mod = sys.modules["projects.mmdet3d_plugin.VAD.VAD_transformer"]
    _rotate = _mod.rotate
    setattr(_mod, "rotate", _hook_function(ch, _rotate, "rotate"))
    ch._capture_calls["rotate.rotate"] = []
