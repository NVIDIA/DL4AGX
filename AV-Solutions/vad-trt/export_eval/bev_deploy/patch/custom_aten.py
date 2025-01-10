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

import torch
from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help

from bev_deploy.infer_shape import infer

def custom_inverse_handler(g, *args, **kwargs):
    version_major = int(torch.__version__.split(".")[0])
    if version_major < 2:
        # signature is different between pytorch 1.x and 2.x
        # 1.x: handler(g: torch._C.Graph, n: torch._C.Node, *args, **kwargs)
        # 2.x: handler(g: torch._C.Graph, *args, **kwargs)
        # _, args = args[0], args[1:]
        pass
    # shape = args[0]
    return g.op("Reshape", g.op("custom_op::InversePlugin", args[0]), shape_i=[])

register_custom_op_symbolic("::inverse", custom_inverse_handler, 1)

def custom_grid_sample_handler(g, *args, **kwargs):
    version_major = int(torch.__version__.split(".")[0])
    if version_major < 2:
        # signature is different between pytorch 1.x and 2.x
        # 1.x: handler(g: torch._C.Graph, n: torch._C.Node, *args, **kwargs)
        # 2.x: handler(g: torch._C.Graph, *args, **kwargs)
        # _, args = args[0], args[1:]
        pass
    # inject extra ops to align the behavior
    # TODO: carefully handle the flags
    grid = g.op("Transpose", args[1], perm_i=[0, 4, 1, 2, 3]) # NDHW3 -> N3DHW
    B = g.op("Constant", value_t=torch.tensor(10.0).to(torch.float32))
    grid = g.op("Mul", grid, B)
    return g.op("custom_op::GridSamplePlugin", 
                args[0], grid, 
                mode_i=0, padding_mode_i=0, align_corners_i=1)

register_custom_op_symbolic("::grid_sampler", custom_grid_sample_handler, 1)

def infer_grid_sample(node):
    feat = node.inputs[0]
    grid = node.inputs[1]     
    out_shape = [feat.shape[0], feat.shape[1]] +  grid.shape[2:]
    node.outputs[0].shape = out_shape

infer.G_HANDLERS["GridSamplePlugin"] = infer_grid_sample
