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

from ..custom_op import register

@register
def custom_ms_deform_attn_op_handler(g, *args, **kwargs):
    version_major = int(torch.__version__.split(".")[0])
    if version_major < 2:
        # signature is different between pytorch 1.x and 2.x
        # 1.x: handler(g: torch._C.Graph, n: torch._C.Node, *args, **kwargs)
        # 2.x: handler(g: torch._C.Graph, *args, **kwargs)
        # _, args = args[0], args[1:]
        pass
    node_name = kwargs["name"] # _node_get(g.original_node, "name")
    if node_name == "MultiScaleDeformableAttnFunction_fp32" or node_name == "MultiScaleDeformableAttnFunction":
        n, args = args[0], list(args[1:6])
        args[1] = g.op("Cast", args[1], to_i=6)
        args[2] = g.op("Cast", args[2], to_i=6)
        return g.op("custom_op::MultiScaleDeformableAttentionPlugin", *args)

# register_custom_op_symbolic("prim::PythonOp", custom_ms_deform_attn_op_handler, 1)

def optimize():
    pass
