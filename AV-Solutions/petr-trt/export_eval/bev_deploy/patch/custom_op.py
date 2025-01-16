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

HANDLERS = []

def register(callback):
    global HANDLERS
    HANDLERS.append(callback)
    return callback

def custom_handler(g, *args, **kwargs):
    global HANDLERS

    version_major = int(torch.__version__.split(".")[0])
    if version_major < 2:
        # signature is different between pytorch 1.x and 2.x
        # 1.x: handler(g: torch._C.Graph, n: torch._C.Node, *args, **kwargs)
        # 2.x: handler(g: torch._C.Graph, *args, **kwargs)
        # _, args = args[0], args[1:]
        pass
    node_name = kwargs["name"] # _node_get(g.original_node, "name")

    for callback in HANDLERS:
        ret = callback(g, *args, **kwargs)
        if ret is not None:
            return ret

    return sym_help._unimplemented(node_name, "unimplemented")

register_custom_op_symbolic("prim::PythonOp", custom_handler, 1)
