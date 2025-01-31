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

from typing import Any
import onnx
import onnxsim
import onnx_graphsurgeon as gs

op_set = set()

def valid_shape(shape):
    if isinstance(shape, list):
        if len(shape) == 0:
            return False
        for item in shape:
            if isinstance(item, str):
                return False
    if shape is None:
        return False

    return True

G_HANDLERS = {}

class BaseHandler(object):
    def __init__(self) -> None:
        pass

    def __call__(self, node) -> Any:
        pass

def _infer_shape_msda(node):
    # https://github.com/open-mmlab/mmcv/blob/0d3cdc6ee016b6cca6a1220fcbc8f2a6ee3426cc/mmcv/ops/csrc/pytorch/cuda/ms_deform_attn_cuda.cu#L232C1-L248C75
    # value, spatial_shapes, level_start_index, sampling_loc, attn_weight
    B, spatial_size, num_heads, channels = node.inputs[0].shape
    num_levels = node.inputs[1].shape[0]
    num_query = node.inputs[3].shape[1]
    num_point = node.inputs[3].shape[4]
    node.outputs[0].shape = [B, num_query, num_heads * channels]
    node.outputs[0].dtype = node.inputs[0].dtype
    return True

def _inner_infer_shape(node):
    if node.op in ("MultiScaleDeformableAttentionPlugin", ):
        return _infer_shape_msda(node)
    elif node.op in G_HANDLERS:
        return G_HANDLERS[node.op](node)
    print(node.op)
    return False

def infer_shape(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    sim_inferred, _ = onnxsim.simplify(inferred)
    g = gs.import_onnx(sim_inferred)
    g.toposort().cleanup()
    # exit after 300 anyway
    might_missing = set()
    for i in range(300):
        print(i)
        modified = False
        for n in g.nodes:
            op_set.add(n.op)
            in_valid = True
            for v in n.inputs:
                in_valid = in_valid and valid_shape(v.shape)

            out_valid = True
            for v in n.outputs:
                out_valid = out_valid and valid_shape(v.shape)
            
            if (in_valid and not out_valid) or n.op in ("Shape", ):
                if _inner_infer_shape(n):
                    modified = True
                    break
                might_missing.add(n.op)
            else:
                # print(n.op, n.name)
                # print(", ".join([f"{i.name}, {i.shape}" for i in n.inputs]))
                # print(", ".join([f"{i.name}, {i.shape}" for i in n.outputs]))
                pass
        model_ = gs.export_onnx(g)
        if modified:
            inferred = onnx.shape_inference.infer_shapes(model_)
            sim_inferred, _ = onnxsim.simplify(inferred)
            g = gs.import_onnx(sim_inferred)
        else:
            break
    print(might_missing)
    return sim_inferred
    