# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import onnx
import numpy as np
import onnx_graphsurgeon as gs
import onnxsim as sim

model = gs.import_onnx(onnx.load(sys.argv[1]))

def parse_lvl_block(name):
    parts = name.split("/")
    key = "/".join(parts[0:3])
    return key

def find_constant(node):
    for t in node.inputs:
        if isinstance(t, gs.ir.tensor.Constant):
            return t
    raise RuntimeError("node contains no constant tensor")

blocks = {}

# gather nodes according to block name
for node in model.nodes:
    if "dcn" in node.name:
        key = parse_lvl_block(node.name)
        if key not in blocks:
            blocks[key] = []
        blocks[key].append(node)

# manually update every block
for k, v in blocks.items():
    block_input = v[0].inputs[0]

    # note: currently the order is hard-coded
    off_mm = v[3]
    off_add = v[4]
    off_reshape = v[5]

    vp_mm = v[0]
    vp_mm_w = find_constant(vp_mm)
    vp_mm_w._values = np.concatenate([vp_mm_w.values, find_constant(off_mm).values], axis=1)
    vp_mm.outputs[0].shape[2] = vp_mm_w._values.shape[1]

    vp_add = v[1]
    vp_add_w = find_constant(vp_add)
    vp_add_w._values = np.concatenate([vp_add_w.values, find_constant(off_add).values])
    vp_add.outputs[0].shape[2] = vp_mm_w._values.shape[1]

    vp_reshape = v[2] # 1, HW, -1, so no need to change
    vp_reshape.outputs[0].shape[3] = vp_mm_w._values.shape[1]

    plugin = v[6]
    plugin.op = "DCNv4FuseOffset_Plugin" # use fused plugin
    plugin.inputs = [plugin.inputs[0]]   # offset_mask is merged to value_proj, so only 1 input remains
                                         # params for DCNv4 remains the same

    plugin_reshape = v[7]
    plugin_output_proj = v[8]

out_pth = sys.argv[1][:-5] + "_fused.onnx"
model.cleanup()
model.toposort()
model, _ = sim.simplify(gs.export_onnx(model))
onnx.save(model, out_pth)
