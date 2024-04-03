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

import onnx_graphsurgeon as gs
import onnx
import numpy as np

# The following names are the 4 feature maps from the encoder
# and shared by segmentation and depth estimation head
# You can find them with onnx visualization tools like netron.app
tensor_name1 = "input.72"
tensor_shape1 = (1,32,256,256)

tensor_name2 = "input.148"
tensor_shape2 = (1,64,128,128)

tensor_name3 = "input.224"
tensor_shape3 = (1,160,64,64)

tensor_name4 = "input.292"
tensor_shape4 = (1,256,32,32)

graph1 = gs.import_onnx(onnx.load("onnx_files/mtmi_slim.onnx"))
tensors = graph1.tensors()

graph1.outputs = []
graph1.outputs.append(tensors[tensor_name1].to_variable(dtype=np.float32, shape=tensor_shape1))
graph1.outputs.append(tensors[tensor_name2].to_variable(dtype=np.float32, shape=tensor_shape2))
graph1.outputs.append(tensors[tensor_name3].to_variable(dtype=np.float32, shape=tensor_shape3))
graph1.outputs.append(tensors[tensor_name4].to_variable(dtype=np.float32, shape=tensor_shape4))
graph1.cleanup()
onnx.save(gs.export_onnx(graph1.cleanup().toposort()), "onnx_files/mtmi_encoder.onnx")

tensor_name_seg = "onnx::ArgMax_1963"
tensor_shape_seg = (1,19,1024,1024)

graph2 = gs.import_onnx(onnx.load("onnx_files/mtmi_slim.onnx"))
tensors = graph2.tensors()
graph2.inputs = []
graph2.inputs.append(tensors[tensor_name1].to_variable(dtype=np.float32, shape=tensor_shape1))
graph2.inputs.append(tensors[tensor_name2].to_variable(dtype=np.float32, shape=tensor_shape2))
graph2.inputs.append(tensors[tensor_name3].to_variable(dtype=np.float32, shape=tensor_shape3))
graph2.inputs.append(tensors[tensor_name4].to_variable(dtype=np.float32, shape=tensor_shape4))
graph2.outputs = [tensors[tensor_name_seg].to_variable(dtype=np.float32, shape=tensor_shape_seg)]
graph2.cleanup()
onnx.save(gs.export_onnx(graph2.cleanup().toposort()), "onnx_files/mtmi_seg_head.onnx")

tensor_name_depth = "onnx::Mul_1912"
tensor_shape_depth = (1,1,1024,1024)

graph3 = gs.import_onnx(onnx.load("onnx_files/mtmi_slim.onnx"))
tensors = graph3.tensors()
graph3.inputs = []
graph3.inputs.append(tensors[tensor_name1].to_variable(dtype=np.float32, shape=tensor_shape1))
graph3.inputs.append(tensors[tensor_name2].to_variable(dtype=np.float32, shape=tensor_shape2))
graph3.inputs.append(tensors[tensor_name3].to_variable(dtype=np.float32, shape=tensor_shape3))
graph3.inputs.append(tensors[tensor_name4].to_variable(dtype=np.float32, shape=tensor_shape4))
graph3.outputs = [tensors[tensor_name_depth].to_variable(dtype=np.float32, shape=tensor_shape_depth)]
graph3.cleanup()
onnx.save(gs.export_onnx(graph3.cleanup().toposort()), "onnx_files/mtmi_depth_head.onnx")
