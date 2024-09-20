#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

import argparse
import os
import numpy as np
from collections import OrderedDict
import onnx
from onnxsim import simplify
import onnxruntime as ort
import onnx_graphsurgeon as gs


def parse_args():
    parser = argparse.ArgumentParser(
        "Simplifies an ONNX model with custom TRT plugin via ORT and onnxsim.\n"
        "  1. Ensures that the custom op is supported as a TRT plugin in ORT (`trt.plugins` domain).\n"
        "  2. Ensures that all nodes have I/O tensor types.\n"
        "  3. Infer tensor shapes with ORT and update the graph accordingly with onnx-graphsurgeon.\n"
        "  4. Apply onnxsim to simplify model with inferred shapes."
    )
    parser.add_argument("--onnx", type=str, required=True, help="Input ONNX model path.")
    parser.add_argument("--plugins", type=str, nargs="+", default=None,
                        help="A space-separated list with paths to .so plugins.")
    parser.add_argument("--custom_ops", type=str, nargs="+", default=None,
                        help="A space-separated list with custom ops to ensure ORT support.")
    parser.add_argument(
        "--keep_intermediate_files",
        action="store_true",
        help="Indicates whether to keep or delete intermediate ONNX files."
    )
    args = parser.parse_args()
    return args

intermediate_generated_files = []


def ensure_ort_support(onnx_path, custom_ops_default=None):
    trt_plugin_domain = "trt.plugins"
    trt_plugin_version = 1

    graph = gs.import_onnx(onnx.load(onnx_path))
    has_custom_op = False
    custom_ops = custom_ops_default or []
    for node in graph.nodes:
        # Note: nodes with module="ai.onnx*" have domain=None. Sometimes, custom ops will have that domain, so
        # the user would need to force those layers to be in the 'trt.plugins' domain by using the --custom_ops flag.
        if node.op in custom_ops or node.domain:
            has_custom_op = True
            custom_ops.append(node.op)
            node.domain = trt_plugin_domain
    custom_ops = np.unique(custom_ops)

    if has_custom_op:
        model = gs.export_onnx(graph)
        model.opset_import.append(onnx.helper.make_opsetid(trt_plugin_domain, trt_plugin_version))
        onnx_path = onnx_path.replace(".onnx", "_ort_support.onnx")
        intermediate_generated_files.append(onnx_path)
        onnx.save(model, onnx_path)
        print(f"Custom ops detected: {custom_ops}!")
    else:
        print(
            "Custom op not found in model! Model may already contain custom op with 'trt.plugins' domain."
        )
    return onnx_path


def ensure_all_tensors_type(onnx_path):
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        # Ensure all input tensors have a type
        for inp in node.inputs:
            if isinstance(inp, gs.Variable) and inp.dtype is None:
                inp.dtype = "float32"

        # Ensure all output tensors have a type
        if node.op == "Identity":
            node_inps = [inp for inp in node.inputs if isinstance(inp, gs.Constant)]
            for out in node.outputs:
                if out.dtype is None:
                    out.dtype = node_inps[0].dtype if node_inps else "float32"
        elif node.op in ["Constant", "ConstantOfShape"]:
            for out in node.outputs:
                if out.dtype is None:
                    out.dtype = node.attrs['value'].dtype or "float32"
        elif node.op == "Shape":
            for out in node.outputs:
                if out.dtype is None:
                    out.dtype = "int64"
        elif node.op in ["Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal", "Not"]:
            for out in node.outputs:
                if out.dtype is None:
                    out.dtype = "bool"
        elif node.op == "Where":
            for out in node.outputs:
                if out.dtype is None:
                    out.dtype = node.inputs[1].dtype
        else:
            for out in node.outputs:
                if out.dtype is None:
                    out.dtype = node.inputs[0].dtype or "float32"

    model = gs.export_onnx(graph)
    onnx_path = onnx_path.replace(".onnx", "_dtype_fix.onnx")
    intermediate_generated_files.append(onnx_path)
    onnx.save(model, onnx_path)
    return onnx_path


def get_ort_tensor_shapes(onnx_path, plugin_paths=[]):
    """Returns ORT tensor shapes for all tensors in the graph."""
    model = onnx.load(onnx_path)

    # Load dummy inputs
    def _load_dummy_data(input_info, data_size=5):
        data = []
        for _ in range(data_size):
            data_sample = {}
            for inp in input_info:
                inp_shape = inp.shape
                # If ONNX model has dynamic batch size, fix it to 1.
                if isinstance(inp_shape[0], str) and not inp_shape[0].isdigit():
                    inp_shape[0] = 1
                data_sample[inp.name] = np.random.rand(*inp_shape).astype(np.float32)
            data.append(data_sample)
        return data

    graph = gs.import_onnx(model)
    input_info = [inp for inp in graph.inputs]
    inp_dict = _load_dummy_data(input_info)[0]

    # Get output names
    onnx_outputs_names = [out.name for out in model.graph.output]

    # Expose all ONNX tensors as outputs
    for node in model.graph.node:
        for output in node.output:
            if output not in onnx_outputs_names:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    onnx_path = onnx_path.replace(".onnx", "_infer_shapes_tmp.onnx")
    intermediate_generated_files.append(onnx_path)
    onnx.save(model, onnx_path)

    # ======== Initialize and run ORT session ========
    sess_options = ort.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.log_severity_level = 1
    EP = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if plugin_paths:
        trt_ep_options = {"trt_extra_plugin_lib_paths": ";".join(plugin_paths)}
        EP.insert(0, ("TensorrtExecutionProvider", trt_ep_options))

    session = ort.InferenceSession(onnx_path, sess_options, providers=EP)
    onnx_outputs_names = [x.name for x in session.get_outputs()]
    onnx_out = session.run(onnx_outputs_names, inp_dict)
    ort_outs = OrderedDict(zip(onnx_outputs_names, onnx_out))

    return ort_outs


def main():
    args = parse_args()
    onnx_path = args.onnx

    # ======== Ensure that custom ops are supported by ORT as TRT plugins ========
    onnx_path = ensure_ort_support(onnx_path, args.custom_ops)

    # ======== Ensure that all nodes have I/O tensor types ========
    onnx_path = ensure_all_tensors_type(onnx_path)

    # ======== Infer shapes in ONNX model with custom TRT op =========
    # Obtain shapes with ORT
    print("Getting tensor shapes!")
    ort_tensors = get_ort_tensor_shapes(onnx_path, args.plugins)

    # Update shapes in ONNX model
    print("Updating shapes in ONNX model.")
    graph = gs.import_onnx(onnx.load(onnx_path))
    for node in graph.nodes:
        for out in node.outputs:
            if out.name in ort_tensors:
                out.shape = ort_tensors[out.name].shape
                out.dtype = ort_tensors[out.name].dtype
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx_path = args.onnx.replace(".onnx", "_post.onnx")
    intermediate_generated_files.append(onnx_path)
    onnx.save(model, onnx_path)

    # ======== Simplify ONNX model with inferred shapes =========
    output_path = onnx_path.replace(".onnx", "_simp.onnx")
    print(f"Simplifying ONNX model with inferred shapes! Saving in {output_path}.")
    model_simp, check = simplify(model)
    onnx.save(model_simp, output_path)

    # ======== Check if intermediate files should be deleted ========
    if not args.keep_intermediate_files:
        for file in intermediate_generated_files:
            os.remove(file)


if __name__ == '__main__':
    main()
