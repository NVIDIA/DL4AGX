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

import ctypes
from typing import Dict, List, Tuple

import argparse
import numpy as np
import onnx
from onnxsim import simplify
import onnx_graphsurgeon as gs
import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser(
        "Simplifies an ONNX model with custom TensorRT (TRT) plugins. Steps: \n"
        "  1. Automatically detect custom TRT ops in the ONNX model.\n"
        "  2. Ensure that the custom ops are supported as a TRT plugin in ONNX-Runtime (`trt.plugins` domain).\n"
        "  3. Use ONNX-GraphSurgeon to update all tensor types and shapes in the ONNX graph.\n"
        "  4. Apply onnxsim to simplify model with inferred shapes."
    )
    parser.add_argument("--onnx", type=str, required=True, help="Input ONNX model path.")
    parser.add_argument("--trt_plugins", type=str, default=None,
                        help=("Specifies custom TensorRT plugin library paths in .so format (compiled shared library). "
                              "For multiple paths, separate them with a semicolon, i.e.: 'lib_1.so;lib_2.so'."))
    args = parser.parse_args()
    return args


def get_custom_layers(onnx_path: str, trt_plugins: str = None) -> Tuple[List[str], Dict]:
    """Gets custom layers in ONNX file.

    Args:
        onnx_path: Path to the input ONNX model.
        trt_plugins: Paths to custom TensorRT plugins.

    Returns:
        List of custom layers.
        Dict containing tensor names and shapes.
    """
    # Initialize TensorRT plugins
    if trt_plugins is not None:
        for plugin in trt_plugins.split(";"):
            ctypes.CDLL(plugin)

    # Create builder and network
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network()

    # Parse ONNX file
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            error_str = [str(parser.get_error(error)) for error in range(parser.num_errors)]
            raise Exception(f"Failed to parse ONNX file: {''.join(error_str)}")

    # Obtain plugin layer names and all tensor shapes
    custom_layers = []
    all_tensor_shapes = {}
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        if "PLUGIN" in str(layer.type):
            custom_layers.append(layer.name)

        for i in range(layer.num_inputs):
            input_tensor = layer.get_input(i)
            if input_tensor:
                inp_shape = ["unk" if (s == -1) else s for s in input_tensor.shape]
                all_tensor_shapes[input_tensor.name] = inp_shape

        for i in range(layer.num_outputs):
            output_tensor = layer.get_output(i)
            if output_tensor:
                out_shape = ["unk" if (s == -1) else s for s in output_tensor.shape]
                all_tensor_shapes[output_tensor.name] = out_shape

    return custom_layers, all_tensor_shapes


def infer_types_in_graph(graph: gs.Graph, custom_ops: List[str]):
    """Infers tensor types in ONNX GS graph."""

    def _map_int_to_onnx_type(i: int):
        """Returns the ONNX type equivalent to the given integer."""
        int_onnx_type_to_string = {
            1: "float32",  # onnx.TensorProto.FLOAT,
            2: "uint8",  # onnx.TensorProto.UINT8,
            3: "int8",  # onnx.TensorProto.INT8,
            4: "uint16",  # onnx.TensorProto.UINT16,
            5: "int16",  # onnx.TensorProto.INT16,
            6: "int32",  # onnx.TensorProto.INT32,
            7: "int64",  # onnx.TensorProto.INT64,
            8: "string",  # onnx.TensorProto.STRING,
            9: "bool",  # onnx.TensorProto.BOOL,
            10: "float16",  # onnx.TensorProto.FLOAT16,
            11: "float64",  # onnx.TensorProto.DOUBLE,
            12: "uint32",  # onnx.TensorProto.UINT32,
            13: "uint64",  # onnx.TensorProto.UINT64,
            14: "complex64",  # onnx.TensorProto.COMPLEX64,
            15: "complex128",  # onnx.TensorProto.COMPLEX128,
            16: "bfloat16",  # onnx.TensorProto.BFLOAT16,
        }
        return int_onnx_type_to_string.get(i, None)

    def _map_onnx_type_to_string(onnx_type: str):
        """Returns the string equivalent of the given ONNX type."""
        onnx_type_to_string = {
            "tensor(float)": "float32",
            "tensor(uint8)": "uint8",
            "tensor(int8)": "int8",
            "tensor(uint16)": "uint16",
            "tensor(int16)": "int16",
            "tensor(int32)": "int32",
            "tensor(int64)": "int64",
            "tensor(string)": "string",
            "tensor(bool)": "bool",
            "tensor(float16)": "float16",
            "tensor(double)": "float64",
            "tensor(uint32)": "uint32",
            "tensor(uint64)": "uint64",
            "tensor(complex64)": "complex64",
            "tensor(complex128)": "complex128",
            "tensor(bfloat16)": "bfloat16",
        }
        return onnx_type_to_string.get(onnx_type, None)

    for node in graph.nodes:
        if node.op in custom_ops:
            # Ensure all input tensors have a type
            for inp in node.inputs:
                if isinstance(inp, gs.Variable) and inp.dtype is None:
                    inp.dtype = "float32"

            # Ensure all output tensors have a type
            for out in node.outputs:
                if out.dtype is None:
                    out.dtype = node.inputs[0].dtype or "float32"
        else:
            schema = onnx.defs.get_schema(node.op)

            # Ensure all input tensors have a type
            for inp, inp_schema in zip(node.inputs, schema.inputs):
                if inp.dtype is None:
                    inp.dtype = (
                        "float32"
                        if "tensor(float)" in list(inp_schema.types)
                        else _map_onnx_type_to_string(list(inp_schema.types)[0])
                    )

            # Ensure all output tensors have a type
            if node.op in ["Constant", "ConstantOfShape"]:
                for out in node.outputs:
                    if out.dtype is None:
                        out.dtype = node.attrs["value"].dtype or "float32"
            elif node.op == "Where":
                for out in node.outputs:
                    if out.dtype is None:
                        out.dtype = node.inputs[1].dtype
            elif node.op == "Cast":
                for out in node.outputs:
                    if out.dtype is None:
                        out.dtype = _map_int_to_onnx_type(node.attrs["to"])
            else:
                # Check if the node has more than 1 output schema. If so, then replicate it N times to match the
                #   actual number of node outputs.
                schema_outputs = (
                    schema.outputs
                    if len(schema.outputs) > 1
                    else schema.outputs * len(node.outputs)
                )
                for out, out_schema in zip(node.outputs, schema_outputs):
                    if out.dtype is None:
                        out.dtype = (
                            node.inputs[0].dtype or "float32"
                            if "tensor(float)" in list(out_schema.types)
                            else _map_onnx_type_to_string(list(out_schema.types)[0])
                        )


def infer_shapes(graph: gs.Graph, all_tensor_shapes: Dict) -> None:
    """Updates tensor shapes in ORT graph."""
    for node in graph.nodes:
        for out in node.outputs:
            if out.name in all_tensor_shapes:
                out.shape = all_tensor_shapes[out.name]
    graph.cleanup().toposort()


def load_ort_supported_model(
    onnx_path: str, trt_plugins: str = None, use_external_data_format: bool = False
) -> Tuple[onnx.onnx_pb.ModelProto, bool, List[str]]:
    """Ensures that ONNX model is supported by ORT if it contains custom ops.

    Args:
        onnx_path: Path to the input ONNX model.
        trt_plugins: Paths to custom TensorRT plugins.
        use_external_data_format: If True, separate data path will be used to store the weights of the quantized model.

    Returns:
        Loaded ONNX model supported by ORT.
        Boolean indicating whether the model has custom ops or not.
        List of custom ops in the ONNX model.
    """
    trt_plugin_domain = "trt.plugins"
    trt_plugin_version = 1

    custom_layers, all_tensor_shapes = get_custom_layers(onnx_path, trt_plugins) or []
    has_custom_op = True if custom_layers else False

    onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)

    custom_ops = []
    if has_custom_op:
        graph = gs.import_onnx(onnx_model)
        for node in graph.nodes:
            if node.name in custom_layers:
                custom_ops.append(node.op)
                node.domain = trt_plugin_domain
        custom_ops = np.unique(custom_ops)

        # Ensure that all nodes have I/O tensor types
        infer_types_in_graph(graph, custom_ops)

        # Ensure that all tensors in the graph have a shape
        infer_shapes(graph, all_tensor_shapes)

        onnx_model = gs.export_onnx(graph)
        onnx_model.opset_import.append(
            onnx.helper.make_opsetid(trt_plugin_domain, trt_plugin_version)
        )

    return onnx_model, has_custom_op, custom_ops


def main():
    args = parse_args()
    onnx_path = args.onnx

    model, has_custom_op, custom_ops = load_ort_supported_model(onnx_path, args.trt_plugins)

    if has_custom_op:
        # Save model with types and shapes in a new file
        print(f"Found {len(custom_ops)} custom ops: {custom_ops}")
        onnx_path = onnx_path.replace(".onnx", "_post.onnx")
        onnx.save(model, onnx_path)

        # Simplify ONNX model with inferred shapes
        print(f"Simplifying ONNX model with inferred shapes...")
        model_simp, check = simplify(model)
        if check:
            output_path = onnx_path.replace(".onnx", "_simp.onnx")
            onnx.save(model_simp, output_path)
            print(f"Simplified model was validated and saved in {output_path}")
        else:
            print(f"Simplified ONNX model could not be validated.")
    else:
        print("No custom ops found!")


if __name__ == '__main__':
    main()
