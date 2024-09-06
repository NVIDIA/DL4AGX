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
        "  2. If the precisions of the plugin inputs are given, ensure them by adding Cast nodes.\n"
        "  3. Infer tensor shapes with ORT and update the graph accordingly with onnx-graphsurgeon.\n"
        "  4. Apply onnxsim to simplify model with inferred shapes."
    )
    parser.add_argument("--onnx", type=str, required=True, help="Input ONNX model path.")
    parser.add_argument("--plugins", type=str, nargs="+", default=None,
                        help="A space-separated list with paths to .so plugins.")
    parser.add_argument("--custom_ops", type=str, nargs="+", default=None,
                        help="A space-separated list with custom ops to ensure ORT support.")
    parser.add_argument(
        "--plugins_precision", type=str, nargs="+", default=None,
        help="A space-separated list indicating the precision for each custom op. If the precision is something other"
             " than fp32, a Cast node will be added.\n"
             "Each item should have the format <op_type>:<precision> (assumes all inputs and outputs and that precision)"
             " or <op_type>:[<inp1_precision>,<inp2_precision>]:[<out1_precision>] (specifies all inputs and outputs"
             " individually), where precision can be fp32 (default), fp16, or int32.\n"
             "   Example 1: op_type_1:fp16 op_type_2:fp32.\n"
             "   Example 2: op_type_1:[fp16,int32,fp16]:[fp16] op_type_2:fp32.")
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
    return onnx_path, custom_ops


def add_fp16_fp32_cast(onnx_path, custom_ops_to_cast):
    """Adds Cast nodes to the inputs and outputs of all required ops."""
    name_dict = {}

    def _get_unique_name(old_name):
        if old_name not in name_dict:
            name_dict[old_name] = 0
            return old_name
        name_dict[old_name] = name_dict[old_name] + 1
        return old_name + "_" + str(name_dict[old_name])

    def _add_cast_node_inp(tensor, precision="fp16", suffix=""):
        if precision == "fp16":
            onnx_precision = int(onnx.TensorProto.FLOAT16)
            np_precision = "float16"
        elif precision == "fp32":
            onnx_precision = int(onnx.TensorProto.FLOAT)
            np_precision = "float32"
        else:
            onnx_precision = int(onnx.TensorProto.INT32)
            np_precision = "int32"

        cast_out = gs.Variable(
            name=_get_unique_name(tensor.name + f"_{precision}{suffix}"),
            dtype=np_precision,
            shape=tensor.shape,
        )
        cast_node = gs.Node(
            op="Cast",
            name=_get_unique_name(tensor.name + f"_cast_to_{precision}{suffix}"),
            attrs={"to": onnx_precision},
            inputs=[tensor],
            outputs=[cast_out],
        )
        graph.nodes.append(cast_node)
        return cast_out

    def _add_cast_node_out(tensor, inp_precision="fp16", out_precision="fp32", suffix=""):
        cast_precision = int(onnx.TensorProto.FLOAT16) if out_precision == "fp16" else int(onnx.TensorProto.FLOAT)
        np_precision = "float16" if inp_precision == "fp16" else "float32"

        cast_inp = gs.Variable(
            name=_get_unique_name(tensor.name + f"_{inp_precision}{suffix}"),
            dtype=np_precision,
            shape=tensor.shape,
        )
        cast_node = gs.Node(
            op="Cast",
            name=_get_unique_name(tensor.name + f"_cast_to_{out_precision}{suffix}"),
            attrs={"to": cast_precision},
            inputs=[cast_inp],
            outputs=[tensor],
        )
        graph.nodes.append(cast_node)
        return cast_inp

    def _is_constant(tensor):
        return isinstance(tensor, gs.Constant) or (tensor.inputs and tensor.inputs[0].op == "Constant")

    def _check_precision_list(precisions, num_tensors, node_name):
        if isinstance(precisions, list):
            assert len(precisions) == num_tensors, \
                f"Number of inputs or outputs doesn't match list of precisions for {node_name}."
        else:
            # Propagate single precision to all inputs
            precisions = [precisions] * num_tensors
        return precisions

    graph = gs.import_onnx(onnx.load(onnx_path))
    castable_nodes = [n for n in graph.nodes if n.op in custom_ops_to_cast.keys()]

    for node in castable_nodes:
        # Cast all inputs to FP16
        inp_precisions = custom_ops_to_cast[node.op]["inp_precisions"]
        inp_precisions = _check_precision_list(inp_precisions, len(node.inputs), node.name)
        for inp_idx, inp in enumerate(node.inputs):
            inp_precision = inp_precisions[inp_idx]
            if inp_precision != "fp32":
                cast_out = _add_cast_node_inp(inp, precision=inp_precision)
                node.inputs[inp_idx] = cast_out
            else:
                node.inputs[inp_idx].dtype = "float32"

        # Cast all outputs back to FP32
        out_precisions = custom_ops_to_cast[node.op]["out_precisions"]
        out_precisions = _check_precision_list(out_precisions, len(node.outputs), node.name)
        for out_idx, out in enumerate(node.outputs):
            out_precision = out_precisions[out_idx]
            if out_precision != "fp32":
                cast_inp = _add_cast_node_out(out, inp_precision=out_precision)
                node.outputs[out_idx] = cast_inp
            else:
                node.outputs[out_idx].dtype = "float32"

    graph.cleanup().toposort()

    new_onnx_path = onnx_path.replace(".onnx", "_castFP16.onnx")
    onnx.save(gs.export_onnx(graph), new_onnx_path)
    return new_onnx_path


def ensure_plugin_io_precisions(onnx_path, plugins_precision):
    custom_ops_to_cast = {}
    for trt_plugin_precision in plugins_precision:
        assert ":" in trt_plugin_precision, (
            "Plugin precision is incorrectly formatted."
            " Please check that it's in the format <op_type>:<precision> or "
            " <op_type>:[<inp1_precision>,<inp2_precision>]:[out1_precision]."
        )
        if "[" not in trt_plugin_precision:
            op_type, precision = trt_plugin_precision.split(":")
            print(f"Assuming that all inputs/outputs in {op_type} have the same precision of {precision}.")
            custom_ops_to_cast[op_type] = {"inp_precisions": precision, "out_precisions": precision}
        else:
            op_type, inp_precisions_str, out_precisions_str = trt_plugin_precision.split(":")
            print(
                f"Precisions for {op_type} were given for each input ({inp_precisions_str}) and output ({out_precisions_str}).")
            custom_ops_to_cast[op_type] = {
                "inp_precisions": inp_precisions_str.strip('[]').split(','),
                "out_precisions": out_precisions_str.strip('[]').split(',')
            }
    if custom_ops_to_cast:
        onnx_path = add_fp16_fp32_cast(onnx_path, custom_ops_to_cast)
        intermediate_generated_files.append(onnx_path)
    return onnx_path


def ensure_all_tensors_type(onnx_path, custom_ops):
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if node.op == "Identity":
            node_inps = [inp for inp in node.inputs if isinstance(inp, gs.Constant)]
            for out in node.outputs:
                if out.dtype is None:
                    out.dtype = node_inps[0].dtype if node_inps else "float32"
        elif node.op == "Constant":
            dtype = node.attrs['value'].dtype
            for out in node.outputs:
                if out.dtype is None:
                    out.dtype = dtype or "float32"
        elif node.op in custom_ops:
            node_inps = [inp for inp in node.inputs if isinstance(inp, gs.Variable) and inp.dtype]
            for out in node.outputs:
                if out.dtype is None:
                    out.dtype = node_inps[0].dtype if node_inps else "float32"
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
    onnx_path, custom_ops = ensure_ort_support(onnx_path, args.custom_ops)

    # ======== Ensure plugin precision ========
    if args.plugins_precision:
        onnx_path = ensure_plugin_io_precisions(onnx_path, args.plugins_precision)

    # ======== Ensure that all tensors have an output type ========
    onnx_path = ensure_all_tensors_type(onnx_path, custom_ops)

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
