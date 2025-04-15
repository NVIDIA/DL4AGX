import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_path", type=str, required=True, default="/workspace/UniAD/onnx/uniad_tiny_imgx0.25_cp.repaired.quant_exclude_MatMul_ops_trt10.9.0.34.onnx")
args = parser.parse_args()

onnx_path = args.onnx_path
model = onnx.load(onnx_path)

for graph_input in model.graph.input:
    if "prev_track_intances" in graph_input.name:
        for idx, d in enumerate(graph_input.type.tensor_type.shape.dim):
            if idx==0:
                print(f" Setting '{graph_input.type.tensor_type.shape.dim[idx].dim_value} {graph_input.type.tensor_type.shape.dim[idx].dim_param}'")
                print(f" theorectically to '{0} {graph_input.name}_dynamic' ...")
                graph_input.type.tensor_type.shape.dim[idx].dim_value = 0
                graph_input.type.tensor_type.shape.dim[idx].dim_param = f'{graph_input.name}_dynamic'
                print(f" in reality to '{graph_input.type.tensor_type.shape.dim[idx].dim_value} {graph_input.type.tensor_type.shape.dim[idx].dim_param}'!!!")

print(f"cleaning inferred shapes...")
del model.graph.value_info[:]

onnx.save(model, onnx_path.replace(".onnx", ".dynamic.onnx"))
print(f"✅ Saved: {onnx_path.replace('.onnx', '.dynamic.onnx')}")