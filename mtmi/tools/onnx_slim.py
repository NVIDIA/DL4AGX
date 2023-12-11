import onnx
from onnxsim import simplify, model_info

model = onnx.load("onnx_files/mtmi.onnx")
model_simp, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"
simplified_path = "onnx_files/mtmi_slim.onnx"
onnx.save(model_simp, simplified_path)
model_info.print_simplifying_info(model, model_simp)
print(f"Simplified onnx model saved to {simplified_path}")