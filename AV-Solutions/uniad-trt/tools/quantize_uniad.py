import numpy as np
from modelopt.onnx.quantization import quantize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_path", type=str, required=True, default="/workspace/UniAD/onnx/uniad_tiny_imgx0.25_cp.repaired.onnx")
parser.add_argument("--cali_data_path", type=str, required=True, default="/workspace/UniAD/data/uniad_cali_data_minShapes901_182.npz")
parser.add_argument("--trt_plugins", type=str, required=True, default="/workspace/UniAD/plugins/lib/libuniad_plugin_trt10.9.0.34_x86_cu118.so")
args = parser.parse_args()

onnx_path = args.onnx_path
cali_data_path = args.cali_data_path
output_path = onnx_path.replace(".onnx", ".quant_exclude_MatMul_ops_trt10.9.0.34.onnx")
calibration_eps = ["cuda:0", "cpu", "trt"]
trt_plugins = args.trt_plugins
UNIAD_SHAPES = "prev_track_intances0:901x512,prev_track_intances1:901x3,prev_track_intances3:901,prev_track_intances4:901,prev_track_intances5:901,prev_track_intances6:901,prev_track_intances8:901,prev_track_intances9:901x10,prev_track_intances11:901x4x256,prev_track_intances12:901x4,prev_track_intances13:901,prev_timestamp:1,prev_l2g_r_mat:1x3x3,prev_l2g_t:1x3,prev_bev:2500x1x256,timestamp:1,l2g_r_mat:1x3x3,l2g_t:1x3,img:1x6x3x256x416,img_metas_can_bus:18,img_metas_lidar2img:1x6x4x4,command:1,use_prev_bev:1,max_obj_id:1"
print(f"Loading npz: {cali_data_path}")
npz_data = np.load(cali_data_path, allow_pickle=True)
calib_data = {key: npz_data[key] for key in npz_data.files}
print("Calibration data keys:", calib_data.keys())
print(f"Quantizing ONNX: {onnx_path}")
quantize(
    onnx_path=onnx_path,
    calibration_data=calib_data,
    calibration_eps=calibration_eps,
    output_path=output_path,
    op_types_to_exclude=['MatMul'],
    trt_plugins=trt_plugins,
    calibration_method="entropy",
    simplify_model=True,
    calibration_shapes=UNIAD_SHAPES,
)
print(f"✅ Done. Quantized model saved at: {output_path}")

# to run
# conda activate bianjiangy_modelopt_newest
# pip install --upgrade onnxruntime-gpu==1.21.0
# pip install /data2/bjyang/TensorRT/TensorRT-10.9.0.33_x86_cu118/python/tensorrt-10.9.0.33-cp312-none-linux_x86_64.whl
# LD_LIBRARY_PATH=/data2/bjyang/cudnn-linux-x86_64-9.8.0.87_cuda12-archive/lib:/data2/bjyang/TensorRT/TensorRT-10.9.0.33_x86_cu118/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=1 python quantize_uniad.py
# pip install /data2/bjyang/TensorRT/TensorRT-10.10.0.19_x86_cu118/python/tensorrt-10.10.0.19-cp312-none-linux_x86_64.whl
# LD_LIBRARY_PATH=/data2/bjyang/cudnn-linux-x86_64-9.8.0.87_cuda12-archive/lib:/data2/bjyang/TensorRT/TensorRT-10.10.0.19_x86_cu118/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=1 python quantize_uniad.py