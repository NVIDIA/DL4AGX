# About
Instructions on how to reproduce the results in the main [README](../README.md#results).

# Steps to reproduce
To reproduce the `FP32`, `FP16`, and `QDQ_BEST (ModelOpt PTQ)` results:
1. Run `./deploy_trt.sh` to build/save the TRT engine and obtain the runtime;
2. Run `./evaluate_trt.sh` to evaluate the TRT engine's accuracy.

To reproduce the `BEST (TensorRT PTQ)` results:
```sh
$ cd /workspace/BEVFormer_tensorrt
$ python /mnt/results/onnx2trt_calib_npz.py configs/bevformer/plugin/bevformer_tiny_trt_p2.py \
    --onnx_path=/mnt/models/bevformer_tiny_epoch_24_cp2_op13_post_simp.onnx \
    --int8 --fp16 \
    --calibrator entropy \
    --calibration_data_path=/workspace/BEVFormer_tensorrt/data/nuscenes/calib_data.npz \
    --trt_plugins=$PLUGIN_PATH
$ python tools/bevformer/evaluate_trt.py \
    configs/bevformer/plugin/bevformer_tiny_trt_p2.py \
    /mnt/models/bevformer_tiny_epoch_24_cp2_op13_post_simp_TRT_PTQ.engine \
    --trt_plugins=$PLUGIN_PATH
```
