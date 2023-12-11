# MTMI-Inference



## Environment

To be added.
```
pip install -r requirements.txt
```

## Onnx scripts

The original onnx model has been exported by running `python tools/pytorch2onnx_seg.py configs/mtformer/mtformer_export.py --checkpoint latest.pth` under [MTMI:mtmi_inference](https://gitlab-master.nvidia.com/boyinz/mtmi/-/tree/mtmi_inference). We use lfs to store the onnx file under [onnx_files/mtmi.onnx](https://gitlab-master.nvidia.com/boyinz/mtmi-inference/-/tree/main/onnx_files).

To simplify the onnx file:
```
python tools/onnx_slim.py
```

To split the onnx model into encoder, depth decoder and semantic segmentation decoder:
```
python tools/onnx_split.py
```

Build engine with "outputIOFormats=fp32:chw32" for MIT-b0 encoder:
```
trtexec --onnx=onnx_files/mtmi_encoder.onnx --fp16 --saveEngine=mtmi_encoder_fp16_gpu.plan --outputIOFormats=fp32:chw32 --verbose
```

Build DLA loadable with "inputIOFormats=int8:chw32" and "outputIOFormats=int8:dla_linear" for depth decoder and semseg decoder onnx model.
```
trtexec --onnx=onnx_files/mtmi_depth_head.onnx --int8 --saveEngine=mtmi_depth_i8_dla.bin --useDLACore=0 --inputIOFormats=int8:chw32 --outputIOFormats=int8:dla_linear --buildOnly --safe --verbose --buildDLAStandalone
trtexec --onnx=onnx_files/mtmi_seg_head.onnx --int8 --saveEngine=mtmi_seg_i8_dla.bin --useDLACore=0 --inputIOFormats=int8:chw32 --outputIOFormats=int8:dla_linear --buildOnly --safe --verbose --buildDLAStandalone

```

## Build inference app

### Build the app
```bash
$ cd inference_app
$ bash build.sh
$ cd ..
```

### Run the app
```bash
$ LD_LIBRARY_PATH=/home/nvidia/workspace/boyin/TensorRT-8.6.11.3/lib:$LD_LIBRARY_PATH ./inference_app/infer configs/config_p1.yaml
```

## Authors and acknowledgment
to be added

## License
to be added


