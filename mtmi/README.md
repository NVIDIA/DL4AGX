# MTMI-Inference



## Environment

To be added.
```bash
$ pip install -r requirements.txt
```

Prepare Tensorrt Environment:
```bash
$ TRTDIR=PATH_TO_TRT_8.6.12.3
$ export PATH=$TRTDIR/bin:$PATH
$ export LD_LIBRARY_PATH=$TRTDIR/lib:$LD_LIBRARY_PATH
```

## Onnx scripts

The original onnx model has been exported by running `python tools/pytorch2onnx_seg.py configs/mtformer/mtformer_export.py --checkpoint latest.pth` under [MTMI:mtmi_inference](https://gitlab-master.nvidia.com/boyinz/mtmi/-/tree/mtmi_inference). We use lfs to store the onnx file under [onnx_files/mtmi.onnx](https://gitlab-master.nvidia.com/boyinz/mtmi-inference/-/tree/main/onnx_files).

To simplify the onnx file:
```bash
$ python tools/onnx_slim.py
```

To split the onnx model into encoder, depth decoder and semantic segmentation decoder:
```bash
$ python tools/onnx_split.py
```

Build engine with "outputIOFormats=fp32:chw32" for MIT-b0 encoder:
```bash
$ trtexec --onnx=onnx_files/mtmi_encoder.onnx --fp16 --saveEngine=engines/mtmi_encoder_fp16_gpu.plan --outputIOFormats=fp32:chw32 --verbose
```


## PTQ

In order to get better accuracy, we need to do calibrations and then build the DLA int8 engines.

### Generate intermediate features for calibration
```bash
$ python tools/batch_preprocessing_onnx.py --onnx=PATH_TO_ONNX --image_path=PATH_TO_IMAGE_FILES --output_path=PATH_TO_SAVE_OUTPUTS
```

### Do calibration
Please refer to the argparse in the python file to enter correct paths, by default you will get the calibration file under `calibration/` and tmp engine(will not be used) under `engines/`
```bash
$ python tools/build_dla.py
```

Note that for now you need to manually modify the last two lines in the cache file for segmentation due to an unknown bug in tensorrt, ask Le for more information and trackign for the nvbugs:
```
input.408: 3e99a6d6
onnx::ArgMax_1963: 3e98d50a
```

### Build DLA loadables using the calibration caches

Build DLA loadable with "inputIOFormats=int8:chw32" and "outputIOFormats=int8:dla_linear" for depth decoder and semseg decoder onnx model.
```bash
$ trtexec --onnx=onnx_files/mtmi_depth_head.onnx --int8 --saveEngine=engines/mtmi_depth_i8_dla.bin --useDLACore=0 --inputIOFormats=int8:chw32 --outputIOFormats=int8:dla_linear --buildOnly --verbose --buildDLAStandalone --calib=calibration/calibration_cache_depth.bin
$ trtexec --onnx=onnx_files/mtmi_seg_head.onnx --int8 --saveEngine=engines/mtmi_seg_i8_dla.bin --useDLACore=0 --inputIOFormats=int8:chw32 --outputIOFormats=int8:dla_linear --buildOnly --verbose --buildDLAStandalone --calib=calibration/calibration_cache_seg_mod.bin
```

## Build inference app

### Build the app
```bash
$ cd inference_app
$ mkdir build
$ cd build
$ cmake ../
$ make
```

### Run the app
```bash
$ ./inference_app/infer configs/config_p1.yaml
```

Sample logs:
```yaml
Average file read time: 34.8031 milliseconds
Average input copy time: 1.7319 milliseconds
Average launch time: 0 milliseconds
Average wall clock time without file read: 39.8018 milliseconds
Average wall clock time: 74.6054 milliseconds
Average prepare DLA clock time: 7.8857 milliseconds
```
Where the 39.8ms stats the compute time in the gpu+dla pipeline

### Visualize results
```
python tools/load_depth.py
python tools/load_seg.py
```

Then you will get image results at `results/`

## Authors and acknowledgment
to be added

## License
to be added


