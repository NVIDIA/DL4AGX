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

Build engine with "outputIOFormats=fp16:chw32" for MIT-b0 encoder:
```bash
$ trtexec --onnx=onnx_files/mtmi_encoder.onnx --fp16 --saveEngine=engines/mtmi_encoder_fp16.engine --outputIOFormats=fp16:chw32 --verbose
```

## PTQ

In order to get better accuracy, we need to generate calibrations first and then build the DLA int8 engines.

### Prepare intermediate features for calibration
```bash
$ python tools/batch_preprocessing_onnx.py --onnx=PATH_TO_ONNX --image_path=PATH_TO_IMAGE_FILES --output_path=PATH_TO_SAVE_INTERMEDIATE_FEATURES
```

### Do calibration
Please refer to the argparse in the python file to enter correct paths, by default you will get the calibration file under `calibration/` and tmp engine(will not be used) under `engines/`
```bash
$ python tools/build_dla.py --onnx=PATH_TO_HEAD_ONNX --image-path=PATH_TO_IMAGE_FILES --output-path=PATH_TO_SAVE_ENGINES --cache-path=PATH_TO_SAVE_CALIBRATION_FILES
```

Note that for now you need to manually modify the last two lines in the cache file for segmentation due to an unknown bug in tensorrt, ask Le for more information and tracking for the nvbugs:
```
input.408: 3e99a6d6
onnx::ArgMax_1963: 3e98d50a
```

### Build DLA loadables using the calibration caches

Build DLA loadable with "inputIOFormats=int8:chw32" and "outputIOFormats=int8:dla_linear" for depth decoder and semseg decoder onnx model.
```bash
$ trtexec --onnx=onnx_files/mtmi_depth_head.onnx --int8 --saveEngine=engines/mtmi_depth_i8_dla.bin --useDLACore=0 --inputIOFormats=int8:chw32 --outputIOFormats=int8:dla_linear --buildOnly --verbose --buildDLAStandalone --calib=calibration/calibration_cache_depth.bin
$ trtexec --onnx=onnx_files/mtmi_seg_head.onnx --int8 --saveEngine=engines/mtmi_seg_i8_dla.bin --useDLACore=1 --inputIOFormats=int8:chw32 --outputIOFormats=int8:dla_linear --buildOnly --verbose --buildDLAStandalone --calib=calibration/calibration_cache_seg_mod.bin
```

## Build inference app

### Build the app
```bash
$ cd inference_app
$ sh build.sh orin
```

### Run the app
```bash
$ ./inference_app/mtmiapp
```

Sample logs:
```yaml

```

### Visualize results
Run the following python scripts to obtain visualization results from dumped binary result data.
```
python tools/visualize.py
```

Then you will get image results at `results/`
