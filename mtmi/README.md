# MTMI-Inference
## Introduction
This application is to demostrate the deployment of a multi-task network on orin platform. 
To improve latency and throughput, we leveraging different compute devices on the SoC with CUDA, TensorRT and cuDLA.
For more details, you may refer to our webinar at [link](https://info.nvidia.com/autonomous-vehicle-multi-task-model-Inference.html)

## Onnx scripts

The original onnx model has been exported by running `python tools/pytorch2onnx_seg.py configs/mtformer/mtformer_export.py --checkpoint latest.pth`.

### Prerequisite
You may install dependencies with `pip install onnx onnxruntime onnx-graphsurgeon onnx-simplifier`

Step1: Simplify the onnx file. This is to manipulate the onnx graph, remove redundant nodes, do constant folding etc. For more detail, you may refer to [link](https://github.com/daquexian/onnx-simplifier)
```bash
python tools/onnx_slim.py
```

Step2: Split the onnx model into encoder, depth decoder and semantic segmentation decoder. Since encoder, depth and segmentation heads are assigned to different device, we also split the whole onnx graph into 3 sub-graphs and handle them separately.
```bash
python tools/onnx_split.py
```

## Post-training quantization

To better utilize DLA, we should use Int8. 
In order to obtain scale information, we utilize tensorrt calibration apis.

### Prepare intermediate features for calibration
We first use onnxruntime to produce intermediate features shared by the two decoders. And these features will be cached for next steps.
```bash
python tools/batch_preprocessing_onnx.py --onnx=PATH_TO_ONNX --image_path=PATH_TO_IMAGE_FILES --output_path=PATH_TO_SAVE_INTERMEDIATE_FEATURES
```

### Perform Post-training quantization
Then we can load from the cached feature map as input, and feed them into calibators.
With default arguments, you will get the calibration file under `calibration/`.

```bash
python tools/create_calibration_cache.py --onnx=PATH_TO_HEAD_ONNX --image-path=PATH_TO_IMAGE_FILES --output-path=PATH_TO_SAVE_ENGINES --cache-path=PATH_TO_SAVE_CALIBRATION_FILES
```

Note that for now you need to manually modify the last two lines in the cache file for segmentation due to an unknown bug which will leads to slower inference with DLA. Assume the modified calibration file is at `calibration/calibration_cache_seg_mod.bin`
```
input.408: 3e99a6d6
onnx::ArgMax_1963: 3e98d50a
```

## Build TensorRT engine and DLA loadables using the calibration caches
On orin platform:

Build engine with "outputIOFormats=fp16:chw32" for MIT-b0 encoder:
```bash
trtexec --onnx=onnx_files/mtmi_encoder.onnx \
        --fp16 \
        --saveEngine=engines/mtmi_encoder_fp16.engine \
        --outputIOFormats=fp16:chw32 \
        --verbose
```

Build DLA loadable with "inputIOFormats=int8:chw32" and "outputIOFormats=int8:dla_linear" for depth decoder and semantic segmentation decoder onnx model.
```bash
trtexec --onnx=onnx_files/mtmi_depth_head.onnx \
        --int8 \
        --saveEngine=loadables/mtmi_depth_i8_dla.loadable \
        --useDLACore=0 \
        --inputIOFormats=int8:chw32 \
        --outputIOFormats=int8:dla_linear \
        --buildOnly \
        --verbose \
        --buildDLAStandalone \
        --calib=calibration/calibration_cache_depth.bin
trtexec --onnx=onnx_files/mtmi_seg_head.onnx \
        --int8 \
        --saveEngine=loadables/mtmi_seg_i8_dla.loadable \
        --useDLACore=1 \
        --inputIOFormats=int8:chw32 \
        --outputIOFormats=int8:dla_linear \
        --buildOnly \
        --verbose \
        --buildDLAStandalone \
        --calib=calibration/calibration_cache_seg_mod.bin
```

## Build and run inference app
The inference app is designed as the following steps:
1. Initialization
2. For each round
    1. load next image, if no more image, move to the first one
    2. preprocess
    3. run backbone on gpu
    4. quantize the feature map
    5. run depth head on dla0, and run segmentation head on dla1 at the same time.
    6. if last round, dump output data to results/ for visualization
3. cleanup and exit

### Build the app
In this section, we will cross compile the application on x86 for orin platform.

Docker: nvcr.io/drive-priority/driveos-pdk/drive-agx-orin-linux-aarch64-pdk-build-x86:6.0.8.1-0006.
Launch docker with 
```bash
docker run --gpus all -it --network=host --rm -v DL4AGX_DIR:/DL4AGX nvcr.io/drive-priority/driveos-pdk/drive-agx-orin-linux-aarch64-pdk-build-x86:6.0.8.1-0006
```
Inside the docker, you can run:
```bash
cd /DL4AGX/mtmi/inference_app
sh build.sh
```
And you will get the executable file at `/DL4AGX/mtmi/inference_app/build/orin/mtmiapp`

### Run the app
Please make sure you have the following files on Orin.
```yaml
engines/
  mtmi_depth_i8_dla.loadable
  mtmi_encoder_fp16.engine
  mtmi_seg_i8_dla.loadable
results/  # folder to hold result data
tests/
  1.png   # input images, must crop to 1024x1024
  ...
mtmiapp   # the demo app
```

Then run the demo with
```bash
$ ./mtmiapp
```

Sample logs:
```
...
loop: 1 backbone elapsed: 29.799711 chrono elapsed: 29.965000
...
```

### Visualize results
Run the following python scripts to obtain visualization results from dumped binary result data.
```
python tools/visualize.py
```
Then you will get image results in `results/`
