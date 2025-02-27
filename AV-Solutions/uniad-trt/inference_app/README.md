# C++ Inference Application

## Environment
The inference application with `TensorRT::enqueueV2` is tested on NVIDIA DRIVE platforms and X86 platform using `TensorRT 8.6`. 

Since `TensorRT::enqueueV2` is being deprecated, we also demonstrate how to use `TensorRT::enqueueV3` to run engine inference with inputs or outputs with data-dependent-shape (DDS). The inference application with `TensorRT::enqueueV3` is tested on NVIDIA DRIVE platforms and X86 platform using `TensorRT 10.8`.
## Dependencies
The inference application will use [cuOSD](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/cuOSD) and [STB](https://github.com/nothings/stb) as submodules. Notice that since cuOSD do not have separate repo, we need to manually download it from [here](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/cuOSD) and put it under [the dependencies folder](../../common/dependencies/), i.e.,
```
common/
├── dependencies/
│   ├── stb/
│   ├── cuOSD/
```
## Build TensorRT plugins and the inference application

### Plugins and Application Compilation
To deploy UniAD-tiny with TensorRT, we first need to compile TensorRT plugins for `MultiScaleDeformableAttnTRT`, `InverseTRT` and `RotateTRT` operators which are not supported by Native TensorRT, and then we need to build the C++ inference application. 

Please `cd` into `enqueueV2` or `enqueueV3` folder and run the following commands to compile:
```
cd enqueueV<2 or 3>/
mkdir ./build && cd ./build/
cmake .. -DTENSORRT_PATH=<path_to_TensorRT> -DTARGET_GPU_SM=<GPU_compute_capability> && make -j$(nproc)
```

Then the ```uniad``` and ```libuniad_plugin.so``` should be generated under the ```./build``` folder


### Engine Build
To build TensorRT engine, run the following commands
```
MIN=901
OPT=901
MAX=1150
SHAPES=prev_track_intances0:${MIN}x512,prev_track_intances1:${MIN}x3,prev_track_intances3:${MIN},prev_track_intances4:${MIN},prev_track_intances5:${MIN},prev_track_intances6:${MIN},prev_track_intances8:${MIN},prev_track_intances9:${MIN}x10,prev_track_intances11:${MIN}x4x256,prev_track_intances12:${MIN}x4,prev_track_intances13:${MIN}

LD_LIBRARY_PATH=${TRT_PATH}/lib:$LD_LIBRARY_PATH \
${TRT_PATH}/bin/trtexec \
  --onnx=<path_to_ONNX> \
  --saveEngine=<path_to_engine> \
  --staticPlugins=<path_to_libuniad_plugin.so> \
  --verbose \
  --profilingVerbosity=detailed \
  --useCudaGraph \
  --tacticSources=+CUBLAS \
  --minShapes=${SHAPES//${MIN}/${MIN}} \
  --optShapes=${SHAPES//${MIN}/${OPT}} \
  --maxShapes=${SHAPES//${MIN}/${MAX}} \
  --skipInference
```

## Test inference application
The overview of the inference pipeline is shown in the following figure. Apart from inputs explicitly listed, there are other state variables which are updated after inference is done on the current frame and are used as inputs for the next frame.
<img src="../assets/engine_infer.png" width="1024">

The inference application will read images directly from jpg files, while it does need metadata generated in the [Generate Preprocessed Data](../documents/data_prep.md#generate-preprocessed-data) step. 

The [Generate Preprocessed Data](../documents/data_prep.md#generate-preprocessed-data) step will generate the following metadata:
timestamp    | l2g_r_mat | l2g_t | command | img_metas_can_bus | img_metas_lidar2img | img_metas_scene_token | info.txt
--------------------- | ---- | -------- | --- | --------------------------------| ------- | ---- | -------
the timestamp of the current frame | lidar2global rotation matrix | lidar2global translation | navigation command | image can bus | lidar2image matrixs | scene ids | image file paths 

Organize the data files to follow the file pattern:
```
uniad_trt_input/
|── timestamp/
|── l2g_r_mat/
|── l2g_t/
|── command/
|── img_metas_can_bus/
|── img_metas_lidar2img/
|── img_metas_scene_token/
|── info.txt
```

The ```uniad_trt_input``` folder is used as ```<input_path>```.


### Run inference
Run the following command to run inference on the input data and to generate output results. Notice that for the application to correctly locate and use the data, you need to call the application at the [enqueueV2](./enqueueV2) or [enqueueV3](./enqueueV3) folder of this repo and to create a soft link of data folder under the [enqueueV2](./enqueueV2) or [enqueueV3](./enqueueV3) folder. The data soft link should be linking to:
```
enqueueV2/ or enqueueV3/
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-trainval/
│   ├── infos/
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_val.pkl
│   ├── others/
│   │   ├── motion_anchor_infos_mode6.pkl
```
The inference command is:
```
LD_LIBRARY_PATH=<path_to_TensorRT>/lib/:$LD_LIBRARY_PATH ./build/uniad <engine_path> ./build/libuniad_plugin.so <input_path> <output_path> <number_frame>
```
This command will read the raw images and the dumped metadata as input, run infernece using the engine and generate visualization results under the ```<output_path>``` folder.

<- Last Page: [UniAD-tiny Traning and Exportation](../documents/train_export.md)
