# C++ Inference Application
In this README file, we will cover the following topics:
1) Inference application environments and platforms;
2) How to build the inference application;
3) How to prepare the data for inference;
4) How to run inference application.

## Environments and platforms
The inference application is using ```TensorRT 8.6.13.3``` and has been tested on Orin DOS Linux platforms.
## Submodules
The inference application will use [cuOSD](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/cuOSD) and [STB](https://github.com/nothings/stb) as submodules, please make sure to clone the latest masters under [dependencies folder](./dependencies/). The folder should looks like:
```
dependencies/
|── stb/
|── cuOSD/
```
## Build inference application
Modify the [CMakeLists.txt](./CMakeLists.txt) to set the TensorRT root path, compute capability for cuOSD, and the path to the ```lib``` folder containing TensorRT plugin.
```
set(TENSORRT_INCLUDE_DIRS <path_to_TRT>/include/)
set(TENSORRT_LIBRARY_DIRS <path_to_TRT>/lib/)
...
set(TARGET_GPU_SM <GPU_arch>)
...
set(TENSORRT_PLUGIN_LIB_PTH <path_to_TRT_plugin_lib>)
```
Then compile the inference application.
```
mkdir ./build/ && cd ./build/
cmake .. -DCMAKE_TENSORRT_PATH=<path_to_TRT> && make -j$(nproc)
```
Then the ```uniad``` should be generated under the ```./build``` folder
## How to run inference
### Prepare data
The inference application will read images directly from jpg files, while it does need metadata generated in the ```Generate Preprocessed Data``` step. 

The ```Generate Preprocessed Data``` step will generate the following metadata:
timestamp    | l2g_r_mat | l2g_t | command | img_metas_can_bus | img_metas_lidar2img | img_metas_scene_token | info.txt
--------------------- | ---- | -------- | --- | --------------------------------| ------- | ---- | -------
the timestamp of the current frame | lidar2global rotation matrix | lidar2global translation | navigation command | image can bus | lidar2image matrixs | scene ids | image file paths 

Organize the data files so that is follows the file pattern:
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

Please also download the [SimHei font](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/libraries/cuOSD/data/simhei.ttf) and locate it in the [```tools```](../tools/) folder in the root directory of the repository for correct font in the visulization.
```
inference_app/
tools/
|── simhei.ttf
```


### Run inference
Run the following command to run inference on the input data and generate output results.
```
cd ..
LD_LIBRARY_PATH=<path_to_TRT>/lib/:$LD_LIBRARY_PATH LD_PRELOAD=<path_to_TRT_plugin_so_file> ./inference_app/build/uniad <TRT_engine_path> <input_path> <output_path> <number_of_frames_to_inference>
```
This command will read the raw images and the dumped metadata as input, run infernece using the engine and generate visualization results under the ```<output_path>``` folder.

<- Last Page: [ONNX and Engine Build](../documents/onnx_engine_build.md)
