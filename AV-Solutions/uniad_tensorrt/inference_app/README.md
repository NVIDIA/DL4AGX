# C++ Inference Application
In this README file, we will cover the following topics:
1) Inference application environments and platforms;
2) How to build the plugins and the inference application;
3) How to build the engines;
4) How to prepare the data for inference;
5) How to run inference application.

## Environments and platforms
The inference application is using ```TensorRT 8.6.13.3``` and has been tested on Orin DOS Linux platforms.
## Submodules
The inference application will use [cuOSD](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/cuOSD) and [STB](https://github.com/nothings/stb) as submodules. Notice that since cuOSD do not have separate repo, we need to manually download it from the [link](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/cuOSD). Please make sure to clone the latest masters under [dependencies folder](./dependencies/). The folder should looks like:
```
dependencies/
|── stb/
|── cuOSD/
```
## Build TensorRT plugins and the inference application

To deploy UniAD-tiny with TensorRT, we first need to compile TensorRT plugins for `MultiScaleDeformableAttnTRT`, `InverseTRT` and `RotateTRT` operators that are not supported by Native TensorRT, and then we need to build the C++ inference application. To achieve that, we will use the [CMakeLists.txt](./CMakeLists.txt) to compile the plugin library and the inference app:
```
mkdir ./build && cd ./build/
cmake .. -DTENSORRT_PATH=<path_to_TensorRT> -DCUDA_ARCH=<GPU_arch> && make -j$(nproc)
```

Then the ```uniad``` and ```libuniad_plugin.so``` should be generated under the ```./build``` folder


#### Engine Build
To build TensorRT engine, please see the instructions at [run_trtexec.sh](../tools/run_trtexec.sh). Please modify TensorRT version (`TRT_VERSION`), TensorRT path (`TRT_PATH`), and other information like ONNX file path inside `run_trtexec.sh` if needed.
```
cd ../tools/
./run_trtexec.sh
```

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


### Run inference
Run the following command to run inference on the input data and generate output results. Notice that for the application to correctly locate and use the data, you need to call the application at the [root dir](../) of this repo.
```
cd ..
LD_LIBRARY_PATH=<path_to_TensorRT>/lib/:$LD_LIBRARY_PATH ./inference_app/build/uniad <engine_path> ./inference_app/build/libuniad_plugin.so <input_path> <output_path> <number_frame>
```
This command will read the raw images and the dumped metadata as input, run infernece using the engine and generate visualization results under the ```<output_path>``` folder.

<- Last Page: [UniAD Tiny Traning and Exportation](../documents/tiny_train_export.md)
