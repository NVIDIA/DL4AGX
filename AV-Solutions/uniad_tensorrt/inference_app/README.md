# UniAD-TRT
[UniAD](https://arxiv.org/abs/2212.10156) is a Unified Autonomous Driving algorithm framework which follows an end-to-end manner, taking multi view vision input and could output planning results directly. Unid achieves SOTA performance in many autonomous driving tasks especially on planning task. The code for UniAD can be found [here](https://github.com/OpenDriveLab/UniAD).

<img src="./media/pipeline.png" width="1024">

(This image comes from the [UniAD repo](https://github.com/OpenDriveLab/UniAD/tree/main))

This application is a sample application to demostrate the deployment of UniAD on NVIDIA Drive Orin platform using TensorRT. 

## How to build
This project is using TensorRT 8.6.13.3 on Orin DOS Linux and X86 Linux platforms. (A docker file is provided at [DockerFile](./uniad_trt.dockerfile) for X86 platforms.)
### Submodules
This project will use [cuOSD](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/cuOSD) and [STB](https://github.com/nothings/stb) as submodules, please make sure to clone the latest masters under [dependencies folder](./dependencies/). The folder should looks like:
```
dependencies/
|-- stb/
|-- cuOSD/
```
### Build TensorRT plugin
The first step to build the inference application is to build TensorRT plugins that UniAD will be using. Those plugins will be built based on the code (latest master) from [BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt).

We need to modify the [CMakeLists.txt](./dependencies/BEVFormer_tensorrt/TensorRT/CMakeLists.txt) before compile for the TensorRT plugin. Please refer to [customized BEVFormer_tensorrt CMakeLists.txt](./tools/CMakeLists.txt) and replace the original [CMakeLists.txt](./dependencies/BEVFormer_tensorrt/TensorRT/CMakeLists.txt) by the customized one.

Be sure to modify the TensorRT root path at Line 16 & 17, and the compute compability at Line 63.
```
set(TENSORRT_INCLUDE_DIRS <path_to_TRT>/include/)
set(TENSORRT_LIBRARY_DIRS <path_to_TRT>/lib/)
...
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_<GPU_arch>,code=compute_<GPU_arch>)
```
Then compile the TensoRT plugin.
```
cd ./dependencies/BEVFormer_tensorrt/TensorRT/build/
rm -rf ./*
cmake .. -DCMAKE_TENSORRT_PATH=<path_to_TRT> && make -j$(nproc) && make install
```
You should see the generated libtensorrt_ops.so under ```./dependencies/BEVFormer_tensorrt/TensorRT/lib``` folder, which we will beusing in the inference.

### Generate TensorRT engine
Then, we need to geterate the TensorRT engine. For FP32 engine, run the command:
```
LD_LIBRARY_PATH=<path_to_TRT>/lib/:$LD_LIBRARY_PATH <path_to_TRT>/bin/trtexec --onnx=<onnx_file> --saveEngine=<engine_save_path> --staticPlugins=./dependencies/BEVFormer_tensorrt/TensorRT/lib/libtensorrt_ops.so --verbose --dumpLayerInfo --useCudaGraph --minShapes=prev_track_intances0:900x512,prev_track_intances1:900x3,prev_track_intances3:900,prev_track_intances4:900,prev_track_intances5:900,prev_track_intances6:900,prev_track_intances8:900,prev_track_intances9:900x10,prev_track_intances11:900x4x256,prev_track_intances12:900x4,prev_track_intances13:900  --optShapes=prev_track_intances0:901x512,prev_track_intances1:901x3,prev_track_intances3:901,prev_track_intances4:901,prev_track_intances5:901,prev_track_intances6:901,prev_track_intances8:901,prev_track_intances9:901x10,prev_track_intances11:901x4x256,prev_track_intances12:901x4,prev_track_intances13:901 --maxShapes=prev_track_intances0:1150x512,prev_track_intances1:1150x3,prev_track_intances3:1150,prev_track_intances4:1150,prev_track_intances5:1150,prev_track_intances6:1150,prev_track_intances8:1150,prev_track_intances9:1150x10,prev_track_intances11:1150x4x256,prev_track_intances12:1150x4,prev_track_intances13:1150 --skipInference > <build_log>
```
Currently TensorRT has bugs when generating FP16 engines, after the bug fix releases, add ```--fp16``` flag to generate FP16 engines.

### Build inference application
Modify the [CMakeLists.txt](./CMakeLists.txt) to set the TensorRT root path (Line 14 & 15), compute capability for cuOSD (Line 20), and the path to the ```lib``` folder containing TensorRT plugin (Line 42).
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
Please download the nuScenes data directly from the [nuScenes website](https://www.nuscenes.org/download).

The inference application will read images directly from jpg files, while it does need metadata to be prepared.

Run metadata preprocess scripts:
```
python3 ./tools/process_metadata.py --config <path_to_config> --dump_meta_pth <input_path> --num_frame <num_of_frames_to_inference>
```

The script will generate the following metadata:
timestamp    | l2g_r_mat | l2g_t | command | img_metas_can_bus | img_metas_lidar2img | img_metas_scene_token | info.txt
--------------------- | ---- | -------- | --- | --------------------------------| ------- | ---- | -------
the timestamp of the current frame | lidar2global rotation matrix | lidar2global translation | navigation command | image can bus | lidar2image matrixs | scene ids | image file paths 

Organize the data files so that is follows the file pattern:
```
uniad-trt/
|--data/
|----timestamp/
|----l2g_r_mat/
|----l2g_t/
|----command/
|----img_metas_can_bus/
|----img_metas_lidar2img/
|----img_metas_scene_token/
|----info.txt
```

The ```./data/``` folder is used as ```<input_path>```.

Please also download the [SimHei font](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/libraries/cuOSD/data/simhei.ttf) and locate it in the [```tools```](./tools/) folder for correct font in the visulization.


### Run inference
Run the following command to run inference on the input data and generate output results.
```
cd ./build/
LD_LIBRARY_PATH=<path_to_TRT>/lib/:$LD_LIBRARY_PATH LD_PRELOAD=<path_to_TRT_plugin_so_file> ./uniad <TRT_engine_path> <input_path> <output_path> <number_of_frames_to_inference>
```
This command will read the raw images and the dumped metadata as input, run infernece using the engine and generate visualization results.

A runtime analysis will be provided after warmup. Here is an example on Orin DriveOS Linux 6.0.8.1, TensorRT 8.6.13.3 in FP32 mode:
```
[timer:  Inference]: 	115.24742 ms
```

Notice that there is no collusion correction or BBOX NMS in the post-process. The visuaizer is visualizing the raw planning trajectory and BBOX pridiction output.

### Generate video
```
cd <output_path>
ffmpeg -framerate 5 -i "./dumped_video_results/%d.jpg" uniad-inference.mp4
```

### Video examples
Here are some examples on the UniAD inference output:

![](./media/uniad-inference.gif)

The numbers draw on the BBOXs are the confidence scores.

In the BEV view, the light green BBOX is indicating the ego car

The light green lines in the BEV view and images are indicating the planning trajectory.

In the BEV view, the white lines in the BBOXs is indicating the heading of the objects, and the lines starting from the center of the BBOXs are illustrating the velocities of the objects.

## Reference
1. https://github.com/OpenDriveLab/UniAD/tree/main
2. https://github.com/DerryHub/BEVFormer_tensorrt/tree/main
3. https://github.com/nothings/stb/tree/master
4. https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion
5. https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/cuOSD
6. https://github.com/Mandylove1993/CUDA-FastBEV/tree/main
