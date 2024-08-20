# UniAD-TensorRT
[UniAD](https://arxiv.org/abs/2212.10156) is a Unified Autonomous Driving algorithm framework which follows end-to-end manners, taking multi-view vision input and could output planning results directly. UniAD achieves SOTA performance in many autonomous driving tasks especially on planning task. The code for UniAD can be found [here](https://github.com/OpenDriveLab/UniAD).

<img src="./assets/pipeline.png" width="1024">

(Image taken from the [UniAD repo](https://github.com/OpenDriveLab/UniAD/tree/main))

This repo demonstrate how to deploy UniAD on NVIDIA Drive Orin platform using TensorRT. To make it more practical, we trained a tiny version UniAD (refered as ```UniAD-tiny```) with smaller backbone and build engines based on that model.

## Table of Contents:
1. [Getting Started](#start)
   - [Project Installation](./documents/proj_installation.md)
   - [Environment Preparation](./documents/env_prep.md)
   - [Data Preparation](./documents/data_prep.md)
   - [UniAD Tiny Training](./documents/tiny_training.md)
   - [Export ONNX and Build Engine](./documents/onnx_engine_build.md)
   - [Inference Application](./inference_app/README.md)
2. [Results demonstration](#results)
3. [Reference](#ref)

## Getting Started <a name="start"></a>

Please see the following documents for environment installation, UniAD training, ONNX exporting and engine building, and inference application in C++.
   - [Project Installation](./documents/proj_installation.md)
   - [Environment Preparation](./documents/env_prep.md)
   - [Data Preparation](./documents/data_prep.md)
   - [UniAD Tiny Training](./documents/tiny_training.md)
   - [Export ONNX and Build Engine](./documents/onnx_engine_build.md)
   - [Inference Application](./inference_app/README.md)

## Results demonstration <a name="results"></a>
The inference application will read raw images as input and generate output visualizations to showcase the planning trajectory and dynamic object detection. Notice that there is no collusion correction or BBOX NMS in the post-process. The visuaizer is visualizing the raw planning trajectory and BBOX pridiction outputs.

Here are some examples for the inference output:

![](./assets/uniad-inference.gif)

1) The numbers draw on the BBOXs are the confidence scores;
2) In the BEV view, the light green BBOX is indicating the ego car;
3) The light green lines in the BEV view and images are indicating the planning trajectory.
4) In the BEV view, the white lines in the BBOXs are indicating the heading of the objects
5) In the BEV view, the lines starting from the center of the BBOXs are illustrating the velocities of the objects.

## Reference <a name="ref"></a>
1. [UniAD Paper: Planning-oriented Autonomous Driving](https://arxiv.org/abs/2212.10156)
2. [UniAD Repository](https://github.com/OpenDriveLab/UniAD/tree/main)
3. [BEVFormer_tensorrt Repository](https://github.com/DerryHub/BEVFormer_tensorrt/tree/main)
4. [STB Repository](https://github.com/nothings/stb/tree/master)
5. [CUDA-BEVFusion Repository](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion)
6. [cuOSD Repository](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/cuOSD)
7. [CUDA-FastBEV Repository](https://github.com/Mandylove1993/CUDA-FastBEV/tree/main)
8. [NuScenes DEveloper Kit](https://github.com/nutonomy/nuscenes-devkit.git)
9. [MMdetection3d](https://github.com/open-mmlab/mmdetection3d)