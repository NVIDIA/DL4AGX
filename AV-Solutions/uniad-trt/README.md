# UniAD-TensorRT
[UniAD](https://arxiv.org/abs/2212.10156) is a Unified Autonomous Driving algorithm framework which follows end-to-end manners, taking multi-view vision input and could output planning results directly. UniAD achieves SOTA performance in many autonomous driving tasks especially on planning task. 

<img src="./assets/pipeline.png" width="1024">

(Image taken from the [UniAD repo](https://github.com/OpenDriveLab/UniAD/tree/main))

This repo demonstrates how to deploy UniAD on NVIDIA Drive Orin platform using TensorRT. Specifically, we trained a tiny version of UniAD (`UniAD-tiny`) and provide step by step workflow including model training, ONNX model export, and inference with a sample C++ application.

## Table of Contents
1. [Getting Started](#start)
   - [Project Setup](#proj_setup)
   - [Environment Preparation](#env_setup)
   - [Data Preparation](#data_prepare)
   - [Model Training and Exportation](#uniad_tiny_train_export)
   - [Inference Application](#inference_app)
2. [Results](#results)
3. [Reference](#ref)

## Getting Started <a name="start"></a>

### Project Setup <a name="proj_setup"></a>
The project for UniAD deployment can be created based on the [UniAD repo](https://github.com/OpenDriveLab/UniAD) and [BEVFormer_tensorrt repo](https://github.com/DerryHub/BEVFormer_tensorrt/tree/main), on top of them, you will need to copy our prepared tool/config/helper files and apply several patches to make the project environment compatible and deployable. Please follow the instructions at [Project Setup](./documents/proj_setup.md) to set up the project code base from multiple sources. 

### Environment Preparation <a name="env_setup"></a>
We provide a dockfile for your convenience to prepare the environment for both training and deployment, please follow the instructions at [Environment Preparation](./documents/env_prep.md) to setup the environment needed in the following steps.

### Data Preparation <a name="data_prepare"></a>
In order to prepare data used for exporting to ONNX and inference with TensorRT, please follow the instructions at [Data Preparation](./documents/data_prep.md) to prepare the data for the project. For preparing training data, please refer to the instructions from [UniAD](https://github.com/OpenDriveLab/UniAD/blob/main/docs/DATA_PREP.md).

### Model Training and Exportation <a name="uniad_tiny_train_export"></a>
For efficiency when deploying a UniAD model on DRIVE platform, we trained a tiny version of UniAD(`UniAD-tiny`), with a smaller ResNet backbone and reduced image size & bev size. After training, the model needs to be exported from Pytorch to ONNX format. Please see [Model Training and Exportation](./documents/train_export.md) for details.

### Inference Application <a name="inference_app"></a>

The inference is showcased with a C++ sample application, it loads raw images and other data as input, runs inference with a built TensorRT engine, and outputs the results of tracking and planning with visualization. Please follow the instructions at [Inference Application](./inference_app/README.md) on how to build the TensorRT engine, compile and run the inference application.


## Results <a name="results"></a>
The inference application will generate output visualizations to showcase the planning trajectory and dynamic object detection. Notice that post-processing such as collusion correction is not implemented in the current sample. Raw planning trajectory and object detection results are visualized.


![](./assets/uniad-inference.gif)

In the visualization, green lines depict planning trajectories of the ego car in green bounding box. Detection of other objects are visualized in bounding boxes with different colors, and with white heading and confidence scores marked. The lines starting from the center of those bounding boxes indicate the velocities of the objects. 

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