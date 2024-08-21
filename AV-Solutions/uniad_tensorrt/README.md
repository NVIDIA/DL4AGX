# UniAD-TensorRT
[UniAD](https://arxiv.org/abs/2212.10156) is a Unified Autonomous Driving algorithm framework which follows end-to-end manners, taking multi-view vision input and could output planning results directly. UniAD achieves SOTA performance in many autonomous driving tasks especially on planning task. The code for UniAD can be found [here](https://github.com/OpenDriveLab/UniAD).

<img src="./assets/pipeline.png" width="1024">

(Image taken from the [UniAD repo](https://github.com/OpenDriveLab/UniAD/tree/main))

This repo demonstrate how to deploy UniAD on NVIDIA Drive Orin platform using TensorRT. Specifically, we trained a tiny version of UniAD (`UniAD-tiny`) and provide step by step workflow including model training, ONNX model export, and inference with a sample C++ application.

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

To deploy the UniAD on NVIDIA Drive Orin platform, we will need to train the UniAD, export ONNX from the trained model, build engine based on the ONNX and finally run an inference applictaion to inference the engine in C++. In this section, we will go through how to deploy the UniAD step by step.

### Project Setup <a name="proj_setup"></a>
To setup and re-create the deployment project, based on the [UniAD repo](https://github.com/OpenDriveLab/UniAD) and [BEVFormer_tensorrt repo](https://github.com/DerryHub/BEVFormer_tensorrt/tree/main), you will need to copy our prepared tool/config/helper files and apply several patches to make the project environment compatible and deployable.

Please follow the instructions at [Project Setup](./documents/proj_setup.md) to set up the project code base from multiple sources. 

### Environment Preparation <a name="env_setup"></a>
We provide a dockfile for your convenience to prepare the environment for both deployment and training under pytorch-1.12 with corresponding mmcv/mmdet/mmseg/mmdet3d.

Please follow the instructions at [Environment Preparation](./documents/env_prep.md) to setup the environment needed in the following steps.

### Data Preparation <a name="data_prepare"></a>
Except preparing dataset for training like [UniAD](https://github.com/OpenDriveLab/UniAD/blob/main/docs/DATA_PREP.md), we also provide guidance to generate ready-to-use data for ONNX exportation and C++ inference with TensorRT engine.

Please follow the instructions at [Data Preparation](./documents/data_prep.md) to prepare the data for the project.

### Model Training and Exportation <a name="uniad_tiny_train_export"></a>
For efficiency when deploying a UniAD model on DRIVE platform, we trained a tiny version of UniAD(`UniAD-tiny`), with a smaller ResNet backbone and reduced image size & bev size. To deploy UniAD Tiny, we provide steps for Pytorch to ONNX conversion.

We have included instructions to train and export UniAD tiny, please see [UniAD Tiny Traning and Exportation](./documents/tiny_train_export.md) for details.

### Inference Application <a name="inference_app"></a>
The inference application is a C++ sample application that runs inference with the TensorRT engine, it will take the raw images and dumped metadata as input, run TensorRT engine inference, and visualize the UniAD's output. 

Please follow the instructions at [Inference Application](./inference_app/README.md) to build the TensorRT engine using the ONNX and to build the inference application to run inference.


## Results <a name="results"></a>
The inference application will generate output visualizations to showcase the planning trajectory and dynamic object detection. Notice that there is no collusion correction or BBOX NMS in the post-process. The visuaizer is visualizing the raw planning trajectory and BBOX pridiction outputs.

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