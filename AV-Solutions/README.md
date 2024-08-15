# Autonomous Vehicle Solutions
This folder contains samples for autonomous vehicle on NVIDIA DRIVE platform, including deployment of SOTA methods with TensorRT and inference application design. More is on the way. Please stay tuned.

## Multi-task model inference on multiple devices
[Multi-task model inference on multiple devices](./mtmi/) is to demostrate the deployment of a multi-task network on NVIDIA Drive Orin platform using both GPU and DLA. Please refer to our webinar on [Optimizing Multi-task Model Inference for Autonomous Vehicles](https://www.nvidia.com/en-us/on-demand/session/other2024-inferenceauto/)

## StreamPETR-TensorRT
[StreamPETR-TensorRT](./streampetr-trt/) is a sample application to demostrate the deployment of [StreamPETR](https://github.com/exiawsh/StreamPETR/tree/main) on NVIDIA Drive Orin platform using TensorRT. 

## UniAD-TensorRT
[UniAD](https://arxiv.org/abs/2212.10156) is a Unified Autonomous Driving algorithm framework which follows an end-to-end manner, taking multi view vision input and could output planning results directly. Unid achieves SOTA performance in many autonomous driving tasks especially on planning task. [UniAD-TensorRT](./uniad-trt/) is a sample application to demostrate the deployment of [UniAD](https://github.com/OpenDriveLab/UniAD) on NVIDIA Drive Orin platform using TensorRT. 
