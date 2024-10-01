# Autonomous Vehicle Solutions
This folder contains samples for autonomous vehicle on NVIDIA DRIVE platform, including deployment of SOTA methods with TensorRT and inference application design. More is on the way. Please stay tuned.

## Multi-task model inference on multiple devices
[Multi-task model inference on multiple devices](./mtmi/) is to demonstrate the deployment of a multi-task network on NVIDIA Drive Orin platform using both GPU and DLA. Please refer to our webinar on [Optimizing Multi-task Model Inference for Autonomous Vehicles](https://www.nvidia.com/en-us/on-demand/session/other2024-inferenceauto/)

## StreamPETR-TensorRT
[StreamPETR-TensorRT](./streampetr-trt/) is a sample application to demonstrate the deployment of [StreamPETR](https://github.com/exiawsh/StreamPETR/tree/main) on NVIDIA Drive Orin platform using TensorRT.

## UniAD-TensorRT
[UniAD](https://arxiv.org/abs/2212.10156) is an end-to-end model for autonomous driving. [UniAD-TensorRT](./uniad-trt/) demostrates the deployment of [UniAD](https://github.com/OpenDriveLab/UniAD) on NVIDIA Drive Orin platform using TensorRT. 

## DCNv4-TensorRT
[DCNv4-TensorRT](./dcnv4-trt/) is a sample application to demonstrate the deployment and optimization of [Deformable Convolution v4 (DCNv4)](https://github.com/OpenGVLab/DCNv4) on NVIDIA Drive Orin platform using TensorRT with multiple plugin implementations.
