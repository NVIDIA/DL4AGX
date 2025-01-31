# Autonomous Vehicle Solutions
This folder contains samples for autonomous vehicle on NVIDIA DRIVE platform, including deployment of SOTA methods with TensorRT and inference application design. More is on the way. Please stay tuned.

## ONNX Export Guidance for TensorRT
As AV models are being developed with more complexity, one of the major challenges in deploying such a model with TensorRT is to be able to export the model to ONNX format from training framework. [ONNX Export Guidance for TensorRT](./onnx-export-guidance/) provides useful tips with examples on how to export TensorRT-friendly ONNX models based on [TorchScript-based ONNX exporter](https://pytorch.org/docs/stable/onnx_torchscript.html).

## Sparsity in INT8
[Sparsity in INT8](./SparsityINT8/) contains the PyTorch codebase for sparsity INT8 training and TensorRT inference, demonstrating the workflow for leveraging both structured sparsity and quantization for more efficient deployment. Please refer to ["Sparsity in INT8: Training Workflow and Best Practices for NVIDIA TensorRT Acceleration"](https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/) for more details..

## Multi-task model inference on multiple devices
[Multi-task model inference on multiple devices](./mtmi/) is to demonstrate the deployment of a multi-task network on NVIDIA Drive Orin platform using both GPU and DLA. Please refer to our webinar on [Optimizing Multi-task Model Inference for Autonomous Vehicles](https://www.nvidia.com/en-us/on-demand/session/other2024-inferenceauto/)

## StreamPETR-TensorRT
[StreamPETR-TensorRT](./streampetr-trt/) is a sample application to demonstrate the deployment of [StreamPETR](https://github.com/exiawsh/StreamPETR/tree/main) on NVIDIA Drive Orin platform using TensorRT.

## UniAD-TensorRT
[UniAD](https://arxiv.org/abs/2212.10156) is an end-to-end model for autonomous driving. [UniAD-TensorRT](./uniad-trt/) demostrates the deployment of [UniAD](https://github.com/OpenDriveLab/UniAD) on NVIDIA Drive Orin platform using TensorRT. 

## DCNv4-TensorRT
[DCNv4-TensorRT](./dcnv4-trt/) is a sample application to demonstrate the deployment and optimization of [Deformable Convolution v4 (DCNv4)](https://github.com/OpenGVLab/DCNv4) on NVIDIA Drive Orin platform using TensorRT with multiple plugin implementations.

## BEVFormer: INT8 explicit quantization for TensorRT
[BEVFormer-INT8-EQ](./bevformer-int8-eq) is an end-to-end example to demonstrate the explicit quantization and deployment of [BEVFormer](https://github.com/fundamentalvision/BEVFormer) on NVIDIA GPUs using TensorRT.

## Far3D-TensorRT
[Far3D-TensorRT](./far3d-trt) is a sample application to demonstrate the deployment of [Far3D](https://github.com/megvii-research/Far3D) on the NVIDIA Drive Orin platform using TensorRT.

## Llama-3.1-8B-TensorRT-LLM
[Llama-3.1-8B-TensorRT-LLM](./Llama-3.1-8B-trtllm) is an example that walks through the setup process for deploying the Llama-3.1-8B model with NVIDIA TensorRT-LLM on the NVIDIA Drive Orin platform.

## VAD-TensorRT
[VAD-TensorRT](./vad-trt) is a sample application to demonstrate the deployment of [VADv1](https://github.com/hustvl/VAD.git) on the NVIDIA Drive Orin platform using TensorRT.

## PETRv1&v2-TensorRT
[PETRv1&v2-TensorRT](./petr-trt) is a sample application to demonstrate the deployment of [PETRv1&v2](https://github.com/megvii-research/PETR) on the NVIDIA Drive Orin platform using TensorRT.

