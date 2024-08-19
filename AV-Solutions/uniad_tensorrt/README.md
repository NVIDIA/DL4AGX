# UniAD-TRT
[UniAD](https://arxiv.org/abs/2212.10156) is a Unified Autonomous Driving algorithm framework which follows an end-to-end manner, taking multi view vision input and could output planning results directly. UniAD achieves SOTA performance in many autonomous driving tasks especially on planning task. The code for UniAD can be found [here](https://github.com/OpenDriveLab/UniAD).

<img src="./media/pipeline.png" width="1024">

(This image comes from the [UniAD repo](https://github.com/OpenDriveLab/UniAD/tree/main))

This application is a sample application to demostrate the deployment of UniAD on NVIDIA Drive Orin platform using TensorRT. 

## Run the project

Please see the [installation guide](./installation.md) for instruction on how to export ONNX, build engine, and run inference in C++.

This project is running with TensorRT 8.6.13.3 and tested on Orin DOS Linux and X86 Linux platforms.

## Inference results
When running the inference application, images will be generated to showcase the planning trajectory and dynamic object detection. A runtime analysis will also be provided. Here is an example on Orin DriveOS Linux 6.0.8.1, TensorRT 8.6.13.3 in FP32 mode:
```
[timer:  Inference]: 	115.24742 ms
```

Notice that there is no collusion correction or BBOX NMS in the post-process. The visuaizer is visualizing the raw planning trajectory and BBOX pridiction output.

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
