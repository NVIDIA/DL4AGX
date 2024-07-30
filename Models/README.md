# Models
This folder contains implementations of efficient model architecture designs with improved accuracy, faster inference speed, and better resource utilization, which make them especially suitable on edge devices such as in autonomous vehicle applications.

## DEST: Depth Estimation with Simplified Transformer
[DEST](./DEST/) employs a GPU-friendly, simplified attention block design, reducing model size and computation by over 80% while increasing accuracy and speed, validated on depth estimation and semantic segmentation tasks. 
For more details about the method, check out our spotlighted [paper](https://arxiv.org/abs/2204.13791) published at [2022 CVPR Workshop on Transformers for Vision](https://sites.google.com/view/t4v-cvpr22/home?authuser=0).

## Covolutional Self-Attention 
[Convolutional Self-Attention](./ConvSelfAttention/) uniquely identifies one-to-many feature relationships using only convolutions and simple tensor manipulations, enabling seamless operation in TensorRT’s restricted mode and making it ideal for safety-critical autonomous vehicle applications. 
Please refer to our [blogpost](https://developer.nvidia.com/blog/emulating-the-attention-mechanism-in-transformer-models-with-a-fully-convolutional-network/) for more details. 

## ReduceFormer
[ReduceFormer](./ReduceFormer/) simplifies transformer architectures for vision tasks by using reduction and element-wise multiplication, enhancing inference performance and making it ideal for edge devices and high-throughput cloud computing.
For more details about ReduceFormer, please refer to our spotlighted [paper](https://arxiv.org/abs/2406.07488) published at [2024 CVPR Workshop on Transformers for Vision](https://sites.google.com/view/t4v-cvpr24).

## Swin-Free
[Swin-Free](./SwinFree/) uses size-varying windows across stages, instead of shifting windows, to achieve cross-connection among local windows. With this simple design change, Swin-Free runs faster than the Swin Transformer at inference with better accuracy. For detail, please refer to [Swin-Free: Achieving Better Cross-Window Attention and Efficiency with Size-varying Window](https://arxiv.org/abs/2306.13776).