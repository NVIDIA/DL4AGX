# Preprocessing Pipeline

## Introduction
Pipeline creator is a sample application to create image processing pipelines with ops such as resize and normalization. The pipeline is built using Nvidia's [DALI](https://github.com/NVIDIA/DALI) library. This app serializes the pipeline to a binary file, which can be later loaded in applications such as calibration for creating engine and inference where image processing is needed.

## Usage
### Input parameters
```Shell
./pipelinecreator
Mandatory params:
  --inputDim=C,H,W     Input dimensions 
  --outputPipeline     Serialized DALI pipeline in binary file
Optional params:
  --meanVal=X,Y,Z      Mean values for preprocessing (default = 0,0,0)
  --stdVal=X,Y,Z       Standard deviation values for preprocessing (default = 1,1,1) 
  --cpu                Pipeline in CPU mode (default = GPU mode)
```

### Examples
Build GPU pipeline for SSD-based object detection model:
```Shell
./pipelinecreator --inputDim=3,300,300 --meanVal=127.5,127.5,127.5 --stdVal=127.5,127.5,127.5 --outputPipeline=/path/to/pipe_SSD
```
Build CPU pipeline for DeepLabV3-based ego-lane segmentation model:
```Shell
./pipelinecreator --inputDim=3,165,547 --meanVal=0,0,0 --stdVal=1,1,1 --outputPipeline=/path/to/pipe_DeepLab --cpu
```