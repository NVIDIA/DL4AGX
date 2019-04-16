# Engine Creator

## Introduction

Engine creator is a sample application designed to build TensorRT engine from UFF model. You can use the code to build the TensorRT optimized engine and run inference on test images. This app requires image preprocessing pipeline (e.g., resize, normalization), which can be built by Pipeline Creator app. For object detection model trained on [COCO](http://cocodataset.org/#home) dataset with 80 categories, the engine creatore is able to output the detection results in JSON format in accordance with COCO standard. The JSON result can be used with the Python evaluation script we provide to obtain the benchmark results. Note that for evaluation, the inference by TensorRT is integrated in [DALI](https://github.com/NVIDIA/DALI) pipeline.  

The engine creator supports fp32, fp16 and int8 precision. For int8 calibration, we use images from COCO dataset. 

## DNN Networks
### Object detection
The detectors are based on [Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd) with ResNet18 backbone. The input size to the detector is 300x300. We provide ResNet18-based pretrained model that detects the following three categories:
1. Car
2. Pedestrian
3. Cyclist 

The input blob name for the aforementioned models is "Input", and the output blob name is "NMS". In addition, we also provide the pretrained model on COCO with MobileNet-V1 and V2 backbones for quantitative evaluation purpose. Note that for object detector used in this app, a library for TensorRT plugin is required. Please refer to the instructions on [compilation](../README.MD) for details. 

### Lane degmentation
The lane segmentation model is based on [DeepLabv3](https://github.com/tensorflow/models/tree/master/research/deeplab) with ResNet18 backbone. This model is able to detect the ego-lane and is trained on [KITTI](http://www.cvlibs.net/datasets/kitti/eval_road.php) dataset. The input size to this model is 240x795. The input blob name for this model is "ImageTensor" and the output blob name is "logits/semantic/BiasAdd".

## Getting Started
### Prepare DNN model
1. Train the model and convert to UFF format. Please refer to the training recipes for details.
2. Extract the UFF models into a directory  
   e.g) \<model_dir\> = /path/to/models/

### Commandline tool
```Shell
./enginecreator
Mandatory params:
  --uffModel=<file>        UFF file
  --inputBlob=<name>       Input blob names
  --inputDim=C,H,W         Input dimensions
  --outputBlob=<name>      Output blob name
  --outputEngine           Serialized TensorRT engine file to be built         

Optional params:
  --calibFolder            Image folder for int8 calibration
  --calibJSON              Annotation JSON file for calibration images
  --calibSize              Number of calibration images (default = 500)
  --testFolder             Image folder for test (e.g. val2017)
  --testJSON               Annotation JSON file for test images (e.g. instances_val2017)
  --evalJSON               Output JSON file of detection results          
  --fp16                   Run in fp16 mode (default = false). Permits 16-bit kernels
  --int8                   Run in int8 mode (default = false).      
  --batch                  Set batch size (default = 50)
  --device                 Set GPU device (default = 0, change to 1 for iGPU)
  --useDLA                 Create engine for DLA
  --workspace              Set workspace size in gigabytes (default = 4 GB)
  --pipeline               Image preprocessing DALI pipeline, required when calibration or inference is needed 
  --plugin                 TensorRT plugin for specific models (required for SSD model, absolute path needed)
  --trtoplib               TensorRT op plugin for DALI
```

### Prepare data for int8 Calibration (Optional)
Here below we provide an example to show how to use images from [MS COCO dataset](http://cocodataset.org) for int8 calibration for the object detector.

1. Download the [dataset](http://cocodataset.org/#download). In this case we use the images from "2017 Val images". The annotations of those images are in "2017 Train/Val annotations". 
2. Extract the data into a directory, <data_dir>  
   e.g) \<data_dir\> = /data/COCO/, put the images (val2017) and the annotation JSON file (instances_val2017.json) under this folder

### Example directory structure on the target
```
/home/nvidia/
|_ createEngine
|  |_ create_engine
|_ models
   |_ detection
      |_ ssd_resnet18.uff
      |_ ssd_mobileV1.uff
      |_ ssd_mobileV2.uff
   |_ segmentation
      |_ deeplabv3_resnet18.uff
|_ data
   |_ COCO
      |_ val2017
         |_ *.jpg
      |_ instances_val2017.json
```

### How to build engine
__Note that to run engines on Xavier's iGPU, --device=1 is needed. For int8 calibration, the first time the engine is created, a calibration table file "CalibrationTableSSD" will be written to the current folder, if any setting has changed when you create another int8 engine, this file needs to be deleted__  

Build ResNet18-SSD engine in fp32 precision:
```Shell
./enginecreator --uffModel=/path/to/ssd_res18_kitti.uff --outputEngine=/path/to/ssd_resnet18_fp32.engine --inputBlob=Input --inputDim=3,300,300 --outputBlob=NMS --pipeline=/path/to/pipe_SSD --plugin=/path/to/libflattenconcatplugin.so --trtoplib=/path/to/libtrtinferop.so--device=1
```

Build ResNet18-SSD engine with int8 quantization:
```Shell
./enginecreator --uffModel=/path/to/ssd_res18_kitti.uff --outputEngine=/path/to/ssd_resnet18_int8.engine --inputBlob=Input --inputDim=3,300,300 --outputBlob=NMS --pipeline=/path/to/pipe_SSD --int8 --calibFolder=/path/to/COCO/val2017 --calibJSON=/path/to/COCO/instances_val2017.json --plugin=/path/to/libflattenconcatplugin.so --trtoplib=/path/to/libtrtinferop.so --device=1
```

Build DeepLabV3 segmentation model in fp16 precision for DLA:
Build engine:
```Shell
./enginecreator --uffModel=/path/to/model_resnet18_240x795.uff --outputEngine=/path/to/deeplabv3_resnet18_fp16.engine --inputBlob=ImageTensor --inputDim=3,240,795 --outputBlob=logits/semantic/BiasAdd --fp16 --useDLA --trtoplib=/path/to/libtrtinferop.so --device=1
```

### How to build engine and write detection results to JSON file
Build MobileNetV1-SSD engine and perform detection on COCO val 2017 set:
```Shell
./enginecreator --uffModel=/path/to/ssd_mobileV1.uff --outputEngine=/path/to/ssd_mobileV1_fp32.engine --inputBlob=Input  --inputDim=3,300,300 --outputBlob=NMS --pipeline=/path/to/pipe_SSD --testFolder=/path/to/COCO/val2017 --testJSON=/path/to/COCO/instances_val2017.json --plugin=/path/to/libflattenconcatplugin.so --evalJSON=/path/to/result.json --trtoplib=/path/to/libtrtinferop.so --device=1
```

Build MobileNetV2-SSD engine with int8 quantization and perform detection on COCO val 2017 set:
```Shell
./enginecreator --uffModel=/path/to/ssd_mobileV2.uff --outputEngine=/path/to/ssd_mobileV2_int8.engine --inputBlob=Input --inputDim=3,300,300 --outputBlob=NMS --pipeline=/path/to/pipe_SSD --int8 --calibFolder=/path/to/COCO/val2017 --calibJSON=/path/to/COCO/instances_val2017.json --testFolder=/path/to/COCO/val2017 --testJSON=/path/to/COCO/instances_val2017.json --plugin=/path/to/libflattenconcatplugin.so --evalJSON=/path/to/result.json --trtoplib=/path/to/libtrtinferop.so --device=1
```

## Object Detection Benchmark on COCO Dataset
After the detection results in JSON file are generated on target, we can send the JSON file to the host for performance evaluation.
### COCO API Installation and apply cocoeval.patch for COCO evaluation
```Shell
./build_coco_api.sh
```
If error "pycocotools/\_mask.c: No such file or directory" occurs, please install cython by "pip install cython".  
The evaluation script also requires matplotlib and python-tk packages. They can be installed by "pip install matplotlib" and "sudo apt-get install python-tk".

### How to run evaluation script
After the results are written into JSON, coco_eval.py in utils folder can be used for benchmark on COCO dataset. 

```shell
Mandatory params:
  --annotation_file=<file> -a   Full path to annotation file
  --result_file=<file>     -r   Full path to JSON file with detection results
```
To run the evaluation:
```Shell
python coco_eval.py -a instances_val2017.json -r result.json
```
### Current evaluation results
AP@IoU=0.50:0.95 on COCO 2017 validation set (5000 images)

| Network                   | Input Size     | Precision | AP                 | AP-small  | AP-medium | AP-large  |
|:-------------------------:|:--------------:|:---------:|:------------------:|:---------:|:---------:|:---------:|
| SSD(Mobilenet-V1)         | 300x300        | FP32      | 26.0               | 2.8       | 22.2      | 55.6      |
| SSD(Mobilenet-V1)         | 300x300        | FP16      | 26.0               | 2.8       | 22.2      | 55.6      |
| SSD(Mobilenet-V1)         | 300x300        | INT8      | 25.8               | 2.8       | 21.9      | 55.2      |
| SSD(Mobilenet-V2)         | 300x300        | FP32      | 27.4               | 3.7       | 22.9      | 59.0      |
| SSD(Mobilenet-V2)         | 300x300        | FP16      | 27.4               | 3.7       | 22.9      | 59.0      |
| SSD(Mobilenet-V2)         | 300x300        | INT8      | 26.8               | 3.4       | 22.4      | 58.0      |
| SSD(ResNet18)             | 300x300        | FP32      | 27.4               | 3.7       | 22.9      | 59.0      |
| SSD(ResNet18)             | 300x300        | FP16      | 27.4               | 3.7       | 22.9      | 59.0      |
| SSD(ResNet18)             | 300x300        | INT8      | 26.8               | 3.4       | 22.4      | 58.0      |