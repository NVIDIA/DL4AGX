# UFF Conversion for SSD Models

Once the SSD model is trained it needs to be converted to a TRT-compatable format in order to deploy it.
The `convert_to_trt.py` Python script handles converting a frozen `.pb` to a TRT-compatable UFF file that's ready for import into a TRT engine.
This script changes the SSD graph structure to use TRT-friendly plugin operations, saves a UFF version of the network, and builds a dummy TRT engine to ensure the integrity of the UFF file.

## Usage

The first step is to have the `FlattenConcat` plugin compiled.
If you haven't already done this, the code and instructions to compile the object file are located in the `//plugins/FlattenConcatPlugin` directory.

In short run:
```sh
dazel build //plugins/FlattenConcatPlugin/...
```

The .so file will be in `//bazel-bin/plugins/FlattenConcatPlugin` and named `libflattenconcatplugin.so`

Once the custom plugin is created and an SSD-based model is trained with the frozen inference graph saved in protobuf file, the aforementioned script can now be used to create the UFF version of the network.

`convert_to_trt.py` usage:
```bash
python convert_to_trt.py -m <full path to frozen_inference_graph.pb> \
--n_classes <number of classes (not including the "background" class)> \
--input_dims <n channels> <width> <height> \
--feature_dims <list of extracted feature dimensions (see below)> \
-o <output directory> \
-fc <full path to libflattenconcat.so>
```

The list of `feature_dims` should be located using `tensorboard`.
For instance, using `tensorboard`, one can locate the ouput shape of a `BoxEncodingPredictor` node in the training graph.
![tensorboard output][tb_features]
The dimension for this feature extractor is 19 (from the 24x19x19x12).
Assuming that the other features (in order) are 10, 5, 3, 2, and 1, then the input for the `feature_dims` would be `19 10 5 3 2 1`.
More details for the `NMS` and `GridAnchor` plugins can be found in the TRT users manual.

## Models

At present, the script has only been successfully tested on the following Google object detection models:

* [MobileNet v1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
* [MobileNet v2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
* [Inception v2](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)

The Resnet18 model that ships with this application is a custom version of the Resnet models found in Google API.
This custom version required altering the MobileNet feature extractor to use Resnet rather than MobileNet as a backbone (the `fpn`-based feature extractors would require another custom plugin), making a new Resnet18 configuration, and training from scratch using the [COCO](http://cocodataset.org/#home) dataset.

The code that contains most of the logic to create the UFF file is located in `utils/model.py`.

## Requirements

* numpy
* Pillow
* pycuda
* tensorflow-gpu
* [tensorrt](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/index.html)
* [uff](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/uff/uff.html)
* [graphsurgeon](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/graphsurgeon/graphsurgeon.html)

[tb_features]: data/feature_dim_example.png "19"