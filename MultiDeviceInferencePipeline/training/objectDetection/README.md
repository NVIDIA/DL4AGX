# Object Detection Training and TRT Conversion

This section explains how to train an SSD-based object detection model using Google's object detection API and how to use the provided Python script to then export this model into a usable format for deployment with TRT. Tensorflow and TensorRT are assumed to have been installed.


Please read all the instructions and look at the contents of the `ssdConvertUFF` before selecting which models to use. Follow the steps below for training and converting model.

### 1. Download and install Google's [object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

```bash
git clone https://github.com/tensorflow/models.git
cd models
git checkout f2b702a056ba08a2f2344425f116a673a302abdd
```

In the cloned directory, git apply ssd_resnet18.patch located in `sampleDriveOS/training/objectDetection`.

```bash
git apply /path/to/driveos_sample.patch
```

### 2. Download the [COCO dataset (object detection annotations)](http://cocodataset.org/#download) and the [KITTI dataset (2D object detection)](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)

#### COCO

* [Training](http://images.cocodataset.org/zips/train2017.zip)
* [Validation](http://images.cocodataset.org/zips/val2017.zip)
* [Testing](http://images.cocodataset.org/zips/test2017.zip)
* [Train/Val Annotation](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [Testing Annotation](http://images.cocodataset.org/annotations/image_info_test2017.zip)

#### KITTI

* [All Data](http://www.cvlibs.net/download.php?file=data_object_image_2.zip)
* [Annotation](http://www.cvlibs.net/download.php?file=data_object_label_2.zip)

### 3. Use the API's dataset [conversion script](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py) to create the requisite tfrecords (be sure to enable the desired classes for the [KITTI conversion script](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_kitti_tf_record.py))

For the COCO dataset:
```bash
OD_DIRECTORY=<path to the tensorflow model directory>
TRAIN_IMAGE_DIR=<path to the directory containing the training images in JPEG format>
VAL_IMAGE_DIR=<path to the directory containing the validation images in JPEG format>
TEST_IMAGE_DIR=<path to the directory containing the testing images in JPEG format>
TRAIN_ANNOTATIONS_FILE=<location of training annotations JSON file (`instances_train2017.json`)>
VAL_ANNOTATIONS_FILE=<location of validation annotations JSON file (`instances_val2017.json`)>
TESTDEV_ANNOTATIONS_FILE=<location of testing annotations JSON file (`image_info_test-dev2017.json`)>
OUTPUT_DIR=<output directory>


python $OD_DIRECTORY/object_detection/dataset_tools/create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
```

For the KITTI dataset:

Modify the label map file (`kitti_label_map.pbtxt`) to have the appropriate number of classes (i.e. the `bicycle` class is absent in the default label map).
```bash
OD_DIRECTORY=<path to the tensorflow model directory>
DATA_DIR=<top-level directory containing the images (i.e. `data_object_image_2` and `training` should be directly under this directory)>
OUTPUT_DIR=<output directory>
LABEL_MAP_PATH="${OD_DIRECTORY}/object_detection/data/kitti_label_map.pbtxt"
VALIDATION_SET_SIZE=500
CLASSED_TO_USE="car,pedestrian,bicycle,dontcare"
# Sky, Building, Road, Sidewalk, Fence, Vegetation, Pole, Car, Sign, Pedestrian, Cyclist, Void

python $OD_DIRECTORY/object_detection/dataset_tools/create_kitti_tf_record.py \
        --data_dir="${DATA_DIR}" \
        --output_path="${OUTPUT_DIR}" \
        --label_map_path="${LABEL_MAP_PATH}" \
        --validation_set_size=$VALIDATION_SET_SIZE \
        --classes_to_use=$CLASSED_TO_USE
```

### 4. Follow the instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md) to train a selected object detection network on the COCO dataset

* highly recommend using SSD-based MobileNet, Inception, or Resnet18
* be sure to properly [configure the pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md) (start from one of the provided configurations in the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) or the template for COCO in `demo_pipeline.config`)
  * you can specify "ssd_resnet18_v1" for the feature extractor type
  * point the config file to the appropriate `<dataset>_label_map.pbtxt` located in `models/research/object_detection/data`

```bash
OD_DIRECTORY=<path to the tensorflow model directory>
PIPELINE_CONFIG_PATH=<path to the pipeline config>
MODEL_DIR=<output directory>
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
CWD=$(pwd)

python $OD_DIRECTORY/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
```

### 5. Fine-tune the trained model using the KITTI tfrecords

This involves changing the pipeline config to point to the KITTI dataset rather than the COCO dataset and possibly modifying the label map file to have the appropriate number of classes (i.e. the `bicycle` class is absent in the default label map).

E.g.:
```bash
model {
  ssd {
    num_classes: 3
...
...
  fine_tune_checkpoint: "<PATH TO COCO OR KITTI>/model.ckpt-<TRAINING ITERATION>"
  fine_tune_checkpoint_type: "detection"
}
train_input_reader {
  label_map_path: "<PATH TO>/KITTI/kitti_label_map.pbtxt"
  tf_record_input_reader {
    input_path: "<PATH TO>/KITTI/train/kitti.tfrecord"
  }
}
eval_input_reader: {
  tf_record_input_reader {
    input_path: "<PATH TO>/KITTI/val/kitti.tfrecord"
  }
  label_map_path: "<PATH TO>/KITTI/kitti_label_map.pbtxt"
  shuffle: false
  num_readers: 1
  num_epochs: 1
  sample_1_of_n_examples: 1
}
```

### 6. Export the Tensorflow inference graph

Convert the training graph to an inference graph:
```bash
OD_DIRECTORY=<path to the tensorflow model directory>
INPUT_TYPE=image_tensor
INPUT_SHAPE="-1,-1,-1,3"
PIPELINE_CONFIG_PATH=<path to the pipeline config>
TRAINED_CKPT_PREFIX=<path to model checkpoint>
EXPORT_DIR=<output directory>

python $OD_DIRECTORY/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --input_shape=${INPUT_SHAPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR} \
    --write_inference_graph=True
```

### 7. Convert the model to the UFF format using the script in the `ssdConvertUFF` folder

Follow the instructions in the `ssdConvertUFF` folder