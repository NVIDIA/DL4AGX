#!/usr/bin/env python3
######################################################################################################
# Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: DL4AGX/MultiDeviceInferencePipeline/training/objectDetection/ssdConvertUFF/convert_to_trt.py
# Description: Script to convert a pb file to a trt engine (with uff in the middle)
#####################################################################################################
import sys
import os
import ctypes
import argparse
import glob

import numpy as np
import tensorrt as trt
from PIL import Image

# Utility functions
import utils.inference as inference_utils  # TRT/TF inference wrappers
import utils.model as model_utils  # UFF conversion

# Model used for inference
# MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
# MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
# MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Precision command line argument -> TRT Engine datatype
TRT_PRECISION_TO_DATATYPE = {
    # 8: trt.DataType.INT8,
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}

TRT_PRECISION_TO_LABEL = {
    # 8: 'INT8',
    16: 'HALF',
    32: 'FLOAT'
}


def main(args):
    # Loading FlattenConcat plugin library using CDLL has a side
    # effect of loading FlattenConcat plugin into internal TensorRT
    # PluginRegistry data structure. This will be needed when parsing
    # network into UFF, since some operations will need to use this plugin
    try:
        ctypes.CDLL(args.flatten_concat)
    except FileNotFoundError:
        print("Error: {}\n{}".format("Could not find {}".format(args.flatten_concat),
                                     "Make sure you have compiled FlattenConcat custom plugin layer"))
        sys.exit(1)

    model_path = args.model
    model_filename = os.path.basename(model_path)
    uff_path = os.path.join(args.output_dir, model_filename[:model_filename.rfind('.')] + '.uff')
    trt_engine_path = os.path.join(
        args.output_dir,
        model_filename[:model_filename.rfind('.')] + '{}.trt'.format(TRT_PRECISION_TO_LABEL[args.precision]))

    model_utils.model_to_uff(
        model_path,
        uff_path,
        n_classes=args.n_classes + 1,  # +1 for background
        input_dims=args.input_dims,
        feature_dims=args.feature_dims)

    # TODO: create calibrator here!

    inference_utils.TRTInference(trt_engine_path,
                                 uff_path,
                                 TRT_PRECISION_TO_DATATYPE[args.precision],
                                 input_shape=args.input_dims,
                                 batch_size=args.max_batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a TF pb file to a TRT engine.')
    parser.add_argument('-p',
                        '--precision',
                        type=int,
                        choices=[32, 16],
                        default=32,
                        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-m', '--model', help='model file')
    parser.add_argument('-o', '--output_dir', help='output directory')
    parser.add_argument('-fc', '--flatten_concat', help='path of built FlattenConcat plugin')
    # parser.add_argument('-voc', '--voc_dir', default=None,
    #     help='VOC2007 root directory (for calibration)')
    parser.add_argument('-b', '--max_batch_size', type=int, default=64, help='max TensorRT engine batch size')
    parser.add_argument('-c', '--n_classes', type=int, default=90, help='number of classes')
    parser.add_argument('-d',
                        '--input_dims',
                        type=int,
                        default=[3, 300, 300],
                        nargs=3,
                        help='channel, height, and width of input')
    parser.add_argument(
        '-f',
        '--feature_dims',
        type=int,
        default=[19, 10, 5, 3, 2, 1],
        nargs='+',
        help=
        'feature extractor dimensions (inspect training graph and look for spatial dimension of the "BoxPredictor_*/BoxEncodingPredictor/BiasAdd" nodes)'
    )

    # Parse arguments passed
    args = parser.parse_args()

    main(args)
