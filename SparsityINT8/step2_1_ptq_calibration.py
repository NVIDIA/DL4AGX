#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This script uses the Calibrator API provided by Polygraphy to calibrate an ONNX model via PTQ and build a
  quantized TensorRT engine that runs in INT8 precision. This script also allows for sparse weights to be used.

Calibrator:
- Function: https://github.com/NVIDIA/TensorRT/blob/8e756f163f83d54389c7ff82235e57a518f6eb03/tools/Polygraphy/polygraphy/backend/trt/calibrator.py#L29
- Calibrator options: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Int8/pyInt8.html
"""

import os
from argparse import ArgumentParser
from train_utils import data_loading

import sys
sys.path.append("./vision/references/classification/")
try:
    import utils as utils_vision
except ImportError:
    print("Error importing pytorch's vision repository!")

from infer_engine import infer
from polygraphy.backend.trt import Calibrator, CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, \
    TrtRunner, SaveEngine

import onnx
import onnx_graphsurgeon as gs
from polygraphy.logger import G_LOGGER
import tensorrt as trt


ARGPARSER = ArgumentParser('This script calibrates an ONNX model and generates a calibration cache and a quantized TRT engine for deployment.')
ARGPARSER.add_argument('--onnx_path', type=str, default="./model.onnx")
ARGPARSER.add_argument('--output_dir', '-o', type=str, default='./converted',
                       help='Output directory to save the ONNX file with appropriate batch size.')
ARGPARSER.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size for calibration')
ARGPARSER.add_argument('--calibrator_type', '-c', type=str, default='entropy',
                       help='Options={entropy (trt.IInt8EntropyCalibrator2), minmax (trt.IInt8MinMaxCalibrator)}')
ARGPARSER.add_argument('--onnx_input_name', type=str, default="input.1", help='Input tensor name in ONNX file.')
ARGPARSER.add_argument("--is_dense_calibration", dest="is_dense_calibration", action="store_true",
                       help="True if we should activate Dense QAT training instead of Sparse.")
# Dataloader
ARGPARSER.add_argument('--data_dir', '-d', type=str, default='/media/Data/imagenet_data',
                       help='Directory containing the tfrecords data.')
ARGPARSER.add_argument("--test_data_size", type=int, default=None,
                       help='Indicates how much validation data should be used for accuracy evaluation.'
                            'If None, use all. Otherwise, use a subset.')
ARGPARSER.add_argument('--calib_data_size', type=int, default=None,
                       help='Indicates how much validation data should be used for calibration.'
                            'If None, use all. Otherwise, use a subset.')
# Dataloader arguments from 'vision' repo
ARGPARSER.add_argument("--cache-dataset", dest="cache_dataset", action="store_true",
                       help="Cache the datasets for quicker initialization. It also serializes the transforms",)
ARGPARSER.add_argument("--test-only", dest="test_only", action="store_true", help="Only test the model")
ARGPARSER.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
ARGPARSER.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
ARGPARSER.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
ARGPARSER.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
ARGPARSER.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
ARGPARSER.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
ARGPARSER.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
ARGPARSER.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")
ARGPARSER.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
ARGPARSER.add_argument("--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)")
ARGPARSER.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
ARGPARSER.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")


# The data loader argument to `Calibrator` can be any iterable or generator that yields `feed_dict`s.
# A `feed_dict` is just a mapping of input names to corresponding inputs.
def calib_data(val_batches, input_name):
    for iteration, (images, labels) in enumerate(val_batches):
        yield {input_name: images.numpy()}


def main(args):
    # ======== Ensure ONNX file and output dir exist =========
    assert os.path.exists(args.onnx_path), f"ONNX model {args.onnx_path} does not exist!"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ======== Create ONNX with batch size BS =========
    new_onnx_filename = args.onnx_path.replace(".onnx", f"_bs{args.batch_size}.onnx").split("/")[-1]
    new_onnx_path = os.path.join(args.output_dir, new_onnx_filename)
    graph = gs.import_onnx(onnx.load(args.onnx_path))
    input_shape = graph.inputs[0].shape
    input_shape[0] = args.batch_size

    import subprocess as sp
    sp.run(["polygraphy", "surgeon", "sanitize", args.onnx_path, "-o", new_onnx_path, "--override-input-shapes",
            "{}:{}".format(graph.inputs[0].name, str(input_shape))])

    # ======== Load data ========
    print("---------- Loading data ----------")
    _, data_loader_test, data_loader_calib = data_loading(
        args.data_dir, args.batch_size, args,
        test_data_size=args.test_data_size, val_data_size=args.calib_data_size
    )

    # ======== TensorRT Deployment ========
    # Set Calibrator
    calibration_cache_path = new_onnx_path.replace(".onnx", "_calibration.cache")
    print("CALIBRATOR = {}".format(args.calibrator_type))
    if args.calibrator_type == "entropy":
        # This is the default calibrator (BaseClass=trt.IInt8EntropyCalibrator2)
        calibrator = Calibrator(data_loader=calib_data(data_loader_calib, args.onnx_input_name),
                                cache=calibration_cache_path)
    elif args.calibrator_type == "minmax":
        calibrator = Calibrator(data_loader=calib_data(data_loader_calib, args.onnx_input_name),
                                cache=calibration_cache_path, BaseClass=trt.IInt8MinMaxCalibrator)
    else:
        raise("Calibrator of type {} not supported!".format(args.calibrator_type))

    # Build engine from ONNX model by enabling INT8 and sparsity weights, and providing the calibrator.
    print("Sparse: {}".format(not args.is_dense_calibration))
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(new_onnx_path),
        config=CreateConfig(
            int8=True,
            calibrator=calibrator,
            sparse_weights=not args.is_dense_calibration
        )
    )

    # Trigger engine saving
    engine_path = new_onnx_path.replace(".onnx", "_ptq.engine")
    build_engine = SaveEngine(build_engine, path=engine_path)

    # Calibrate engine
    #   When we activate our runner, it will calibrate and build the engine. If we want to
    #   see the logging output from TensorRT, we can temporarily increase logging verbosity:
    with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(build_engine) as runner:
        print("Calibrated engine!")

        # Infer PTQ engine and evaluate its accuracy
        log_file = engine_path.split("/")[-1].replace(".engine", "_accuracy.txt")
        infer(engine_path, data_loader_test, batch_size=args.batch_size, log_file=log_file)

        print("Inference succeeded for model {}!".format(args.onnx_path))


if __name__ == "__main__":
    args = ARGPARSER.parse_args()

    utils_vision.init_distributed_mode(args)
    if args.distributed:
        print("Running distributed script with world size of {}".format(args.world_size))
    else:
        print("Running script in non-distributed manner!")

    main(args)
