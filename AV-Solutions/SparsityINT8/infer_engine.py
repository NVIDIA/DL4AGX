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

import os
import argparse
import numpy as np
import torch
from train_utils import data_loading

import tensorrt as trt
import pycuda.driver as cuda
cuda.init()

import sys
sys.path.append("./vision/references/classification/")
try:
    import utils as utils_vision
except ImportError:
    print("Error importing pytorch's vision repository!")

TRT_DYNAMIC_DIM = -1


class HostDeviceMem(object):
    """Simple helper data class to store Host and Device memory."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine: trt.ICudaEngine, batch_size: int) -> [list, list, list]:
    """
    Function to allocate buffers and bindings for TensorRT inference.

    Args:
        engine (trt.ICudaEngine):
        batch_size (int): batch size to be used during inference.

    Returns:
        inputs (List): list of input buffers.
        outputs (List): list of output buffers.
        dbindings (List): list of device bindings.
    """
    inputs = []
    outputs = []
    dbindings = []

    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        if binding_shape[0] == TRT_DYNAMIC_DIM:  # dynamic shape
            size = batch_size * abs(trt.volume(binding_shape))
        else:
            size = abs(trt.volume(binding_shape))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings
        dbindings.append(int(device_mem))

        # Append to the appropriate list (input/output)
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, dbindings


def infer(
    engine_path: str,
    val_batches: torch.utils.data.DataLoader,
    batch_size: int = 8,
    log_file: str = "engine_accuracy.log"
) -> None:
    """
    Performs inference in TensorRT engine.

    Args:
        engine_path (str): path to the TensorRT engine.
        val_batches (torch.utils.data.DataLoader): validation dataset (batches).
        batch_size (int): batch size used for inference and dataset batch splitting.
        log_file (str): filename to save logs.

    Raises:
        RuntimeError: raised when loading images in the host fails.
    """

    def override_shape(shape: tuple) -> tuple:
        """Overrides batch dimension if dynamic."""
        if TRT_DYNAMIC_DIM in shape:
            shape = tuple(
                [batch_size if dim == TRT_DYNAMIC_DIM else dim for dim in shape]
            )
        return shape

    # Open engine as runtime
    with open(engine_path, "rb") as f, trt.Runtime(
        trt.Logger(trt.Logger.ERROR)
    ) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        ctx = cuda.Context.attach()
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, dbindings = allocate_buffers(engine, batch_size)
        ctx.detach()

        # Initiate test_accuracy
        metric_logger = utils_vision.MetricLogger(delimiter="  ")

        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            # Resolves dynamic shapes in the context
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                binding_shape = engine.get_binding_shape(binding_idx)
                if engine.binding_is_input(binding_idx):
                    binding_shape = override_shape(binding_shape)
                    context.set_binding_shape(binding_idx, binding_shape)
                    input_shape = binding_shape

            # Loop over number of steps to evaluate entire validation dataset
            for step, example in enumerate(val_batches):
                images, labels = example
                if step % 100 == 0 and step != 0:
                    print(
                        "Evaluating batch {}: {:.4f}, {:.4f}".format(
                            step, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
                        )
                    )
                try:
                    # Load images in Host (pad, flatten, and copy to page-locked buffer in Host)
                    images_padded = images.numpy()
                    labels_padded = labels
                    if images.shape[0] != batch_size:
                        # Pad images tensor so it's in the shape that pagelocked_buffer is expecting
                        pad_size = batch_size - images.shape[0]
                        padding = [images_padded[0] for _ in range(pad_size)]
                        images_padded = np.concatenate((images_padded, padding), axis=0)
                        padding_label = [labels[0] for _ in range(pad_size)]
                        labels_padded = np.concatenate((labels_padded, padding_label), axis=0)
                    data = images_padded.astype(np.float32).ravel()
                    pagelocked_buffer = inputs[0].host
                    np.copyto(pagelocked_buffer, data)
                except RuntimeError:
                    raise RuntimeError(
                        "Failed to load images in Host at step {}".format(step)
                    )

                inp = inputs[0]
                # Transfer input data from Host to Device (GPU)
                cuda.memcpy_htod(inp.device, inp.host)
                # Run inference
                context.execute_v2(dbindings)
                # Transfer predictions back to Host from GPU
                out = outputs[0]
                cuda.memcpy_dtoh(out.host, out.device)

                # Split 1-D output of length N*labels into 2-D array of (N, labels)
                batch_outs = np.array(np.split(np.array(out.host), batch_size))
                # Update test accuracy
                acc1, acc5 = utils_vision.accuracy(torch.tensor(batch_outs), torch.tensor(labels_padded), topk=(1, 5))
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

            # Print final accuracy and save to log file
            print("\n======================================\n")
            result_str = "Top-1,5 accuracy: {:.4f}, {:.4f}\n".format(
                metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
            )
            print(result_str)
            # Save logs to file
            results_dir = "/".join(engine_path.split("/")[:-1])
            with open(os.path.join(results_dir, log_file), "w") as log_file:
                log_file.write(result_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on TensorRT engines for Imagenet-based Classification models.")
    parser.add_argument("-e", "--engine", type=str, default="", help="Path to TensorRT engine")
    parser.add_argument("-d", "--data_dir", default="/media/Data/imagenet_data", type=str,
                        help="Path to directory of input images (val data).")
    parser.add_argument("-b", "--batch_size", default=1, type=int,
                        help="Number of inputs to send in parallel (up to max batch size of engine).")
    parser.add_argument("--log_file", type=str, default="engine_accuracy.log", help="Filename to save logs.")
    parser.add_argument('--val_data_size', type=int, default=None,
                        help='Indicates how much validation data should be used for accuracy eval. '
                             'If None, use all. Otherwise, use a subset.')
    # Dataloader arguments from 'vision' repo
    parser.add_argument("--cache-dataset", dest="cache_dataset", action="store_true",
                        help="Cache the datasets for quicker initialization. It also serializes the transforms")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Only test the model")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument("--interpolation", default="bilinear", type=str,
                        help="the interpolation method (default: bilinear)")
    parser.add_argument("--val-resize-size", default=256, type=int,
                        help="the resize size used for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int,
                        help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int,
                        help="the random crop size used for training (default: 224)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument("--ra-reps", default=3, type=int,
                        help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    
    args = parser.parse_args()

    utils_vision.init_distributed_mode(args)
    # distributed = args.world_size > 1
    if args.distributed:
        print("Running distributed script with world size of {}".format(args.world_size))
    else:
        print("Running script in non-distributed manner!")

    # Load the test data and pre-process input
    print("---------- Loading data ----------")
    _, _, val_batches = data_loading(
        args.data_dir, args.batch_size, args,
        train_data_size=1, test_data_size=1, val_data_size=args.val_data_size
    )

    # Perform inference
    if args.engine:
        infer(args.engine, val_batches, batch_size=args.batch_size, log_file=args.log_file)
    else:
        raise Exception("Please indicate a TRT engine via --engine.")
