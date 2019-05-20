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
# File: DL4AGX/MultiDeviceInferencePipeline/training/objectDetection/ssdConvertUFF/utils/engine.py
# Description: Utility functions for building/saving/loading TensorRT Engines
#####################################################################################################
import sys
import os

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

from utils.model import ModelData


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """Allocates host and device buffer for TRT engine inference.

    This function is similair to the one in ../../common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.

    Args:
        engine (trt.ICudaEngine): TensorRT engine

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    binding_to_type = {"Input": np.float32, "NMS": np.float32, "NMS_1": np.int32}

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = binding_to_type[str(binding)]
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def build_engine(uff_model_path,
                 trt_logger,
                 trt_engine_datatype=trt.DataType.FLOAT,
                 batch_size=1,
                 input_shape=[3, 300, 300],
                 silent=False):
    with trt.Builder(trt_logger) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 30
        if trt_engine_datatype == trt.DataType.HALF:
            builder.fp16_mode = True
        builder.max_batch_size = batch_size

        parser.register_input(ModelData.INPUT_NAME, input_shape)
        parser.register_output("MarkOutput_0")
        parser.parse(uff_model_path, network)

        if not silent:
            print("Building TensorRT engine. This may take few minutes.")

        return builder.build_cuda_engine(network)


def save_engine(engine, engine_dest_path):
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine
