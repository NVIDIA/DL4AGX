# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import argparse
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, batch_size, cache_file):
        super().__init__()
        self.cache_file = cache_file
        self.data = calibration_data
        self.batch_size = batch_size
        self.current_index = 0
        self.device_inputs = [None, None, None, None]
        
        # Initialize device inputs with zeros
        index = 0
        for tensor_list in calibration_data:
            tensor_shape = tensor_list[0].shape
            size = int(np.dtype(np.float32).itemsize * np.prod(tensor_shape) * self.batch_size)
            self.device_inputs[index] = cuda.mem_alloc(size)
            index+=1
    
    def set_image_batcher(self, image_batcher):
        pass

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        if self.current_index + self.batch_size > len(self.data[0]):  # Using the first tensor's data length as reference
            return None
        for idx, name in enumerate(names):
            batch = self.data[idx][self.current_index:self.current_index+self.batch_size]
            # np.copyto(self.device_inputs[idx], batch[0])
            cuda.memcpy_htod(self.device_inputs[idx], np.ascontiguousarray(batch))
            
        self.current_index += self.batch_size
        return [int(d_input) for d_input in self.device_inputs]

    def read_calibration_cache(self):
        # If cache exists, use it to skip calibration. Otherwise, perform calibration
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def read_calibration_cache_size(self):
        # Obtain the size of the calibration cache
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return len(f.read())
        return -1

def build_engine_from_onnx(onnx_file_path, cache_file, calibration_data=None, dla_core=0):
    # Create builder, network, and parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1)  # 1 = EXPLICIT_BATCH
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Load the ONNX model and parse it
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Configure the builder settings
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # Use FP16 precision if desired
    config.set_flag(trt.BuilderFlag.DIRECT_IO)
    # config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = dla_core
    config.engine_capability = trt.EngineCapability.DLA_STANDALONE

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    
    if calibration_data:
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = PythonEntropyCalibrator(calibration_data=calibration_data, batch_size=8, cache_file=cache_file)
        config.int8_calibrator = calibrator

        formats = 1 << int(trt.TensorFormat.CHW32)
        for input in inputs:
            input.allowed_formats = formats
            input.dtype = trt.DataType.INT8

        for output in outputs:
            # output.allowed_formats = 1 << int(trt.TensorFormat.DLA_LINEAR)
            # output.dtype = trt.DataType.HALF
            output.allowed_formats = formats
            output.dtype = trt.DataType.INT8
    
    # Build and return the engine
    return builder.build_engine(network, config)

def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--onnx', type=str, default="onnx_files/mtmi_seg_head.onnx",
                        help='Path to load onnx file')
    parser.add_argument('--image-path', type=str, help='Path to load images from')
    parser.add_argument('--output-path', type=str, default="engines/seg.bin",
                        help='Path to save engine to')
    parser.add_argument('--cache-path', type=str, default="calibration/calibration_cache_seg.bin",
                        help='Path to save calibration to')
    args = parser.parse_args()

    img_pth = args.image_path
    onnx_file_path = args.onnx
    feat_path = args.feat_path
    cache_path = args.cache_path
    
    # load calibration data
    calibration_data = [[], [], [], []]
    dir_names = ['left', 'right']
    count = 0
    for file_name in os.listdir(img_pth):
        npy_name = file_name[:-4] + '.npy'
        for j in range(2):
            dir_name = dir_names[j]
            for i in range(4):
                feature = np.load(os.path.join(feat_path, dir_name, str(i), npy_name))
                calibration_data[i].append(feature)
        count+=1

    print("before building engine")
    _ = build_engine_from_onnx(onnx_file_path, cache_path, calibration_data=calibration_data, dla_core=0)

if __name__ == '__main__':
    main()