# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import os
import argparse
import numpy as np
np.random.seed(0)
np.set_printoptions(precision=5, suppress=True, linewidth=120)

import tensorrt as trt
import pycuda.driver as cuda
cuda.init()

logger = trt.Logger(trt.Logger.VERBOSE)

class CustomProfiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
    
    def report_layer_time(self, layer_name, ms):
        print(layer_name, ms)

class InferTrt(object):
    def __init__(self, stream=None):
        self.cuda_ctx = cuda.Device(0).retain_primary_context()

        self.cuda_ctx.push()

        self.builder = trt.Builder(logger)
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.opt = self.builder.create_optimization_profile()

        self.config = self.builder.create_builder_config()
        self.config.add_optimization_profile(self.opt)
        # self.config.max_workspace_size = 2 << 34
        self.config.builder_optimization_level = 5
        self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # self.config.set_flag(trt.BuilderFlag.FP16)  # control this
        self.stream = cuda.Stream() if stream is None else stream
        self.cuda_ctx.pop()

    def from_onnx(self, onnx_mod):
        parser = trt.OnnxParser(self.network, logger)
        result = parser.parse(onnx_mod.SerializeToString())
        if not result:
            print("failed parsing onnx")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(-1)

    def build(self):
        self.buf = self.builder.build_serialized_network(self.network, self.config)
        self._build_engine()
        
    def _build_engine(self):
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.buf)
        self.context = self.engine.create_execution_context()
        # self.context.profiler = CustomProfiler()
        self.names = []
        n_io = self.engine.num_io_tensors
        for i in range(n_io):
            self.names.append(self.engine.get_tensor_name(i))

    def write(self, path):
        with open(path, "wb") as fp:
            fp.write(self.buf)

    def read(self, path):
        with open(path, "rb") as fp:
            self.buf = fp.read()
        self._build_engine()

    def __str__(self):
        # show some meta info?
        # inspector = self.engine.create_engine_inspector()
        # print('trt_engine layer_info:\n{}'.format(
        #     inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        # ))
        n_io = self.engine.num_io_tensors
        metas = []
        for i in range(n_io):
            tname = self.engine.get_tensor_name(i)
            tshape = str(self.engine.get_tensor_shape(tname))
            tdtype = str(self.engine.get_tensor_dtype(tname))
            m = f"{tname} {tshape} {tdtype}"
            metas.append(m)
        return "\n".join(metas)

    def benchmark(self, args):
        pass

    def forward(self, args, stream=None):
        # args are torch.Tensor.data_ptr(), please always beware dtype
        self.cuda_ctx.push()
        for i in range(len(self.names)):
            self.context.set_tensor_address(self.names[i], args[i])
        handle = self.stream.handle if stream is None else stream.cuda_stream
        self.context.execute_async_v3(stream_handle=handle)
        # beware, we need explicit sync out-of-the function
        # self.stream.synchronize()
        self.cuda_ctx.pop()
        return

    def sync(self):
        self.stream.synchronize()
