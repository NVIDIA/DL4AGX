#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# This file is modified from "BEVFormer_tensorrt/tools/bevformer/onnx2trt.py":
# - Enabled the use of calibration_data.npz (the same data used by ModelOpt for explicit quantization).
# - Greatly simplified the Calibrator class and script in general.
#
# Copyright (c) 2023 Derry Lin (DerryHub). All rights reserved.
# Apache-2.0 [see LICENSE for details]
#

import pycuda.autoinit
import argparse
import numpy as np
from mmcv import Config
import onnx
from torch.utils.data import Dataset
import os
import sys
sys.path.append(".")
from det2trt.quantization import get_calibrator
from det2trt.convert import build_engine
from third_party.bev_mmdet3d.datasets.builder import build_dataloader
from modelopt.onnx.utils import get_input_names, get_input_shapes


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument("config", help="config file path")
    parser.add_argument(
        "--onnx_path", required=True, type=str, help="Input onnx model without Q/DQ nodes."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--max_workspace_size", type=int, default=1)
    parser.add_argument(
        "--calibration_data_path",
        type=str,
        default="/media/Data/nuScenes_v1.0/data_calib/calib_data.npz",
        help="Calibration data in npz format. If None, random data for calibration will be used.",
    )
    parser.add_argument(
        "--calibrator", type=str, default=None, help="[legacy, entropy, minmax]"
    )
    # parser.add_argument("--plugin", default="/workspace/BEVFormer_tensorrt/TensorRT/lib/libtensorrt_ops.so")
    parser.add_argument(
        "--trt_plugins",
        type=str,
        default="/workspace/BEVFormer_tensorrt/TensorRT/lib/libtensorrt_ops.so",
        help=(
            "Specifies custom TensorRT plugin library paths in .so format (compiled shared library). "
            'For multiple paths, separate them with a semicolon, i.e.: "lib_1.so;lib_2.so". '
            "If this is not None, the TensorRTExecutionProvider is invoked, so make sure that the TensorRT libraries "
            "are in the PATH or LD_LIBRARY_PATH variables."
        ),
    )
    parser.add_argument("--output", default=None, help="Path to save calibrated TensorRT engine.")
    parser.add_argument("--verbose", action="store_true", help="If verbose, print all the debug info.")
    args = parser.parse_args()
    return args


class DictDataset(Dataset):
    def __init__(self, data_dict, onnx_path):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.length = len(next(iter(data_dict.values())))  # Assumes all arrays have the same length
        onnx_model = onnx.load(onnx_path)
        self.input_names = get_input_names(onnx_model)
        self.input_shapes = get_input_shapes(onnx_model)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Create list of model inputs with appropriate batch size for the input iterator
        n_itr = int(
            self.data_dict[self.input_names[0]].shape[0] / max(1, self.input_shapes[self.input_names[0]][0])
        )
        calibration_data_list = [{}] * n_itr
        for input_name in self.input_names:
            for idx, calib_data in enumerate(
                    np.array_split(self.data_dict[input_name], n_itr, axis=0)
            ):
                calibration_data_list[idx][input_name] = calib_data

        return {key: calibration_data_list[index][key] for key in self.keys}


def main():
    args = parse_args()
    config_file = args.config
    config = Config.fromfile(config_file)

    calibration_data = None
    if args.calibration_data_path:
        # Load calibration data from .npz file
        calibration_data_npz = np.load(args.calibration_data_path, allow_pickle=True)
        # Convert the NpzFile object to a regular dictionary
        calibration_data = {key: calibration_data_npz[key] for key in calibration_data_npz.files}

    dataset = DictDataset(calibration_data, args.onnx_path)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )

    class Calibrator(get_calibrator(args.calibrator)):
        def __init__(self, *args, **kwargs):
            super(Calibrator, self).__init__(*args, **kwargs)

        def decode_data(self, data):
            for name in self.names:
                if name in ["image", "prev_bev", "use_prev_bev", "can_bus", "lidar2img"]:
                    tensor = data[name].numpy().reshape(-1).astype(np.float32)
                    assert self.host_device_mem_dic[name].host.nbytes == tensor.nbytes, \
                        f"Bytes difference: {self.host_device_mem_dic[name].host.nbytes} vs {tensor.nbytes}"
                    self.host_device_mem_dic[name].host = tensor
                else:
                    raise RuntimeError(f"Cannot find input name {name}.")

    calibrator = Calibrator(config, loader, dataset.length)

    dynamic_input = None
    if config.dynamic_input.dynamic_input:
        dynamic_input = [
            config.dynamic_input.min,
            config.dynamic_input.reg,
            config.dynamic_input.max,
        ]
    output = args.output or os.path.join(args.onnx_path.replace(".onnx", "_IQ_PTQ.engine"))
    build_engine(
        args.onnx_path,
        output,
        dynamic_input=dynamic_input,
        int8=args.int8,
        fp16=args.fp16,
        max_workspace_size=args.max_workspace_size,
        calibrator=calibrator,
    )


if __name__ == "__main__":
    main()
