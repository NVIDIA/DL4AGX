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

import argparse
import numpy as np
from modelopt.onnx.quantization import quantize


def parse_args():
    parser = argparse.ArgumentParser(description="Create calibration data.")
    parser.add_argument(
        "--onnx_path", required=True, type=str, help="Input onnx model without Q/DQ nodes."
    )
    parser.add_argument(
        "--calibration_data_path",
        type=str,
        help="Calibration data in npz format. If None, random data for calibration will be used.",
    )
    parser.add_argument(
        "--calibration_method",
        type=str,
        default="entropy",
        choices=["max", "entropy"],
        help=(
            "Calibration method choices for int8: {entropy (default), max}."
        ),
    )
    parser.add_argument(
        "--op_types_to_quantize",
        type=str,
        default=[],
        nargs="+",
        help="A space-separated list of node types to quantize.",
    )
    parser.add_argument(
        "--op_types_to_exclude",
        type=str,
        default=[],
        nargs="+",
        help="A space-separated list of node types to exclude from quantization.",
    )
    parser.add_argument(
        "--trt_plugins",
        type=str,
        default=None,
        help=(
            "Specifies custom TensorRT plugin library paths in .so format (compiled shared library). "
            'For multiple paths, separate them with a semicolon, i.e.: "lib_1.so;lib_2.so". '
            "If this is not None, the TensorRTExecutionProvider is invoked, so make sure that the TensorRT libraries "
            "are in the PATH or LD_LIBRARY_PATH variables."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help=(
            "Output filename to save the converted ONNX model. If None, save it in the same dir as"
            " the original ONNX model with an appropriate suffix."
        ),
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    calibration_data = None
    if args.calibration_data_path:
        # Load calibration data from .npz file
        calibration_data_npz = np.load(args.calibration_data_path, allow_pickle=True)
        # Convert the NpzFile object to a regular dictionary
        calibration_data = {key: calibration_data_npz[key] for key in calibration_data_npz.files}

    quantize(
        args.onnx_path,
        calibration_data=calibration_data,
        calibration_method=args.calibration_method,
        op_types_to_quantize=args.op_types_to_quantize,
        op_types_to_exclude=args.op_types_to_exclude,
        trt_plugins=args.trt_plugins,
        output_path=args.output_path,
    )


if __name__ == '__main__':
    main()
