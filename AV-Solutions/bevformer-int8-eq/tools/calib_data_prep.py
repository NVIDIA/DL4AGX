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
import copy
import numpy as np
from mmcv import Config
from collections import defaultdict
import onnxruntime as ort
import os
import sys
sys.path.append("..")
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Create calibration data.")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--num_samples", default=600, type=int, help="Number of samples in data.")
    parser.add_argument(
        "--onnx_path", required=True, type=str, help="Input onnx model without Q/DQ nodes."
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
    parser.add_argument("--output", default=None, help="Path to save calibration data.")
    parser.add_argument("--verbose", action="store_true", help="If verbose, print all the debug info.")
    args = parser.parse_args()
    return args


def _add_data_to_dict(inputs_dict, key, data, verbose=False):
    if len(inputs_dict.get(key, [])) == 0:
        inputs_dict[key] = data
    else:
        inputs_dict[key] = np.concatenate((inputs_dict[key], data), axis=0)
    if verbose:
        print(f"{key} shape: {np.shape(inputs_dict[key])}")


def main():
    args = parse_args()
    config_file = args.config
    config = Config.fromfile(config_file)

    dataset = build_dataset(cfg=config.data.quant)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )

    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 1

    # Create TensorRT provider options
    assert "TensorrtExecutionProvider" in ort.get_available_providers(), "TensorrtExecutionProvider not available!"

    EP = [
        ('TensorrtExecutionProvider', {
            "device_id": 0,
            "trt_extra_plugin_lib_paths": args.trt_plugins,
        }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
    ]
    ort_session = ort.InferenceSession(args.onnx_path, sess_options=session_opts, providers=EP)
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]

    expected_inputs = list(config.input_shapes.keys())
    assert expected_inputs == input_names, \
        f"Expected inputs ({expected_inputs}) are different than ONNX inputs ({input_names})!"

    if args.verbose:
        print(f"Expected inputs: {expected_inputs}!")

    prev_bev = np.random.randn(config.bev_h_ * config.bev_w_, 1, config._dim_)
    prev_frame_info = {
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }
    inputs_dict = defaultdict(lambda: [])
    num_samples = 0
    for data in loader:
        img = data["img"][0].data[0].numpy()
        img_metas = data["img_metas"][0].data[0]

        use_prev_bev = np.array([1.0])
        if img_metas[0]["scene_token"] != prev_frame_info["scene_token"]:
            use_prev_bev = np.array([0.0])
        prev_frame_info["scene_token"] = img_metas[0]["scene_token"]
        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        if use_prev_bev[0] == 1:
            img_metas[0]["can_bus"][:3] -= prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= prev_frame_info["prev_angle"]
        else:
            img_metas[0]["can_bus"][-1] = 0
            img_metas[0]["can_bus"][:3] = 0
        can_bus = img_metas[0]["can_bus"]
        lidar2img = np.stack(img_metas[0]["lidar2img"], axis=0)

        if args.verbose:
            print("========")
        _add_data_to_dict(inputs_dict, "image", img.astype(np.float32), args.verbose)
        _add_data_to_dict(inputs_dict, "prev_bev", prev_bev.astype(np.float32), args.verbose)
        _add_data_to_dict(inputs_dict, "use_prev_bev", use_prev_bev.astype(np.float32), args.verbose)
        _add_data_to_dict(inputs_dict, "can_bus", can_bus.astype(np.float32), args.verbose)
        _add_data_to_dict(inputs_dict, "lidar2img", np.expand_dims(lidar2img, axis=0).astype(np.float32), args.verbose)

        # Execute ORT to obtain the updated 'prev_bev' 'prev_frame_info'
        inp_dict = {
            "image": img.astype(np.float32),
            "prev_bev": prev_bev.astype(np.float32),
            "use_prev_bev": use_prev_bev.astype(np.float32),
            "can_bus": can_bus.astype(np.float32),
            "lidar2img": np.expand_dims(lidar2img, axis=0).astype(np.float32)
        }
        ort_output = ort_session.run([], inp_dict)
        out_dict = dict(zip(output_names, ort_output))
        prev_bev = out_dict["bev_embed"]
        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle

        # Check if the required number of samples has been reached
        num_samples += 1
        if num_samples == args.num_samples:
            break

    # Save the dictionary to an .npz file
    output = args.output or os.path.join(config.data.quant["data_root"], f"calib_data.npz")
    print(f"Saving calibration data in {output}")
    np.savez(output, **inputs_dict)


if __name__ == "__main__":
    main()
