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

import onnxruntime as ort
import mmcv
import numpy as np
import os
import sys
import argparse

def run_onnx_model(input_data, onnx_path):
    # Start a new ONNX Runtime inference session
    sess = ort.InferenceSession(onnx_path)

    # Assume the model has a single input and output.
    # Adjust this if your model has more inputs/outputs.
    input_name = sess.get_inputs()[0].name
    output_names = [
        sess.get_outputs()[0].name, 
        sess.get_outputs()[1].name, 
        sess.get_outputs()[2].name, 
        sess.get_outputs()[3].name]
    
    # Run the model
    result = sess.run(output_names, {input_name: input_data})

    # Return the result
    return result


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--onnx', type=str, default="onnx_files/mtmi_encoder.onnx",
                        help='Path to load onnx file')
    parser.add_argument('--image-path', type=str, default="", help='Path to load images from')
    parser.add_argument('--output-path', type=str, default="calibration/",
                        help='Path to save results to')
    args = parser.parse_args()

    onnx_path = args.onnx
    image_path = args.image_path
    if len(image_path) == 0:
        raise RuntimeError("Please provide image_path")

    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], 
        std=[58.395, 57.12, 57.375], 
        to_rgb=True)

    dir_name = ['left', 'right']

    count = 0
    for file_name in os.listdir(image_path):
        count += 1
        img_path = os.path.join(image_path, file_name)
        raw_img = mmcv.imread(img_path)
        for i in range(2):
            img = raw_img[:, i * 512: (i+2) * 512, :]
            cur_path = os.path.join(output_path, dir_name[i])
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)
            # print(img.shape)
            img_norm = mmcv.imnormalize(img.astype(np.float32), 
                                        np.array(img_norm_cfg['mean'], dtype=np.float32), 
                                        np.array(img_norm_cfg['std'], dtype=np.float32), 
                                        img_norm_cfg['to_rgb'])
            img = np.moveaxis(img_norm, -1, 0)
            img = img.reshape([1, 3, 1024, 1024])
            onnx_output = run_onnx_model(img, onnx_path)
            for j in range(4):
                feat_path = os.path.join(output_path, dir_name[i], str(j))
                if not os.path.exists(feat_path):
                    os.makedirs(feat_path)
                np.save(os.path.join(output_path, dir_name[i], str(j), file_name[:-4]+'.npy'), onnx_output[j])
                print("successfully save to ", os.path.join(output_path, dir_name[i], str(j), file_name[:-4]+'.npy'))
    print("total number :", count)

if __name__ == '__main__':
    main()
