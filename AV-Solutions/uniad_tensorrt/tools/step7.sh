# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.



#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="./BEVFormer_tensorrt/det2trt/models"
DEST_DIR="./UniAD/projects/mmdet3d_plugin/uniad"

# List of files to copy
FILES=(
    "functions/__init__.py"
    "functions/bev_pool_v2.py"
    "functions/grid_sampler.py"
    "functions/inverse.py"
    "functions/modulated_deformable_conv2d.py"
    "functions/multi_head_attn.py"
    "functions/multi_scale_deformable_attn.py"
    "functions/rotate.py"
    "modules/cnn/__init__.py"
    "modules/cnn/dcn.py"
    "modules/feedforward_network.py"
    "modules/multi_head_attention.py"
    "utils/__init__.py"
    "utils/onnx_ops.py"
    "utils/register.py"
    "utils/scp_layer.py"
    "utils/test_trt_ops/__init__.py"
    "utils/test_trt_ops/base_test_case.py"
    "utils/test_trt_ops/test_bev_pool_v2.py"
    "utils/test_trt_ops/test_grid_sampler.py"
    "utils/test_trt_ops/test_inverse.py"
    "utils/test_trt_ops/test_modulated_deformable_conv2d.py"
    "utils/test_trt_ops/test_multi_head_attn.py"
    "utils/test_trt_ops/test_multi_scale_deformable_attn.py"
    "utils/test_trt_ops/test_rotate.py"
    "utils/test_trt_ops/utils.py"
    "utils/trt_ops.py"
)

# Copy each file to the destination directory, preserving directory structure
for file in "${FILES[@]}"; do
    # Create the directory structure in the destination if it doesn't exist
    mkdir -p "${DEST_DIR}/$(dirname "${file}")"
    
    # Copy the file
    cp "${SOURCE_DIR}/${file}" "${DEST_DIR}/${file}"
    
    # Print the action
    echo "Copied ${file} to ${DEST_DIR}/${file}"
done

echo "All files copied successfully."
