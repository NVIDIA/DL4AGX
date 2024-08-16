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
SOURCE_DIR="."
DEST_DIR="./UniAD"

# List of files to copy
FILES=(
    "projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py"
    "projects/configs/stage2_e2e/tiny_imgx0.25_e2e_dump_trt_input.py"
    "projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py"
    "projects/configs/stage1_track_map/tiny_imgx0.25_track_map.py"
    "README.md"
    "tools/deploy.py"
    "tools/process_metadata.py"
    "tools/run_trtexec.sh"
    "tools/uniad_deploy.sh"
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
