#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="./dependencies/BEVFormer_tensorrt/det2trt/models"
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
