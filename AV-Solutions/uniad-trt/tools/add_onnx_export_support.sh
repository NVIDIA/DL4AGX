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
    "tools/export_onnx.py"
    "tools/process_metadata.py"
    "tools/uniad_export_onnx.sh"
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
