#!/bin/bash

# Check if TRT_VERSION and PLATFORM are provided
if [ -z "$TRT_VERSION" ] || [ -z "$PLATFORM" ]; then
  echo "Error: TRT_VERSION and PLATFORM must be specified."
  echo "Usage: TRT_VERSION=10.7 PLATFORM=x86_cu118 ./compile_plugins_x86.sh"
  exit 1
fi

# Set variables
TRT_PATH=/workspace/TensorRT-${TRT_VERSION}_${PLATFORM}
BUILD_DIR="./build"
OUTPUT_LIB="../lib/lib_uniad_plugins_trt${TRT_VERSION}_${PLATFORM}.so"

# Clean and prepare the build directory
echo "Cleaning and preparing the build directory..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit

# Run cmake and make
echo "Running cmake and make..."
cmake .. -DTRT_PTH="$TRT_PATH"
if [ $? -ne 0 ]; then
  echo "Error: cmake failed. Please check the TRT_PATH: $TRT_PATH"
  exit 1
fi

make -j$(nproc)
if [ $? -ne 0 ]; then
  echo "Error: make failed."
  exit 1
fi

# Move the generated library
echo "Moving the generated library..."
mkdir -p ../lib
mv libtensorrt_ops.so "$OUTPUT_LIB"

echo "Build completed successfully. Library saved to: $OUTPUT_LIB"
