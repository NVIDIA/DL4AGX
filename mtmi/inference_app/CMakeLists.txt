# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 2.8.7)
project(mtmi_inference)
find_package(CUDA REQUIRED)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the yaml-cpp package
find_package(yaml-cpp REQUIRED)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_C_COMPILER /usr/bin/gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()

set(TENSORRT_INCLUDE_DIRS /home/nvidia/workspace/boyin/TensorRT-8.6.12.3/include/)
set(TENSORRT_LIBRARY_DIRS /home/nvidia/workspace/boyin/TensorRT-8.6.12.3/lib/)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-Wno-deprecated-declarations -O0 -g")
add_compile_options(-W)
add_compile_options(-std=c++11)

set(SMS 75 80 86 87)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
      -ccbin ${CMAKE_CXX_COMPILER}
      -Xcompiler -DWIN_INTERFACE_CUSTOM
      -Xlinker -lsocket
  )
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
      -ccbin ${CMAKE_CXX_COMPILER}
      -Xcompiler -DWIN_INTERFACE_CUSTOM
      -Xcompiler -I/usr/include/
      -Xlinker -lsocket
      -Xlinker -rpath=/usr/lib/x86_64-linux-gnu/
      -Xlinker -rpath=/usr/lib/
      -Xlinker -L/usr/lib/x86_64-linux-gnu/
      -Xlinker -L/usr/lib/
  )
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_87,code=compute_87)
if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  include_directories(
      ${CUDA_INCLUDE_DIRS}
      ${TENSORRT_INCLUDE_DIRS}
      include
  )
  link_directories(
    ${TENSORRT_LIBRARY_DIRS}
    ${CUDA_LIBRARY_DIRS}
  )
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  include_directories(
      ${CUDA_INCLUDE_DIRS}
      ${TENSORRT_INCLUDE_DIRS}
      include
  )
  link_directories(
    ${TENSORRT_LIBRARY_DIRS}
    ${CUDA_LIBRARY_DIRS}
  )
endif()

file(GLOB_RECURSE BUILD_FILES
    src/*.cpp
    kernels/*.cu
)

cuda_add_executable(infer infer.cpp ${BUILD_FILES})

target_link_libraries(infer
    nvinfer
    nvparsers
    nvonnxparser
    yaml-cpp
    cudla
    cudart
    pthread
)