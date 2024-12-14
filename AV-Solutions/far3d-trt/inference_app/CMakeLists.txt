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

cmake_minimum_required(VERSION 3.10)
project(far3d)

option(TENSORRT_ROOT "Path to TensorRT extraction" "/data/TensorRT")
option(TARGET "x86_64")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-fPIC -g -O0")
set(CMAKE_CXX_FLAGS "-fPIC -g -O0")
set(CUOSD_LIB_PTH ./dependencies/Lidar_AI_Solution/libraries/cuOSD)

find_package (Eigen3 REQUIRED NO_MODULE)

FILE(GLOB_RECURSE far3d_src "lib/*")
FILE(GLOB_RECURSE far3d_hdr "include/*")
set(STB_LIB_PATH ./dependencies/stb)

if(${TARGET} MATCHES "orin")
    set(CMAKE_C_COMPILER   /usr/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu")
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

    set(CUDA_TOOLKIT_ROOT_DIR     /usr/local/cuda-11.4/)
    set(CUDA_TOOLKIT_TARGET_NAME  "aarch64-linux")
    set(CUDA_TOOLKIT_INCLUDE      /usr/local/cuda-11.4/targets/aarch64-linux/include)
    set(CUDA_CUDART_LIBRARY       /usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so)    
    set(CUDA_HOST_COMPILER  "/usr/bin/aarch64-linux-gnu-g++" CACHE STRING "host compiler" FORCE)

    set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
    set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu)

    set(PLATFORM_LIBRARY_DIRS 
        "/drive/drive-linux/lib-target"
        "/usr/local/cuda-11.4/targets/aarch64-linux/lib/"
        "/usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/"
    )
    set(PLATFORM_LIBRARIES cudla)

else(${TARGET} MATCHES "orin")

    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT}/include)
    set(TENSORRT_LIBRARY_DIRS ${TENSORRT_ROOT}/lib)
    set(PLATFORM_LIBRARIES "")

ENDIF(${TARGET} MATCHES "orin")

find_package(CUDA REQUIRED)

cuda_add_library(far3d SHARED "${far3d_src};${far3d_hdr}")
set_target_properties(far3d PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(far3d
    PUBLIC
        ${TENSORRT_INCLUDE_DIRS}
        "${CMAKE_CURRENT_LIST_DIR}/include"
)

target_link_directories(far3d
    PUBLIC
        ${TENSORRT_LIBRARY_DIRS}
        ${PLATFORM_LIBRARY_DIRS}
)

target_link_libraries(far3d
    nvinfer
    nvinfer_plugin
    Eigen3::Eigen
    ${CUDA_LIBRARIES}
    ${PLATFORM_LIBRARIES}
)

file(GLOB_RECURSE VISUALIZER_FILES 
  ${CUOSD_LIB_PTH}/src/*.cpp
  ${CUOSD_LIB_PTH}/src/*.cu
)

cuda_add_executable(main "src/main.cpp" ${VISUALIZER_FILES})
target_link_libraries(main
    far3d
)

target_include_directories(main
    PUBLIC
        ${CUOSD_LIB_PTH}/src
        ${STB_LIB_PATH}
)