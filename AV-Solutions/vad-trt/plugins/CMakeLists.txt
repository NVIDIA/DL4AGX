# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.16)
project(bev_deploy LANGUAGES C CXX CUDA)

option(TARGET "x86_64")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-fPIC -O0 -g")
set(CMAKE_CXX_FLAGS "-fPIC -O0 -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 --expt-relaxed-constexpr --compiler-options -fPIC -O2 -Xptxas=-v")

if(${TARGET} MATCHES "aarch64")
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

    set(ARCH 87)

    include_directories(
        ${CUDA_INCLUDE_DIRS} 
        ${CUDA_TOOLKIT_INCLUDE}
        ${TENSORRT_INCLUDE_DIRS})
    link_directories(
        /drive/drive-linux/lib-target
        /usr/local/cuda-11.4/targets/aarch64-linux/lib/
        /usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
        ${TENSORRT_LIBRARY_DIRS})
    set(DEPS ${DEPS} nvcudla cudla cuda cudart)
    find_package(CUDA REQUIRED)
elseif(${TARGET} MATCHES "x86_64")
    option(TRT_ROOT "$ENV{TRT_ROOT}")
    # linux x86
    find_package(CUDA REQUIRED)
    
    set(ARCH 86)
    message(STATUS ${TRT_ROOT})
    set(TENSORRT_INCLUDE_DIRS ${TRT_ROOT}/include/)
    set(TENSORRT_LIBRARY_DIRS ${TRT_ROOT}/lib/)
    
    set(DEPS ${DEPS} cuda)
    include_directories(
        ${CUDA_INCLUDE_DIRS} 
        ${TENSORRT_INCLUDE_DIRS})
    link_directories(
        ${TENSORRT_LIBRARY_DIRS})
else()
endif()

# uncomment macro DEBUG for more verbose logging information inside plugins
# add_definitions(-DDEBUG)

# uncomment below macro to enable debug output in cub
# add_definitions(-DCUB_LOG -DCUB_DETAIL_DEBUG_ENABLE_LOG)

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/kernel ${OpenCV_INCLUDE_DIRS})

set(plugins_cu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/multi_scale_deform_attn/ms_deform_attn_kernel.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/rotate/rotateKernel.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/select_and_pad/select_and_pad_kernel.cu)

cuda_add_library(plugins_cu ${plugins_cu_srcs} OPTIONS -arch=sm_${ARCH})

set_target_properties(plugins_cu PROPERTIES CUDA_ARCHITECTURES $ARCH)
set_target_properties(plugins_cu PROPERTIES CUDA_STANDARD 14)
set_target_properties(plugins_cu PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(plugins SHARED ${CMAKE_CURRENT_SOURCE_DIR}/multi_scale_deform_attn/ms_deform_attn.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/rotate/rotate_plugin.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/select_and_pad/select_and_pad.cpp)

target_link_libraries(plugins plugins_cu nvinfer ${CUDA_LIBRARIES} ${DEPS} cublas)
