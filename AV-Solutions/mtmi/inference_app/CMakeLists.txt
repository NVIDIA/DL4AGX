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

cmake_minimum_required(VERSION 3.14)
project(mtmiapp LANGUAGES CXX CUDA)

option(TARGET "x86_64")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-fPIC -g -O2")
set(CMAKE_CXX_FLAGS "-fPIC -g -O2")

if(${TARGET} MATCHES "orin")
    set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu")
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4/)
    set(CUDA_TOOLKIT_INCLUDE /usr/local/cuda-11.4/targets/aarch64-linux/include)
    set(CUDA_CUDART_LIBRARY /usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so)
    set(CUDA_HOST_COMPILER "/usr/bin/aarch64-linux-gnu-g++" CACHE STRING "host compiler" FORCE)

    set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
    set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu)
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
    add_definitions(-DUSE_ORIN)
    find_package(CUDA REQUIRED)
else()    
    set(TENSORRT_INCLUDE_DIRS ${TRTROOT}/include/)
    set(TENSORRT_LIBRARY_DIRS ${TRTROOT}/lib/)
    
    set(DEPS ${DEPS} cuda)
    include_directories(
        ${CUDA_INCLUDE_DIRS} 
        ${TENSORRT_INCLUDE_DIRS})
    link_directories(
        ${TENSORRT_LIBRARY_DIRS})
    find_package(CUDA REQUIRED)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
cuda_add_library(mtmicu
                 ${CMAKE_SOURCE_DIR}/kernels/kernels.cu)
add_executable(mtmiapp main.cpp 
               ${CMAKE_SOURCE_DIR}/src/lodepng.cpp
               ${CMAKE_SOURCE_DIR}/src/cudla_context.cpp)
target_link_libraries(mtmiapp mtmicu nvinfer ${CUDA_LIBRARIES} ${DEPS})
