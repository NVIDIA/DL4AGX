#  SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#  http://www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

cmake_minimum_required(VERSION 3.12)
project(uniad LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

find_package(CUDA REQUIRED)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)

# change the TensorRT root
set(TENSORRT_INCLUDE_DIRS <path_to_TRT>/include/)
set(TENSORRT_LIBRARY_DIRS <path_to_TRT>/lib/)
# change the TensorRT root
# change the computation capability (for cuOSD)
set(TARGET_GPU_SM <GPU_arch>)
# change the computation capability (for cuOSD)
# change the path to the lib folder for TensorRT plugin
set(TENSORRT_PLUGIN_LIB_PTH <path_to_TRT_plugin_lib>)
# change the path to the lib folder for TensorRT plugin
set(STB_LIB_PTH ./dependencies/stb)
set(CUOSD_LIB_PTH ./dependencies/cuOSD)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG "-Wno-deprecated-declarations -O0 -g -DENABLE_TEXT_BACKEND_STB")
add_compile_options(-W)
add_compile_options(-std=c++11)

set(SMS 75 80 86 87)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

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

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${TARGET_GPU_SM},code=compute_${TARGET_GPU_SM})
include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
  include/
  src/
)
link_directories(
  ${TENSORRT_LIBRARY_DIRS}
  ${CUDA_LIBRARY_DIRS}
  /usr/lib/x86_64-linux-gnu/
  /usr/lib/
)

if(NOT CMAKE_TENSORRT_PATH)
  message( "Need: -DCMAKE_TENSORRT_PATH=/path/to/TensorRT")
endif()

include_directories(
  include
  src
  /usr/local/cuda/include
  /usr/local/include
  ${STB_LIB_PTH}
  ${CUOSD_LIB_PTH}/src
)
link_directories(
  ${CMAKE_TENSORRT_PATH}/lib
  ${TENSORRT_PLUGIN_LIB_PTH}
  )
include_directories(${CMAKE_TENSORRT_PATH}/include)

file(GLOB_RECURSE KERNEL_FILES 
  src/uniad.cpp
  src/tensorrt.cpp
  src/tensor.cu
  src/pre_process.cu
)

file(GLOB_RECURSE VISUALIZE_FILES 
  ${CUOSD_LIB_PTH}/src/*.cpp
  ${CUOSD_LIB_PTH}/src/*.cu
  src/visualize.cu
)

cuda_add_library(uniad_kernel SHARED 
  ${KERNEL_FILES}
)

target_compile_options(uniad_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${ARCH}>)

target_link_libraries(uniad_kernel
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libnvinfer_plugin.so
  libtensorrt_ops.so
)

set_target_properties(uniad_kernel PROPERTIES CUDA_ARCHITECTURES "${ARCH}")

cuda_add_executable(uniad 
  src/main.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(uniad
  uniad_kernel
  libnvinfer.so
  libnvinfer_plugin.so
  libtensorrt_ops.so
)
