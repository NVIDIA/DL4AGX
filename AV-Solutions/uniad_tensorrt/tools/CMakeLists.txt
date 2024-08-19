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



cmake_minimum_required(VERSION 3.12)

project(tensorrt_ops LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_C_COMPILER /usr/bin/gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()

set(TENSORRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu/)
set(TENSORRT_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu/)
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
      -Xcompiler -I/usr/aarch64-linux-gnu/include/
      -Xlinker -lsocket
      -Xlinker -rpath=/usr/lib/aarch64-linux-gnu/
      -Xlinker -rpath=/usr/aarch64-linux-gnu/lib/
      -Xlinker -L/usr/lib/aarch64-linux-gnu/
      -Xlinker -L/usr/aarch64-linux-gnu/lib/
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
    /usr/lib/aarch64-linux-gnu
    /usr/aarch64-linux-gnu/lib/
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
    /usr/lib/x86_64-linux-gnu/
    /usr/lib/
  )
endif()

execute_process(COMMAND nvcc -lcuda ${CMAKE_CURRENT_SOURCE_DIR}/cudaComputeVersion.cu -o /tmp/cudaComputeVersion)
execute_process(COMMAND /tmp/cudaComputeVersion
        RESULT_VARIABLE CUDA_RETURN_CODE
        OUTPUT_VARIABLE ARCH)
execute_process(COMMAND rm /tmp/cudaComputeVersion)

if(NOT CMAKE_TENSORRT_PATH)
    message( "Need: -DCMAKE_TENSORRT_PATH=/path/to/TensorRT")

endif()

include_directories(common)
include_directories(/usr/local/cuda/include)
include_directories(/usr/local/include)

message(STATUS "Linking TensorRT directory: ${CMAKE_TENSORRT_PATH}")
link_directories(${CMAKE_TENSORRT_PATH}/lib)
include_directories(${CMAKE_TENSORRT_PATH}/include)

file(GLOB_RECURSE TENSORRT_OPS_SRCS common/*.cpp common/*.cu plugin/*/*.cu plugin/*/*.cpp)

set(SHARED_TARGET tensorrt_ops)
add_library(${SHARED_TARGET} SHARED ${TENSORRT_OPS_SRCS})
target_compile_options(${SHARED_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${ARCH}>)

target_link_libraries(${SHARED_TARGET} PUBLIC cudnn nvinfer cublas)
set_target_properties(${SHARED_TARGET} PROPERTIES CUDA_ARCHITECTURES "${ARCH}")

install(
        TARGETS ${SHARED_TARGET}
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
)
