cmake_minimum_required(VERSION 3.12)
project(uniad LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

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

if(NOT TENSORRT_PATH)
  message( "Need: -DTENSORRT_PATH=/path/to/TensorRT")
endif()

set(TENSORRT_INCLUDE_DIRS ${TENSORRT_PATH}/include/)
set(TENSORRT_LIBRARY_DIRS ${TENSORRT_PATH}/lib/)
set(TARGET_GPU_SM 87)
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

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${TARGET_GPU_SM},code=compute_${TARGET_GPU_SM})
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

include_directories(
  include
  src
  /usr/local/cuda/include
  /usr/local/include
  ${STB_LIB_PTH}
  ${CUOSD_LIB_PTH}/src
  ${TENSORRT_INCLUDE_DIRS}
  ../dependencies/BEVFormer_tensorrt/TensorRT/common
)
link_directories(${TENSORRT_LIBRARY_DIRS})

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

file(GLOB_RECURSE PLUGIN_FILES 
  ../dependencies/BEVFormer_tensorrt/TensorRT/common/*.cpp 
  ../dependencies/BEVFormer_tensorrt/TensorRT/common/*.cu 
  ../dependencies/BEVFormer_tensorrt/TensorRT/plugin/*/*.cu 
  ../dependencies/BEVFormer_tensorrt/TensorRT/plugin/*/*.cpp
)

set(SHARED_TARGET uniad_plugin)

add_library(${SHARED_TARGET} SHARED ${PLUGIN_FILES})
target_compile_options(${SHARED_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${TARGET_GPU_SM}>)

target_link_libraries(${SHARED_TARGET} PUBLIC cudnn nvinfer cublas)
set_target_properties(${SHARED_TARGET} PROPERTIES CUDA_ARCHITECTURES "${TARGET_GPU_SM}")

install(
  TARGETS ${SHARED_TARGET}
  LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/build/
)

cuda_add_library(uniad_kernel SHARED 
  ${KERNEL_FILES}
)

target_compile_options(uniad_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${TARGET_GPU_SM}>)

target_link_libraries(uniad_kernel
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libnvinfer_plugin.so
)

set_target_properties(uniad_kernel PROPERTIES CUDA_ARCHITECTURES "${TARGET_GPU_SM}")

cuda_add_executable(uniad 
  src/main.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(uniad
  uniad_kernel
  libnvinfer.so
  libnvinfer_plugin.so
)
