cmake_minimum_required(VERSION 3.16)
project(dcnv4_plugin LANGUAGES CXX CUDA)

option(TARGET "x86_64")
option(USE_PTX "Enable ptx kernels" "ON")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-fPIC -O2")
set(CMAKE_CXX_FLAGS "-fPIC -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 --expt-relaxed-constexpr --compiler-options -fPIC -O2 -Xptxas=-v")

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

    IF(NOT $ENV{TRT_ROOT} STREQUAL "")
        message($ENV{TRT_ROOT})
        set(TENSORRT_INCLUDE_DIRS $ENV{TRT_ROOT}/include/)
        set(TENSORRT_LIBRARY_DIRS $ENV{TRT_ROOT}/lib/)
    ELSE()
        message("DEFAULT TRTROOT")
        set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
        set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu)
    ENDIF()

    set(ARCH 87)
    set(PTX_ARCH sm_87)

    include_directories(
        ${CUDA_INCLUDE_DIRS} 
        ${CUDA_TOOLKIT_INCLUDE}
        ${TENSORRT_INCLUDE_DIRS})
    link_directories(
        /drive/drive-linux/lib-target/
        /usr/local/cuda-11.4/targets/aarch64-linux/lib/
        /usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
        ${TENSORRT_LIBRARY_DIRS})
    set(DEPS ${DEPS} nvcudla cudla cuda cudart)
    add_definitions(-DUSE_ORIN)
    find_package(CUDA REQUIRED)
else()    
    set(TENSORRT_INCLUDE_DIRS $ENV{TRT_ROOT}/include/)
    set(TENSORRT_LIBRARY_DIRS $ENV{TRT_ROOT}/lib/)

    set(ARCH 86)         # change this according to your platform
    set(PTX_ARCH sm_86)

    set(DEPS ${DEPS} cuda cudart)

    find_package(CUDA REQUIRED)
    find_package(Eigen3 REQUIRED)
    # define macro DEBUG for more verbose logging information inside plugins
    # add_definitions(-DDEBUG)
    
    include_directories(
        ${CUDA_INCLUDE_DIRS} 
        ${TENSORRT_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR})
    link_directories(
        ${TENSORRT_LIBRARY_DIRS})
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/3rd_party/stb/)

cuda_add_library(DCNv4_app_cu ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess.cu                 
                 OPTIONS -arch=sm_${ARCH})
set_target_properties(DCNv4_app_cu PROPERTIES CUDA_ARCHITECTURES $ARCH)
set_target_properties(DCNv4_app_cu PROPERTIES CUDA_STANDARD 14)
set_target_properties(DCNv4_app_cu PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${CMAKE_SOURCE_DIR}/plugins)
add_executable(DCNv4_app ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(DCNv4_app DCNv4_app_cu DCNv4_plugin DCNv4_plugin_cu nvinfer ${CUDA_LIBRARIES} ${DEPS})
