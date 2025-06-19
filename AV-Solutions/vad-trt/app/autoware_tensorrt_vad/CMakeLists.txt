# CMakeの最小バージョンを指定
cmake_minimum_required(VERSION 3.14)

# プロジェクト名を設定
project(autoware_tensorrt_vad)

find_package(autoware_cmake REQUIRED)
autoware_package(
    SKIP_LINT
  )

# Set C++17 standard to match Autoware Universe
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wno-deprecated-declarations)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Find additional dependencies for components
find_package(CUDA REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# TensorRTの設定
option(TRT_ROOT "$ENV{TRT_ROOT}")
if(TRT_ROOT)
    set(TENSORRT_INCLUDE_DIRS ${TRT_ROOT}/include/)
    set(TENSORRT_LIBRARY_DIRS ${TRT_ROOT}/lib/)
else()
    # デフォルトのTensorRTパス
    set(TENSORRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
    set(TENSORRT_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu)
endif()

# CUDAの設定
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# インクルードディレクトリの設定
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
)

# リンクディレクトリの設定
link_directories(
    ${CUDA_LIBRARY_DIRS}
    ${TENSORRT_LIBRARY_DIRS}
)

# メインライブラリの追加（ヘッダーファイルは除外）
ament_auto_add_library(${PROJECT_NAME}_lib SHARED
    lib/tensor.cpp
)

# ライブラリのリンク設定
target_link_libraries(${PROJECT_NAME}_lib
    nvinfer
    ${CUDA_LIBRARIES}
    cuda
    cudart
)

# コンポーネントの追加
ament_auto_add_library(${PROJECT_NAME}_component SHARED
    src/vad_node.cpp
)

target_link_libraries(${PROJECT_NAME}_component
    ${PROJECT_NAME}_lib
)

rclcpp_components_register_node(${PROJECT_NAME}_component
    PLUGIN "autoware::tensorrt_vad::VadNode"
    EXECUTABLE vad_node
)

# テスト用の関数定義
function(add_testcase filepath)
  get_filename_component(filename ${filepath} NAME)
  string(REGEX REPLACE ".cpp" "" test_name ${filename})
  ament_add_gmock(${test_name} ${filepath})
  target_include_directories(${test_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/lib
      ${CMAKE_CURRENT_SOURCE_DIR}/test
      ${CUDA_INCLUDE_DIRS}
      ${TENSORRT_INCLUDE_DIRS}
  )
  target_link_directories(${test_name} PRIVATE
      ${CUDA_LIBRARY_DIRS}
      ${TENSORRT_LIBRARY_DIRS}
  )
  target_sources(${test_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/tensor.cpp
  )
  target_link_libraries(${test_name}
      ${PROJECT_NAME}_lib
      ${CUDA_LIBRARIES}
      dl
      pthread
      nvinfer
      nvonnxparser
      nvinfer_plugin
  )
  ament_target_dependencies(${test_name} ${${PROJECT_NAME}_FOUND_BUILD_DEPENDS})
endfunction()

# テストの追加
if(BUILD_TESTING)
    # find_package(ament_lint_auto REQUIRED)
    # ament_lint_auto_find_test_dependencies()
    add_testcase(test/test_vad_integration.cpp)
endif()

ament_auto_package(
    INSTALL_TO_SHARE
        launch
        config
        test
)
