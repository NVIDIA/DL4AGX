# CMakeの最小バージョンを指定
cmake_minimum_required(VERSION 3.16)

# プロジェクト名を設定
project(VAD_TESTS CXX)

# C++標準を設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDAとTensorRTの最小限の設定
find_package(CUDA REQUIRED)

# YAML-CPPの設定
find_package(yaml-cpp REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(autoware_perception_msgs REQUIRED)
find_package(autoware_planning_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRTのパスをシステムパスに直接設定
# findコマンドの結果から、ライブラリは標準のシステムディレクトリにインストールされていることが判明
set(TENSORRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TENSORRT_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu)
message(STATUS "Using system-installed TensorRT from: ${TENSORRT_LIBRARY_DIRS}")


# テストターゲットを作成
add_executable(
  vad_tests
  test_vad_model.cpp
  test_vad_interface.cpp
  test_vad_integration.cpp
  ../lib/vad_interface.cpp
  ../lib/vad_model.cpp
)

# ターゲットごとにインクルードディレクトリを絶対パスで指定
get_filename_component(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib ABSOLUTE)
target_include_directories(vad_tests PRIVATE
    ${LIB_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${tf2_msgs_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${geometry_msgs_INCLUDE_DIRS}
    ${rclcpp_INCLUDE_DIRS}
    ${nav_msgs_INCLUDE_DIRS}
    ${tf2_ros_INCLUDE_DIRS}
    ${tf2_eigen_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)

# ターゲットごとにリンクディレクトリを指定
target_link_directories(vad_tests PRIVATE
    ${CUDA_LIBRARY_DIRS}
    ${TENSORRT_LIBRARY_DIRS}
)

# コンパイル対象のソースファイルを追加
if(EXISTS "${LIB_DIR}/tensor.cpp")
    target_sources(vad_tests PRIVATE "${LIB_DIR}/tensor.cpp")
    message(STATUS "Added ${LIB_DIR}/tensor.cpp to sources.")
else()
    message(WARNING "tensor.cpp not found. Some functionalities might be unavailable.")
endif()

# Google Test & Mock を探してリンクする
find_package(GTest REQUIRED)

# GTestとgmockをリンクする関数
function(find_and_link_gmock_to_target target)
    # モダンなCMakeターゲットが存在するか試す
    if(TARGET GTest::gmock)
        message(STATUS "Found and linked modern GTest::gmock, GTest::gtest, and GTest::gtest_main targets.")
        target_link_libraries(${target} GTest::gmock GTest::gtest GTest::gtest_main)
    else()
        # 古いバージョン用のフォールバック
        message(STATUS "GTest::gmock target not found. Falling back to find_library.")
        find_library(GMOCK_LIBRARY NAMES gmock)
        if(GMOCK_LIBRARY)
            message(STATUS "Found gmock library: ${GMOCK_LIBRARY}")
            # 古いシステムではgtest_mainを明示的に探す必要がある場合がある
            find_library(GTEST_MAIN_LIBRARY NAMES gtest_main)
            if(GTEST_MAIN_LIBRARY)
                target_link_libraries(${target} ${GMOCK_LIBRARY} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARY})
            else()
                target_link_libraries(${target} ${GMOCK_LIBRARY} ${GTEST_LIBRARIES})
            endif()
        else()
            message(FATAL_ERROR "Could not find gmock library. Please install libgmock-dev.")
        endif()
    endif()
endfunction()

find_and_link_gmock_to_target(vad_tests)


ament_target_dependencies(vad_tests
  rclcpp
  tf2
  tf2_ros
  tf2_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

# 必要なライブラリをすべてリンク
# キーワード付き（PRIVATE）で統一
target_link_libraries(vad_tests
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    dl
    pthread
    nvinfer
    nvonnxparser
    nvinfer_plugin
    yaml-cpp
)

# GoogleTestの自動検出を有効化
include(GoogleTest)
gtest_discover_tests(vad_tests)

# カスタムテストターゲット
add_custom_target(run_tests
    COMMAND ./vad_tests
    DEPENDS vad_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running VadModel tests"
)

# ヘルプ用のカスタムターゲット
add_custom_target(test_help
    COMMAND ${CMAKE_COMMAND} -E echo "Available test targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  make vad_tests     - Build tests"
    COMMAND ${CMAKE_COMMAND} -E echo "  make run_tests     - Build and run tests"
    COMMAND ${CMAKE_COMMAND} -E echo "  ctest              - Run CTest"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Direct execution:"
    COMMAND ${CMAKE_COMMAND} -E echo "  ./vad_tests        - Run tests directly"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Recommended build process:"
    COMMAND ${CMAKE_COMMAND} -E echo "  mkdir build"
    COMMAND ${CMAKE_COMMAND} -E echo "  cd build"
    COMMAND ${CMAKE_COMMAND} -E echo "  cmake .. -DTRT_ROOT=/path/to/tensorrt"
    COMMAND ${CMAKE_COMMAND} -E echo "  make vad_tests"
    COMMAND ${CMAKE_COMMAND} -E echo "  ./vad_tests"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Note: -DTARGET is not required for tests"
    COMMAND ${CMAKE_COMMAND} -E echo "      tests use standard CUDA detection"
    COMMENT "VadModel test help"
)

# テスト実行設定
enable_testing()
add_test(NAME VadModelTests COMMAND vad_tests)

target_compile_definitions(vad_tests PRIVATE UNIT_TEST)
