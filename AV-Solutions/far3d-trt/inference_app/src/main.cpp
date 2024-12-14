/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <NvInferPlugin.h>

#include "far3d.hpp"
#include "layout.hpp"
#include "tensor.hpp"

#include <cuosd.h>

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <dlfcn.h>

#include <utility>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using far3d::CRC;
using far3d::CCHW;
using far3d::RC;
using far3d::OwningTensor;
using nv::CPU;
using nv::GPU;
using nv::makeShape;

template<class DISK_DTYPE= float, class DEST_DTYPE>
bool loadFromBinaryFile(const std::string& path, DEST_DTYPE* dst_ptr, size_t volume)
{
    const size_t expected_disk_size = sizeof(DISK_DTYPE) * volume;
    std::ifstream file(path, std::ios::binary);
    if( file.fail() ) {
        std::cerr << path << " missing!" << std::endl;
        return false;
    }
    file.seekg(0);
    const size_t start_pos = file.tellg();

    if(std::is_same<DISK_DTYPE, DEST_DTYPE>::value)
    {
        file.read(reinterpret_cast<char*>(dst_ptr), expected_disk_size);
    }else {
        // Considered a static buffer but it's probably more expensive to do a chunked read
        // instead of a bulk read with dynamic allocation.
        std::vector<DISK_DTYPE> buffer(volume);
        file.read(reinterpret_cast<char*>(buffer.data()), expected_disk_size);
        for(size_t i = 0; i < volume; ++i)
        {
            dst_ptr[i] = static_cast<DEST_DTYPE>(buffer[i]);
        }
    }
    const size_t current_pos = file.tellg();
    file.seekg(0, std::ios_base::end);
    const size_t end_pos = file.tellg();
    if(end_pos != current_pos)
    {
        return false;
    }

    return true;
}

template<class DISK_DTYPE= float, class DEST_DTYPE>
bool saveToBinaryFile(const std::string& path, const DEST_DTYPE* src_ptr, size_t volume)
{
    const size_t write_size = sizeof(DISK_DTYPE) * volume;
    std::ofstream file(path, std::ios::binary);
    if( file.fail() ) {
        std::cerr << path << " missing!" << std::endl;
        return false;
    }

    if(std::is_same<DISK_DTYPE, DEST_DTYPE>::value)
    {
        file.write(reinterpret_cast<const char*>(src_ptr), write_size);
    }else {
        // Considered a static buffer but it's probably more expensive to do a chunked read
        // instead of a bulk read with dynamic allocation.
        std::vector<DISK_DTYPE> buffer(volume);
        for(size_t i = 0; i < volume; ++i)
        {
            buffer[i] = static_cast<DEST_DTYPE>(src_ptr[i]);
        }
        file.write(reinterpret_cast<const char*>(buffer.data()), write_size);
        
    }
    return true;
}

template<class DISK_DTYPE= float, class TENSOR_DTYPE, class LAYOUT>
bool loadTensorFromBinaryFile(const std::string& path, nv::Tensor<TENSOR_DTYPE, LAYOUT, CPU>& tensor)
{
    return loadFromBinaryFile<DISK_DTYPE>(path, tensor.getData(), tensor.numel());
}

template<class DISK_DTYPE= float, class TENSOR_DTYPE, class LAYOUT>
bool saveTensorToBinaryFile(const std::string& path, const nv::Tensor<TENSOR_DTYPE, LAYOUT, CPU>& tensor)
{
    return saveToBinaryFile<DISK_DTYPE>(path, tensor.getData(), tensor.numel());
}

enum DetectionIndex{
    X = 0,
    Y = 1,
    Z = 2,
    X_size = 3,
    Y_size = 4,
    Z_size = 5,
    YAW = 6, 
};

/*

                                          up z
                        front x            ^
                               /           |
                              /            |
                (x1, y0, z1) + ----------- + (x1, y1, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z1)+ ----------- +    + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)
*/

const Eigen::Matrix<float, 3, 9> corner_coefficients = (Eigen::Matrix<float, 9, 3>()
                         << -0.5, -0.5,  0.0, // rear right bottom
                            +0.5, -0.5,  0.0, // front right bottom
                            +0.5, +0.5,  0.0, // front left bottom
                            -0.5, +0.5,  0.0, // front right bottom
                            -0.5, -0.5, +1.0,
                            +0.5, -0.5, +1.0,
                            +0.5, +0.5, +1.0,
                            -0.5, +0.5, +1.0,
                            0.0, 0.0, 0.0 // Centroid for debugging
                            ).finished().transpose();

const std::array<std::pair<int, int>, 12> edge_indices{
    std::make_pair(0,1), // bottom rectangle
    std::make_pair(1,2),
    std::make_pair(2,3),
    std::make_pair(3,0),
    std::make_pair(4,5), // top rectangle
    std::make_pair(5,6),
    std::make_pair(6,7),
    std::make_pair(7,4),
    std::make_pair(0,4),
    std::make_pair(1,5),
    std::make_pair(2,6),
    std::make_pair(3,7)
};

void createCuboidFromDetection(const float* detection, Eigen::Matrix<float, 3, 9>& cuboid)
{
    const Eigen::Vector3f centroid(detection[DetectionIndex::X],
                                        detection[DetectionIndex::Y], 
                                        detection[DetectionIndex::Z]);

    const Eigen::Vector3f shape(detection[DetectionIndex::X_size], 
                                detection[DetectionIndex::Y_size], 
                                detection[DetectionIndex::Z_size]);
    const float yaw = detection[DetectionIndex::YAW];
    // broadcasted colwise multiplication of coefficient matrix by the shape
    cuboid = corner_coefficients.array().colwise() * shape.array();
    // Rotate by yaw
    const Eigen::AngleAxisf rotation(yaw, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
    cuboid = rotation.matrix() * cuboid;
    cuboid.colwise() += centroid;
}

bool projectCuboidToImage(const Eigen::Matrix<float, 3, 9>& cuboid, 
                                Eigen::Matrix<float, 3, 9>& uv_coordinates, 
                                const Eigen::Matrix4f& camera_matrix)
{
    Eigen::Matrix<float, 3, 9> camera_cuboid = cuboid;
    
    uv_coordinates = (camera_matrix * camera_cuboid.colwise().homogeneous()).colwise().hnormalized();
    if(uv_coordinates(2,8) < 0)
    {
        return false;
    }
    // This could probably be done via expressions to avoid uv_coordinates having 3 rows, TODO later
    uv_coordinates.array().rowwise() /= uv_coordinates.row(2).array();
    return true;
}

bool inImage(float x, float y, int width, int height)
{
    return x > 0 && x < width && y > 0 && y < height;
}

int clamp(int x, int max_val, int min_val = 0)
{
    return std::min(std::max(min_val, x), max_val);
}

void drawCuboid(const Eigen::Matrix<float, 3, 9>& uv_coordinates, int width, int height, cuOSDContext_t context)
{
    for(const std::pair<int, int> index_pair: edge_indices)
    {
        float x0 = uv_coordinates(0, index_pair.first);
        float y0 = uv_coordinates(1, index_pair.first);
        float x1 = uv_coordinates(0, index_pair.second);
        float y1 = uv_coordinates(1, index_pair.second);
        if(inImage(x0, y0, width, height) || inImage(x1, y1, width, height))
        {
            cuosd_draw_line(context, x0, y0, x1, y1, 2, {0, 255, 0, 255});
        }
    }
}

const uint32_t NUM_CAMERAS = 7;
// Called with args:
// 1. encoder engine file
// 2. decoder engine file
// 3. path input prefix file
int main(int argc, const char** argv)
{
    far3d::Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::IPluginRegistry* registry = getPluginRegistry();
    initLibNvInferPlugins(&logger,"");
    
    far3d::Inference inference(argv[1], argv[2], *runtime);

    far3d::Inference::BBoxes_t d_bboxes;
    far3d::Inference::Labels_t d_labels;
    far3d::Inference::Scores_t d_scores;

    far3d::Inference::BBoxes_t::CPU_t h_bboxes;
    far3d::Inference::Labels_t::CPU_t h_labels;
    far3d::Inference::Scores_t::CPU_t h_scores;

    far3d::OwningTensor<float, RC, CPU> ego_pose(makeShape(4,4));
    far3d::OwningTensor<float, CRC, CPU> intrinsics(makeShape(NUM_CAMERAS,4,4));
    far3d::OwningTensor<float, CRC, CPU> extrinsics(makeShape(NUM_CAMERAS,4,4));
    far3d::OwningTensor<float, CRC, CPU> img2lidar(makeShape(NUM_CAMERAS,4,4));
    far3d::OwningTensor<float, CRC, CPU> lidar2img(makeShape(NUM_CAMERAS,4,4));

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);

    std::string line;
    std::ifstream file_list(argv[3]);
    int iteration_count = 0;
    const nvinfer1::Dims input_image_dims = inference.getInputDims();
    const int32_t height = input_image_dims.d[2];
    const int32_t width = input_image_dims.d[3];

    // This buffer is used for reading the uint8_t images from disk, 
    far3d::OwningTensor<uint8_t, far3d::CHWC, CPU> h_image_read_buffer(input_image_dims);
    far3d::OwningTensor<uint8_t, far3d::CHWC, GPU> d_image_render_buffer(input_image_dims);
    far3d::OwningTensor<float, far3d::CHWC, CPU> h_image_buffer(input_image_dims);
    far3d::OwningTensor<float, far3d::CHWC, GPU> d_image_buffer(input_image_dims);
    float timestamp;
    Eigen::Matrix<float, 3, 9> cuboid;
    Eigen::Matrix<float, 3, 9> uv_coordinates;
    std::vector<std::shared_ptr<cuOSDContext>> draw_contexts;
    for(int i = 0; i < NUM_CAMERAS; ++i)
    {
        cuOSDContext_t cuosd_context = cuosd_context_create();
        draw_contexts.push_back(std::shared_ptr<cuOSDContext>(cuosd_context, &cuosd_context_destroy));
    }
    
    const size_t image_buffer_size = h_image_read_buffer.numel();
    const size_t image_stride = h_image_read_buffer.numel(1);
    
    while(std::getline(file_list, line))
    {
        std::cout << "Processing example: " <<  line << std::endl;
        loadTensorFromBinaryFile<uint8_t>(line + "img.bin", h_image_read_buffer);
        float* casted_pixels = h_image_buffer.getData();
        // TODO have the network take in uint8_t and cast in the network
        const uint8_t* read_pixels = h_image_read_buffer.getData();
        for(size_t i = 0; i < image_buffer_size; ++i)
        {
            casted_pixels[i] = static_cast<float>(read_pixels[i]);
        }
        d_image_buffer.copyFrom(h_image_buffer, stream);
        d_image_render_buffer.copyFrom(h_image_read_buffer, stream);
        loadTensorFromBinaryFile<float>(line + "intrinsics.bin", intrinsics);
        loadTensorFromBinaryFile<float>(line + "extrinsics.bin", extrinsics);
        loadTensorFromBinaryFile<float>(line + "img2lidar.bin", img2lidar);
        loadTensorFromBinaryFile<float>(line + "lidar2img.bin", lidar2img);
        loadFromBinaryFile<double>(line + "timestamp.bin", &timestamp, 1);
        loadTensorFromBinaryFile<float>(line + "ego_pose.bin", ego_pose);
        inference.forward(d_image_buffer, timestamp, ego_pose, intrinsics, extrinsics, img2lidar, lidar2img, d_bboxes, d_labels, d_scores, stream);
        h_bboxes.copyFrom(d_bboxes, stream);
        h_labels.copyFrom(d_labels, stream);
        h_scores.copyFrom(d_scores, stream);
        err = cudaStreamSynchronize(stream);
        saveTensorToBinaryFile<float>(line + "bboxes.bin", h_bboxes);
        saveTensorToBinaryFile<int32_t>(line + "labels.bin", h_labels);
        saveTensorToBinaryFile<float>(line + "scores.bin", h_scores);
        const int32_t det_stride = h_bboxes.getShape()[1];
        float* conf = h_scores.getData();
        for(int camera_index = 0; camera_index < NUM_CAMERAS; ++camera_index)
        {
            cuOSDContext_t context = draw_contexts[camera_index].get();
            // lidar2img is camera_intrinsics @ lidar2cam thus it has the transform and intrinsic projection built in
            Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> camera_intrinsics(lidar2img.getData() + 16 * camera_index);
            for(int det_index = 0; det_index < h_bboxes.getShape()[0]; ++det_index)
            {
                if(conf[det_index] > 0.20)
                {
                    createCuboidFromDetection(h_bboxes.getData() + det_stride * det_index, cuboid);
                    if(projectCuboidToImage(cuboid, uv_coordinates, camera_intrinsics))
                    {
                        drawCuboid(uv_coordinates, width, height, context);
                        cuosd_draw_point(context, uv_coordinates(0, 8), uv_coordinates(1,8), 2, {255,0 ,0, 255});
                    }
                }
            }
            uint8_t* render_buffer = d_image_render_buffer.getData() + image_stride * camera_index;
            cuosd_apply(context, render_buffer, nullptr, width, width*3, height, cuOSDImageFormat::RGB,stream, true);    
        }
                
        h_image_read_buffer.copyFrom(d_image_render_buffer, stream);
        cudaStreamSynchronize(stream);
        for(int camera_index = 0; camera_index < NUM_CAMERAS; ++camera_index)
        {
            uint8_t* render_buffer = h_image_read_buffer.getData() + image_stride * camera_index;
            std::string name = line + "image_" + std::to_string(camera_index) + ".jpg";
            stbi_write_jpg(name.c_str(), width, height, 3, render_buffer, 100);
        }
    }
    
}