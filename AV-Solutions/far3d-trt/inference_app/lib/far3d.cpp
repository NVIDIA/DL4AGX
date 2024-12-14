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

#include "far3d.hpp"

#include "NvInferRuntime.h"
#include "NvInferRuntimeBase.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include <eigen3/Eigen/Dense>
 

size_t dtypeSizeLookup(const nvinfer1::DataType dtype){
    switch(dtype)
    {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 4;
        case nvinfer1::DataType::kINT8: return 4;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 4;
        case nvinfer1::DataType::kUINT8: return 4;
        case nvinfer1::DataType::kFP8: return 4;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kBF16: return 4;
        case nvinfer1::DataType::kINT64: return 4;
        case nvinfer1::DataType::kINT4: throw std::runtime_error("We do not currently support INT4 inference");
#endif
    }       
}


namespace far3d
{


    Logger::~Logger(){}

    void Logger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept
    {
        std::cout << msg << std::endl;
    }

    // Technically more efficient to re-use the binary_buffer by not putting this into a function, but this code is much cleaner
    nvinfer1::ICudaEngine* loadEngineFromDisk(nvinfer1::IRuntime& runtime, const std::string& file_path)
    {
        std::ifstream encoder_stream(file_path, std::ios::binary);
        encoder_stream.seekg(0, std::ifstream::beg);
        const size_t begin = encoder_stream.tellg();
        encoder_stream.seekg(0, std::ifstream::end);
        const size_t end = encoder_stream.tellg();
        encoder_stream.seekg(0);
        const size_t size = end - begin;
        std::vector<char> binary_buffer(size);
        encoder_stream.read(binary_buffer.data(), size);
        nvinfer1::ICudaEngine* engine = runtime.deserializeCudaEngine(binary_buffer.data(), size);
        return engine;
    }

    Inference::Inference(const std::string& encoder_file_path, const std::string& decoder_file_path, nvinfer1::IRuntime& runtime)
    {
        this->m_encoder_engine = std::shared_ptr<nvinfer1::ICudaEngine>(loadEngineFromDisk(runtime, encoder_file_path));
        this->m_decoder_engine = std::shared_ptr<nvinfer1::ICudaEngine>(loadEngineFromDisk(runtime, decoder_file_path));

        this->setupBuffers();
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        m_upload_stream.reset(stream, &cudaStreamDestroy);
        cudaEvent_t event = nullptr;
        err = cudaEventCreate(&event);
        m_upload_done_event.reset(event, &cudaEventDestroy);
        err = cudaMalloc(reinterpret_cast<void**>(&d_prev_exists), sizeof(float));
        float initial_value = 0.0;
        err = cudaMemcpy(d_prev_exists, &initial_value, sizeof(float), cudaMemcpyHostToDevice);
    }

    void Inference::setupBuffers()
    {
        const int32_t num_encoder_io_tensors = this->m_encoder_engine->getNbIOTensors();
        this->m_encoder_executor.reset(this->m_encoder_engine->createExecutionContext());
        for(int32_t i = 0; i < num_encoder_io_tensors; ++i)
        {
            std::string name = this->m_encoder_engine->getIOTensorName(i);
            const nvinfer1::Dims dims = this->m_encoder_engine->getTensorShape(name.c_str());
            
            if(this->m_encoder_engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT)
            {
                this->m_image_input_dims = dims;
            }else {
                
                const nvinfer1::DataType dtype = this->m_encoder_engine->getTensorDataType(name.c_str());
                // currently only support float buffers
                size_t size = nv::numel(dims);
                size *= dtypeSizeLookup(dtype);
                void* ptr = nullptr;
                cudaError_t err = cudaMalloc(&ptr, size);
                m_transition_buffers[name] = std::shared_ptr<void>(ptr, &nv::cudaFreeWrapper);
                this->m_encoder_executor->setTensorAddress(name.c_str(), ptr);
            }
        }// loop encoder io tensors

        nvinfer1::Dims label_shape =this->m_decoder_engine->getTensorShape("labels");
        this->m_num_output_dets = label_shape.d[0];
        this->m_decoder_executor.reset(this->m_decoder_engine->createExecutionContext());

        const int32_t num_decoder_bindings = this->m_decoder_engine->getNbIOTensors();
        for(int i = 0; i < num_decoder_bindings; ++i)
        {
            const std::string name = this->m_decoder_engine->getIOTensorName(i);
            auto itr = m_transition_buffers.find(name);
            if(itr != m_transition_buffers.end())
            {
                this->m_decoder_executor->setInputTensorAddress(name.c_str(), itr->second.get());
            }
        }
        
        this->setupStateBuffers();
        this->setupInputBuffers();        
    }

    void Inference::setupStateBuffers()
    {
        m_memory_embedding.reshape(this->m_decoder_engine->getTensorShape("memory_embedding_out"));
        m_memory_reference_point.reshape(this->m_decoder_engine->getTensorShape("memory_reference_point_out"));
        m_memory_egopose.reshape(this->m_decoder_engine->getTensorShape("memory_egopose_out"));
        m_memory_velo.reshape(this->m_decoder_engine->getTensorShape("memory_velo_out"));
        m_memory_timestamp.reshape(this->m_decoder_engine->getTensorShape("memory_timestamp_out"));
        
        this->m_decoder_executor->setInputTensorAddress("memory_embedding", m_memory_embedding.getData());
        this->m_decoder_executor->setInputTensorAddress("memory_reference_point", m_memory_reference_point.getData());
        this->m_decoder_executor->setInputTensorAddress("memory_egopose", m_memory_egopose.getData());
        this->m_decoder_executor->setInputTensorAddress("memory_velo", m_memory_velo.getData());
        this->m_decoder_executor->setInputTensorAddress("memory_timestamp", m_memory_timestamp.getData());

        this->m_decoder_executor->setTensorAddress("memory_embedding_out", m_memory_embedding.getData()); 
        this->m_decoder_executor->setTensorAddress("memory_reference_point_out", m_memory_reference_point.getData());
        this->m_decoder_executor->setTensorAddress("memory_egopose_out", m_memory_egopose.getData());
        this->m_decoder_executor->setTensorAddress("memory_velo_out", m_memory_velo.getData());
        this->m_decoder_executor->setTensorAddress("memory_timestamp_out", m_memory_timestamp.getData());


        // initialize all states to zero
        size_t size = nv::numel(m_memory_embedding.getShape());
        cudaMemset(m_memory_embedding.getData(), 0, size * sizeof(float));

        size = nv::numel(m_memory_reference_point.getShape());
        cudaMemset(m_memory_reference_point.getData(), 0, size * sizeof(float));

        size = nv::numel(m_memory_egopose.getShape());
        cudaMemset(m_memory_egopose.getData(), 0, size * sizeof(float));

        size = nv::numel(m_memory_velo.getShape());
        cudaMemset(m_memory_velo.getData(), 0, size * sizeof(float));

        size = nv::numel(m_memory_timestamp.getShape());
        cudaMemset(m_memory_timestamp.getData(), 0, size * sizeof(float));
    }

    void Inference::setupInputBuffers()
    {
        // Lastly we setup the input buffers that are used for auxilliary data
        m_ego_pose.reshape(this->m_decoder_engine->getTensorShape("ego_pose"));
        m_ego_pose_inv.reshape(this->m_decoder_engine->getTensorShape("ego_pose_inv"));
        m_intrinsics.reshape(this->m_decoder_engine->getTensorShape("intrinsics"));
        m_extrinsics.reshape(this->m_decoder_engine->getTensorShape("extrinsics"));
        m_img2lidar.reshape(this->m_decoder_engine->getTensorShape("img2lidar"));
        m_lidar2img.reshape(this->m_decoder_engine->getTensorShape("lidar2img"));
        
        cudaMalloc(reinterpret_cast<void**>(&d_timestamp), sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&d_prev_exists), sizeof(float));
        cudaMallocHost(reinterpret_cast<void**>(&h_timestamp), sizeof(float));
        h_ego_pose_inv.reshape(m_ego_pose_inv.getShape());
        
        this->m_decoder_executor->setInputTensorAddress("timestamp", d_timestamp);
        this->m_decoder_executor->setInputTensorAddress("prev_exists", d_prev_exists);
        this->m_decoder_executor->setInputTensorAddress("ego_pose", m_ego_pose.getData());
        this->m_decoder_executor->setInputTensorAddress("ego_pose_inv", m_ego_pose_inv.getData());
        this->m_decoder_executor->setInputTensorAddress("intrinsics", m_intrinsics.getData());
        this->m_decoder_executor->setInputTensorAddress("extrinsics", m_extrinsics.getData());
        this->m_decoder_executor->setInputTensorAddress("img2lidar", m_img2lidar.getData());
        this->m_decoder_executor->setInputTensorAddress("lidar2img", m_lidar2img.getData());
    }

    bool Inference::forward(InputTensor<CHWC, GPU> image, 
                                              float timestamp,
                                              InputTensor<RC> ego_pose,
                                              InputTensor<CRC> intrinsics,
                                              InputTensor<CRC> extrinsics,
                                              InputTensor<CRC> img2lidar,
                                              InputTensor<CRC> lidar2img, 
                                              BBoxes_t& bboxes, Labels_t& labels, Scores_t& scores,
                                              cudaStream_t stream)
    {
        // We can directly do a forward pass on the image encoder from the input image buffer
        this->m_encoder_executor->setInputTensorAddress("img", image.getData());
        bool success = this->m_encoder_executor->enqueueV3(stream);
        if(!success)
        {
            return false;
        }
        // These uploads should overlap with the above encoder forward pass
        m_ego_pose.copyFrom(ego_pose, stream);
        m_intrinsics.copyFrom(intrinsics, stream);
        m_extrinsics.copyFrom(extrinsics, stream);
        m_img2lidar.copyFrom(img2lidar, stream);
        m_lidar2img.copyFrom(lidar2img, stream);
        *h_timestamp = timestamp;
        cudaError_t err = cudaMemcpyAsync(this->d_timestamp, this->h_timestamp, sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

        // Calculate ego_pose_inv
        Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> ego_pose_view(const_cast<float*>(ego_pose.getData()));
        Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> ego_pose_inv_view(this->h_ego_pose_inv.getData());
        ego_pose_inv_view = ego_pose_view.inverse();
        err = cudaMemcpyAsync(this->m_ego_pose_inv.getData(), this->h_ego_pose_inv.getData(), this->h_ego_pose_inv.numel() * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

        // There should be very few situations where we are waiting on these uploads to finish, but this is the most correct way 
        // of ensuring the data is in place before it is needed
        err = cudaEventRecord(this->m_upload_done_event.get(), m_upload_stream.get());
        err = cudaStreamWaitEvent(stream, this->m_upload_done_event.get());
        
        // Ensure output buffers are sufficiently sized
        bboxes.reshape(makeShape(this->m_num_output_dets, 7));
        labels.reshape(makeShape(this->m_num_output_dets));
        scores.reshape(makeShape(this->m_num_output_dets));

        // Set output tensor addresses
        const nvinfer1::DataType dlabels_dtype = this->m_decoder_engine->getTensorDataType("labels");

        this->m_decoder_executor->setTensorAddress("bboxes", bboxes.getData());
        this->m_decoder_executor->setTensorAddress("labels", labels.getData());
        this->m_decoder_executor->setTensorAddress("scores", scores.getData());


        success = this->m_decoder_executor->enqueueV3(stream);
        // need to ensure the upload is complete before we leave this function just to ensure the user calling code
        // doesn't do anything to those input buffers
        err = cudaEventSynchronize(this->m_upload_done_event.get());
        if(this->m_iteration_counter == 0)
        {
            // TODO future improvement would be to allocate this with pinned memory and to do this asynchronously
            // but this only happens once after the first iteration
            float prev_exists = 1.0;
            err = cudaMemcpy(d_prev_exists, &prev_exists, sizeof(float), cudaMemcpyHostToDevice);
        }
        ++this->m_iteration_counter;
        return success;
    }
}

