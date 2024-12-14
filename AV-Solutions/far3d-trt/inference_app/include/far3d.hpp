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

#ifndef FAR3D_HPP
#define FAR3D_HPP
#include "tensor.hpp"
#include "layout.hpp"

#include <NvInferRuntime.h>

#include <unordered_map>

namespace far3d
{
    using nv::Tensor;
    using nv::CPU;
    using nv::GPU;
    using nv::OwningTensor;
    using nv::makeShape;
    using nv::layout::LayoutBase;
    
    
    template<class Layout>
    using StateTensor = OwningTensor<float, Layout, GPU>;
    template<class Layout, class XPU = CPU>
    using InputTensor = Tensor<const float, Layout, XPU>;

    template<class Layout>
    using TempBuffer = StateTensor<Layout>;

    // Camera, Channel, Height, Width
    struct CCHW: LayoutBase<4>{};
    // Camera, Height, Width, Channel
    struct CHWC: LayoutBase<4>{};

    // query, channel
    struct QC: LayoutBase<2>{};
    
    // query, row, col
    // Used for representing 4x4 transformation matrix
    struct QRC: LayoutBase<3>{};

    // Camera, row, col
    struct CRC: LayoutBase<3>{};

    struct RC: LayoutBase<2>{};

    // Num, Detection, Box
    // Box expected to be 7 elements
    struct DB: LayoutBase<2>{};
    struct Det: LayoutBase<1>{};
    struct Elements: LayoutBase<1>{};


    class Logger: public nvinfer1::ILogger
    {
        public:
        ~Logger() override;
        void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
    };

    
    class Inference
    {
    public:
        static constexpr const int32_t kNUM_CAMERAS = 7;
        using BBoxes_t = OwningTensor<float, DB, GPU>;
        using Labels_t = OwningTensor<int32_t, Det, GPU>;
        using Scores_t = OwningTensor<float, Det, GPU>;

        Inference(const std::string& encoder_file_path, const std::string& decoder_file_path, nvinfer1::IRuntime& runtime);

        // Inputs:
        // Image: camera=7, channel=3, height, width
        // timestamp: 
        // ego_pose: row=4, col=4
        // intrinsics: camera=7, row=4, col=4
        // extrinsics: camera=7, row=4, col=4
        // img2lidar: camera=7, row=4, col=4
        // lidar2img: camera=7, row=4, col=4
        // In an automotive application it is highly likely that the image would already be on the GPU from the ISP
        // whereas these other inputs would likely be on the CPU.  So this is an example of an API with a
        // GPU image input tensor and CPU input tensors for all others.  Internally we own device side buffers
        // to upload to and we overlap the upload with the image encoder forward pass.
        bool forward(InputTensor<CHWC, GPU> image, 
                                              float timestamp,
                                              InputTensor<RC> ego_pose,
                                              InputTensor<CRC> intrinsics,
                                              InputTensor<CRC> extrinsics,
                                              InputTensor<CRC> img2lidar,
                                              InputTensor<CRC> lidar2img, 
                                              BBoxes_t& bboxes, Labels_t& labels, Scores_t& scores,
                                              cudaStream_t stream);
        nvinfer1::Dims getInputDims() const{return m_image_input_dims;}

    private:

        nvinfer1::Dims m_image_input_dims;
        

        void setupBuffers();
        void setupStateBuffers();
        void setupInputBuffers();

        Logger m_logger;

        int32_t m_num_output_dets = 0;
        
        
        std::shared_ptr<nvinfer1::ICudaEngine> m_encoder_engine;
        std::shared_ptr<nvinfer1::ICudaEngine> m_decoder_engine;

        std::shared_ptr<nvinfer1::IExecutionContext> m_encoder_executor;
        std::shared_ptr<nvinfer1::IExecutionContext> m_decoder_executor;

        // These buffers are used as intermediates between the encoder and the decoder
        std::unordered_map<std::string, std::shared_ptr<void>> m_transition_buffers;

        // These are device side buffers for uploading the transformer decoder model inputs
        TempBuffer<RC> m_ego_pose;
        TempBuffer<RC> m_ego_pose_inv;
        TempBuffer<CRC> m_intrinsics;
        TempBuffer<CRC> m_extrinsics;
        TempBuffer<CRC> m_img2lidar;
        TempBuffer<CRC> m_lidar2img;
        // Host side buffer for storing the inverse transform
        OwningTensor<float, RC, CPU> h_ego_pose_inv;

        // internal state tensors
        StateTensor<QC> m_memory_embedding;
        StateTensor<QC> m_memory_reference_point;
        StateTensor<QRC> m_memory_egopose;
        StateTensor<QC> m_memory_velo;
        StateTensor<QC> m_memory_timestamp;
        // output state tensors, cannot operate in place on these tensors without corrupting the internal state
        StateTensor<QC> m_memory_embedding_out;
        StateTensor<QC> m_memory_reference_point_out;
        StateTensor<QRC> m_memory_egopose_out;
        StateTensor<QC> m_memory_velo_out;
        StateTensor<QC> m_memory_timestamp_out;
        // This only changes once per instantiation of this object since for embedded 
        // inference we don't support resetting / multiple sequences
        // Thus there isn't a huge need to optimize how we set the value
        float* d_prev_exists = nullptr;
        float* d_timestamp = nullptr;
        float* h_timestamp = nullptr;
        
        std::shared_ptr<CUstream_st> m_upload_stream;
        std::shared_ptr<CUevent_st> m_upload_done_event;
        int32_t m_iteration_counter = 0;
    };
}

#endif // FAR3D_HPP