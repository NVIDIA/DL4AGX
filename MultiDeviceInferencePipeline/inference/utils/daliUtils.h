/***************************************************************************************************
 * Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Project: MultiDeviceInferencePipeline > Inference 
 * 
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/utils/daliUtils.h
 * 
 * Description: Utility functions for working with DALI
 ***************************************************************************************************/
#pragma once
#include "common/macros.h"
#include "dali/core/common.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/util/image.h"
#include "dali/util/user_stream.h"

#include <fstream>
#include <string>
#include <vector>

namespace multideviceinferencepipeline
{
namespace inference
{
namespace utils
{
inline void readSerializedFileToString(std::string filename, std::string& data)
{
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good())
    {
        std::stringstream buffer;
        buffer << file.rdbuf();
        data = buffer.str();
        file.close();
    }
    else
    {
        std::cerr << "File " << filename << " not valid" << std::endl;
        return;
    }
}

template <typename T>
inline void CPUTensorListToRawData(dali::TensorList<dali::CPUBackend>* tl, std::vector<T>* data)
{
    memcpy(data->data(), tl->raw_mutable_data(), sizeof(T) * data->size());
}

template <typename T>
inline void GPUTensorListToRawData(dali::TensorList<dali::GPUBackend>* tl, std::vector<T>* data, bool async = false, cudaStream_t s = NULL)
{
    if (async)
    {
        CHECK(cudaMemcpyAsync(data->data(), tl->raw_mutable_data(), sizeof(T) * data->size(), cudaMemcpyDeviceToHost, s));
    }
    else
    {
        CHECK(cudaMemcpy(data->data(), tl->raw_mutable_data(), sizeof(T) * data->size(), cudaMemcpyDeviceToHost));
    }
}

inline void makeJPEGBatch(std::vector<std::string>& jpegNames, dali::TensorList<dali::CPUBackend>* tl, int n)
{
    dali::ImgSetDescr jpegs;
    dali::LoadImages(jpegNames, &jpegs);

    const auto nImgs = jpegs.nImages();
    std::vector<std::vector<::dali::Index> > shape(n);
    for (int i = 0; i < n; i++)
    {
        shape[i] = {jpegs.sizes_[i % nImgs]};
    }

    tl->template mutable_data<dali::uint8>();
    tl->Resize(shape);
    for (int i = 0; i < n; i++)
    {
        memcpy(tl->template mutable_tensor<dali::uint8>(i),
               jpegs.data_[i % nImgs], jpegs.sizes_[i % nImgs]);
    }
}

inline void duplicateCPUTensorLists(std::vector<dali::TensorList<dali::CPUBackend>*>& srcs, std::vector<dali::TensorList<dali::CPUBackend>*>& copies)
{
    copies.reserve(srcs.size());
    for (size_t i = 0; i < srcs.size(); i++)
    {
        copies.emplace_back(new dali::TensorList<dali::CPUBackend>());
    }

    for (size_t i = 0; i < srcs.size(); i++)
    {
        copies[i]->Copy<dali::CPUBackend>(*srcs[i], NULL);
    }
}
} // namespace utils
} // namespace inference
} // namespace multideviceinferencepipeline
