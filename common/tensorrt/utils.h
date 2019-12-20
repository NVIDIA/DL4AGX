/**************************************************************************
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
 * File: DL4AGX/common/tensorrt/utils.h
 * Description: Utilites to use with TensorRT
 *************************************************************************/
#pragma once
#include "NvInfer.h"
#include "NvInferVersion.h"
#include <cstring>
#include <iostream>
#include <numeric>

namespace common
{
namespace tensorrt
{
#if NV_TENSORRT_MAJOR < 6
inline void enableDLA(nvinfer1::IBuilder* b, int dlaID)
{
    b->allowGPUFallback(true);
    b->setFp16Mode(true);
    b->setDefaultDeviceType(static_cast<nvinfer1::DeviceType>(dlaID));
}
#endif

inline int parseDLA(int argc, char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg.substr(0, 9) == "--useDLA=" && arg.size() > 9)
            return std::stoi(argv[i] + 9);
    }
    return -1;
}

#if NV_TENSORRT_MAJOR >= 6
    inline void enableDLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
        }
        if (allowGPUFallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        if (!builder->getInt8Mode() && !config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            builder->setFp16Mode(true);
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }
}
#endif

    
inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename A, typename B>
inline A divUp(A m, B n)
{
    return (m + n - 1) / n;
}

} // namespace tensorrt
} // namespace common
