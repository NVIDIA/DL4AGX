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
 * Project: MultiDeviceInferencePipeline > enginecreator
 * 
 * File: DL4AGX/MultiDeviceInferencePipeline/DALIStream/DALIStream.h
 * 
 * Description: Build a DALI/TRT Inference Pipeline  
 ***************************************************************************************************/
#pragma once
// TRT dependencies
#include "NvInfer.h"
//#include "common/common.h"

// DALI dependencies
#include "dali/core/common.h"
#include "dali/pipeline/data/allocator.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/pipeline.h"
#include "dali/plugin/plugin_manager.h"
#include "dali/util/image.h"
#include "dali/util/user_stream.h"

// JSON dependencies
#include "common/datasets/coco/cocoJSON.h"

namespace multideviceinferencepipeline
{
namespace enginecreator
{
class DALIStream
{
public:
    DALIStream(int batchSize,
               const nvinfer1::Dims3& inputDims,
               const std::string& pipe,
               std::vector<common::datasets::coco::data::image> files,
               const std::string& calibDirectory,
               dali::Pipeline* ploaded_pipe,
               dali::DeviceWorkspace& ws,
               bool asyncDali);
    void reset();
    bool next();
    float* getBatch();
    int getBatchSize() const;
    int getImageSize() const;
    nvinfer1::Dims getDims() const;
    // static functions
    static void makeJPEGBatch(dali::TensorList<dali::CPUBackend>* tl,
                              nvinfer1::DimsNCHW& imageDims,
                              std::vector<std::string>& jpegNames);
    static void readJPEGImages(std::vector<float>& processedData,
                               std::string& serialized,
                               nvinfer1::DimsNCHW& imageDims,
                               std::vector<std::string>& jpegNames,
                               dali::Pipeline* ppreprocessPipe,
                               dali::DeviceWorkspace& preprocessWS,
                               bool asyncDali);
    static void doInference(std::vector<float>& detectionOut,
                            std::vector<int>& keepCount,
                            nvinfer1::DimsNCHW& imageDims,
                            std::vector<std::string>& jpegNames,
                            dali::Pipeline& inferencePipe,
                            dali::DeviceWorkspace& inferenceWS,
                            bool asyncDali);
    static void buildInferencePipe(std::string& preprocessSerialized,
                                   nvinfer1::DimsNCHW& imageDims,
                                   std::string& savedEngine,
                                   std::string& libPlugin,
                                   std::string& inputBlob,
                                   std::vector<std::string>& outputBindings,
                                   std::string& inferenceSerialized,
                                   int device,
                                   bool pipelinedDali,
                                   int queueDepthDali,
                                   bool asyncDali);

private:
    int mBatchSize{0};
    int mImageSize{0};
    std::vector<common::datasets::coco::data::image>::iterator mFileIter;
    nvinfer1::DimsNCHW mDims;
    std::vector<float> mBatch;
    std::vector<common::datasets::coco::data::image> mFiles;
    std::string mCalibDirectory;
    int mImageCount{0};
    std::string mPipe;
    dali::Pipeline* mpLoadedPipe;
    dali::DeviceWorkspace mDALIWS;
    bool mAsyncDali;
};
} // namespace enginecreator
} // namespace multideviceinferencepipeline
