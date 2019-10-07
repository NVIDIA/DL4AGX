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
 * File: DL4AGX/MultiDeviceInferencePipeline/DALIStream/DALIStream.cpp
 * 
 * Description: Build a DALI/TRT Inference Pipeline  
 ***************************************************************************************************/
#include "MultiDeviceInferencePipeline/enginecreator/DALIStream/DALIStream.h"
#include "dali/util/image.h"

#include "common/common.h"

using namespace multideviceinferencepipeline;

//Constructor
enginecreator::DALIStream::DALIStream(int batchSize,
                                      const nvinfer1::Dims3& inputDims,
                                      const std::string& pipe,
                                      std::vector<common::datasets::coco::data::image> files,
                                      const std::string& calibDirectory,
                                      dali::Pipeline* ploaded_pipe,
                                      dali::DeviceWorkspace& ws,
                                      bool asyncDali)
    : mBatchSize(batchSize)
    , mFiles(files)
    , mCalibDirectory(calibDirectory)
    , mPipe(pipe)
    , mpLoadedPipe(ploaded_pipe)
    , mDALIWS(ws)
    , mAsyncDali(asyncDali)
{
    mDims.d[0] = batchSize;      // Batch Size
    mDims.d[1] = inputDims.d[0]; // Channels
    mDims.d[2] = inputDims.d[1]; // Height
    mDims.d[3] = inputDims.d[2]; // Width

    mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
    mBatch.resize(common::tensorrt::volume(mDims), 0);
    reset();
    std::cout << "DALIStream: Using " << files.size() << " images (nchw=" << mDims.d[0] << "x"
              << mDims.d[1] << "x" << mDims.d[2] << "x" << mDims.d[3] << ")" << std::endl;
}

void enginecreator::DALIStream::reset() { mFileIter = mFiles.begin(); }

bool enginecreator::DALIStream::next()
{
    if (mFileIter == mFiles.end())
    {
        return false;
    }
    assert(mBatchSize > 0);
    std::cout << "Processing batch " << mImageCount / mBatchSize << std::endl;
    std::vector<std::string> jpegNames;
    for (int i = 0; (i < mBatchSize) && (mFileIter != mFiles.end()); ++i)
    {
        std::string filename = mCalibDirectory + "/" + (*mFileIter).file_name;
        mFileIter++;
        mImageCount++;
        jpegNames.push_back(filename);
    }

    // read JPEG images
    std::vector<float> data(common::tensorrt::volume(mDims), 0.f);

    readJPEGImages(data, mPipe, mDims, jpegNames, mpLoadedPipe, mDALIWS, mAsyncDali);
    std::copy_n(data.data(), mDims.n() * mImageSize, getBatch());
    return true;
}

float* enginecreator::DALIStream::getBatch() { return mBatch.data(); }
int enginecreator::DALIStream::getBatchSize() const { return mBatchSize; }
int enginecreator::DALIStream::getImageSize() const { return mImageSize; }
nvinfer1::Dims enginecreator::DALIStream::getDims() const { return mDims; }

void enginecreator::DALIStream::makeJPEGBatch(dali::TensorList<dali::CPUBackend>* tl, nvinfer1::DimsNCHW& imageDims, std::vector<std::string>& jpegNames)
{
    int batchSize = imageDims.d[0];
    dali::ImgSetDescr jpegs_;
    dali::LoadImages(jpegNames, &jpegs_);
    const auto nImgs = jpegs_.nImages();
    std::vector< std::vector<::dali::Index> > shape(batchSize);
    for (int i = 0; i < batchSize; i++)
    {
        shape[i] = {jpegs_.sizes_[i % nImgs]};
    }
    tl->template mutable_data<dali::uint8>();
    tl->Resize(shape);
    for (int i = 0; i < batchSize; i++)
    {
        memcpy(tl->template mutable_tensor<dali::uint8>(i),
               jpegs_.data_[i % nImgs], jpegs_.sizes_[i % nImgs]);
    }
}

void enginecreator::DALIStream::readJPEGImages(std::vector<float>& processedData,
                                               std::string& serialized,
                                               nvinfer1::DimsNCHW& imageDims,
                                               std::vector<std::string>& jpegNames,
                                               dali::Pipeline* preprocessPipe,
                                               dali::DeviceWorkspace& preprocessWS,
                                               bool asyncDali)
{
    // read into data
    dali::TensorList<dali::CPUBackend> dataDALI;
    makeJPEGBatch(&dataDALI, imageDims, jpegNames);
    preprocessPipe->SetExternalInput("raw_jpegs", dataDALI);
    preprocessPipe->RunCPU();
    preprocessPipe->RunGPU();
    preprocessPipe->Outputs(&preprocessWS);
    dali::TensorList<dali::GPUBackend>& tl = preprocessWS.Output<dali::GPUBackend>(0);
    if (asyncDali)
    {
        CHECK(cudaMemcpyAsync(processedData.data(), tl.raw_mutable_data(), sizeof(float) * processedData.size(), cudaMemcpyDeviceToHost));
    }
    else
    {
        CHECK(cudaMemcpy(processedData.data(), tl.raw_mutable_data(), sizeof(float) * processedData.size(), cudaMemcpyDeviceToHost));
    }
}

void enginecreator::DALIStream::buildInferencePipe(std::string& preprocessSerialized,
                                                   nvinfer1::DimsNCHW& imageDims,
                                                   std::string& savedEngine,
                                                   std::string& libPlugin,
                                                   std::string& inputBlob,
                                                   std::vector<std::string>& outputBindings,
                                                   std::string& inferenceSerialized,
                                                   int device,
                                                   bool pipelinedDali,
                                                   int queueDepthDali,
                                                   bool asyncDali)
{
    int num_thread = 1;
    int nBatch = imageDims.d[0];

    dali::Pipeline preprocessPipe(preprocessSerialized, nBatch, num_thread, device, pipelinedDali, queueDepthDali, asyncDali);
    // add TRT op
    std::vector<std::string> inputBlobs, outputBlobs;
    std::vector<std::string> plugins;
    std::string engineFile(savedEngine);
    std::string engineData;
    common::loadEngine(engineFile, engineData);

    inputBlobs.push_back(inputBlob);
    outputBlobs = outputBindings;
    plugins.push_back(libPlugin);

    dali::OpSpec inferOp("TensorRTInfer");

    inferOp.AddArg("device", "gpu")
        .AddArg("inference_batch_size", nBatch)
        .AddArg("engine", engineData)
        .AddArg("plugins", plugins)
        .AddArg("num_outputs", outputBlobs.size())
        .AddArg("input_nodes", inputBlobs)
        .AddArg("output_nodes", outputBlobs)
        .AddArg("log_severity", 4);

    inferOp.AddInput("preprocessed_images", "gpu");

    for (size_t i = 0; i < outputBlobs.size(); ++i)
    {
        inferOp.AddOutput(outputBlobs[i], "gpu");
    }

    preprocessPipe.AddOperator(inferOp);
    inferenceSerialized = preprocessPipe.SerializeToProtobuf();
}

void enginecreator::DALIStream::doInference(std::vector<float>& detectionOut, std::vector<int>& keepCount, nvinfer1::DimsNCHW& imageDims, std::vector<std::string>& jpegNames, dali::Pipeline& inferencePipe, dali::DeviceWorkspace& inferenceWS, bool asyncDali)
{
    // read into data
    dali::TensorList<dali::CPUBackend> dataDALI;
    makeJPEGBatch(&dataDALI, imageDims, jpegNames);
    inferencePipe.SetExternalInput("raw_jpegs", dataDALI);
    inferencePipe.RunCPU();
    inferencePipe.RunGPU();
    inferencePipe.Outputs(&inferenceWS);
    dali::TensorList<dali::GPUBackend>& tl0 = inferenceWS.Output<dali::GPUBackend>(0);
    dali::TensorList<dali::GPUBackend>& tl1 = inferenceWS.Output<dali::GPUBackend>(1);
    if (asyncDali)
    {
        CHECK(cudaMemcpyAsync(detectionOut.data(), tl0.raw_mutable_data(), sizeof(float) * detectionOut.size(), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpyAsync(keepCount.data(), tl1.raw_mutable_data(), sizeof(int) * keepCount.size(), cudaMemcpyDeviceToHost));
    }
    else
    {
        CHECK(cudaMemcpy(detectionOut.data(), tl0.raw_mutable_data(), sizeof(float) * detectionOut.size(), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(keepCount.data(), tl1.raw_mutable_data(), sizeof(int) * keepCount.size(), cudaMemcpyDeviceToHost));
    }
}