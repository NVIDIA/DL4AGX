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

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include <chrono>
#include <thread>
#include <pthread.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>

#include "cudla_context.h"
#include "kernel.cuh"
#include "lodepng.h"

#define LASTERR()   { \
    cudaDeviceSynchronize(); \
    auto code = cudaGetLastError(); \
    if( code != cudaSuccess) { \
        std::cout << cudaGetErrorString(code) << std::endl; \
    } \
}

using namespace nvinfer1;

class Logger : public ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override {
		// Only print error messages
		if (severity == Severity::kERROR) {
			std::cerr << msg << std::endl;
		}
	}
};

Logger gLogger;

inline unsigned int getElementSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

struct Feat {
    void* ffeat;
    void* ifeat;
    float scale;
    size_t feat_size = 0;

    void allocate() {
        cudaMalloc(&ifeat, feat_size);
    }

    // quantize from float/half value into int8 with formula:
    //   quant_value = cast_to_int8(round(clip(float_value / scale, -128, 127)))
    void quantize(cudaStream_t stream) {
        optimize::convert_half_to_int8(
            ffeat, 
            reinterpret_cast<int8_t*>(ifeat),
            feat_size, scale, stream);
    }
}; // struct Feat

struct Tensor {
    std::string name;
    void* ptr;
    Dims dim;
    int32_t volume = 1;
    DataType dtype;
    TensorIOMode iomode;

    Tensor(std::string name, Dims dim, DataType dtype): 
        name(name), dim(dim), dtype(dtype) 
    {
        if( dim.nbDims == 0 ) {
            volume = 0;
        } else {
            volume = 1;
            for(int i=0; i<dim.nbDims; i++) {
                volume *= dim.d[i];
            }
        }        
    }

    void allocate() {
        cudaMalloc(&ptr, volume * getElementSize(dtype));
    }

    int32_t nbytes() {
        return volume * getElementSize(dtype);
    }
}; // struct Tensor

std::ostream& operator<<(std::ostream& os, Tensor& t) {
    os << "[" << (int)(t.iomode) << "] ";
    os << t.name << ", [";
    
    for( int nd=0; nd<t.dim.nbDims; nd++ ) {
        if( nd == 0 ) {
            os << t.dim.d[nd];
        } else {
            os << ", " << t.dim.d[nd];
        }
    }
    std::cout << "]";
    std::cout << ", type = " << int(t.dtype);
    return os;
}

struct Engine {
    ICudaEngine* engine;
    IExecutionContext* context; 
    std::unordered_map<std::string, std::shared_ptr<Tensor>> bindings;
    bool use_cuda_graph = false;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    Engine(
        std::string engine_path, IRuntime* runtime,
        std::unordered_map<std::string, void*> external_mem
    ) {
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file) {
            throw std::runtime_error("Error opening engine file: " + engine_path);
        }
        engine_file.seekg(0, engine_file.end);
        long int fsize = engine_file.tellg();
        engine_file.seekg(0, engine_file.beg);

        // Read the engine file into a buffer
        std::vector<char> engineData(fsize);

        engine_file.read(engineData.data(), fsize);
        engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
        context = engine->createExecutionContext();

        int nb = engine->getNbIOTensors();
        for( int n=0; n<nb; n++ ) {
            std::string name = engine->getIOTensorName(n);
            Dims d = engine->getTensorShape(name.c_str());
            DataType dtype = engine->getTensorDataType(name.c_str());

            bindings[name] = std::make_shared<Tensor>(name, d, dtype);
            bindings[name]->iomode = engine->getTensorIOMode(name.c_str());
            std::cout << *(bindings[name]) << std::endl;
            if( external_mem.find(name) != external_mem.end() ) {
                auto item = external_mem.at(name);
                bindings[name]->ptr = item;
            } else {
                bindings[name]->allocate();
            }   
            context->setTensorAddress(name.c_str(), bindings[name]->ptr);         
        }
    }

    ~Engine() {
    }

    void EnableCudaGraph(cudaStream_t stream) {        
        // run first time to avoid allocation?
        this->Forward(stream);
        cudaStreamSynchronize(stream);

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        this->Forward(stream);
        cudaStreamEndCapture(stream, &graph);
        this->use_cuda_graph = true;
        cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    }

    void Forward(cudaStream_t stream) {
        if( this->use_cuda_graph ) {
            cudaGraphLaunch(graph_exec, stream);
        } else {
            context->enqueueV3(stream);
        }        
    }
}; // struct Engine

const size_t image_size = 1024 * 1024;

struct App {
    cudaStream_t stream_backbone;
    std::shared_ptr<Engine> backbone;
    cudaEvent_t event_backbone_ready;
    cudaEvent_t event_backbone_ready_stage[2];
    cudaEvent_t bb_s, bb_e; // backbone start and end events
    cudaStream_t stream_head;

    const int INPUT_HxW = 1024 * 1024;
    const int DEPTH_HxW = 1024 * 1024;
    const int SEG_N_CLASS = 19;
    const int SEG_HxW = 1024 * 1024;
    const float DEP_SCALE = 0.00764765; // convert from onnx::Mul_1912: 3bfa9921

    unsigned char* hraw;
    unsigned char* draw;
    float* dimg;

    // These hex binaries are extracted from calibration cache
    // e.g. input.72: 3cb9c64a, and this is hex format of 0.0226776f
    // for more detail for the conversion, please refer to https://en.wikipedia.org/wiki/IEEE_754
    
    uint32_t scales_bin[4] = {0x3cb9c64a, 0x3ceabfec, 0x3d111505, 0x3ded7bfd};
    //                        0.0226776f,  0.028656f, 0.0354204f, 0.115959f
    float* scales = reinterpret_cast<float*>(scales_bin);
    size_t feat_sizes[4] = {32*256*256, 64*128*128, 160*64*64, 256*32*32 };
    std::string mappings[4] = {"input.72", "input.148", "input.224", "input.292"};

    Feat feats[4];

    // preload for better showcase
    std::vector<unsigned char*> images;
    std::vector<std::string> image_names;

    App(IRuntime* runtime) {
        // init backbone
        cudaStreamCreate(&stream_backbone);
        cudaEventCreate(&event_backbone_ready);

        cudaEventCreateWithFlags(&event_backbone_ready_stage[0], cudaEventBlockingSync);
        cudaEventCreateWithFlags(&event_backbone_ready_stage[1], cudaEventBlockingSync);
        cudaEventCreateWithFlags(&bb_s, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&bb_e, cudaEventBlockingSync);

        cudaStreamCreate(&stream_head);
        std::unordered_map<std::string, void*> backbone_mappings;
        backbone = std::make_shared<Engine>("engines/mtmi_encoder_fp16.engine", runtime, backbone_mappings);
        dimg = reinterpret_cast<float*>(backbone->bindings["input"]->ptr);
        cudaMalloc((void**)&draw, image_size * sizeof(unsigned char) * 4);

        for( int i=0; i<4; i++ ) {
            feats[i].scale = scales[i];
            feats[i].feat_size = feat_sizes[i];
            feats[i].allocate();
            feats[i].ffeat = backbone->bindings[mappings[i]]->ptr;
        }

        init_head(runtime);

        const std::filesystem::path img_dir{"tests"};
    
        // directory_iterator can be iterated using a range-for loop
        for (auto const& dir_entry : std::filesystem::directory_iterator{img_dir}) {
            std::string pth_str = dir_entry.path().generic_string();
            if( pth_str.find("png") != -1) {
                std::cout << pth_str << '\n';
                std::vector<unsigned char> image;
                unsigned width, height;
                unsigned error = lodepng::decode(image, width, height, pth_str);
                if( error != 0 ) {
                    std::cerr << "load " << pth_str << " failed" << std::endl;
                    exit(-1);
                }
                unsigned char* paged_image;
                cudaMallocHost(&paged_image, image.size());
                images.emplace_back(paged_image);
                std::memcpy(paged_image, image.data(), image.size());
                image_names.push_back(dir_entry.path().stem());
            }            
        }            
    }

    ~App() {        
    }
    
    void step_prepare(int i) {
        int real_i = i % images.size();
        cudaMemcpyAsync(
            draw, images[real_i],
            image_size * sizeof(unsigned char) * 4,
            cudaMemcpyHostToDevice, stream_backbone);
    }

    void step_backbone() {
        cudaEventRecord(bb_s, stream_backbone);

        // rgba to float, mean and std process
        preprocess_image(draw, dimg, image_size, stream_backbone);

        // encoder
        backbone->Forward(stream_backbone);

        // feat to int8 conversion
        for( int i=0; i<4; i++ ) {
            feats[i].quantize(stream_backbone);
        }

        cudaEventRecord(bb_e, stream_backbone);
    }

#ifdef USE_ORIN
    std::shared_ptr<CudlaContext> seg;
    void* seg_output;

    std::shared_ptr<CudlaContext> dep;
    void* dep_output;

    cudaStream_t stream_seg, stream_dep;
    int stage = 0;

    void init_head(IRuntime* runtime) {
        cudaEventCreateWithFlags(&event_backbone_ready_stage[0], cudaEventBlockingSync);
        cudaEventCreateWithFlags(&event_backbone_ready_stage[1], cudaEventBlockingSync);
        cudaEventCreateWithFlags(&bb_s, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&bb_e, cudaEventBlockingSync);

        cudaStreamCreateWithFlags(&stream_dep, cudaStreamNonBlocking);
        dep = std::make_shared<CudlaContext>("loadables/mtmi_depth_i8_dla.loadable", 0);
        cudaMalloc(&dep_output, DEPTH_HxW);

        // the order is related to built dla engine
        // please always cross-check with meta-info
        dep->bufferPrep(
            {feats[3].ifeat, feats[2].ifeat, feats[1].ifeat, feats[0].ifeat},
            {dep_output},
            stream_dep);

        cudaStreamCreateWithFlags(&stream_seg, cudaStreamNonBlocking);
        seg = std::make_shared<CudlaContext>("loadables/mtmi_seg_i8_dla.loadable", 1);
        cudaMalloc(&seg_output, SEG_HxW * SEG_N_CLASS);        
        seg->bufferPrep(
            {feats[0].ifeat, feats[1].ifeat, feats[2].ifeat, feats[3].ifeat},
            {seg_output},
            stream_seg);
    }

    void step_head_dla() {        
        seg->submitDLATask(stream_seg);
        dep->submitDLATask(stream_dep);
    }

    void forward_sync() {
        // debug purpose, run every input image in a synchronize manner
        for( int r=0; r<this->image_names.size(); r++ ) {
            this->step_prepare(r);
            this->step_backbone();
            cudaEventRecord(event_backbone_ready_stage[0], stream_backbone);
            cudaStreamWaitEvent(stream_seg, event_backbone_ready_stage[0]);
            cudaStreamWaitEvent(stream_dep, event_backbone_ready_stage[0]);
            this->step_head_dla();
            this->vis(r, "results/");
        }
    }

    void forward_async() {
        // manually enable and initialize the cudagraph
        this->backbone->EnableCudaGraph(stream_backbone);

        // trigger first encoder inference
        this->step_prepare(0);
        this->step_backbone();
        cudaEventRecord(event_backbone_ready_stage[0], stream_backbone);
        
        int image_index = 1;

        // quick loop for profiling
        int loop;
        for( loop=0; loop<200; loop++) {
            printf("loop: %d ", loop);
            auto begin_time = ::std::chrono::steady_clock::now();

            // issue the following head inference
            // since this part is non-blocking
            // it will return directly
            int head_event_index = loop % 2;
            cudaStreamWaitEvent(stream_seg, event_backbone_ready_stage[head_event_index]);
            cudaStreamWaitEvent(stream_dep, event_backbone_ready_stage[head_event_index]);
            this->step_head_dla();

            cudaStreamSynchronize(stream_backbone);
            float elapsed = -1.0f;
            cudaEventElapsedTime(&elapsed, bb_s, bb_e);
            printf("backbone elapsed: %f ", elapsed);

            // now we should be able to issue next encoder inference
            this->step_prepare(image_index);
            
            this->step_backbone();
            cudaEventRecord(bb_e, stream_backbone);

            int backbone_event_index = (loop + 1) % 2;
            cudaEventRecord(event_backbone_ready_stage[backbone_event_index], stream_backbone);

            image_index += 1;
            if( image_index >= image_names.size()) {
                // exceed, loop back to the first image
                image_index = 0;
            }

            auto end_time = ::std::chrono::steady_clock::now();
            auto elapsed_serial_ms = ::std::chrono::duration_cast<::std::chrono::microseconds>(end_time - begin_time).count() / 1000.;
            printf("chrono elapsed: %f\n", elapsed_serial_ms);
        }
        // last round
        int head_event_index = loop % 2;
        cudaStreamWaitEvent(stream_seg, event_backbone_ready_stage[head_event_index]);
        cudaStreamWaitEvent(stream_dep, event_backbone_ready_stage[head_event_index]);
        this->step_head_dla();

        cudaStreamSynchronize(stream_backbone);
        cudaStreamSynchronize(stream_seg);
        cudaStreamSynchronize(stream_dep);

        // verification round
        for( int r=0; r<images.size(); r++) {
            this->step_prepare(r);
            this->step_backbone();
            cudaEventRecord(event_backbone_ready_stage[0], stream_backbone);

            cudaStreamWaitEvent(stream_seg, event_backbone_ready_stage[0]);
            cudaStreamWaitEvent(stream_dep, event_backbone_ready_stage[0]);
            this->step_head_dla();
            this->vis(r, "results/");            
        }
    }

    // export results from dla buffer
    void vis(int r, std::string result_dir) {
        std::string prefix = result_dir + image_names[r];
        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream_head);

        // depth output        
        float* ddep; cudaMalloc(&ddep, DEPTH_HxW * sizeof(float));

        // convert int8 to float val
        optimize::convert_int8_to_float(
            reinterpret_cast<int8_t*>(dep_output),
            ddep,             
            DEPTH_HxW, DEP_SCALE);

        float* hdep = new float[DEPTH_HxW];
        cudaMemcpy(hdep, ddep, DEPTH_HxW * sizeof(float), cudaMemcpyDeviceToHost);
        std::ofstream dep_stream((prefix + "_depth.bin").c_str(), 
                                 std::fstream::out | std::fstream::binary);
        dep_stream.write(reinterpret_cast<char*>(hdep), DEPTH_HxW * sizeof(float));
        dep_stream.close();

        // segmentation output
        char* hseg = new char[SEG_HxW * SEG_N_CLASS];
        cudaMemcpy(hseg, seg_output, SEG_HxW * SEG_N_CLASS, cudaMemcpyDeviceToHost);
        std::ofstream seg_stream((prefix + "_seg.bin").c_str(), 
                                 std::fstream::out | std::fstream::binary);
        seg_stream.write(reinterpret_cast<char*>(hseg), SEG_HxW * SEG_N_CLASS);
        seg_stream.close();

        cudaFree(ddep);
        delete[] hdep;
        delete[] hseg;
    }
#else
    std::shared_ptr<Engine> seg;
    std::shared_ptr<Engine> dep;
    // cudaEvent_t event_backbone_ready;

    void init_head(IRuntime* runtime) {
        std::unordered_map<std::string, void*> head_mappings;
        head_mappings["input.72"] = feats[0].ifeat;
        head_mappings["input.148"] = feats[1].ifeat;
        head_mappings["input.224"] = feats[2].ifeat;
        head_mappings["input.292"] = feats[3].ifeat;
        seg = std::make_shared<Engine>("engines/mtmi_seg_i8.engine", runtime, head_mappings);
        dep = std::make_shared<Engine>("engines/mtmi_depth_i8.engine", runtime, head_mappings);
    }

    void step_head() {
        // wait until backbone finish inference
        cudaStreamWaitEvent(stream_head, event_backbone_ready);
        seg->Forward(stream_head);
        dep->Forward(stream_head);
    }
    
    // export results from internal buffer 
    void vis(int r, std::string result_dir) {
        std::string prefix = result_dir + image_names[r];
        cudaDeviceSynchronize();

        // depth output        
        float* ddep; cudaMalloc(&ddep, DEPTH_HxW * sizeof(float));

        // convert int8 to float val
        optimize::convert_int8_to_float(
            reinterpret_cast<int8_t*>(dep->bindings["onnx::Mul_1912"]->ptr),
            ddep,             
            DEPTH_HxW, DEP_SCALE);

        float* hdep = new float[DEPTH_HxW];
        cudaMemcpy(hdep, ddep, DEPTH_HxW * sizeof(float), cudaMemcpyDeviceToHost);
        std::ofstream dep_stream((prefix + "_depth.bin").c_str(), 
                                 std::fstream::out | std::fstream::binary);
        dep_stream.write(reinterpret_cast<char*>(hdep), DEPTH_HxW * sizeof(float));
        dep_stream.close();

        // segmentation output
        void* seg_ptr = seg->bindings["onnx::ArgMax_1963"]->ptr;
        char* hseg = new char[SEG_HxW * SEG_N_CLASS];
        cudaMemcpy(hseg, seg_ptr, SEG_HxW * SEG_N_CLASS, cudaMemcpyDeviceToHost);
        std::ofstream seg_stream((prefix + "_seg.bin").c_str(), 
                                 std::fstream::out | std::fstream::binary);
        seg_stream.write(reinterpret_cast<char*>(hseg), SEG_HxW * SEG_N_CLASS);
        seg_stream.close();

        cudaFree(ddep);
        delete[] hdep;
        delete[] hseg;
    }
#endif
};

int main() {
    cudaSetDevice(0);
    auto runtime_deleter = [](IRuntime *runtime) {};
	std::unique_ptr<IRuntime, decltype(runtime_deleter)> runtime{
        createInferRuntime(gLogger), runtime_deleter};

    auto app = App(runtime.get());

#ifdef USE_ORIN
    // app.forward_sync();  // use this api if you want to execute in non-pipelined manner
    app.forward_async();    // use this api for pipelined multi-task inference
#else
    for( int r=0; r<app.image_names.size(); r++ ) {
        app.step_prepare(r);
        app.step_backbone();
        cudaEventRecord(app.event_backbone_ready, app.stream_backbone);

        app.step_head();
        app.vis(r, "results/");
    }
#endif
    return 0;
}
