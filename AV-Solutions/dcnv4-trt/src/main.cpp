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

#include <dlfcn.h> // for dlopen

#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <limits>
#include <memory>
#include <chrono>
#include <unordered_map>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace fs = std::filesystem;

void normalize(float* in_out, int HW);

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
        cudaMalloc(&ptr, volume * getElementSize(dtype));
    }

    int32_t nbytes() {
        return volume * getElementSize(dtype);
    }

    void mov(std::shared_ptr<Tensor> other, cudaStream_t stream) {
        // copy from 'other'
        cudaMemcpyAsync(
            ptr, other->ptr, 
            nbytes(), 
            cudaMemcpyHostToDevice,
            stream);
    }

    template<class Htype=float, class Dtype=float>
    void load(std::string fname) {
        size_t hsize = volume * sizeof(Htype);
        size_t dsize = volume * getElementSize(dtype);
        std::vector<char> b1(hsize);
        std::vector<char> b2(dsize);
        std::ifstream file_(fname, std::ios::binary);
        if( file_.fail() ) {
            std::cerr << fname << " missing!" << std::endl;
            return;
        }
        file_.read(b1.data(), hsize);
        Htype* hbuffer = reinterpret_cast<Htype*>(b1.data());
        Dtype* dbuffer = reinterpret_cast<Dtype*>(b2.data());
        
        for( int i=0; i<volume; i++ ) {
            dbuffer[i] = (Dtype)hbuffer[i];
        }

        cudaMemcpy(ptr, b2.data(), dsize, cudaMemcpyHostToDevice);
    }

    template<class Htype=float, class Dtype=float>
    void save(std::string fname) {
        size_t hsize = volume * sizeof(Htype);
        size_t dsize = volume * getElementSize(dtype);
        std::vector<char> b1(hsize);
        std::vector<char> b2(dsize);
        std::ofstream file_(fname, std::ios::binary);
        if( file_.fail() ) {
            std::cerr << fname << " can't open!" << std::endl;
            return;
        }
        
        Htype* hbuffer = reinterpret_cast<Htype*>(b1.data());
        Dtype* dbuffer = reinterpret_cast<Dtype*>(b2.data());
        cudaMemcpy(b2.data(), ptr, dsize, cudaMemcpyDeviceToHost);
        
        for( int i=0; i<volume; i++ ) {
            hbuffer[i] = (Htype)dbuffer[i];
        }
        file_.write(b2.data(), hsize);
        file_.close();
    }

    std::vector<float> cpu() {
        std::vector<float> buffer(volume);
        cudaMemcpy(buffer.data(), ptr, volume * sizeof(float), cudaMemcpyDeviceToHost);
        return buffer;
    }

    std::vector<char> load_ref(std::string fname) {
        size_t bsize = volume * sizeof(float);
        std::vector<char> buffer(bsize);
        std::ifstream file_(fname, std::ios::binary);
        file_.read(buffer.data(), bsize);
        return buffer;
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

Logger gLogger;

class SubNetwork {
    ICudaEngine* engine;
    IExecutionContext* context; 
public:
    std::unordered_map<std::string, std::shared_ptr<Tensor>> bindings;
    bool use_cuda_graph = false;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    SubNetwork(std::string engine_path, IRuntime* runtime) {
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
            context->setTensorAddress(name.c_str(), bindings[name]->ptr);
        }
    }

    void Enqueue(cudaStream_t stream) {
        if( this->use_cuda_graph ) {
            cudaGraphLaunch(graph_exec, stream);
        } else {
            context->enqueueV3(stream);
        }  
    }

    ~SubNetwork() {
    }

    void EnableCudaGraph(cudaStream_t stream) {        
        // run first time to avoid allocation
        this->Enqueue(stream);
        cudaStreamSynchronize(stream);

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        this->Enqueue(stream);
        cudaStreamEndCapture(stream, &graph);
        this->use_cuda_graph = true;
#if CUDART_VERSION < 12000
        cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
#else
        cudaGraphInstantiate(&graph_exec, graph, 0);
#endif
    }
}; // class SubNetwork

void normalize(uint8_t* in, float* out, int HW, cudaStream_t stream);

struct Application {

    const int IMAGE_H = 224;
    const int IMAGE_W = 224;
    const int IMAGE_HxW = 224 * 224;
    const int IMAGENET_1K_N_CLASS = 1000;
    void* buf_d; // buffer
    void* buf_h; // buffer on host side
    void* out_h; 
    cudaStream_t stream;
    std::shared_ptr<SubNetwork> backbone;

    cudaEvent_t events[3];
    float elapsed[4] = {0.0f};
    float elapsed_total = 0.0f;
    int total_call = 0;

    Application(std::string engine_path) {
        printf("nvinfer: %d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
        cudaSetDevice(0);

        auto runtime_deleter = [](IRuntime *runtime) { /* runtime->destroy(); */ };
        std::unique_ptr<IRuntime, decltype(runtime_deleter)> runtime{createInferRuntime(gLogger), runtime_deleter};
        backbone = std::make_shared<SubNetwork>(engine_path, runtime.get());
        cudaStreamCreate(&stream);
        cudaMalloc(&buf_d, sizeof(uint8_t) * IMAGE_HxW * 3);
        cudaMallocHost(&buf_h, sizeof(uint8_t) * IMAGE_HxW * 3);
        cudaMallocHost(&out_h, sizeof(float) * 1000);
        for( int i=0; i<3; i++ ) {
            cudaEventCreate(&events[i]);
        }
        backbone->EnableCudaGraph(stream);
    }

    ~Application() {
        cudaStreamDestroy(stream);
        cudaFree(buf_d);
    }

    int inference(std::string pth) {
        auto t0 = ::std::chrono::steady_clock::now();

        int iw, ih, channels;
        unsigned char *img = stbi_load(pth.c_str(), &iw, &ih, &channels, 3);
        int ow = IMAGE_H;
        int oh = IMAGE_W;        
        stbir_resize_uint8_linear(img, iw, ih, 0, (unsigned char*)buf_h, ow, oh, 0, STBIR_RGB);
        auto t1 = ::std::chrono::steady_clock::now();

        cudaEventRecord(events[0], stream);
        cudaMemcpyAsync(buf_d, buf_h, sizeof(uint8_t) * IMAGE_HxW * 3, cudaMemcpyHostToDevice, stream);
        normalize(
            (uint8_t*)buf_d, 
            reinterpret_cast<float*>(this->backbone->bindings["input"]->ptr),
            IMAGE_HxW, stream);
        cudaEventRecord(events[1], stream);
        this->backbone->Enqueue(stream);       
        cudaMemcpyAsync(
            out_h, 
            this->backbone->bindings["output"]->ptr,
            sizeof(float) * IMAGENET_1K_N_CLASS, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(events[2], stream);
        cudaStreamSynchronize(stream);

        auto t2 = ::std::chrono::steady_clock::now();

        total_call += 1;
        if( total_call > 10 ) {
            elapsed_total += 1.0f;

            auto elapsed_load_and_resize = ::std::chrono::duration_cast<::std::chrono::microseconds>(t1 - t0).count() / 1000.;
            elapsed[0] += elapsed_load_and_resize;

            auto elapsed_engine_forward = ::std::chrono::duration_cast<::std::chrono::microseconds>(t2 - t1).count() / 1000.;
            elapsed[1] += elapsed_engine_forward;

            float _tmp_val = -1.0f; 
            cudaEventElapsedTime(&_tmp_val, events[0], events[1]); elapsed[2] += _tmp_val;
            cudaEventElapsedTime(&_tmp_val, events[1], events[2]); elapsed[3] += _tmp_val;
        }

        int max_class = -1;
        float* scores = reinterpret_cast<float*>(out_h);
        float max_score = std::numeric_limits<float>::min();
        for( int i=0; i<IMAGENET_1K_N_CLASS; i++ ) {
            if( scores[i] > max_score ) {
                max_score = scores[i];
                max_class = i;
            }
        }

        stbi_image_free(img);
        return max_class;
    }
}; // struct Application

int main(int argc, char** argv) {
    if( argc != 4 ) {
        printf("usage: ./DCNv4_app plugin_file engine_file file_name");
        exit(-1);
    }
    void* so_handle = dlopen(argv[1], RTLD_NOW);

    Application app{std::string(argv[2])};
    int pred = -1;    
    pred = app.inference(std::string(argv[3]));
    printf("%s: %d\n", argv[3], pred);
    dlclose(so_handle);
    return 0;
}
