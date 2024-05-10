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

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <unordered_map>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>

#include "memory.cuh"
 
#define LASTERR()   { \
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
        // in some cases we want to load from different dtype
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
        // file_.read(b1.data(), hsize);
        Htype* hbuffer = reinterpret_cast<Htype*>(b1.data());
        Dtype* dbuffer = reinterpret_cast<Dtype*>(b2.data());
        cudaMemcpy(b2.data(), ptr, dsize, cudaMemcpyDeviceToHost);
        // in some cases we want to load from different dtype
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

class Duration {
    // stat
    std::vector<float> stats;
    cudaEvent_t b, e;
    std::string m_name;
public:
    Duration(std::string name): m_name(name) {
        cudaEventCreate(&b);
        cudaEventCreate(&e);
    }

    void MarkBegin(cudaStream_t s) {
        cudaEventRecord(b, s);
    }

    void MarkEnd(cudaStream_t s) {
        cudaEventRecord(e, s);
    }

    float Elapsed() {
        float val;
        cudaEventElapsedTime(&val, b, e);
        stats.push_back(val);
        return val;
    }
}; // class 

void validate(SubNetwork& net, std::string frame_dir, std::string key) {
    auto ref = net.bindings[key]->load_ref(frame_dir + key + ".bin");
    float* f_ref = reinterpret_cast<float*>(ref.data());
    auto fresult = net.bindings[key]->cpu();
    std::cout << key << std::endl;
    for(int n=0; n<16; n++ ) {
        printf("%12.6f ", fresult[n]);
    }
    printf("\n");
    for(int n=0; n<16; n++ ) {
        printf("%12.6f ", f_ref[n]);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    // handle multi-gpu
    printf("nvinfer: %d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
    cudaSetDevice(0);

    auto runtime_deleter = [](IRuntime *runtime) { /* runtime->destroy(); */ };
	std::unique_ptr<IRuntime, decltype(runtime_deleter)> runtime{createInferRuntime(gLogger), runtime_deleter};
    SubNetwork backbone(std::string("./engines/simplify_extract_img_feat.engine"), runtime.get());
    SubNetwork pts_head(std::string("./engines/simplify_pts_head_memory.engine"), runtime.get());
    Memory mem;

    cudaStream_t stream; cudaStreamCreate(&stream);
    mem.mem_stream = stream;
    mem.pre_buf = (float*)pts_head.bindings["pre_memory_timestamp"]->ptr;
    mem.post_buf = (float*)pts_head.bindings["post_memory_timestamp"]->ptr;

    // events for measurement
    Duration dur_backbone("backbone");
    Duration dur_ptshead("ptshead");

    const std::filesystem::path data_dir{"data"};

    int n_frames = 0;
    for (auto const& dir_entry : std::filesystem::directory_iterator{data_dir}) {
        n_frames ++;
    }
    printf("Total frames: %d\n", n_frames);
    
    bool is_first_frame = true;

    backbone.EnableCudaGraph(stream);
    pts_head.EnableCudaGraph(stream);

    for( int f=1; f<n_frames; f++ ) {
        // load data
        char buf[5] = {0};
        sprintf(buf, "%04d", f);        
        std::string frame_dir = "./data/" + std::string(buf) + "/";
        std::cout << frame_dir << std::endl;
        // img
        backbone.bindings["img"]->load(frame_dir + "img.bin");

        dur_backbone.MarkBegin(stream);
        // inference
        backbone.Enqueue(stream);
        dur_backbone.MarkEnd(stream);

        cudaMemcpyAsync(
            pts_head.bindings["x"]->ptr,
            backbone.bindings["img_feats"]->ptr, 
            backbone.bindings["img_feats"]->nbytes(), 
            cudaMemcpyDeviceToDevice, stream);

        pts_head.bindings["pos_embed"]->load(frame_dir + "pos_embed.bin");
        pts_head.bindings["cone"]->load(frame_dir + "cone.bin");
        
        // load double timestamp from file
        double stamp_current = 0.0;
        char stamp_buf[8];
        std::ifstream file_(frame_dir + "data_timestamp.bin", std::ios::binary);
        file_.read(stamp_buf, sizeof(double));
        stamp_current = reinterpret_cast<double*>(stamp_buf)[0];
        std::cout << "stamp: " << stamp_current << std::endl;

        if( is_first_frame ) {
            // binary is stored as double
            pts_head.bindings["pre_memory_timestamp"]->load<double, float>(frame_dir + "prev_memory_timestamp.bin");

            // start from dumped values
            pts_head.bindings["pre_memory_embedding"]->load(frame_dir + "init_memory_embedding.bin");
            pts_head.bindings["pre_memory_reference_point"]->load(frame_dir + "init_memory_reference_point.bin");
            pts_head.bindings["pre_memory_egopose"]->load(frame_dir + "init_memory_egopose.bin");
            pts_head.bindings["pre_memory_velo"]->load(frame_dir + "init_memory_velo.bin");

            is_first_frame = false;
        } else {
            // update timestamp
            mem.StepPre(stamp_current);
        }        

        pts_head.bindings["data_ego_pose"]->load(frame_dir + "data_ego_pose.bin");
        pts_head.bindings["data_ego_pose_inv"]->load(frame_dir + "data_ego_pose_inv.bin");

        // inference
        dur_ptshead.MarkBegin(stream);
        pts_head.Enqueue(stream);
        dur_ptshead.MarkEnd(stream);
        mem.StepPost(stamp_current);

        // copy mem_post to mem_pre for next round
        pts_head.bindings["pre_memory_embedding"]->mov(pts_head.bindings["post_memory_embedding"], stream);
        pts_head.bindings["pre_memory_reference_point"]->mov(pts_head.bindings["post_memory_reference_point"], stream);
        pts_head.bindings["pre_memory_egopose"]->mov(pts_head.bindings["post_memory_egopose"], stream);
        pts_head.bindings["pre_memory_velo"]->mov(pts_head.bindings["post_memory_velo"], stream);
        
        cudaStreamSynchronize(stream);

        std::cout << "backbone: " << dur_backbone.Elapsed() 
                  << ", ptshead: " << dur_ptshead.Elapsed() 
                  << std::endl;

        validate(pts_head, frame_dir, "all_bbox_preds"); 
        validate(pts_head, frame_dir, "all_cls_scores"); 

        // dump preds and labels
        pts_head.bindings["all_bbox_preds"]->save(frame_dir + "/all_bbox_preds_trt.bin");
        pts_head.bindings["all_cls_scores"]->save(frame_dir + "/all_cls_scores_trt.bin");
    }

    return 0;
}
