#!/bin/bash

# Setup virtual environment
python3.8 -m venv trtllm_0.13
source trtllm_0.13/bin/activate


# Setup model/engine path
export path_engine=$PWD/DeepSeek-R1-Distill-Llama-8B_int4_awq_kv_int8
export path_model=$PWD/DeepSeek-R1-Distill-Llama-8B


# Disable rotary_scaling in Llama 3.1 for Orin deployment
python json_modifier.py $path_engine/config.json


# git clone TensorRT-LLM codebase
git clone -b v0.13.0 https://github.com/NVIDIA/TensorRT-LLM.git $PWD/TensorRT-LLM
cd $PWD/TensorRT-LLM
git submodule update --init --recursive
cp -r ../batch_manager/ ./cpp/tensorrt_llm/
cp -r ../executor/ ./cpp/tensorrt_llm/
cp -r ../nvrtcWrapper/aarch64-linux-gnu/libtensorrt_llm_nvrtc_wrapper.so ./cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/aarch64-linux-gnu/
git apply ../new_patch.patch


# setup dependency
## NCCL
cd ..
git clone https://github.com/NVIDIA/nccl.git 
cd nccl
git checkout v2.22.3-1 
make -j8 src.build 


# Install pytorch
cd ..
wget https://developer.download.nvidia.cn/compute/redist/jp/v51/pytorch/torch-1.14.0a0+44dac51c.nv23.01-cp38-cp38-linux_aarch64.whl  
python -m pip install --upgrade pip
export TORCH_INSTALL=$PWD/torch*cp38-linux_aarch64.whl
python3 -m pip install --no-cache $TORCH_INSTALL


# Install TensorRT
pip install $PWD/TensorRT-10.4.0.11/python/tensorrt-10.4.0b11-cp38-none-linux_aarch64.whl

# Copy libnvinfer.so file
sudo cp $PWD/TensorRT-10.4.0.11/lib/libnvinfer.so.10.4.0 /usr/lib/aarch64-linux-gnu/libnvinfer.so 

# Build TensorRT-LLM 0.13 from scratch on Orin
## Setup paths for build
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/openmpi/lib:$LD_LIBRARY_PATH 
export PATH_TRT=$PWD/TensorRT-10.4.0.11 
export PATH_NCCL=$PWD/nccl/build 
export LD_LIBRARY_PATH=${PATH_TRT}/lib 
export LD_LIBRARY_PATH=$PWD/TensorRT-10.4.0.11/lib/:$LD_LIBRARY_PATH


## Build from source
cd $PWD/TensorRT-LLM
./scripts/build_wheel.py --trt_root $PATH_TRT --nccl_root $PATH_NCCL --clean --cuda_architectures "87"


# Install TensorRT-LLM 
pip install ./build/tensorrt_llm-0.13.0-cp38-cp38-linux_aarch64.whl


# Fix transformers version
pip install transformers==4.45.0


# Try import TensorRT-LLM
cd ..
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"