FROM nvcr.io/nvidia/tensorrt:24.05-py3

ARG CMAKE_VERSION=3.29.3
ARG NUM_JOBS=8

ENV DEBIAN_FRONTEND noninteractive
ENV PROJECT_DIR=/workspace

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        locales \
        locales-all \
        wget \
        git \
        libcurl4-openssl-dev \
        sudo \
        ssh \
        libssl-dev \
        pbzip2 \
        pv \
        bzip2 \
        unzip \
        devscripts \
        lintian \
        fakeroot \
        dh-make \
        vim && \
    apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
RUN rm -rf /tmp/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install cuda-python==12.3.0 \
                numpy==1.26.3 \
                onnx==1.16.1 \
                onnxsim
RUN pip install --extra-index-url https://pypi.ngc.nvidia.com onnx_graphsurgeon==0.3.27
RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# ======== Install ModelOpt Toolkit for quantization and ORT for CUDA 12 =========
## RUN git clone $PUBLIC_GIT_LINK
ADD modelopt ${PROJECT_DIR}/modelopt
RUN cd ${PROJECT_DIR}/modelopt &&  \
    pip install -e ".[dev]" --extra-index-url https://pypi.nvidia.com --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple

# Need onnxruntime with CUDA 12 to work with this docker, not CUDA 11.8
RUN pip uninstall -y onnxruntime-gpu && \
    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# ======== Prepare repo to convert BEVFormer model from PyTorch to ONNX ========
# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install MMLab
ARG TORCH_CUDA_ARCH_LIST="7.5;6.1;8.6"
ENV FORCE_CUDA="1"
ENV TRT_LIBPATH="/usr/lib/x86_64-linux-gnu"

# Install pytorch-quantization from source (needed for PyTorch 2 support)
# WAR for issue with package installed via "pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com pytorch-quantization"
RUN cd ${PROJECT_DIR} && git clone https://github.com/NVIDIA/TensorRT.git -b release/10.0 && \
    cd TensorRT/tools/pytorch-quantization && \
    MAX_JOBS=4 python setup.py install

ENV DERRYHUB_PROJECT_DIR=${PROJECT_DIR}/BEVFormer_tensorrt
RUN cd ${PROJECT_DIR} && git clone https://github.com/DerryHub/BEVFormer_tensorrt.git

# Install MMCV. First replace c++14 by c++17 to be able to compile PyTorch 2.x then do pip install with MAX_JOBS=4 to prevent workstation from freezing.
RUN cd ${DERRYHUB_PROJECT_DIR} && \
    git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && git checkout v1.5.0 && \
    pip install -r requirements/optional.txt && \
    sed -i "s/c++14/c++17/g" setup.py && \
    MAX_JOBS=4 MMCV_WITH_OPS=1 pip install -v -e .

RUN cd ${DERRYHUB_PROJECT_DIR} && \
    git clone https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && git checkout v2.25.1 && \
    pip install -v -e .

RUN cd ${DERRYHUB_PROJECT_DIR} && \
    git clone https://github.com/open-mmlab/mmdeploy.git && \
    cd mmdeploy && git checkout v0.10.0 && \
    git clone https://github.com/NVIDIA/cub.git third_party/cub && \
    cd third_party/cub && git checkout c3cceac115 && cd .. && \
    git clone https://github.com/pybind/pybind11.git pybind11 && \
    cd pybind11 && git checkout 70a58c5 && \
    cd ${DERRYHUB_PROJECT_DIR}/mmdeploy

# Fix mmdeploy repo for TRT 10 and install package.
RUN perl -pi -e 's/&\(/(int*)&\(/g' ${DERRYHUB_PROJECT_DIR}/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/gather_topk/gather_topk.cpp && \
    perl -pi -e 's/&\(/(int*)&\(/g' ${DERRYHUB_PROJECT_DIR}/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/grid_sampler/trt_grid_sampler.cpp && \
    cd ${DERRYHUB_PROJECT_DIR}/mmdeploy && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_CXX_COMPILER=g++ -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=$TRT_LIBPATH -DCUDNN_DIR=/usr/local/cuda .. && \
    make -j$(nproc) && \
    make install
##    cd ${DERRYHUB_PROJECT_DIR}/mmdeploy && MAX_JOBS=4 pip install -v -e .

RUN cd ${DERRYHUB_PROJECT_DIR}/third_party/bev_mmdet3d && \
    MAX_JOBS=4 python setup.py build develop --user

# Update torch to ONNX script in BEVFormer_tensorrt to enable some constant folding
RUN perl -pi -e 's/keep_initializers_as_inputs=True/keep_initializers_as_inputs=False/g' ${DERRYHUB_PROJECT_DIR}/det2trt/convert/pytorch2onnx.py && \
    perl -pi -e 's/do_constant_folding=False/do_constant_folding=True/g' ${DERRYHUB_PROJECT_DIR}/det2trt/convert/pytorch2onnx.py

RUN sudo apt-get install -y libgl1-mesa-glx

# Set environment and working directory
ENV PATH="${PATH}:/usr/local/bin/ngc-cli"

WORKDIR /mnt
