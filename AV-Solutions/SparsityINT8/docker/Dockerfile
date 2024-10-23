FROM nvcr.io/nvidia/pytorch:22.09-py3
COPY ./scripts /scripts
WORKDIR /scripts

# ======== Give executable permission to run.sh ========
RUN chmod +x /scripts/run_test*.sh

# ======== Install Quantization requirements ========
# Clone vision repo with quantization enabled
RUN git clone -b quantize https://github.com/gcunhase/vision.git && \
    echo "__version__ = '0.12.0+cu113'" > ./vision/torchvision/version.py && \
    pip install torchsummary tensorboard

# Install QAT toolkit
RUN pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com

# Install requirements for PTQ calibration and engine inference (TensorRT)
RUN pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com && \
    pip install pycuda && \
    pip install --upgrade onnx-graphsurgeon

# ======== Install Sparsity requirements ========
RUN pip install tabulate packaging && \
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--permutation_search" ./

# ======== End ========
CMD /bin/bash
