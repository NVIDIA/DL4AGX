pip install --upgrade pip

# ======== Basic requirements ========
pip install numpy
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# ======== Quantization requirements ========
# Clone vision repo with quantization enabled
git clone -b quantize https://github.com/gcunhase/vision.git
echo "__version__ = '0.12.0+cu113'" > ./vision/torchvision/version.py
pip install torchsummary tensorboard

# Install QAT toolkit
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com

# Install requirements for PTQ calibration and engine inference (TensorRT)
pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
pip install pycuda
pip install --upgrade onnx-graphsurgeon

# ======== Sparsity requirements ========
# Install Sparsity toolkit
pip install tabulate packaging
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--permutation_search" ./
