FROM nvcr.io/nvidia/pytorch:23.06-py3

COPY . /ReduceFormer

RUN pip install torch==2.0.1 torchvision==0.15.2 einops opencv-python timm==0.6.13 tqdm torchprofile matplotlib 
RUN pip install git+https://github.com/zhijian-liu/torchpack.git@3a5a9f7ac665444e1eb45942ee3f8fc7ffbd84e5
RUN pip install transformers onnx onnxsim onnxruntime mpi4py ptflops

WORKDIR /ReduceFormer/