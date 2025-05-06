# TensorRT-LLM on DRIVE Orin

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.13.0) is a toolbox for optimizing Large Language Model (LLM) inference. It offers cutting-edge optimizations such as custom attention kernels, plugins, and various quantization techniques, enabling efficient inference on NVIDIA GPUs. In this repository, we demonstrate how to deploy Large Lanuage Model (LLM) on DRIVE Orin platform for developers who is interested in using TensorRT-LLM. Following this repository, we detail the deployment of LLMs on Orin with TensorRT-LLM 0.13. This repository is for evaluation purposes only. 

## Supported Models:
Currently, the following LLMs are supported: 
- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)

## Orin Environment:

Please make sure you have the following Orin environment setup on your device. You can follow the [DRIVE OS Install Guidance](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-installation/index.html) to prepare the device. To get the required DRIVE OS and TensorRT version, please refer to details on the [NVIDIA DRIVE Downloads](https://developer.nvidia.com/drive/downloads) site.

- Drive OS 0.6.9.0
- Python 3.8
- CUDA 11.4
- Ubuntu 20.04
- TensorRT 10.4.0.11


## Generate LLMs with INT4_AWQ on x86

In this example, we are recommending quantizing the LLMs during the deployment. To quantize these LLMs model, please follow this link to install the TensorRT-LLM on an x86 system with at least 16GB GPU memory the following sample commands: 

```
python -m venv trtllm
source trtllm/bin/activate
pip install tensorrt_llm==0.16 
```
Please check if your venv has the following dependency:
- TensorRT-LLM 0.16
- modelOpt 0.19.0

After confirming the above setup, follow the commands below to quantize the Llama model:
```
cd $PWD/TensorRT-LLM/examples/llama 
python convert_checkpoint.py --model_dir $input_model --output_dir $output_model --dtype float16 --use_weight_only --weight_only_precision=int4_awq
```


## Build TensorRT-LLM from Source

After generating the quantized LLMs on x86 system, please copy original LLM checkpoint folder and quantized LLM folder from host to your target DRIVE Orin working directory. Then please run the following command to build TensorRT-LLM from source. 
```
./setup_from_source_{MODEL_NAME}.sh
```

The overall working directory after running `setup_from_source_{MODEL_NAME}.sh` will be as follows:
```
work_dir
├── batch_manager 
├── executor
├── build_from_scoure_changes.patch 
├── setup_from_source.sh 
├── TensorRT-LLM
├── TensorRT-10.4.0.11
├── {MODEL_NAME}
├── {MODEL_NAME}_int4_awq
├── trtllm_0.13 (vitual enviornment)
├── nccl
├── json_modifier.py
```
Note that on Drive Orin, currently TensorRT-LLM v0.13 is used for inference.

## Build LLM Engine and Sample Inference on Drive Orin
After successfully build the TensorRT-LLM from source on target Drive Orin, the following command will build the corresponding LLM engine:
```
trtllm-build --checkpoint_dir $path_engine --output_dir $path_engine/1-gpu/ --gemm_plugin auto --max_batch_size 1 
```

Using the above engine, you can run LLM inference using the following commands:
```
cd TensorRT-LLM/examples/
python run.py --max_output_len 128 --engine_dir $path_engine/1-gpu/ --tokenizer_dir $path_model 
```

Here is a sample inference output for Llama-3.1-8B model 
```
[11/21/2024-16:10:58] [TRT-LLM] [I] Load engine takes: 62.147863149642944 sec
Input [Text 0]: "<|begin_of_text|>Born in north-east France, Soyer trained as a"
Output [Text 0 Beam 0]: " chef in Paris before moving to London in 1848. He was appointed chef to the Prince of Wales in 1850, and later became chef to Queen Victoria. He was a pioneer of French cuisine in England, and his cookery books were very popular. He died in 1868."
```
