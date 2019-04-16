# TensorRTInferOp 

A plugin for DALI [https://github.com/NVIDIA/DALI/](https://github.com/NVIDIA/DALI/) that allows users to include TensorRT engines in DALI pipelines. This lets people use the same DALI GPU accelerated data preprocessing pipelines used to training in inference.

## Compiling

To compile this library just run the applicable command for your platform:

#### x86_64-linux

```sh
dazel run //plugins/dali/TensorRTInferOp:libtensorrtinferop.so
```

#### aarch64-linux

```sh
dazel run //plugins/dali/TensorRTInferOp:libtensorrtinferop.so --config=[D5L/L4T]-toolchain
```

#### aarch64-qnx

```sh
dazel run //plugins/dali/TensorRTInferOp:libtensorrtinferop.so --config=D5Q-toolchain
```

## Usage 

### Op Name: TensorRTInfer

__Perform inference over the TensorRT engine__

#### Arguments:

##### **Required**:

- __input_nodes__ `Vec<string>`: Inputs nodes in the engine

- __output_nodes__ `Vec<string>`: Outputs nodes in the engine

- __engine__ `string`: Path to TensorRT engine file to run inference

##### **Optional**

- __log_severity__ `int` (`nvinfer::Severity`): Logging severity for TensorRT

- __plugins__ `Vec<string>`: Plugin library to load

- __num_outputs__ `int`: Number of outputs

- __inference_batch_size__ `int`: Batch size to run inference

- __use_dla_core__ `int`: DLA core to run inference upon