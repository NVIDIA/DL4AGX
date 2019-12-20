# Matrix Multiplication using Different Memory Types
This app is modified version of the CUDA matrix multiplication sample app shipped with toolkit. We choose either global memory or pinned memory input/output buffers to see the performance implications for various input size matrices. This sample needs to compiled and executed on DRIVE AGX platform.

## Target platform
- DRIVE AGX platform
- DRIVE SW 10.0

## How to compile
Follow the instructions `//docker` to build a base container for the system you develop with.

To compile for Jetson use:

``` sh
dazel build //CUDAMemoryTypesMatMul:cuda_memory_types_mat_mul --config=L4T-toolchain
```

To compile for DRIVE Linux use:

``` sh
dazel build //CUDAMemoryTypesMatMul:cuda_memory_types_mat_mul --config=D5L-toolchain
```

To compile for DRIVE QNX use:

``` sh
dazel build //CUDAMemoryTypesMatMul:cuda_memory_types_mat_mul --config=D5Q-toolchain
```

Then copy this file over to your target system

## Usage

### Choose Global Memory buffers
Run the app with different matrix dimensions by varying `hA`,`wA`,`hB`,`wB` like below.

```sh
./cuda_memory_types_mat_mul -hA=640 -wA=12800 -hB=12800 -wB=640     # To run DRIVE AGX Xavier's iGPU
./cuda_memory_types_mat_mul -hA=640 -wA=12800 -hB=12800 -wB=640 -device=1    # To run DRIVE AGX Pegasus's iGPU
```

### Choose Pinned Memory buffers
Run the app with different matrix dimensions by varying `hA`,`wA`,`hB`,`wB` like below.

```sh
./cuda_memory_types_mat_mul -hA=640 -wA=12800 -hB=12800 -wB=640 -type=1    # To run DRIVE AGX Xavier's iGPU
./cuda_memory_types_mat_mul -hA=640 -wA=12800 -hB=12800 -wB=640 -type=1 -device=1    # To run DRIVE AGX Pegasus's iGPU
```

### Use Shared Memory optimization
Run the app with different matrix dimensions by varying `hA`,`wA`,`hB`,`wB` like below.

```sh
./cuda_memory_types_mat_mul -hA=640 -wA=12800 -hB=12800 -wB=640 -type=0 -enableSharedMemory=1   # To select global memory on DRIVE AGX Xavier's iGPU
./cuda_memory_types_mat_mul -hA=640 -wA=12800 -hB=12800 -wB=640 -type=1 -enableSharedMemory=1   # To select Pinned memory on DRIVE AGX Xavier's iGPU

./cuda_memory_types_mat_mul -hA=640 -wA=12800 -hB=12800 -wB=640 -type=0 -enableSharedMemory=1 -device=1  # To select global memory on DRIVE AGX Pegasus's iGPU
./cuda_memory_types_mat_mul -hA=640 -wA=12800 -hB=12800 -wB=640 -type=1 -enableSharedMemory=1 -device=1  # To select Pinned memory on DRIVE AGX Pegasus's iGPU
```
