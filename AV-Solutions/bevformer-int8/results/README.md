# About
Instructions on how to reproduce the results in the main [README](../README.md#results).

# Steps to reproduce
To reproduce the `FP32`, `FP16`, and `QDQ_BEST` (Implicit and Explicit Quantization) results:
1. Run `./deploy_trt.sh` to build/save the TRT engine and obtain the runtime;
2. Run `./evaluate_trt.sh` to evaluate the TRT engine's accuracy.
