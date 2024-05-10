# StreamPETR Deployment On NVIDIA Drive Orin Platform
[StreamPETR](https://github.com/exiawsh/StreamPETR/) is an efficient camera-only 3D detection algorithm. This detection framework has an object-centric design and utilizes temporal information to enhance the performance with minimum computational overhead. Its long-sequence modeling ability and sparse query design results in significant performance improvement. 

## Getting started
This repository demostrates how to deploy StreamPETR on NVIDIA Drive Orin Platform with TensorRT, in the following two steps:
- [Conversion from pytorch to onnx](conversion/README.md)
- [StreamPETR TensorRT application](inference_app/README.md)

## Deployment strategy
The inference application is structured as depicted in the following diagram. The entire model is divided into an image encoder and a head. This division serves two primary purposes: firstly, it facilitates separate control over precision, and secondly, it allows for the separation of the backbone and head, enabling a cleaner handling of the recursive memory tensor update. During each frame, the application loads the image, ego motion, and timestamp externally. It then establishes the input 'memory' based on the output 'memory' from the previous frame.

<img src="assets/streampetr-framework.png" width="512">

## References
- [Official StreamPETR Repository](https://github.com/exiawsh/StreamPETR/)
- [Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection](https://arxiv.org/abs/2303.11926)