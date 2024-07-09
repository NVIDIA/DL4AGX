# ReduceFormer: Attention with Tensor Reduction by Summation 

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/reduceformer_attn.png" height="360">
</div>

### [Paper](https://arxiv.org/abs/2406.07488) | [Poster](./resources/ReduceFormer_CVPRW_2024_Poster.pdf)
<!-- John Yang, Le An, Su Inn Park   -->

ReduceFormer overcomes the computational demands of transformers in vision tasks, 
offering efficient alternatives to matrix multiplication and Softmax while retaining the benefits of attention mechanisms.
ReduceFormer employs only straightforward operations, such as reduction and element-wise multiplication, 
resulting in a simplified architecture and significantly enhanced inference performance. 
This makes ReduceFormer an ideal choice for both edge devices with limited computational resources and memory bandwidth, and for cloud computing environments where high throughput is crucial.
<hr>

## Prerequistes

We have developed our implementation based on the [EfficientViT](https://github.com/mit-han-lab/efficientvit/tree/master) repository.
Please download the official EfficientViT implementation and integrate our relevant code into it.

To that end, run the following commands

```bash
git clone https://github.com/mit-han-lab/efficientvit.git
cd efficientvit

mkdir docker
cp path/to/reduceformer/docker/Dockerfile docker/
cp path/to/reduceformer/configs/*.yaml configs/cls/imagenet/
mkdir efficientvit/models/reduceformer/
cp path/to/reduceformer/reduceformer/*.py efficientvit/models/reduceformer/
cp path/to/reduceformer/cls_model_zoo.py efficientvit/
```

In order to place ReduceFormer and its config file within the EfficientViT repo as shown below:

```yaml
efficientvit
 ├ configs
 │ └ cls
 │   └ imagenet 
 │     ├ b1_rf.yaml
 │     ├ b2_rf.yaml
 │     └ b3_rf.yaml
 ├ efficientvit
 | ├ cls_model_zoo.py 
 │ ├ models
 │ │ ├ efficientvit
 │ │ └ reduceformer
 │ │   ├ __init__.py
 │ │   ├ backbone.py
 │ │   └ cls.py
...
```
`cls_model_zoo.py` will be overwritten over EfficientViT's as copied into its repository. 


## Training

For training the ReduceFormer Classification model on [ImageNet dataset](https://www.image-net.org/),
please follow the [training](https://github.com/mit-han-lab/efficientvit/blob/master/applications/cls.md) instruction of the original EfficientViT repo. 

Single-node multi-GPU Training command examples:

```bash

torchpack dist-run -np 8 \
python train_cls_model.py configs/cls/imagenet/b1_rf.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --run_config.eval_image_size "[288]" \
    --path .exp/cls/imagenet/b1_r288/

torchpack dist-run -np 8 \
python train_cls_model.py configs/cls/imagenet/b2_rf.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --run_config.eval_image_size "[288]" \
    --path .exp/cls/imagenet/b2_r288/

torchpack dist-run -np 8 \
python train_cls_model.py configs/cls/imagenet/b3_rf.yaml \
    --data_provider.image_size "[128,160,192,224,256,288]" \
    --run_config.eval_image_size "[288]" \
    --path .exp/cls/imagenet/b3_r288/

```

## Testing
After training, you can evaluate your model (e.g. ReduceFormer-B1 variant):

```bash
python eval_cls_model.py --model b1-r288 --image_size 288
```


## Results

ReduceFormer classification models are trained from scratch on ImageNet-1K for 300 epochs, including 20 warmup epochs.

Here are the results we achieved with [the training configurations](configs).

| Model         |  Resolution | Top1 Acc|  Params |  MACs |  Orin (bs1) | 
|----------------------|:----------:|:----------:|:---------:|:----------:|:------------:|
| ReduceFormer-B1 | 224x224 | 79.3 | 9.0M | 0.52G | 0.68ms |
| ReduceFormer-B1 | 256x256 | 80.1 | 9.0M | 0.67G | 0.73ms |
| ReduceFormer-B1 | 288x288 | 80.6 | 9.0M | 0.85G | 0.87ms |
| |
| ReduceFormer-B2 | 224x224 | 81.9 | 24.5M  | 1.68G  | 1.29ms |
| ReduceFormer-B2 | 256x256 | 82.6 | 24.5M  | 2.19G  | 1.41ms |
| ReduceFormer-B2 | 288x288 | 83.0 | 24.5M  | 2.77G  | 1.68ms |
| |
| ReduceFormer-B3 | 224x224 | 83.4 | 48.1M  | 3.87G  | 2.22ms |
| ReduceFormer-B3 | 256x256 | 83.6 | 48.1M  | 5.06G  | 2.43ms |
| ReduceFormer-B3 | 288x288 | 84.2 | 48.1M  | 6.40G  | 3.03ms |

Latency is measured on NVIDIA DRIVE Orin with TensorRT-8.6 in FP16 precision.
For additional findings on DRIVE Orin and related platforms, we direct readers to our paper.

## License
The provided code can be used for research or other non-commercial purposes. For details please check the [LICENSE](LICENSE) file.

