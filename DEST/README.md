# DEST: Depth Estimation with Simplified Transformer

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/attentions.png" height="400">
</div>

***[DEST: Depth Estimation with Simplified Transformer](https://arxiv.org/abs/2204.13791)***<br />
John Yang, Le An, Anurag Dixit, Jinkyu Koo, Su Inn Park  
CVPR Workshop on [Transformers For Vision](https://sites.google.com/view/t4v-cvpr22), 2022

DEST leverages a simplified design of attention block in the transformer that is GPU friendly. Compared to state-of-the-art methods, our model achieves over 80% reduction in terms of model size and computation, while being more accurate and faster. The proposed model was validated on both depth esitimation and semantic segmentation tasks. This repository contains the official Pytorch model implementation and training configuration which can be adapted to your traing workflow. 

<hr>

## Monocular Depth Estimation
For depth estimation, we employ the same setup as that in [PackNet-sfm](https://github.com/TRI-ML/packnet-sfm). For details on environment preparation, data download, and training/evaluation scripts, please refer to the original repo for details. 

### Prerequistes

Run the following commands

```bash
git clone https://github.com/TRI-ML/packnet-sfm.git
cd packnet-sfm

cp path/to/DEST/configs/train_kitti_dest.yaml configs/
cp path/to/DEST/models/*_dest.py packnet_sfm/models/
cp path/to/DEST/networks/DESTNet.py packnet_sfm/networks/depth/
mkdir packnet_sfm/networks/DEST
cp path/to/DEST/networks/DEST/*.py packnet_sfm/networks/DEST/
```

in order to place DEST and its config file within the [PackNet-sfm](https://github.com/TRI-ML/packnet-sfm) implementation as shown below:

```yaml
packnet-sfm
 ├ configs
 │ ...
 │ └ train_kitti_dest.yaml
 ├ packnet_sfm
 │ ...
 │ ├ models
 │ │ ...
 │ │ ├ SfmModel_dest.py
 │ │ ├ SemiSupModel_dest.py
 │ │ └ SelfSupModel_dest.py
 │ ├ networks
 │ │ ...
 │ │ ├ depth
 │ │ │ ...
 │ │ │ └ DESTNet.py
 │ │ └ DEST
 │ │   ├ __init__.py
 │ │   ├ DEST_EncDec.py
 │ │   ├ simplified_attention.py
 │ │   └ simplified_joint_attention.py
...
```

### Modifications to make on PackNet repo
Our work quires ```timm``` library, so please add the following line in `docker/Dockerfile`.

```bash
RUN pip install timm
```

Before building the docker image, we also need to adjust the Python version, CUDNN version, NCCL version, etc. in the Dockerfile according to our machine. Note that the minimum supported Python version is 3.7. Base images can be found from [dockerhub](https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated): 

After properly configuring Dockerfile, please follow [the instructions](https://github.com/TRI-ML/packnet-sfm#install) to build your docker image.

Also, due to [the issues from the PackNet repository](https://github.com/TRI-ML/packnet-sfm/issues/107) during evalution, 
you need to edit the lines of L295, L302 from the file `packnet-sfm/packnet_sfm/models/model_wrapper.py`.

Change lines
```
[L295] depth = inv2depth(inv_depths[0])
...
[L301] inv_depth_pp = post_process_inv_depth(
[L302]     inv_depths[0], inv_depths_flipped[0], method='mean')
```
to
```
[L295] depth = inv2depth(inv_depths)
...
[L301] inv_depth_pp = post_process_inv_depth(
[L302]     inv_depths, inv_depths_flipped, method='mean')
```


### Training

To train DEST from scratch on KITTI dataset, run the following command:
```bash
python scripts/train.py configs/train_kitti_dest.yaml
```

### Evaluation
For the evaluation of DEST model on KITTI dataset, run the following:

```bash
python scripts/eval.py --checkpoint <DEST.ckpt> [--config <config.yaml>]
```

For inference on a single image or folder:
You can also directly run inference on a single image or folder:

```bash
python scripts/infer.py --checkpoint <DEST.ckpt> --input <image or folder> --output <image or folder> [--image_shape <input shape (h,w)>]
```

<hr>

## Semantic Segmentation
For semantic segmentation, our implementation can be readily integrated into [OpenMMLab Semantic Segmentation Toolbox and Benchmark](https://github.com/open-mmlab/mmsegmentation) implementation for training and evaluation. 

Please refer to their instruction for [installations](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) and [dataset preparatation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets).
Our DEST is trained/evaluated on [CityScapes Dataset](https://www.cityscapes-dataset.com/login/). 

### Prerequisites
In order to follow MMSegmentation instructions for training,  refer to the files that are located at ```DEST/semseg/``` and
re-locate the files within the MMSegmentation repository by running the following commands:
```bash
git clone https://github.com/open-mmlab/mmsegmentation.git # first clone the MMSegmentation env
cd mmsegmentation
mkdir configs/dest/

cp path/to/DEST/semseg/dest_simpatt-b0.py configs/_base_/models/
cp path/to/DEST/semseg/schedule_160k_adamw.py configs/_base_/schedules/
cp path/to/DEST/semseg/cityscapes_1024x1024_repeat.py configs/_base_/datasets/
cp path/to/DEST/semseg/dest_simpatt-*_1024x1024_160k_cityscapes.py configs/dest/
cp path/to/DEST/semseg/simplified_attention_mmseg.py mmseg/models/backbones/
cp path/to/DEST/semseg/dest_head.py mmseg/models/decode_heads/
```

You now need to include DEST in their library
```bash
echo 'from .simplified_attention_mmseg import SimplifiedTransformer' >> mmseg/models/backbones/__init__.py
echo 'from .dest_head import DestHead' >> mmseg/models/decode_heads/__init__.py
```

Then, you can start training/evaluating with a desired configuration of DEST.

### Training
Example: train DEST-B1 on CityScapes Dataset:

```bash
# Single-gpu training
python tools/train.py configs/dest/dest_simpatt-b1_1024x1024_160k_cityscapes.py
# Multi-gpu training
./tools/dist_train.sh configs/dest/dest_simpatt-b1_1024x1024_160k_cityscapes.py <GPU_NUM>
```

### Evaluation
After training, you can evaluate the trained model (e.g. DEST-B1)

```bash
# Single-gpu testing
python tools/test.py configs/dest/dest_simpatt-b1_1024x1024_160k_cityscapes.py /path/to/checkpoint_file
# Multi-gpu testing
./tools/dist_test.sh configs/dest/dest_simpatt-b1_1024x1024_160k_cityscapes.py /path/to/checkpoint_file <GPU_NUM>
# Multi-gpu, multi-scale testing
tools/dist_test.sh configs/dest/dest_simpatt-b1_1024x1024_160k_cityscapes.py /path/to/checkpoint_file <GPU_NUM> --aug-test
```


## License
The provided code can be used for research or other non-commercial purposes. For details please check the [LICENSE](LICENSE) file.

## Citation
```
@article{YangDEST,
  title={Depth Estimation with Simplified Transformer},
  author={Yang, John and An, Le and Dixit, Anurag and Koo, Jinkyu and Park, Su Inn},
  journal={arXiv preprint arXiv:2204.13791},
  year={2022}
}
```
