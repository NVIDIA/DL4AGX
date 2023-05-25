# About
This readme includes steps on how to set-up a docker environment.

# Set-up docker env
To build the docker image:
1. Set `$IMAGE_NAME` in [build_docker_image.sh](./build_docker_image.sh) as you wish.
2. Run:
```
./build_docker_image.sh
```

Note that this docker image doesn't include the `tensorrt` python wheel installation, which is needed for PTQ calibration and TRT engine inference only. Please install it manually.

# How to Run
1. Create the container as:
```bash
docker run --runtime=nvidia --name sparsity_container --net=host -ti -v /data1/imagenet:/data1/imagenet $IMAGE_NAME:latest
```

2. Run the sparsity training script. Below is an example with 8 GPUs:
```sh
export WORLD_SIZE=8
export RANK=-1
export LOCAL_RANK=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=$WORLD_SIZE step1_sparse_training.py --model_name=resnet34 --data_dir=/data1/imagenet --batch_size=128 --eval_baseline --eval_sparse
```

**This is intended purely as an instructional example.**  
The number of GPUs, batch size, and other hyper-parameters should be adjusted depending on your system's availability, model, data, and so on.
