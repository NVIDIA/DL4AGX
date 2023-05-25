# Run script distributed to 8 GPUs
WORLD_SIZE=1  # 8
RANK=-1  # This is equivalent to $WORLD_SIZE
LOCAL_RANK=0  # 0,1,2,3,4,5,6,7  # gpu

nvidia-smi

torchrun --nproc_per_node=$WORLD_SIZE step1_sparse_training.py \
         --model_name=resnet34 \
         --data_dir="/data1/imagenet/" \
         --batch_size=128 \
         --output_dir="./weights_qat" \
         --sparse_epoch=1 \
         --sparse_steps_per_epoch=20 \
         --sparse_lr=0.1 \
         --sparse_ckpt="sparse-finetuned_best.pth" \
         --eval_baseline --eval_sparse --rewrite_sparse_weights
