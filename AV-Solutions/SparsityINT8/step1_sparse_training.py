#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import copy
import os
import sys
import argparse
import torch
import torch.utils.data
from torch import nn
import json

# from torchvision import models
sys.path.insert(0, "./vision")
try:
    from torchvision import models
except ImportError:
    print("Error importing pytorch's torchvision repository!")

from train_utils import train_loop, evaluate, data_loading, eval_baseline, prune_trained_model_custom, \
    get_optimizer, get_lr_scheduler

sys.path.append("./vision/references/classification/")
try:
    from train import get_args_parser
    import utils as utils_vision
except ImportError:
    print("Error import pytorch's vision repository!")


def train_sparse(model, args, criterion, data_loader, data_loader_test, data_loader_val):
    # Clone model to make sure the original model is not modified
    model_sparse = copy.deepcopy(model)

    # Set optimizer
    parameters = utils_vision.set_weight_decay(
        model_sparse,
        args.sparse_weight_decay,
        norm_weight_decay=None,
        custom_keys_weight_decay=None,
    )
    optimizer = get_optimizer(args, parameters)

    if args.distributed:
        model_sparse = torch.nn.parallel.DistributedDataParallel(model_sparse, device_ids=[args.gpu])
    else:
        model_sparse = torch.nn.DataParallel(model_sparse.cuda(0))
    model_sparse_without_ddp = model_sparse.module
    # Initialize sparsity mode before loading checkpoints and/or starting training.
    #   Apart from the import statement, it is sufficient to add just the following line of code before the training
    #   phase to augment the model and the optimizer for sparse training/inference:
    prune_trained_model_custom(model_sparse, optimizer, allow_recompute_mask=True)
    model_sparse.to(args.device)

    sparse_ckpt_path = os.path.join(args.output_dir, args.sparse_ckpt)
    if os.path.exists(sparse_ckpt_path) and not args.rewrite_sparse_weights:
        print("> Loading Sparse ckpt from {}!".format(sparse_ckpt_path))
        try:
            load_dict = torch.load(sparse_ckpt_path)
        except Exception:
            print("Loading checkpoint from distributed model. Mapping GPU location to local single-GPU setting.")
            load_dict = torch.load(sparse_ckpt_path, map_location="cuda:0")  # "cuda:{}".format(args.device))
        try:
            model_sparse.load_state_dict(load_dict["model_state_dict"])  # , strict=False)
        except Exception:
            model_sparse_without_ddp.load_state_dict(load_dict["model_state_dict"], strict=False)
    else:
        print("> Fine-tuning stage started...")
        # Set LR scheduler
        lr_scheduler = get_lr_scheduler(args, optimizer)

        # Training loop
        train_loop(model_sparse, model_sparse_without_ddp, criterion, optimizer, data_loader, data_loader_val,
                   torch.device(args.device),
                   lr_scheduler=lr_scheduler, epoch=args.sparse_epoch, args=args,
                   summary_writer_dir=os.path.join(args.output_dir, "logs", "sparse"),
                   save_ckpt_path=sparse_ckpt_path, opset=13,
                   steps_per_epoch=args.sparse_steps_per_epoch)

        # Load BEST model
        if os.path.exists(sparse_ckpt_path):
            print("> Loading Sparse ckpt from {}!".format(sparse_ckpt_path))
            load_dict = torch.load(sparse_ckpt_path)
            model_sparse_without_ddp.load_state_dict(load_dict["model_state_dict"])

    # Evaluate model
    acc1, acc5 = None, None
    if args.eval_sparse:
        with torch.no_grad():
            acc1, acc5, _ = evaluate(model_sparse_without_ddp, criterion, data_loader_test, device="cuda", print_freq=args.print_freq)

    return model_sparse, acc1, acc5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script Sparsifies a Dense model and fine-tunes it.")
    parser.add_argument("--model_name", type=str, default="resnet34",
                        help="See more model names at https://pytorch.org/vision/stable/models.html and "
                             "  https://github.com/pytorch/vision/tree/main/torchvision/models")
    parser.add_argument("--data_dir", type=str, default="/media/Data/imagenet_data", help="Path to ImageNet dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--train_data_size", type=int, default=None,
                        help="Dataset to be used during training."
                             " If None, take the entire train data. Otherwise, take subset.")
    parser.add_argument("--test_data_size", type=int, default=None,
                        help="Dataset to be used for the final model evaluation (to obtain accuracy)."
                             " If None, take the entire val data. Otherwise, take subset.")
    parser.add_argument("--val_data_size", type=int, default=None,
                        help="Dataset to be used during training to check for best checkpoint."
                             " If None, take the entire val data. Otherwise, take subset."
                             " Test and Val data are obtained from the same dataset. The only difference is the number"
                             "   of samples. The motivation behind this is that a small val data should be enough to "
                             "   check for the best checkpoint while removing the time bottleneck of the validation"
                             "   step during training. After training is done, the model can then be evaluated on the "
                             "   complete val data.")
    parser.add_argument("--device", type=str, default="cuda", help="Hardware to run code on.")
    parser.add_argument("--output_dir", type=str, default="./weights_qat",
                        help="Path to save outputs (log files, checkpoint, ...).")
    # Sparse params
    parser.add_argument("--sparse_epoch", type=int, default=30,
                        help="Number of epochs to fine-tune Sparse model.")
    parser.add_argument("--sparse_steps_per_epoch", type=int, default=None,
                        help="Steps per epoch: number of steps = train_data_size/batch_size."
                             " If None, use the entire train data in each epoch. Otherwise, use a subset.")
    parser.add_argument("--sparse_lr", type=float, default=0.1, help="Base learning rate for Sparse workflow.")
    parser.add_argument("--sparse_weight_decay", type=float, default=1e-4, help="Weight decay for Sparse workflow.")
    parser.add_argument("--sparse_momentum", type=float, default=0.9, help="Momentum for Sparse workflow.")
    parser.add_argument("--sparse_ckpt", type=str, default="sparse-finetuned_best.pth",
                        help="Sparse checkpoint filename (must be inside `output_dir` and of type .pth)."
                             " If checkpoint exists, simply load it."
                             " Otherwise, perform Sparse fine-tuning and save checkpoint.")
    # torchvision args
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr_scheduler", default="constant", type=str, help="LR Scheduler, options={multistep, constant, step}")
    parser.add_argument("--lr_warmup_epochs", default=5, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr_warmup_decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr_step_size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr_gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr_min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--label_smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--print_freq", default=20, type=int, help="print frequency")

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--model_ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model_ema_steps", type=int, default=32,
                        help="the number of iterations that controls how often to update the EMA model (default: 32)")
    parser.add_argument("--model_ema_decay", type=float, default=0.99998,
                        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")
    parser.add_argument("--clip_grad_norm", default=None, type=float, help="the maximum gradient norm (default None)")

    # Dataloader arguments from 'vision' repo
    parser.add_argument("--cache-dataset", dest="cache_dataset", action="store_true",
                        help="Cache the datasets for quicker initialization. It also serializes the transforms")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Only test the model")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument("--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    # Distributed
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Eval params
    parser.add_argument("--save_baseline", dest="save_baseline", action="store_true", help="Save baseline model.")
    parser.add_argument("--eval_baseline", dest="eval_baseline", action="store_true", help="Evaluate baseline model.")
    parser.add_argument("--eval_sparse", dest="eval_sparse", action="store_true", help="Evaluate sparse model.")
    parser.add_argument("--rewrite_sparse_weights", dest="rewrite_sparse_weights", action="store_true",
                        help="Rewrite Sparse checkpoint if it exists.")
    args = parser.parse_args()

    # The function below creates args.distributed and args.gpu/rank
    utils_vision.init_distributed_mode(args)
    if args.distributed:
        print("Running distributed script with world size of {}".format(args.world_size))
    else:
        print("Running script in non-distributed manner!")

    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except FileExistsError:
            print("Directory {} exists, not creating it again.".format(args.output_dir))

    # Data loading
    print("---------- Loading data ----------")
    data_loader, data_loader_test, data_loader_val = data_loading(
        args.data_dir, args.batch_size, args,
        args.train_data_size, args.test_data_size, args.val_data_size
    )

    # Set loss criteria
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ############# BASELINE ##################
    assert hasattr(models, args.model_name), print("Model {} not supported!".format(args.model_name))
    model, acc1, acc5 = eval_baseline(args, criterion, data_loader_test)

    # ############### SPARSE ###################
    print("---------- Fine-tuning Dense as Sparse model (FP32) for {} epochs ----------".format(args.sparse_epoch))
    model_sparse, acc1_sparse, acc5_sparse = train_sparse(
        model, args, criterion, data_loader, data_loader_test, data_loader_val
    )

    # ############ Write logs to 'out.log' and Save args into 'args.json' ###############
    results_str = " ------------ Evaluation Results ------------\n"
    if args.eval_baseline:
        results_str += "Baseline: Top-1 {:.3f}%, Top-5: {:.3f}%\n".format(acc1, acc5)
    if args.eval_sparse:
        results_str += "Sparse: Top-1 {:.3f}%, Top-5 {:.3f}%\n".format(acc1_sparse, acc5_sparse)
    results_str += " ------------ CMD -------------------\n"
    results_str += '\n'.join(sys.argv[1:])
    with open(os.path.join(args.output_dir, "out_sparse.log"), 'w') as f:
        f.write(results_str)

    with open(os.path.join(args.output_dir, "args_sparse.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print("End!")
