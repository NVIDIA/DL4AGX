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
    # This repository is needed to add QDQ nodes in residual branches
    # Modified the model definition as instructed in the pytorch-quantization toolkit:
    #   https://github.com/NVIDIA/TensorRT/blob/main/tools/pytorch-quantization/examples/torchvision/models/classification/resnet.py#L154-L155
    from torchvision import models
except ImportError:
    print("Error importing pytorch's torchvision repository!")

from train_utils import train_loop, evaluate, data_loading, eval_baseline, prune_trained_model_custom, collect_stats, compute_amax

sys.path.append("./vision/references/classification/")
try:
    from train import load_data
    import utils as utils_vision
except ImportError:
    print("Error importing pytorch's vision repository!")

# QAT Toolkit
from pytorch_quantization import quant_modules


def train_qat(args, criterion, data_loader, data_loader_test, data_loader_val, data_loader_calib):
    # Enable model quantization: relevant layers will be quantized except residual connections
    quant_modules.initialize()
    try:
        # Instantiate model and quantize residual branches (quantize=True)
        model_qat = models.__dict__[args.model_name](pretrained=True, quantize=True)
    except NotImplementedError:
        print("Model definition doesn't accept `quantize` parameter. Instantiating model without quantizing residual connections.")
        model_qat = models.__dict__[args.model_name](pretrained=True)
    # quant_modules.deactivate()

    if args.distributed:
        model_qat = torch.nn.parallel.DistributedDataParallel(model_qat, device_ids=[args.gpu])
    else:
        model_qat = torch.nn.DataParallel(model_qat.cuda(args.device))
    model_qat_without_ddp = model_qat.module

    # Set optimizer
    optimizer = torch.optim.SGD(model_qat.parameters(), lr=args.qat_lr)

    if not args.is_dense_training:
        print("Training Sparse model!")
        prune_trained_model_custom(model_qat, optimizer, compute_sparse_masks=False)

        sparse_ckpt_path = os.path.join(args.output_dir, args.sparse_ckpt)
        if os.path.exists(sparse_ckpt_path):
            print("> Loading Sparse ckpt from {}!".format(sparse_ckpt_path))
            try:
                load_dict = torch.load(sparse_ckpt_path)
            except Exception:
                print("Loading checkpoint from distributed model. Mapping GPU location to local single-GPU setting.")
                load_dict = torch.load(sparse_ckpt_path, map_location="cuda:{}".format(args.device))
            try:
                model_qat.load_state_dict(load_dict["model_state_dict"])  # , strict=False)
            except Exception:
                model_qat_without_ddp.load_state_dict(load_dict["model_state_dict"], strict=False)

    qat_ckpt_path = os.path.join(args.output_dir, args.qat_ckpt)
    if os.path.exists(qat_ckpt_path) and not args.rewrite_qat_weights:
        print("> Loading QAT ckpt from {}!".format(qat_ckpt_path))
        load_dict = torch.load(qat_ckpt_path)
        model_qat_without_ddp.load_state_dict(load_dict["model_state_dict"])
    else:
        # ======== Model calibration ========
        print("> Calibration started...")
        calibrated_ckpt = os.path.join(args.output_dir, "calibrated_ckpt.pth")
        if os.path.exists(calibrated_ckpt):
            checkpoint = torch.load(calibrated_ckpt, map_location="cuda:{}".format(args.device))
            model_qat_without_ddp.load_state_dict(checkpoint, strict=False)
        else:
            collect_stats(
                model_qat_without_ddp,
                data_loader_calib,
                num_batches=len(data_loader_calib),
            )
            amax_computation_method = "entropy"
            compute_amax(model_qat_without_ddp, method=amax_computation_method)
            # Save the calibrated model
            torch.save(model_qat_without_ddp.state_dict(), calibrated_ckpt)

        # ======== QAT fine-tuning ========
        print("> Fine-tuning started...")

        # Set LR scheduler
        if args.lr_scheduler == "step":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_step_size,
                gamma=args.lr_gamma
            )
        elif args.lr_scheduler == "multistep":
            main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
            )
        else:
            raise ("LR Scheduler {} not supported!".format(args.lr_scheduler))
        if args.lr_warmup_epochs > 0:
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                milestones=[args.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler

        # Training loop
        train_loop(model_qat, model_qat_without_ddp, criterion, optimizer, data_loader, data_loader_val,
                   torch.device("cuda:{}".format(args.device)),
                   lr_scheduler=lr_scheduler, epoch=args.qat_epoch, args=args,
                   summary_writer_dir=os.path.join(args.output_dir, "logs", "quant"),
                   save_ckpt_path=qat_ckpt_path, opset=13,
                   steps_per_epoch=args.qat_steps_per_epoch)

        # Load BEST model
        if os.path.exists(qat_ckpt_path):
            print("> Loading QAT ckpt from {}!".format(qat_ckpt_path))
            load_dict = torch.load(qat_ckpt_path)
            model_qat_without_ddp.load_state_dict(load_dict["model_state_dict"])

    # Evaluate model
    acc1, acc5 = None, None
    if args.eval_qat:
        with torch.no_grad():
            acc1, acc5, _ = evaluate(model_qat_without_ddp, criterion, data_loader_test, device="cuda", print_freq=args.print_freq)

    return model_qat, acc1, acc5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script fine-tunes a sparse or dense model via QAT.")
    parser.add_argument("--model_name", type=str, default="resnet34",
                        help="See more model names at https://pytorch.org/vision/stable/models.html and "
                             "  https://github.com/pytorch/vision/tree/main/torchvision/models")
    parser.add_argument("--data_dir", type=str, default="/media/Data/imagenet_data", help="Path to ImageNet dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--train_data_size", type=int, default=None,
                        help="If None, take the entire train data. Otherwise, take subset.")
    parser.add_argument("--test_data_size", type=int, default=None,
                        help="Dataset to be used for the final model evaluation (to obtain accuracy)."
                             " If None, take the entire val data. Otherwise, take subset.")
    parser.add_argument("--calib_data_size", type=int, default=68,
                        help="Dataset to be used for model calibration."
                             " If None, take the entire val data. Otherwise, take subset.")
    parser.add_argument("--val_data_size", type=int, default=None,
                        help="Dataset to be used during training to check for best checkpoint."
                             " If None, take the entire val data. Otherwise, take subset."
                             " Test and Val data are obtained from the same dataset. The only difference is the number"
                             "   of samples. The motivation behind this is that a small val data should be enough to "
                             "   check for the best checkpoint while removing the time bottleneck of the validation"
                             "   step during training. After training is done, the model can then be evaluated on the "
                             "   complete val data.")
    parser.add_argument("--device", type=int, default=0, help="GPU number.")
    parser.add_argument("--output_dir", type=str, default="./weights_qat",
                        help="Path to save outputs (log files, checkpoint, ...).")
    # Sparse params
    parser.add_argument("--sparse_ckpt", type=str, default="sparse-finetuned_best.pth",
                        help="Sparse checkpoint filename (must be inside `output_dir`). If checkpoint exists, simply "
                             "load it. Otherwise, perform Sparse fine-tuning and save checkpoint.")
    # QAT params
    parser.add_argument("--qat_epoch", type=int, default=10,
                        help="Number of epochs to fine-tune QAT model.")
    parser.add_argument("--qat_steps_per_epoch", type=int, default=500,
                        help="Steps per epoch: number of steps = train_data_size/batch_size."
                             " If None, use the entire train data in each epoch. Otherwise, use a subset."
                             " Note that setting train_data_size=500*bs is equivalent to setting train_data_size=None"
                             " and steps_per_epoch=500. The only difference is that by setting train_data_size "
                             " directly, it will update the train_loop verbose print.")
    parser.add_argument("--qat_lr", type=float, default=0.001, help="Base learning rate for QAT workflow.")
    parser.add_argument("--qat_ckpt", type=str, default="quant-finetuned_best.pth",
                        help="QAT checkpoint filename (must be inside `output_dir` and of type .pth)."
                             " If checkpoint exists, simply load it. "
                             " Otherwise, perform QAT fine-tuning and save checkpoint.")
    # LR scheduler params
    parser.add_argument("--lr_scheduler", default="multistep", type=str, help="LR Scheduler, options={multistep, step}")
    parser.add_argument("--lr_warmup_method", default="constant", type=str, help="Warmup method, options={constant, linear}")
    parser.add_argument("--lr_warmup_epochs", default=1, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr_warmup_decay", default=0.1, type=float, help="the decay for lr")  # 0.01
    parser.add_argument("--lr_step_size", default=4, type=int,
                        help="Decrease lr every step-size epochs. Needed for StepLR.")
    parser.add_argument("--lr_gamma", default=0.1, type=float,
                        help="Decrease lr by a factor of lr-gamma. Needed for both Step and MultiStepR.")
    parser.add_argument('--lr_milestones', nargs='+', type=int, default=[2, 7],
                        help='Milestones for MultiStepLR scheduler. Use like: --milestones 1 2 7')
    # torchvision args
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
    parser.add_argument("--eval_qat", dest="eval_qat", action="store_true", help="Evaluate QAT model.")
    parser.add_argument("--rewrite_qat_weights", dest="rewrite_qat_weights", action="store_true", help="Rewrite QAT checkpoint if it exists.")
    parser.add_argument("--is_dense_training", dest="is_dense_training", action="store_true",
                        help="True if we should activate Dense QAT training instead of Sparse.")

    args = parser.parse_args()

    utils_vision.init_distributed_mode(args)
    if args.distributed:
        print("Running distributed script with world size of {}".format(args.world_size))
    else:
        print("Running script in non-distributed manner!")

    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Data loading
    print("---------- Loading data ----------")
    _, _, data_loader_calib = data_loading(
        args.data_dir, args.batch_size, args,
        val_data_size=args.calib_data_size
    )
    data_loader, data_loader_test, data_loader_val = data_loading(
        args.data_dir, args.batch_size, args,
        args.train_data_size, args.test_data_size, args.val_data_size
    )

    # Set loss criteria
    criterion = nn.CrossEntropyLoss()

    # ############# BASELINE ##################
    assert hasattr(models, args.model_name), print("Model {} not supported!".format(args.model_name))
    model, acc1, acc5 = eval_baseline(args, criterion, data_loader_test)

    # ############# QAT #######################
    if args.is_dense_training:
        print("---------- Fine-tuning Dense as QAT model for {} epochs ----------".format(args.qat_epoch))
    else:
        print("---------- Fine-tuning Sparse as QAT model for {} epochs ----------".format(args.qat_epoch))
    model_qat, acc1_qat, acc5_qat = train_qat(
        args, criterion, data_loader, data_loader_test, data_loader_val, data_loader_calib
    )

    # ############ Write logs to 'out.log' and Save args into 'args.json' ###############
    results_str = " ------------ Evaluation Results ------------\n"
    if args.eval_baseline:
        results_str += "Baseline: Top-1 {:.3f}%, Top-5: {:.3f}%\n".format(acc1, acc5)
    if args.eval_qat:
        results_str += "QAT: Top-1 {:.3f}%, Top-5 {:.3f}%\n".format(acc1_qat, acc5_qat)
    results_str += " ------------ CMD -------------------\n"
    results_str += '\n'.join(sys.argv[1:])
    with open(os.path.join(args.output_dir, "out_qat.log"), 'w') as f:
        f.write(results_str)

    with open(os.path.join(args.output_dir, "args_qat.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print("End!")
