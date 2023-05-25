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
# > prune_trained_model_custom()
# Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
#
# BSD-3-Clause license
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# > train_one_epoch(), evaluate()
# Copyright (c) Soumith Chintala 2016. All rights reserved.
#
# BSD 3-Clause License
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""
Modifications:
- prune_trained_model_custom():
  1. Abstracted 'allow_recompute_mask' and 'allow_permutation' arguments;
  2. Enabled sparse mask computation as optional.
- train_one_epoch(): added steps_per_epoch if condition.
- evaluate():
  1. Returned Top-5 accuracy and loss on top of the Top-1 accuracy;
  2. Commented out "and torch.distributed.get_rank() == 0" if.
"""

import copy
import os
import sys
import time
import torch
import torch.utils.data
from torch import nn
import warnings
from torch.utils.tensorboard import SummaryWriter

# from torchvision import models
sys.path.insert(0, "./vision")
try:
    from torchvision import models
except ImportError:
    print("Error importing pytorch's torchvision repository!")

sys.path.append(os.path.abspath("./vision/references/classification"))
try:
    import utils
    from train import load_data
except ImportError:
    print("Error import pytorch's vision repository!")

# Sparsity Toolkit
sys.path.append("./apex")
try:
    from apex.contrib.sparsity import ASP
except ImportError:
    print("Error importing `apex`!")

# Quantization Toolkit
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
import tqdm


def prune_trained_model_custom(model, optimizer, allow_recompute_mask=False, allow_permutation=True,
                               compute_sparse_masks=True):
    """ Adds mask buffers to model (init_model_for_pruning), augments optimize, and computes masks if .
    Source: https://github.com/NVIDIA/apex/blob/52c512803ba0a629b58e1c1d1b190b4172218ecd/apex/contrib/sparsity/asp.py#L299
    Modifications:
      1) Abstracted 'allow_recompute_mask' and 'allow_permutation' arguments
      2) Enabled sparse mask computation as optional
    """
    asp = ASP()
    asp.init_model_for_pruning(
        model,
        mask_calculator="m4n2_1d",
        verbosity=2,
        whitelist=[torch.nn.Linear, torch.nn.Conv2d],
        allow_recompute_mask=allow_recompute_mask,
        allow_permutation=allow_permutation
    )
    asp.init_optimizer_for_pruning(optimizer)
    if compute_sparse_masks:
        asp.compute_sparse_masks()
    return asp


def data_loading(data_path, batch_size, torchvision_args, train_data_size=None, test_data_size=None, val_data_size=None,
                 distributed=False, drop_last=False):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, torchvision_args)
    dataset_val = copy.deepcopy(dataset_test)
    val_sampler = copy.deepcopy(test_sampler)

    # Take subset if != None
    if train_data_size:
        dataset = torch.utils.data.Subset(dataset, list(range(0, train_data_size)))
        if torchvision_args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
    if test_data_size:
        dataset_test = torch.utils.data.Subset(dataset_test, list(range(0, test_data_size)))
        if torchvision_args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            test_sampler = torch.utils.data.RandomSampler(dataset_test)
    if val_data_size:
        dataset_val = torch.utils.data.Subset(dataset_val, list(range(0, val_data_size)))
        if torchvision_args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
        else:
            val_sampler = torch.utils.data.RandomSampler(dataset_val)

    # Make dataloader
    train_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=train_sampler, drop_last=drop_last)

    test_data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size,
        sampler=test_sampler, drop_last=drop_last)

    val_data_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size,
        sampler=val_sampler, drop_last=drop_last)

    return train_data_loader, test_data_loader, val_data_loader


def eval_baseline(args, criterion, data_loader_test):
    model = models.__dict__[args.model_name](pretrained=True)
    model.to(args.device)

    acc1, acc5, loss = None, None, None
    if args.eval_baseline:
        print("---------- Evaluating baseline model ----------")
        with torch.no_grad():
            acc1, acc5, loss = evaluate(model, criterion, data_loader_test, device=args.device, print_freq=args.print_freq)

    if args.save_baseline:
        ckpt_path = os.path.join(args.output_dir, "baseline.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            "acc1_val": acc1,
            "acc5_val": acc5,
            "loss_val": loss,
            'args': args
        }, ckpt_path)
        export_onnx(model, ckpt_path.replace(".pth", ".onnx"), args.batch_size, val_crop_size=args.val_crop_size)

    return model, acc1, acc5


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, steps_per_epoch=None,
                    model_ema=None, scaler=None):
    """
    Codebase: torch/vision/references/classification/train.py
    Modification: added steps_per_epoch.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        if steps_per_epoch is not None and i >= steps_per_epoch:
            break
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    return metric_logger


def train_loop(model, model_without_ddp, criterion, optimizer, data_loader, data_loader_val, device, epoch, args, summary_writer_dir,
               save_ckpt_path, steps_per_epoch=None, model_ema=None, scaler=None, lr_scheduler=None, opset=13):
    def _save_model(ep, ckpt_path, acc1_val, acc5_val, current_val_loss):
        torch.save({
            'epoch': ep,
            'model_state_dict': model_without_ddp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,  # ADDED
            'loss': criterion,
            "acc1_val": acc1_val,
            "acc5_val": acc5_val,
            "loss_val": current_val_loss,
            'args': args  # ADDED
        }, ckpt_path)
        # Export to ONNX
        export_onnx(model_without_ddp, ckpt_path.replace(".pth", ".onnx"), args.batch_size,
                    val_crop_size=args.val_crop_size, opset_version=opset)

    summary_writer = SummaryWriter(summary_writer_dir)
    best_val_loss = float("inf")

    tick = time.time()
    for e in range(0, epoch):
        print("Epoch {}/{}".format(e, epoch))
        if args.distributed:
            data_loader.sampler.set_epoch(e)
            torch.distributed.barrier()
        metric_logger = train_one_epoch(
            model, criterion, optimizer, data_loader, device, e, args, steps_per_epoch=steps_per_epoch,
            model_ema=model_ema, scaler=scaler
        )
        if lr_scheduler is not None:
            lr_scheduler.step()
        if args.distributed:
            torch.distributed.barrier()

        # ======== Validation step ========
        acc1_val, acc5_val, loss_val = evaluate(model, criterion, data_loader_val, device, print_freq=args.print_freq)
        # Save the BEST model
        current_val_loss = loss_val
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            _save_model(e, save_ckpt_path, acc1_val, acc5_val, current_val_loss)

        # ======== Summary Writer (Tensorboard) log ========
        summary_writer.add_scalar("lr", metric_logger.lr.value, e)

        summary_writer.add_scalar("Loss_train/epoch", metric_logger.loss.global_avg, e)
        summary_writer.add_scalar("Accuracy_top1_train/epoch", metric_logger.acc1.global_avg, e)
        summary_writer.add_scalar("Accuracy_top5_train/epoch", metric_logger.acc5.global_avg, e)

        summary_writer.add_scalar("Loss_val/epoch", loss_val, e)
        summary_writer.add_scalar("Accuracy_top1_val/epoch", acc1_val, e)
        summary_writer.add_scalar("Accuracy_top5_val/epoch", acc5_val, e)

        # Save the FINAL model
        _save_model(e, save_ckpt_path.replace("_best.pth", "_final.pth"), acc1_val, acc5_val, current_val_loss)

    tock = time.time()
    time_min = (tock - tick) / (1000 * 60)
    print("Training took {} minutes or {} hours!".format(time_min, time_min / 60))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    """
    Codebase: torch/vision/references/classification/train.py
    Modifications:
      1) returning Top-5 accuracy and loss on top of the Top-1 accuracy, and
      2) commented out "and torch.distributed.get_rank() == 0" if.
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    print("Num processed samples: {}".format(num_processed_samples))
    print("data_loader len: {}".format(len(data_loader.dataset)))
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        # and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f},"
          f" Acc@5 {metric_logger.acc5.global_avg:.3f},"
          f" loss {metric_logger.loss.global_avg:.5f}")
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, metric_logger.loss.global_avg


def export_onnx(model, onnx_filename, batch_onnx, val_crop_size=224, opset_version=13, verbose=False,
                do_constant_folding=True, trace_model=False):
    model.eval()
    # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    dummy_input = torch.randn(batch_onnx, 3, val_crop_size, val_crop_size, device="cuda")
    try:
        print("Exporting ONNX model with input {} to {} with opset {}!".format(dummy_input.shape, onnx_filename, opset_version))
        model_tmp = model
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            #  '.module' is necessary here because model is wrapped in torch.nn.DataParallel
            model_tmp = model.module
        if trace_model:
            model_tmp = torch.jit.trace(model_tmp, dummy_input)
        torch.onnx.export(
            model_tmp, dummy_input, onnx_filename,
            verbose=verbose,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding
        )
    except ValueError:
        print("Failed to export to ONNX")
        return False

    return True


def get_optimizer(args, parameters):
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.sparse_lr,
            momentum=args.sparse_momentum,
            weight_decay=args.sparse_weight_decay
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=args.sparse_lr,
            momentum=args.sparse_momentum,
            weight_decay=args.sparse_weight_decay,
            eps=0.0316,
            alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.sparse_lr,
            weight_decay=args.sparse_weight_decay
        )
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    return optimizer


def get_lr_scheduler(args, optimizer):
    if args.lr_scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=args.lr_gamma)
    elif args.lr_scheduler == "constant":
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=args.lr_warmup_decay, total_iters=args.sparse_epoch
        )
    elif args.lr_scheduler == "step":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        if args.lr_warmup_epochs > 0:
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
    else:
        raise ("LR Scheduler {} not supported!".format(args.lr_scheduler))

    return lr_scheduler


# ======== Calibration ========
# Source: https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/examples/calibrate_quant_resnet50.ipynb
def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    progress_bar = tqdm.tqdm(total=len(data_loader), leave=True, desc='Evaluation Progress')
    for i, (image, _) in enumerate(data_loader):
        model(image.to(torch.device("cuda:0")))  # .cuda())
        progress_bar.update()
        if i >= num_batches:
            break
    progress_bar.update()

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    """Load calib result"""
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(strict=False, **kwargs)
    model.cuda()
