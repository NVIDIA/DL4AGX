# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This fils is modified from export.py in the original repo.
# Modification: added dcnv4_function_handler for DCNv4 exporting to onnx then tensorrt

# --------------------------------------------------------
# DCNv4
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import time
import argparse

import torch
from tqdm import tqdm

from config import get_config
from models import build_model

from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help

def dcnv4_function_handler(g, *args, **kwargs):
    version_major = int(torch.__version__.split(".")[0])
    if version_major < 2:
        # signature is different between pytorch 1.x and 2.x
        # 1.x: handler(g: torch._C.Graph, n: torch._C.Node, *args, **kwargs)
        # 2.x: handler(g: torch._C.Graph, *args, **kwargs)
        _, args = args[0], args[1:]

    # kh, kw, sh, sw, ph, pw, dh, dw, g, gc, off_s, step, remove_center
    keys = ["kh_i", "kw_i", 
            "sh_i", "sw_i", 
            "ph_i", "pw_i", 
            "dh_i", "dw_i", 
            "group_i", "group_channels_i", 
            "offscale_f", 
            "step_i", "remove_center_i"]
    kw = {}
    # first two are input_feature and offset
    for k, v in zip(keys, args[2:]):
        kw[k] = v
    return g.op("custom_op::DCNv4_Plugin", args[0], args[1], **kw)

register_custom_op_symbolic("prim::PythonOp", dcnv4_function_handler, 1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='internimage_t_1k_224')
    parser.add_argument('--ckpt_dir', type=str,
                        default='/mnt/petrelfs/share_data/huangzhenhang/code/internimage/checkpoint_dir/new/cls')
    parser.add_argument('--onnx', default=False, action='store_true')
    parser.add_argument('--trt', default=False, action='store_true')

    args = parser.parse_args()
    args.cfg = os.path.join('./configs', f'{args.model_name}.yaml')
    args.ckpt = os.path.join(args.ckpt_dir, f'{args.model_name}.pth')
    args.size = int(args.model_name.split('.')[0].split('_')[-1])

    cfg = get_config(args)
    return args, cfg

def get_model(args, cfg):
    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location='cpu')['model']

    model.load_state_dict(ckpt)
    return model

def speed_test(model, input):
    # warmup
    for _ in tqdm(range(100)):
        _ = model(input)

    # speed test
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(100)):
        _ = model(input)
    end = time.time()
    th = 100 / (end - start)
    print(f"using time: {end - start}, throughput {th}")

def torch2onnx(args, cfg):
    model = get_model(args, cfg).cuda()

    # speed_test(model)

    onnx_name = f'{args.model_name}.onnx'
    torch.onnx.export(model,
                      torch.rand(1, 3, args.size, args.size).cuda(),
                      onnx_name,
                      input_names=['input'],
                      output_names=['output'])

    return model

def onnx2trt(args):
    from mmdeploy.backend.tensorrt import from_onnx

    onnx_name = f'{args.model_name}.onnx'
    from_onnx(
        onnx_name,
        args.model_name,
        dict(
            input=dict(
                min_shape=[1, 3, args.size, args.size],
                opt_shape=[1, 3, args.size, args.size],
                max_shape=[1, 3, args.size, args.size],
            )
        ),
        max_workspace_size=2**30,
    )

def check(args, cfg):
    from mmdeploy.backend.tensorrt.wrapper import TRTWrapper

    model = get_model(args, cfg).cuda()
    model.eval()
    trt_model = TRTWrapper(f'{args.model_name}.engine',
                           ['output'])

    x = torch.randn(1, 3, args.size, args.size).cuda()

    torch_out = model(x)
    trt_out = trt_model(dict(input=x))['output']

    print('torch out shape:', torch_out.shape)
    print('trt out shape:', trt_out.shape)

    print('max delta:', (torch_out - trt_out).abs().max())
    print('mean delta:', (torch_out - trt_out).abs().mean())

    speed_test(model, x)
    speed_test(trt_model, dict(input=x))

def main():
    args, cfg = get_args()

    if args.onnx or args.trt:
        torch2onnx(args, cfg)
        print('torch -> onnx: succeess')

    if args.trt:
        onnx2trt(args)
        print('onnx -> trt: success')
        check(args, cfg)

if __name__ == '__main__':
    main()
