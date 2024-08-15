# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import os
import numpy as np
import importlib
import tqdm
import torch
from mmcv import Config
from third_party.uniad_mmdet3d.datasets.builder import build_dataloader, build_dataset

"""
The metadata that needed to be prepared before engine inference:
timestamp, l2g_r_mat, l2g_t, img_metas_scene_token
command, img_metas_can_bus, img_metas_lidar2img
"""

def scene_token_preprocess(scene_token):
    scene_token_list = []
    for ch in scene_token:
        scene_token_list.append(ord(ch))
    return torch.tensor(scene_token_list)

def process_metadata(data_loader, folder, trt_path, onnx_path, stop_id):
    assert stop_id>5
    for sample_id, data in tqdm.tqdm(enumerate(data_loader)):
        if sample_id >= stop_id:
            break
        img_metas = data["img_metas"][0].data
        timestamp = data["timestamp"][0] if data["timestamp"] is not None else None
        trt_inputs = dict()
        trt_inputs["img_metas_lidar2img"] = np.float32(np.stack(img_metas[0][0]["lidar2img"])[None,...])
        trt_inputs["img_metas_scene_token"] = np.float32(scene_token_preprocess(img_metas[0][0]["scene_token"]).cpu().numpy())
        trt_inputs["l2g_t"] = np.float32(data["l2g_t"].cpu().numpy())
        trt_inputs["l2g_r_mat"] = np.float32(data["l2g_r_mat"].cpu().numpy())
        trt_inputs["timestamp"] = np.int64(timestamp[0].cpu().numpy())
        trt_inputs["command"] = np.float32(data["command"][0].cpu().numpy())
        trt_inputs["img_metas_can_bus"] = np.float32(img_metas[0][0]["can_bus"])

        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(trt_path):
            os.makedirs(trt_path)
        if not os.path.exists(onnx_path):
            os.makedirs(onnx_path)

        if sample_id <= 5:
            onnx_inputs = dict()
            onnx_inputs["img_metas_lidar2img"] = np.float32(np.stack(img_metas[0][0]["lidar2img"])[None,...])
            onnx_inputs["img_metas_scene_token"] = np.float32(scene_token_preprocess(img_metas[0][0]["scene_token"]).cpu().numpy())
            onnx_inputs["l2g_t"] = np.float32(data["l2g_t"].cpu().numpy())
            onnx_inputs["l2g_r_mat"] = np.float32(data["l2g_r_mat"].cpu().numpy())
            onnx_inputs["timestamp"] = np.int64(timestamp[0].cpu().numpy())
            onnx_inputs["command"] = np.float32(data["command"][0].cpu().numpy())
            onnx_inputs["img_metas_can_bus"] = np.float32(img_metas[0][0]["can_bus"])
            onnx_inputs['img'] = np.float32(data["img"][0].data[0].cpu().numpy())
            onnx_inputs['gt_segmentation'] = np.float32(data["gt_segmentation"][0].cpu().numpy())
            onnx_inputs['gt_lane_masks'] = np.float32(data["gt_lane_masks"][0].cpu().numpy())
            onnx_inputs['gt_lane_labels'] = np.float32(data["gt_lane_labels"][0].cpu().numpy())
            for key in onnx_inputs:
                if not os.path.exists(os.path.join(onnx_path, key)):
                    os.makedirs(os.path.join(onnx_path, key))
                np.save(os.path.join(onnx_path, key, str(sample_id)+".npy"), onnx_inputs[key])
        
        for key in trt_inputs:
            if not os.path.exists(os.path.join(trt_path, key)):
                os.makedirs(os.path.join(trt_path, key))
            trt_inputs[key] = np.ravel(trt_inputs[key]).astype(np.float32)
            trt_inputs[key] = np.ascontiguousarray(trt_inputs[key]).astype(np.float32)
            trt_inputs[key].tofile(os.path.join(trt_path, key, str(sample_id)+".bin"))
    return
    
def parse_args():
    parser = argparse.ArgumentParser(description="UniAD metadata-processor.")
    parser.add_argument("--config", type=str, default='./projects/configs/stage2_e2e/tiny_imgx0.25_e2e_dump_trt_input.py', help="test config file path")
    parser.add_argument("--dump_folder", type=str, default='./nuscenes_np' ,help="input meta dump folder")
    parser.add_argument("--dump_trt_path", type=str, default='./nuscenes_np/uniad_trt_input' ,help="input meta dump path")
    parser.add_argument("--dump_onnx_path", type=str, default='./nuscenes_np/uniad_onnx_input' ,help="input meta dump path")
    parser.add_argument("--num_frame", type=int, default=69, help="total number of frames to process")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if hasattr(cfg, "plugin") and cfg.plugin and hasattr(cfg, "plugin_dir"):
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        _module_dir = _module_dir.split('/')
        _module_path = _module_dir[0]

        for m in _module_dir[1:]:
            _module_path = _module_path + "." + m
        print("[INFO] Loading mmdet3d plugin from: ", _module_path)
        plg_lib = importlib.import_module(_module_path)

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    process_metadata(data_loader, args.dump_folder, args.dump_trt_path, args.dump_onnx_path, args.num_frame)