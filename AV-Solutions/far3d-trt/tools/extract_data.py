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
# modified from https://github.com/megvii-research/Far3D/blob/main/tools/test.py 


# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import os
import torch

import torch

from mmcv import Config

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config',help='test config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    import importlib
    plugin_dir = cfg.plugin_dir
    _module_dir = os.path.dirname(plugin_dir)
    _module_dir = _module_dir.split('/')
    _module_path = _module_dir[0]

    for m in _module_dir[1:]:
        _module_path = _module_path + '.' + m
    print(_module_path)
    plg_lib = importlib.import_module(_module_path)


    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    cfg.data.test.test_mode = True
            
    set_random_seed(0, deterministic=False)    

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    current_scene_token = None
    with open("data/filelist.txt", "wt") as filelist:
        for i, data in enumerate(data_loader):
            lidar2img = data['lidar2img'][0].data[0][0].unsqueeze(0)
            img = data['img'][0].data[0]
            # convert bgr to rgb image
            img = img.flip(2)
            intrinsics = data['intrinsics'][0].data[0][0].unsqueeze(0)
            extrinsics = data['extrinsics'][0].data[0][0].unsqueeze(0)
            img2lidar = lidar2img.inverse()
            ego_pose_inv = data['ego_pose_inv'][0].data[0][0].unsqueeze(0)
            ego_pose = data['ego_pose'][0].data[0][0].unsqueeze(0)
            timestamp = torch.tensor(data['timestamp'][0].data[0][0])
            scene_token = data['img_metas'][0].data[0][0]['scene_token']
            if current_scene_token is None:
                current_scene_token = scene_token
            else:
                if current_scene_token != scene_token:
                    break
            os.makedirs(f'data/{scene_token}', exist_ok=True)
            basename = "data/{}/{:05}_".format(scene_token, i)
            img.cpu().permute(0,1,3,4,2).contiguous().numpy().astype(np.uint8).tofile(basename + "img.bin")
            intrinsics.cpu().numpy().tofile(basename + "intrinsics.bin")
            extrinsics.cpu().numpy().tofile(basename + "extrinsics.bin")
            lidar2img.cpu().numpy().tofile(basename + "lidar2img.bin")
            img2lidar.cpu().numpy().tofile(basename + "img2lidar.bin")
            ego_pose_inv.cpu().numpy().tofile(basename + "ego_pose_inv.bin")
            ego_pose.cpu().numpy().tofile(basename + "ego_pose.bin")
            timestamp.cpu().numpy().tofile(basename + "timestamp.bin")
            #img_paths = data['img_metas'][0].data[0][0]['filename']
            #filelist.write(' '.join(img_paths))
            filelist.write(basename + "\n")
        

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
