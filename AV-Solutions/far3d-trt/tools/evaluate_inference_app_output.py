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
import glob
import numpy as np
import torch
from mmcv import Config
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.datasets import replace_ImageToTensor

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
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    
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
    trt_outputs = []
    for i, data in enumerate(data_loader):
        scene_token = data['img_metas'][0].data[0][0]['scene_token']
        
        if scene_token != current_scene_token:
            # TODO validation of index
            input_files = glob.glob(os.path.join("data", scene_token) + "/*_bboxes.bin")
            if len(input_files) == 0:
                break
            input_files = sorted(input_files)
            for file in input_files:
                bboxes = torch.from_numpy(np.fromfile(file, dtype=np.float32).reshape(300, 7))
                scores = torch.from_numpy(np.fromfile(file.replace('bboxes', 'scores'), dtype=np.float32))
                labels = torch.from_numpy(np.fromfile(file.replace('bboxes', 'labels'), dtype=np.int32))

                trt_results = dict(pts_bbox=dict(boxes_3d=LiDARInstance3DBoxes(bboxes), scores_3d=scores, labels_3d=labels))
                trt_outputs.append(trt_results)
        current_scene_token = scene_token
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    dataset.evaluate(trt_outputs, **eval_kwargs)
        
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
