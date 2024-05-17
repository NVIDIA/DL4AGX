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

# This fils is modified from test.py in the original repo.
# Modification: adding 2 proxy class TrtEncoderContainer and TrtPtsHeadContainer
#               and then export these two torch.nn.Module to onnx

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch 
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

import argparse
import time
import pickle as pkl
import os
import sys
sys.path.append('./')
import numpy as np

from onnxsim import simplify
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector

from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--section', help='section can be either extract_img_feat or pts_head_memory')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=300, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = MMDataParallel(model, device_ids=[0])
    
    # Wrapper Class for onnx conversion
    class TrtEncoderContainer(torch.nn.Module):
        def __init__(self, mod, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.mod = mod

        def forward(self, img):
            # img: 6, 3, 256, 704
            mod = self.mod
            return mod.extract_img_feat(img, 1)

    # Wrapper Class for onnx conversion
    class TrtPtsHeadContainer(torch.nn.Module):
        def __init__(self, mod, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.mod = mod

        def _post_update_memory(self, data_ego_pose, data_timestamp, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec):
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
            
            # topk proposals
            _, topk_indexes = torch.topk(rec_score, 128, dim=1)
            rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
            rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
            rec_memory = topk_gather(rec_memory, topk_indexes).detach()
            rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
            rec_velo = topk_gather(rec_velo, topk_indexes).detach()

            head = self.mod.pts_bbox_head
            head.memory_embedding = torch.cat([rec_memory, head.memory_embedding], dim=1)
            head.memory_timestamp = torch.cat([rec_timestamp, head.memory_timestamp], dim=1)
            head.memory_egopose= torch.cat([rec_ego_pose, head.memory_egopose], dim=1)
            head.memory_reference_point = torch.cat([rec_reference_points, head.memory_reference_point], dim=1)
            head.memory_velo = torch.cat([rec_velo, head.memory_velo], dim=1)
            head.memory_reference_point = transform_reference_points(head.memory_reference_point, data_ego_pose, reverse=False)            
            head.memory_egopose = data_ego_pose.unsqueeze(1) @ head.memory_egopose

            # cast to float64 out-of-tensorrt
            # head.memory_timestamp -= data_timestamp.unsqueeze(-1).unsqueeze(-1)

        def _pts_head(self, x, pos_embed, cone):
            head = self.mod.pts_bbox_head

            B, N, C, H, W = x.shape
            num_tokens = N * H * W
            memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
            
            # don't do on-the-fly position_embedding
            # head.position_embeding(data, memory_center, topk_indexes, img_metas)

            memory = head.memory_embed(memory)

            # spatial_alignment in focal petr
            memory = head.spatial_alignment(memory, cone)
            pos_embed = head.featurized_pe(pos_embed, memory)

            reference_points = head.reference_points.weight
            reference_points, attn_mask, mask_dict = head.prepare_for_dn(B, reference_points, {})
            query_pos = head.query_embedding(pos2posemb3d(reference_points))
            tgt = torch.zeros_like(query_pos)

            # prepare for the tgt and query_pos using mln.
            query_pos_in = query_pos.detach()
            tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = head.temporal_alignment(query_pos, tgt, reference_points)

            # transformer here is a little different from PETR
            outs_dec, _ = head.transformer(memory, tgt, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)
            outputs_classes = []
            outputs_coords = []
            reference = inverse_sigmoid(reference_points.clone())
            for lvl in range(1):
                outputs_class = head.cls_branches[lvl](outs_dec[lvl])
                tmp = head.reg_branches[lvl](outs_dec[lvl])

                tmp[..., 0:3] += reference[..., 0:3]
                tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            all_cls_scores = torch.stack(outputs_classes)
            all_bbox_preds = torch.stack(outputs_coords)
            all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (head.pc_range[3:6] - head.pc_range[0:3]) + head.pc_range[0:3])

            return pos_embed, reference_points, tgt, temp_memory, temp_pos, \
                   query_pos, query_pos_in, outs_dec, all_cls_scores, all_bbox_preds, rec_ego_pose

        def forward(self, x, pos_embed, cone, 
                    data_timestamp, data_ego_pose, data_ego_pose_inv,
                    memory_embedding, memory_reference_point, memory_timestamp, memory_egopose, memory_velo):
            # x[1, 6, 256, 16, 44]
            # pos_embed[1, 4224, 256]
            # cone[1, 4224, 8]
            head = self.mod.pts_bbox_head

            # memory update before head
            # memory_timestamp += data_timestamp.unsqueeze(-1).unsqueeze(-1)
            memory_egopose = data_ego_pose_inv.unsqueeze(1) @ memory_egopose
            memory_reference_point = transform_reference_points(memory_reference_point, data_ego_pose_inv, reverse=False)
            
            head.memory_timestamp = memory_timestamp[:, :head.memory_len]
            head.memory_reference_point = memory_reference_point[:, :head.memory_len]
            head.memory_embedding = memory_embedding[:, :head.memory_len]
            head.memory_egopose = memory_egopose[:, :head.memory_len]
            head.memory_velo = memory_velo[:, :head.memory_len]
            
            pos_embed, reference_points, tgt, temp_memory, temp_pos, \
            query_pos, query_pos_in, outs_dec, all_cls_scores, all_bbox_preds, rec_ego_pose = \
                self._pts_head(x, pos_embed, cone)
        
            # memory update after head
            self._post_update_memory(data_ego_pose, data_timestamp, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec)

            return all_cls_scores, all_bbox_preds, \
                head.memory_embedding, head.memory_reference_point, head.memory_timestamp, head.memory_egopose, head.memory_velo, \
                reference_points, tgt, temp_memory, temp_pos, query_pos, query_pos_in, outs_dec
                
    model.eval()
    model = model.float()

    if args.section not in ["extract_img_feat", "pts_head_memory"]:
        raise RuntimeError("unknown section {}".format(args.section))
        exit(-1)

    section = args.section
    if args.section == "extract_img_feat":        
        tm = TrtEncoderContainer(model.module)
        arrs = [torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, 6, 3, 256, 704))).float(),]
        input_names=["img"]
        output_names=["img_feats"]

    elif args.section == "pts_head_memory":
        tm = TrtPtsHeadContainer(model.module)
        dmem_init = 640
        arrs = [
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, 6, 256, 16, 44))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, 4224, 256))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, 4224, 8))).float(),  
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, ))).double(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, 4, 4))).float(),  
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, 4, 4))).float(),  
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, dmem_init, 256))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, dmem_init, 3))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, 512, 1))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, dmem_init, 4, 4))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, dmem_init, 2))).float(),
        ]
        input_names=["x", "pos_embed", "cone",
                     "data_timestamp", "data_ego_pose", "data_ego_pose_inv",
                     "pre_memory_embedding",
                     "pre_memory_reference_point",
                     "pre_memory_timestamp",
                     "pre_memory_egopose",
                     "pre_memory_velo",]
        output_names=["all_cls_scores", 
                      "all_bbox_preds", 
                      "post_memory_embedding", 
                      "post_memory_reference_point",
                      "post_memory_timestamp", 
                      "post_memory_egopose", 
                      "post_memory_velo",
                      "reference_points", "tgt", "temp_memory", "temp_pos", "query_pos", "query_pos_in", "outs_dec"]

    tm = tm.float()
    tm.cpu()
    tm.eval()
    tm.training = False
    tm.mod.pts_bbox_head.with_dn = False

    args = tuple(arrs)

    with torch.no_grad():
        torch.onnx.export(
            tm, args,
            "{}.onnx".format(section),
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            verbose=True)

    import onnx
    from onnxsim import simplify
    filename = "{}.onnx".format(section)
    onnx_model = onnx.load(filename)
    onnx_model_simp, check = simplify(onnx_model)
    onnx.save(onnx_model_simp, "simplify_" + filename)

    print(section)

if __name__ == '__main__':
    main()
