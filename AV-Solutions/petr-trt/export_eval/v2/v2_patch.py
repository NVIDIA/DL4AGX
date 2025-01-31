# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import math
import torch
import torch.nn.functional as F
from mmdet.models.utils.transformer import inverse_sigmoid

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

def constantization_PETRv2Head_forward(self, mlvl_feats, img_metas):
    x = mlvl_feats[self.position_level]
    batch_size, num_cams = x.size(0), x.size(1)

    input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
    masks = x.new_ones(
        (batch_size, num_cams, input_img_h, input_img_w))
    for img_id in range(batch_size):
        for cam_id in range(num_cams):
            img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
            masks[img_id, cam_id, :img_h, :img_w] = 0
        
    x = self.input_proj(x.flatten(0, 1))
    x = x.view(batch_size, num_cams, *x.shape[-3:])

    # interpolate masks to have the same spatial shape with x
    masks = F.interpolate(
        masks, size=x.shape[-2:]).to(torch.bool)
    constantization_PETRv2Head_forward.masks = masks

    if self.with_position:
        coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
        if self.with_fpe:
            coords_position_embeding = self.fpe(coords_position_embeding.flatten(0,1), x.flatten(0,1)).view(x.size())

        pos_embed = coords_position_embeding

        if self.with_multiview:
            sin_embed = self.positional_encoding(masks)
            sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
            constantization_PETRv2Head_forward.sin_embed = sin_embed.detach().clone()
            pos_embed = pos_embed + sin_embed
        else:
            pos_embeds = []
            for i in range(num_cams):
                xy_embed = self.positional_encoding(masks[:, i, :, :])
                pos_embeds.append(xy_embed.unsqueeze(1))
            sin_embed = torch.cat(pos_embeds, 1)
            sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
            pos_embed = pos_embed + sin_embed
    else:
        if self.with_multiview:
            pos_embed = self.positional_encoding(masks)
            pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
        else:
            pos_embeds = []
            for i in range(num_cams):
                pos_embed = self.positional_encoding(masks[:, i, :, :])
                pos_embeds.append(pos_embed.unsqueeze(1))
            pos_embed = torch.cat(pos_embeds, 1)

    reference_points = self.reference_points.weight
    query_embeds = self.query_embedding(pos2posemb3d(reference_points))
    constantization_PETRv2Head_forward.query_embeds = query_embeds

    reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1) #.sigmoid()
    constantization_PETRv2Head_forward.reference_points = reference_points

    outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, self.reg_branches)
    outs_dec = torch.nan_to_num(outs_dec)
    
    if self.with_time:
        time_stamps = []
        for img_meta in img_metas:    
            time_stamps.append(np.asarray(img_meta['timestamp']))
        time_stamp = x.new_tensor(time_stamps)
        time_stamp = time_stamp.view(batch_size, -1, 6)
        mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)
    
    outputs_classes = []
    outputs_coords = []
    for lvl in range(outs_dec.shape[0]):
        reference = inverse_sigmoid(reference_points.clone())
        assert reference.shape[-1] == 3
        outputs_class = self.cls_branches[lvl](outs_dec[lvl])
        tmp = self.reg_branches[lvl](outs_dec[lvl])

        tmp[..., 0:2] += reference[..., 0:2]
        tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        tmp[..., 4:5] += reference[..., 2:3]
        tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

        if self.with_time:
            tmp[..., 8:] = tmp[..., 8:] / mean_time_stamp[:, None, None]

        outputs_coord = tmp
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)

    all_cls_scores = torch.stack(outputs_classes)
    all_bbox_preds = torch.stack(outputs_coords)

    all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
    all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
    all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

    outs = {
        'all_cls_scores': all_cls_scores,
        'all_bbox_preds': all_bbox_preds,
        'enc_cls_scores': None,
        'enc_bbox_preds': None, 
    }
    return outs

def fn_head_fwd(args, kwargs):
    batch_size = 1
    x = args[0][0]
    img_metas = args[1]
    time_stamps = []
    for img_meta in img_metas:    
        time_stamps.append(np.asarray(img_meta['timestamp']))
    time_stamp = x.new_tensor(time_stamps)
    time_stamp = time_stamp.view(batch_size, -1, 6)
    mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)

    batch_size, num_cams = 1, 12
    input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]

    mlvl_feats = args[0]
    masks = mlvl_feats[0].new_ones(
        (batch_size, num_cams, input_img_h, input_img_w))
    for img_id in range(batch_size):
        for cam_id in range(num_cams):
            img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
            masks[img_id, cam_id, :img_h, :img_w] = 0

    # interpolate masks to have the same spatial shape with x
    masks = F.interpolate(masks, size=mlvl_feats[0].shape[-2:]).to(torch.bool)
    coords_position_embeding, _ = fn_head_fwd.mod.position_embeding(mlvl_feats, img_metas, masks)    

    args[1][0]["mean_time_stamp"] = mean_time_stamp
    args[1][0]["coords_position_embeding"] = coords_position_embeding
    return args, kwargs

def patch_PETRv2Head_forward(self, mlvl_feats, img_metas):
    x = mlvl_feats[self.position_level]
    batch_size, num_cams = x.size(0), x.size(1)
        
    x = self.input_proj(x.flatten(0, 1))
    x = x.view(batch_size, num_cams, *x.shape[-3:])

    masks = constantization_PETRv2Head_forward.masks
    # pos_embed = constantization_PETRv2Head_forward.pos_embed
    coords_position_embeding = img_metas[0]["coords_position_embeding"]

    pos_embed = self.fpe(coords_position_embeding.flatten(0, 1), x.flatten(0, 1)).view(x.size())
    pos_embed = pos_embed + constantization_PETRv2Head_forward.sin_embed

    query_embeds = constantization_PETRv2Head_forward.query_embeds
    reference_points = constantization_PETRv2Head_forward.reference_points

    outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, self.reg_branches)
    outs_dec = torch.nan_to_num(outs_dec)
    
    # if self.with_time:
    #     time_stamps = []
    #     for img_meta in img_metas:    
    #         time_stamps.append(np.asarray(img_meta['timestamp']))
    #     time_stamp = x.new_tensor(time_stamps)
    #     time_stamp = time_stamp.view(batch_size, -1, 6)
    #     mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)
    
    mean_time_stamp = img_metas[0]["mean_time_stamp"]

    outputs_classes = []
    outputs_coords = []
    for lvl in range(outs_dec.shape[0]):
        reference = inverse_sigmoid(reference_points.clone())
        assert reference.shape[-1] == 3
        outputs_class = self.cls_branches[lvl](outs_dec[lvl])
        tmp = self.reg_branches[lvl](outs_dec[lvl])

        tmp[..., 0:2] += reference[..., 0:2]
        tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        tmp[..., 4:5] += reference[..., 2:3]
        tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

        if self.with_time:
            tmp[..., 8:] = tmp[..., 8:] / mean_time_stamp[:, None, None]

        outputs_coord = tmp
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)

    all_cls_scores = torch.stack(outputs_classes)
    all_bbox_preds = torch.stack(outputs_coords)

    all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
    all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
    all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

    outs = {
        'all_cls_scores': all_cls_scores,
        'all_bbox_preds': all_bbox_preds,
        'enc_cls_scores': None,
        'enc_bbox_preds': None, 
    }
    return outs

def fn_extract_feat(args, kwargs):
    a = kwargs["img"][:, 0: 6, ...].contiguous()
    b = kwargs["img"][:, 6:12, ...].contiguous()
    img_metas = kwargs["img_metas"]
    prev = fn_extract_feat.mod.extract_img_feat(b, img_metas)
    kwargs["img"]  = a
    kwargs["prev"] = prev
    return args, kwargs

def patch_extract_feat(self, img, img_metas, **kwargs):
    prev = kwargs["prev"]
    img_feats = self.extract_img_feat(img, img_metas)
    ret = []
    for i in range(2):
        ret.append(torch.cat([
            img_feats[i].reshape(*prev[i].shape),
            prev[i]], dim=1))
    return ret
