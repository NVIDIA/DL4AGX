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

import torch
import numpy as np

# spatial cross attention
def patch_spatial_cross_attn_forward(
    self,
    query,
    key,
    value,
    residual=None,
    query_pos=None,
    key_padding_mask=None,
    reference_points=None,
    spatial_shapes=None,
    reference_points_cam=None,
    bev_mask=None,
    level_start_index=None,
    flag='encoder',
    **kwargs
):
    if key is None:
        key = query
    if value is None:
        value = key

    if residual is None:
        inp_residual = query
        slots = torch.zeros_like(query)
    if query_pos is not None:
        query = query + query_pos

    bs, num_query, _ = query.size()

    D = reference_points_cam.size(3)
    indexes = (bev_mask.sum(-1) > 0).permute(1, 0, 2).unsqueeze(-1)
    max_len = bev_mask.shape[2]

    # indexes = []
    # for i, mask_per_img in enumerate(bev_mask):
    #     index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
    #     indexes.append(index_query_per_img)
    # max_len = max([len(each) for each in indexes])

    # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
    queries_rebatch = query.new_zeros(
        [bs, self.num_cams, max_len, self.embed_dims])
    # reference_points_rebatch = reference_points_cam.new_zeros(
    #     [bs, self.num_cams, max_len, D, 2])
    reference_points_rebatch = reference_points_cam.clone().view(bs, self.num_cams, max_len, D, 2)

    # for j in range(bs):
    #     for i, reference_points_per_img in enumerate(reference_points_cam):   
    #         index_query_per_img = indexes[i]
    #         queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
    #         reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
    # for j in range(bs):
    #     for i, reference_points_per_img in enumerate(reference_points_cam):   
    #         queries_rebatch[j, i, :] = query[j, :]
    query_ = query.reshape(bs, 1, max_len, self.embed_dims)
    queries_rebatch = query_.repeat(1, self.num_cams, 1, 1)

    num_cams, l, bs, embed_dims = key.shape

    key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
    value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)

    queries = self.deformable_attention(
        query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), 
        key=key, value=value,
        reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), 
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index)
    queries = queries.view(bs, self.num_cams, max_len, self.embed_dims)
    slots = (queries * indexes).sum(1)

    # for j in range(bs):
    #     for i, index_query_per_img in enumerate(indexes):
    #         slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

    count = bev_mask.sum(-1) > 0
    count = count.sum(0) # count = count.permute(1, 2, 0).sum(-1)
    count = torch.clamp(count, min=1.0)
    slots = slots / count[..., None]
    slots = self.output_proj(slots)

    return self.dropout(slots) + inp_residual

patch_spatial_cross_attn_forward._unused = set([
    "key_padding_mask",
    "reference_points",
    "flag",
    "kwargs",
])

def patch_point_sampling(self, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])

    # lidar2img = np.asarray(lidar2img)
    # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    lidar2img = torch.stack(lidar2img).to(reference_points.dtype).to(reference_points.device)
    reference_points = reference_points.clone()

    x = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    y = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    z = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

    reference_points = torch.cat((x, y, z, torch.ones_like(reference_points[..., :1])), -1)

    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

    lidar2img = lidar2img.view(
        1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                        reference_points.to(torch.float32))
    # 8, 1, 6, 20000, 4, 1
    # reference_points_cam = reference_points_cam.squeeze(-1)
    reference_points_cam = reference_points_cam.reshape(D, B, num_cam, num_query, 4)
    eps = 1e-5

    bev_mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

    bev_mask = (bev_mask 
        & (reference_points_cam[..., 1:2] > 0.0)
        & (reference_points_cam[..., 1:2] < 1.0)
        & (reference_points_cam[..., 0:1] < 1.0)
        & (reference_points_cam[..., 0:1] > 0.0))
    # if digit_version(TORCH_VERSION) >= digit_version('1.8'):
    bev_mask = torch.nan_to_num(bev_mask)
    # else:
    #     bev_mask = bev_mask.new_tensor(
    #         np.nan_to_num(bev_mask.cpu().numpy()))

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4)
    # 6, 1, 20000, 8, 1
    # bev_mask = bev_mask.squeeze(-1)
    bev_mask = bev_mask.reshape(num_cam, B, num_query, D)
    # bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    return reference_points_cam, bev_mask

def patch_get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
    # reference points in 3D space, used in spatial cross-attention (SCA)
    if dim == '3d':
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d

    # reference points on 2D bev plane, used in temporal self-attention (TSA).
    elif dim == '2d':
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d

def patch_bevformer_encoder_forward(
    self,
    bev_query,
    key,
    value,
    *args,
    bev_h=None,
    bev_w=None,
    bev_pos=None,
    spatial_shapes=None,
    level_start_index=None,
    valid_ratios=None,
    prev_bev=None,
    shift=0.,
    **kwargs
):
    output = bev_query
    intermediate = []

    ref_3d = self.get_reference_points(
        bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
    # ref_2d = self.get_reference_points(
    #     bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)
    ref_2d = ref_3d[0, 0, :, :2].view(1, -1, 1, 2).clone()  # <--- hard to say why original code won't work

    reference_points_cam, bev_mask = self.point_sampling(
        ref_3d, self.pc_range, kwargs['img_metas'])

    # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
    shift_ref_2d = ref_2d  # .clone()
    shift_ref_2d += shift[:, None, None, :]

    # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
    bev_query = bev_query.permute(1, 0, 2)
    bev_pos = bev_pos.permute(1, 0, 2)
    bs, len_bev, num_bev_level, _ = ref_2d.shape
    if prev_bev is not None:
        prev_bev = prev_bev.permute(1, 0, 2)
        prev_bev = torch.stack(
            [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
        hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
            bs*2, len_bev, num_bev_level, 2)
    else:
        hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
            bs*2, len_bev, num_bev_level, 2)

    for lid, layer in enumerate(self.layers):
        output = layer(
            bev_query,
            key,
            value,
            *args,
            bev_pos=bev_pos,
            ref_2d=hybird_ref_2d,
            ref_3d=ref_3d,
            bev_h=bev_h,
            bev_w=bev_w,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            prev_bev=prev_bev,
            **kwargs)

        bev_query = output
        if self.return_intermediate:
            intermediate.append(output)

    if self.return_intermediate:
        return torch.stack(intermediate)

    return output

def fn_lidar2img(m):
    for i in range(len(m)):
        m[i]["lidar2img"] = torch.from_numpy(np.asarray(m[i]["lidar2img"])).to(torch.float32)
    return m

def fn_canbus(other, m, bev_h, bev_w, grid_length):
    delta_x = np.array([each['can_bus'][0] for each in m])
    delta_y = np.array([each['can_bus'][1] for each in m])
    ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in m])

    # grid_length = [0.3, 0.3]
    grid_length_y = grid_length[0]
    grid_length_x = grid_length[1]
    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    bev_angle = ego_angle - translation_angle
    shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
    shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
    shift_y = shift_y
    shift_x = shift_x
    shift = other.new_tensor([shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy
    m[0]["shift"] = shift
    m[0]["can_bus"] = other.new_tensor([each['can_bus'] for each in m])
    return m
