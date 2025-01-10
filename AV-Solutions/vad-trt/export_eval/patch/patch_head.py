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

# Part of the functions was from https://github.com/hustvl/VAD/blob/main/projects/mmdet3d_plugin/VAD/VAD_head.py
# Licensed under https://github.com/hustvl/VAD/blob/main/LICENSE

import torch

def patch_VADHead_select_and_pad_query(
    self,
    query,
    query_pos,
    query_score,
    score_thresh=0.5,
    use_fix_pad=True
):
    """select_and_pad_query.
    Args:
        query: [B, Q, D].
        query_pos: [B, Q, 2]
        query_score: [B, Q, C].
        score_thresh: confidence threshold for filtering low-confidence query
        use_fix_pad: always pad one query instance for each batch
    Returns:
        selected_query: [B, Q', D]
        selected_query_pos: [B, Q', 2]
        selected_padding_mask: [B, Q']
    """
    # in our case, thresh == 0.0, so no need to do anything here?
    batch_max_qnum = query.shape[1]
    padding_mask = torch.tensor([0.0], device=query_score.device, dtype=torch.float32).repeat(1, batch_max_qnum)
    return query, query_pos, padding_mask

class SelectAndPadFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, *args, **kwargs):
        # beware, for export we use a different P value
        # just to make sure it's big enough
        version_major = int(torch.__version__.split(".")[0])
        return g.op("custom_op::SelectAndPadPlugin", args[0], args[1], args[2], P_i=args[3])

    @staticmethod
    def forward(ctx, feat, flag, invalid, P):
        flag_ = flag.to(torch.bool)
        selected_feat = []
        for i in range(flag.shape[0]):
            # dim = feat.shape[-1]
            valid_pnum = flag[i].sum()
            valid_feat = feat[i, flag_[i]]
            pad_pnum = P - valid_pnum
            # padding_mask = torch.tensor([False], device=feat.device).repeat(P)
            if pad_pnum != 0:
                if valid_feat.dim() == 1:
                    invalid_feat = invalid.repeat(pad_pnum)
                else:
                    invalid_feat = invalid.repeat(pad_pnum, 1)
                valid_feat = torch.cat([valid_feat, invalid_feat], dim=0)
                # valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=feat.device)], dim=0)
                # padding_mask[valid_pnum:] = True
            selected_feat.append(valid_feat)
        selected_feat = torch.stack(selected_feat, dim=0)
        return selected_feat

from bev_deploy.infer_shape.infer import G_HANDLERS

def infer_select_and_pad(node):
    P = node.attrs["P"]
    input_shape = node.inputs[0].shape
    node.outputs[0].shape = [input_shape[0], P, input_shape[2]]
    node.outputs[0].dtype = node.inputs[0].dtype
    return True

G_HANDLERS["SelectAndPadPlugin"] = infer_select_and_pad

def patch_VADHead_select_and_pad_pred_map(
    self,
    motion_pos,
    map_query,
    map_score,
    map_pos,
    map_thresh=0.5,
    dis_thresh=None,
    pe_normalization=True,
    use_fix_pad=False
):
    """select_and_pad_pred_map.
    Args:
        motion_pos: [B, A, 2]
        map_query: [B, P, D].
        map_score: [B, P, 3].
        map_pos: [B, P, pts, 2].
        map_thresh: map confidence threshold for filtering low-confidence preds
        dis_thresh: distance threshold for masking far maps for each agent in cross-attn
        use_fix_pad: always pad one lane instance for each batch
    Returns:
        selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
        selected_map_pos: [B*A, P1(+1), 2]
        selected_padding_mask: [B*A, P1(+1)]
    """
    # let's assume B=1 here
    if dis_thresh is None:
        raise NotImplementedError('Not implement yet')

    # use the most close pts pos in each map inst as the inst's pos
    batch, num_map = map_pos.shape[:2]
    map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
    min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
    min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
    min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
    min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

    # select & pad map vectors for different batch using map_thresh
    map_score = map_score.sigmoid()
    map_max_score = map_score.max(dim=-1)[0]
    map_idx = map_max_score > map_thresh

    batch_max_pnum = 16  # hard to say if 16 is enough
    for i in range(map_score.shape[0]):
        pnum = map_idx[i].sum()
        if pnum > batch_max_pnum:
            batch_max_pnum = pnum

    # patch_VADHead_select_and_pad_pred_map.max_pnum = max(patch_VADHead_select_and_pad_pred_map.max_pnum, batch_max_pnum)

    dim = map_query.shape[-1]

    # original code
    # selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
    # for i in range(map_score.shape[0]):
    #     
    #     valid_pnum = map_idx[i].sum()
    #     valid_map_query = map_query[i, map_idx[i]]
    #     valid_map_pos = min_map_pos[i, map_idx[i]]
    #     pad_pnum = batch_max_pnum - valid_pnum
    #     padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
    #     if pad_pnum != 0:
    #         valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
    #         valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
    #         padding_mask[valid_pnum:] = True
    #     selected_map_query.append(valid_map_query)
    #     selected_map_pos.append(valid_map_pos)
    #     selected_padding_mask.append(padding_mask)

    # selected_map_query = torch.stack(selected_map_query, dim=0)
    # selected_map_pos = torch.stack(selected_map_pos, dim=0)
    # selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

    # replaced with SelectAndPadFunction
    selected_map_query = SelectAndPadFunction.apply(
        map_query, map_idx.to(torch.int32), torch.zeros(dim, device=map_query.device, dtype=map_query.dtype), batch_max_pnum)
    selected_map_pos = SelectAndPadFunction.apply(
        min_map_pos, map_idx.to(torch.int32), torch.zeros(2, device=map_query.device, dtype=map_query.dtype), batch_max_pnum)
    selected_padding_mask = SelectAndPadFunction.apply(
        torch.zeros(size=(batch, num_map, 1), device=map_query.device, dtype=torch.float32), 
        map_idx.to(torch.int32), 
        torch.ones(size=(1, ), device=map_query.device, dtype=map_query.dtype), 
        batch_max_pnum).reshape(batch, batch_max_pnum)

    # generate different pe for map vectors for each agent
    num_agent = motion_pos.shape[1]
    selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
    selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
    selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
    # move lane to per-car coords system
    selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
    if pe_normalization:
        selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

    # filter far map inst for each agent
    map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
    valid_map_inst = (map_dis <= dis_thresh).to(torch.float32)  # [B, A, max_P]
    invalid_map_inst = (valid_map_inst == 0.0)
    selected_padding_mask = selected_padding_mask + invalid_map_inst

    selected_map_query = selected_map_query.flatten(0, 1)
    selected_map_pos = selected_map_pos.flatten(0, 1)
    selected_padding_mask = selected_padding_mask.flatten(0, 1)

    num_batch = selected_padding_mask.shape[0]
    feat_dim = selected_map_query.shape[-1]
    if use_fix_pad:
        pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
        pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
        pad_lane_mask = torch.tensor([0.0], device=selected_padding_mask.device, dtype=torch.float32).unsqueeze(0).repeat(num_batch, 1)
        selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
        selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
        selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

    return selected_map_query, selected_map_pos, selected_padding_mask

# patch_VADHead_select_and_pad_pred_map.max_pnum = 16
# from torchvision.transforms.functional import rotate
from projects.mmdet3d_plugin.VAD.VAD_transformer import rotate

def patch_VADPerceptionTransformer_get_bev_features(
    self,
    mlvl_feats,
    bev_queries,
    bev_h,
    bev_w,
    grid_length=[0.512, 0.512],
    bev_pos=None,
    prev_bev=None,
    **kwargs
):
    bs = mlvl_feats[0].size(0)
    bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
    bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

    # obtain rotation angle and shift with ego motion

    # delta_x = np.array([each['can_bus'][0] for each in kwargs['img_metas']])
    # delta_y = np.array([each['can_bus'][1] for each in kwargs['img_metas']])
    # ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])

    # grid_length_y = grid_length[0]
    # grid_length_x = grid_length[1]
    # translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    # translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    # bev_angle = ego_angle - translation_angle
    # shift_y = translation_length * \
    #     np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
    # shift_x = translation_length * \
    #     np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
    # shift_y = shift_y * self.use_shift
    # shift_x = shift_x * self.use_shift
    # shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

    # replaced with pre-calc shift
    shift = kwargs["img_metas"][0]["shift"]

    if prev_bev is not None:
        if prev_bev.shape[1] == bev_h * bev_w:
            prev_bev = prev_bev.permute(1, 0, 2)
        if self.rotate_prev_bev:
            for i in range(bs):
                # num_prev_bev = prev_bev.size(1)
                rotation_angle = kwargs['img_metas'][i]['can_bus'][0, -1]
                tmp_prev_bev = prev_bev[:, i].reshape(
                    bev_h, bev_w, -1).permute(2, 0, 1)
                tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                    bev_h * bev_w, 1, -1)
                # prev_bev[:, i] = tmp_prev_bev[:, 0]
                # TODO: this assume bs == 1
                prev_bev = tmp_prev_bev

    # add can bus signals
    # can_bus = bev_queries.new_tensor([each['can_bus'] for each in kwargs['img_metas']])
    can_bus = kwargs["img_metas"][0]["can_bus"]
    can_bus = self.can_bus_mlp(can_bus)[None, :, :]
    bev_queries = bev_queries + can_bus * int(self.use_can_bus)

    feat_flatten = []
    spatial_shapes = []
    for lvl, feat in enumerate(mlvl_feats):
        bs, num_cam, c, h, w = feat.shape
        spatial_shape = (h, w)
        feat = feat.flatten(3).permute(1, 0, 3, 2)
        if self.use_cams_embeds:
            feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
        feat = feat + self.level_embeds[None,
                                        None, lvl:lvl + 1, :].to(feat.dtype)
        spatial_shapes.append(spatial_shape)
        feat_flatten.append(feat)

    feat_flatten = torch.cat(feat_flatten, 2)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

    feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

    bev_embed = self.encoder(
        bev_queries,
        feat_flatten,
        feat_flatten,
        mlvl_feats=mlvl_feats,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        prev_bev=prev_bev,
        shift=shift,
        **kwargs
    )
    return bev_embed

from mmdet.models.utils.transformer import inverse_sigmoid

def patch_VADHead_forward(self,
    mlvl_feats,
    img_metas,
    prev_bev=None,
    only_bev=False,
    ego_his_trajs=None,
    ego_lcf_feat=None,
):
    """Forward function.
    Args:
        mlvl_feats (tuple[Tensor]): Features from the upstream
            network, each is a 5D-tensor with shape
            (B, N, C, H, W).
        prev_bev: previous bev featues
        only_bev: only compute BEV features with encoder. 
    Returns:
        all_cls_scores (Tensor): Outputs from the classification head, \
            shape [nb_dec, bs, num_query, cls_out_channels]. Note \
            cls_out_channels should includes background.
        all_bbox_preds (Tensor): Sigmoid outputs from the regression \
            head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
            Shape [nb_dec, bs, num_query, 9].
    """

    bs, num_cam, _, _, _ = mlvl_feats[0].shape
    dtype = mlvl_feats[0].dtype
    object_query_embeds = self.query_embedding.weight.to(dtype)

    if self.map_query_embed_type == 'all_pts':
        map_query_embeds = self.map_query_embedding.weight.to(dtype)
    elif self.map_query_embed_type == 'instance_pts':
        map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
        map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
        map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

    bev_queries = self.bev_embedding.weight.to(dtype)

    bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                            device=bev_queries.device).to(dtype)
    bev_pos = self.positional_encoding(bev_mask).to(dtype)
        
    if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
        return self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                            self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )
    else:
        outputs = self.transformer(
            mlvl_feats,
            bev_queries,
            object_query_embeds,
            map_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,
            map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
            map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev)

    bev_embed, hs, init_reference, inter_references, \
        map_hs, map_init_reference, map_inter_references = outputs

    hs = hs.permute(0, 2, 1, 3)
    outputs_classes = []
    outputs_coords = []
    outputs_coords_bev = []
    outputs_trajs = []
    outputs_trajs_classes = []

    map_hs = map_hs.permute(0, 2, 1, 3)
    map_outputs_classes = []
    map_outputs_coords = []
    map_outputs_pts_coords = []
    map_outputs_coords_bev = []

    for lvl in range(hs.shape[0]):
        if lvl == 0:
            reference = init_reference
        else:
            reference = inter_references[lvl - 1]
        reference = inverse_sigmoid(reference)
        outputs_class = self.cls_branches[lvl](hs[lvl])
        tmp = self.reg_branches[lvl](hs[lvl])

        # TODO: check the shape of reference
        assert reference.shape[-1] == 3
        tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
        tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
        tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
        tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
        tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                            self.pc_range[0]) + self.pc_range[0])
        tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                            self.pc_range[1]) + self.pc_range[1])
        tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                            self.pc_range[2]) + self.pc_range[2])

        # TODO: check if using sigmoid
        outputs_coord = tmp
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)

    for lvl in range(map_hs.shape[0]):
        if lvl == 0:
            reference = map_init_reference
        else:
            reference = map_inter_references[lvl - 1]
        reference = inverse_sigmoid(reference)
        map_outputs_class = self.map_cls_branches[lvl](
            map_hs[lvl].view(bs,self.map_num_vec, self.map_num_pts_per_vec,-1).mean(2)
        )
        tmp = self.map_reg_branches[lvl](map_hs[lvl])
        # TODO: check the shape of reference
        assert reference.shape[-1] == 2
        tmp[..., 0:2] += reference[..., 0:2]
        tmp = tmp.sigmoid() # cx,cy,w,h
        map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
        map_outputs_coords_bev.append(map_outputs_pts_coord.clone().detach())
        map_outputs_classes.append(map_outputs_class)
        map_outputs_coords.append(map_outputs_coord)
        map_outputs_pts_coords.append(map_outputs_pts_coord)
        
    if self.motion_decoder is not None:
        batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
        # motion_query
        motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
        mode_query = self.motion_mode_query.weight  # [fut_mode, D]
        # [M, B, D], M=A*fut_mode
        motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
        if self.use_pe:
            motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
            motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
            motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
            motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]
        else:
            motion_pos = None

        if self.motion_det_score is not None:
            motion_score = outputs_classes[-1]
            max_motion_score = motion_score.max(dim=-1)[0]
            invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
            invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
        else:
            invalid_motion_idx = None

        motion_hs = self.motion_decoder(
            query=motion_query,
            key=motion_query,
            value=motion_query,
            query_pos=motion_pos,
            key_pos=motion_pos,
            key_padding_mask=invalid_motion_idx)

        if self.motion_map_decoder is not None:
            # map preprocess
            motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
            motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
            map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
            map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
            map_score = map_outputs_classes[-1]
            map_pos = map_outputs_coords_bev[-1]
            map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(
                motion_coords, map_query, map_score, map_pos,
                map_thresh=self.map_thresh, dis_thresh=self.dis_thresh,
                pe_normalization=self.pe_normalization, use_fix_pad=True)
            map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]
            ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

            # position encoding
            if self.use_pe:
                (num_query, batch) = ca_motion_query.shape[:2] 
                motion_pos = torch.zeros((num_query, batch, 2), device=motion_hs.device)
                motion_pos = self.pos_mlp(motion_pos)
                map_pos = map_pos.permute(1, 0, 2)
                map_pos = self.pos_mlp(map_pos)
            else:
                motion_pos, map_pos = None, None
            
            ca_motion_query = self.motion_map_decoder(
                query=ca_motion_query,
                key=map_query,
                value=map_query,
                query_pos=motion_pos,
                key_pos=map_pos,
                key_padding_mask=key_padding_mask)
        else:
            ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

        batch_size = outputs_coords_bev[-1].shape[0]
        chn = motion_hs.shape[2]
        # motion_hs: 1800, 1, 256 -> 1, 1800, 256
        motion_hs = motion_hs.permute(1, 0, 2)
        motion_hs = motion_hs.reshape(batch_size, num_agent, self.fut_mode, chn)

        # ca_motion_query: 1, 1800, 256
        # ca_motion_query = ca_motion_query.squeeze(0)  # 1800, 256
        ca_motion_query = ca_motion_query.reshape(batch_size, num_agent, self.fut_mode, -1)

        motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]
    else:
        raise NotImplementedError('Not implement yet')

    outputs_traj = self.traj_branches[0](motion_hs)
    outputs_trajs.append(outputs_traj)
    outputs_traj_class = self.traj_cls_branches[0](motion_hs)
    outputs_traj_class = outputs_traj_class.reshape(1, 300, 6) # TODO: hard coded here
    outputs_trajs_classes.append(outputs_traj_class)
    # (batch, num_agent) = motion_hs.shape[:2]
            
    map_outputs_classes = torch.stack(map_outputs_classes)
    map_outputs_coords = torch.stack(map_outputs_coords)
    map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

    outputs_classes = torch.stack(outputs_classes)
    outputs_coords = torch.stack(outputs_coords)
    outputs_trajs = torch.stack(outputs_trajs)
    outputs_trajs_classes = torch.stack(outputs_trajs_classes)
    # return map_outputs_classes, map_outputs_coords, map_outputs_pts_coords, \
    #     outputs_classes, outputs_coords, outputs_traj, outputs_trajs_classes

    # planning
    (batch, num_agent) = motion_hs.shape[:2]
    if self.ego_his_encoder is not None:
        ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
    else:
        ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
    # Interaction
    ego_query = ego_his_feats
    ego_pos = torch.zeros((batch, 1, 2), device=ego_query.device)
    ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)
    agent_conf = outputs_classes[-1]
    agent_query = motion_hs.reshape(batch, num_agent, -1)
    agent_query = self.agent_fus_mlp(agent_query) # [B, A, fut_mode, 2*D] -> [B, A, D]
    agent_pos = outputs_coords_bev[-1]
    agent_query, agent_pos, agent_mask = self.select_and_pad_query(
        agent_query, agent_pos, agent_conf,
        score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
    )
    agent_pos_emb = self.ego_agent_pos_mlp(agent_pos)
    # ego <-> agent interaction
    ego_agent_query = self.ego_agent_decoder(
        query=ego_query.permute(1, 0, 2),
        key=agent_query.permute(1, 0, 2),
        value=agent_query.permute(1, 0, 2),
        query_pos=ego_pos_emb.permute(1, 0, 2),
        key_pos=agent_pos_emb.permute(1, 0, 2),
        key_padding_mask=agent_mask)

    # ego <-> map interaction
    ego_pos = torch.zeros((batch, 1, 2), device=agent_query.device)
    ego_pos_emb = self.ego_map_pos_mlp(ego_pos)
    map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
    map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
    map_conf = map_outputs_classes[-1]
    map_pos = map_outputs_coords_bev[-1]
    # use the most close pts pos in each map inst as the inst's pos
    batch, num_map = map_pos.shape[:2]
    map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
    min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
    min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
    min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
    min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]
    map_query, map_pos, map_mask = self.select_and_pad_query(
        map_query, min_map_pos, map_conf,
        score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
    )
    map_pos_emb = self.ego_map_pos_mlp(map_pos)
    ego_map_query = self.ego_map_decoder(
        query=ego_agent_query,
        key=map_query.permute(1, 0, 2),
        value=map_query.permute(1, 0, 2),
        query_pos=ego_pos_emb.permute(1, 0, 2),
        key_pos=map_pos_emb.permute(1, 0, 2),
        key_padding_mask=map_mask)

    if self.ego_his_encoder is not None and self.ego_lcf_feat_idx is not None:
        ego_feats = torch.cat(
            [ego_his_feats,
                ego_map_query.permute(1, 0, 2),
                ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
            dim=-1
        )  # [B, 1, 2D+2]
    elif self.ego_his_encoder is not None and self.ego_lcf_feat_idx is None:
        ego_feats = torch.cat(
            [ego_his_feats,
                ego_map_query.permute(1, 0, 2)],
            dim=-1
        )  # [B, 1, 2D]
    elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is not None:                
        ego_feats = torch.cat(
            [ego_agent_query.permute(1, 0, 2),
                ego_map_query.permute(1, 0, 2),
                ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
            dim=-1
        )  # [B, 1, 2D+2]
    elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is None:                
        # hit this one
        ego_feats = torch.cat(
            [ego_agent_query.permute(1, 0, 2), ego_map_query.permute(1, 0, 2)],
            dim=-1
        )  # [B, 1, 2D]  

    # Ego prediction
    outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
    outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], self.ego_fut_mode, self.fut_ts, 2)

    outs = {
        'bev_embed': bev_embed,
        'all_cls_scores': outputs_classes,
        'all_bbox_preds': outputs_coords,
        'all_traj_preds': outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1),
        'all_traj_cls_scores': outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1),
        'map_all_cls_scores': map_outputs_classes,
        'map_all_bbox_preds': map_outputs_coords,
        'map_all_pts_preds': map_outputs_pts_coords,
        'enc_cls_scores': None,
        'enc_bbox_preds': None,
        'map_enc_cls_scores': None,
        'map_enc_bbox_preds': None,
        'map_enc_pts_preds': None,
        'ego_fut_preds': outputs_ego_trajs,
    }

    return outs
