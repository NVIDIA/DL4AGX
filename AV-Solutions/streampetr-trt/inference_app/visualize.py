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

# For NMSFreeCoder and denormalize_bbox
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import glob
import sys
import os
from pathlib import Path
import pickle as pkl

import numpy as np
import cv2 as cv
import torch

# Reference: https://github.com/exiawsh/StreamPETR/blob/main/projects/mmdet3d_plugin/core/bbox/util.py
def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]

    # size
    w = normalized_bboxes[..., 3:4]
    l = normalized_bboxes[..., 4:5]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

# Reference: https://github.com/exiawsh/StreamPETR/blob/main/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py
class NMSFreeCoder(object):
    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode='floor')
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores >= self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

class_colors = [
    (  0,   0, 200), # 'car', 
    (  0,   0,  70), # 'truck', 
    (  0,  80, 100), # 'construction_vehicle', 
    (  0,  60, 100), # 'bus', 
    (  0,   0, 110), # 'trailer', 
    (180, 165, 180), # 'barrier',
    (  0,   0, 230), # 'motorcycle', 
    (119,  11,  32), # 'bicycle', 
    (  0,   0, 200), # 'pedestrian', 
    (153, 153, 153), # 'traffic_cone' 
]

def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def box2corners(box):
    box = box.numpy()
    # cx, cy, cz, w, l, h, rot, vx, vy
    x, y = box[0], box[1]
    w, l = box[3], box[4]
    R = rotz(box[6] - np.pi / 2.0)
    corners = np.array([
        [     0, l / 2, l / 2,  -l / 2, -l / 2,      0, 0],
        [-w / 2,-w / 2, w / 2,   w / 2, -w / 2, -w / 2, 0],
        [     0,     0,     0,       0,      0,      0, 0],
    ])
    corners = np.dot(R, corners)    
    corners[0, :] += (51.2 + x)
    corners[1, :] += (51.2 + y)
    return (corners * 10).astype(np.int32)[0:2, :]

if __name__ == "__main__":
    # these numbers are from configs
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]
    decoder = NMSFreeCoder(
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=300,
        voxel_size=voxel_size,
        num_classes=10
    )

    SCORE_THRESH = 0.5
    NUSCENES_DIR = os.environ.get("NUSCENES_ROOT")
    lidar_tops = sorted(list(glob.glob(NUSCENES_DIR + "/samples/LIDAR_TOP/*.bin")))
    demo_dir = Path(sys.argv[1])

    for i in range(1, 11):
        white = np.ones(shape=(1024, 1024, 3), dtype=np.uint8) * 255
        frame_dir = demo_dir / "{:04d}".format(i)

        # please make sure lidar frames are matched. You may check the data['meta'] in StreamPETR
        lidar_top_dir = lidar_tops[i]

        lidar_data = np.fromfile(lidar_top_dir, dtype=np.float32).reshape(-1, 5)

        all_bbox_preds = torch.from_numpy(np.fromfile(str(frame_dir / "all_bbox_preds_trt.bin"), dtype=np.float32).reshape(1,1,428,10))
        all_cls_scores = torch.from_numpy(np.fromfile(str(frame_dir / "all_cls_scores_trt.bin"), dtype=np.float32).reshape(1,1,428,10))
        ret = decoder.decode(dict(all_cls_scores=all_cls_scores, all_bbox_preds=all_bbox_preds))

        bboxes = ret[0]['bboxes']
        scores = ret[0]['scores']
        labels = ret[0]['labels']
        for j in range(0, 300):
            # cx, cy, cz, w, l, h, rot, vx, vy
            box = bboxes[j]
            if scores[j] < SCORE_THRESH:
                continue
            center = int((float(box[0]) + 51.2) * 10), int((float(box[1]) + 51.2) * 10)
            color = list(class_colors[labels[j]])[::-1]
            corners = box2corners(box).T
            cv.polylines(white, [corners], False, color, 2)

        lidar_data_bev = np.array((lidar_data + 51.2) * 10).astype(np.int32)
        for j in range(lidar_data.shape[0]):
            pt = lidar_data_bev[j]
            if 0 <= pt[0] < 1024 and 0 <= pt[1] < 1024:
                white[pt[1], pt[0], :] = (50, 50, 50)

        out_dir = str(frame_dir / "vis.jpg")
        cv.imwrite(out_dir, white)
