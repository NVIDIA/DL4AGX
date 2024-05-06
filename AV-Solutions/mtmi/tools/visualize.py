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

import numpy as np
import cv2
import matplotlib 
from pathlib import Path

PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]]
PALETTE = np.array(PALETTE)

def colorize(value, cmap='jet', vmin=None, vmax=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    value_int = np.clip(value * 255, 0, 255).astype(np.int32)
    indices = np.arange(0, 255)
    cmapper = matplotlib.cm.get_cmap(cmap)
    colors = cmapper(indices, bytes=True)
    mapped_colors = colors[value_int]
    return mapped_colors

img_dir = "tests/"
result_dir = "results/"

images = Path(img_dir).glob("*.png")

for img in images:
    stem = img.stem
    prefix = result_dir + stem    
    im = cv2.imread(str(img))

    # segmentation
    seg_result = np.fromfile(prefix + "_seg.bin", dtype=np.int8).reshape(19, 1024, 1024)
    seg_i = np.argmax(seg_result, axis=0)
    seg_im = PALETTE[seg_i]

    # depth
    depth_result = 1.0 - np.fromfile(prefix + "_depth.bin", dtype=np.float32)
    depth_im = colorize(depth_result * 80, vmin=1e-3, vmax=80).reshape(1024, 1024, 4)

    cv2.imwrite(prefix + "_vis.png", 
                np.concatenate([im, seg_im[:, :, 0:3], depth_im[:, :, 0:3]], axis=1))
