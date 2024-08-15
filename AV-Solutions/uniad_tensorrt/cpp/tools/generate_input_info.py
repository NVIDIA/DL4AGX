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

import pickle as pkl
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate info in txt files")
    parser.add_argument('--info_pkl_pth', help='input info pkl file path')
    parser.add_argument('--prefix_pth', help='original image path')
    parser.add_argument('--info_file_pth', help='output txt file path')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    infos = pkl.load(open(args.info_pkl_pth, "rb"))

    # re-order the infos
    start_infos = []
    token2info = {}
    ordered_infos = []
    for info in infos["infos"]:
        if info["prev"] == "":
            start_infos.append(info)
        token2info[info["token"]] = info
    
    for info in start_infos:
        ordered_infos.append(info)
        curr_info = info
        while curr_info["next"] != "" and curr_info["next"] in token2info.keys():
            ordered_infos.append(token2info[curr_info["next"]])
            curr_info = token2info[curr_info["next"]]
    
    if ordered_infos[0] not in start_infos:
        print("Incorrect order!")
    for i in range(len(ordered_infos)-1):
        # i, i+1
        if ordered_infos[i]["next"] == "":
            if ordered_infos[i+1]["prev"] != "":
                print("Incorrect order!")
        else:
            if ordered_infos[i]["next"] != ordered_infos[i+1]["token"]:
                print("Incorrect order!")
            elif ordered_infos[i]["token"] != ordered_infos[i+1]["prev"]:
                print("Incorrect order!")

    # dump to txt files
    prefix_pth = args.prefix_pth
    dump_info_str = [f"{ os.path.join(prefix_pth, _info['cams']['CAM_FRONT']['data_path']) };{ os.path.join(prefix_pth, _info['cams']['CAM_FRONT_RIGHT']['data_path']) };{ os.path.join(prefix_pth, _info['cams']['CAM_FRONT_LEFT']['data_path']) };{ os.path.join(prefix_pth, _info['cams']['CAM_BACK']['data_path']) };{ os.path.join(prefix_pth, _info['cams']['CAM_BACK_LEFT']['data_path']) };{ os.path.join(prefix_pth, _info['cams']['CAM_BACK_RIGHT']['data_path']) };\n" for _info in ordered_infos]
    info_file = open(args.info_file_pth, "w")
    info_file.writelines(dump_info_str)
    info_file.close()
