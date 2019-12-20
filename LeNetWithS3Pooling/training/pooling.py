##########################################################################
# Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: //LeNetWithS3Pooling/training/pooling.py
# Description: Implementation of S3Pooling 
##########################################################################
import numpy as np
import random
import torch
from torchsummary import summary
import torch.nn.functional as F

class StochasticPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(StochasticPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grid_size = kernel_size

        # Reference: https://arxiv.org/pdf/1611.05138.pdf
        # First, perform with stride=1 and maintain resolution
        # Hence, padding zeroes only on the right and bottom
        self.padding = torch.nn.ConstantPad2d((0,1,0,1),0)
    

    def forward(self, x, s3pool_flag=False):     
        # If S3Pool flag is enabled or training mode: Run S3Pooling
        if s3pool_flag or self.training:
        
            # Compute spatial dimensions from input feature map tensor
            h, w = x.shape[-2:]
            n_h = h // self.grid_size
            n_w = w // self.grid_size
            n_h = int(n_h)
            n_w = int(n_w)

            # Reference: https://arxiv.org/pdf/1611.05138.pdf
            # First, perform with stride=1 and maintain resolution
            # Hence, padding only on the right and bottom
            x = self.padding(x)

            # First step : perform maxpooling
            x = F.max_pool2d(x, self.kernel_size, 1)

            w_indices = []
            h_indices = []

            # Second step : Perform stochastic S3Pooling

            for i in range(n_w):

                # Calculate offset
                position_offset = self.grid_size * i

                # Max range for Boundary case
                if i + 1 < n_w:
                    max_range = self.grid_size
                else:
                    max_range = w - position_offset
                
                # Pick random w index from [ position_offset to grid size ]
                # Don't use random at inference time for exporting to IR
                if not self.training:
                    w_index = torch.LongTensor([0])
                else:
                    w_index = torch.LongTensor(1).random_(0, max_range)
                w_indices.append(torch.add(w_index, position_offset))
            
            for j in range(n_h):

                # Calculate offset 
                position_offset = self.grid_size * j

                # Max range for Boundary case
                if j + 1 < n_h:
                    max_range = self.grid_size
                else:
                    max_range = h - position_offset

                # Pick random h index from [position offset to grid_size]
                # Don't use random at inference time for exporting to IR
                if not self.training:
                    h_index = torch.LongTensor([0])
                else:
                    h_index = torch.LongTensor(1).random_(0, max_range)
                h_indices.append(torch.add(h_index, position_offset))
            
            # Gather all the h, w indicies from S3Pooling step
            h_indices = torch.cat(h_indices, dim = 0)
            w_indices = torch.cat(w_indices, dim = 0)

            #output = x
            # Pick values corresponding to h, w indices calculated
            output = x[:, :, h_indices.cuda()][:, :, :, w_indices.cuda()]
            print(x.shape, output.shape)
        else:
            # If S3Pooling flag disabled and inference time, perform average pooling
            # Use AvgPooling
            output = F.avg_pool2d(x, self.kernel_size, self.stride)
        
        return output
        


        
