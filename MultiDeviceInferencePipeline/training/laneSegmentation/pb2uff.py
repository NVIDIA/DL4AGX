######################################################################################################
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
# File: DL4AGX/MultiDeviceInferencePipeline/training/laneSegmentation/pb2uff.py
# Description: Convert pb file to uff file
#####################################################################################################
import tensorrt as trt
import uff
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate UFF file from protobuf file.")
    parser.add_argument("-p",
                        "--pb_file_name",
                        type=str,
                        required=True,
                        help="""A protobuf file containing a frozen tensorflow graph""")
    parser.add_argument("-u", "--uff_filename", type=str, required=True, help="""Output UFF file""")
    parser.add_argument("-o", "--out_tensor_names", type=str, required=True, help="""Output Tensor names""")
    args, unknown_args = parser.parse_known_args()

    out_tensor_names = [args.out_tensor_names]

    uff.from_tensorflow_frozen_model(args.pb_file_name,
                                     out_tensor_names,
                                     output_filename=args.uff_filename,
                                     text=True,
                                     quiet=False,
                                     write_preprocessed=True,
                                     debug_mode=False)


if __name__ == '__main__':
    main()
