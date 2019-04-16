###################################################################
# The following code was taken from the rules_go repo (Apache License)
###################################################################

# Copyright 2014 The Bazel Authors. All rights reserved.
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

load("//tools/nvcc:private/constants.bzl", CUDA_ACCEPTABLE_SRC_EXTENSIONS = "CUDA_ACCEPTABLE_SRC_EXTENSIONS", 
                                           CUDA_ACCEPTABLE_BIN_EXTENSIONS = "CUDA_ACCEPTABLE_BIN_EXTENSIONS", 
                                           CUDA_ACCEPTABLE_HDR_EXTENSIONS = "CUDA_ACCEPTABLE_HDR_EXTENSIONS")

def as_iterable(v):
    if type(v) == "list":
        return v
    if type(v) == "tuple":
        return v
    if type(v) == "depset":
        return v.to_list()
    fail("Cannot convert {} to an iterable type".format(v))

def split_srcs(srcs):
    sources = struct(
        cu = [],
        headers = [],
        bin = []
    )
    ext_pairs = (
        (sources.cu, CUDA_ACCEPTABLE_SRC_EXTENSIONS),
        (sources.headers, CUDA_ACCEPTABLE_HDR_EXTENSIONS),
        (sources.bin, CUDA_ACCEPTABLE_BIN_EXTENSIONS),
    )
    extmap = {}
    for outs, exts in ext_pairs:
        for ext in exts:
            ext = ext[1:]  # strip the dot
            if ext in extmap:
                break
            extmap[ext] = outs
    for src in as_iterable(srcs):
        extouts = extmap.get(src.extension)
        if extouts == None:
            fail("Unknown source type {0}".format(src.basename))
        extouts.append(src)
    return sources

def join_srcs(source):
    return source.cu + source.headers + source.bin