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



#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-27596}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to ./projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/export_onnx.py \
    $CFG \
    $CKPT \
    --launcher pytorch ${@:4} \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/eval.$T