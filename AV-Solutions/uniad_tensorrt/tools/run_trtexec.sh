#!/bin/bash

# Define variables
TRT_VERSION=8.6.13.3
FP=fp32
DAT_PATH=./dumped_inputs/uniad_trtexec_fp64
MODEL=uniad_tiny_imgx0.25_cp.repaired
TRT_PATH=./TensorRT
MIN=901
OPT=901
MAX=1150
ONNX_PATH=./onnx
PLUGINS_PATH=./tools/tensorrt_plugin
ENGINE_PATH=./trt
LOGS_PATH=./logs

mkdir ${ENGINE_PATH}
mkdir ${LOGS_PATH}

# Construct shapes
SHAPES="prev_track_intances0:${MIN}x512,prev_track_intances1:${MIN}x3,prev_track_intances3:${MIN},prev_track_intances4:${MIN},prev_track_intances5:${MIN},prev_track_intances6:${MIN},prev_track_intances8:${MIN},prev_track_intances9:${MIN}x10,prev_track_intances11:${MIN}x4x256,prev_track_intances12:${MIN}x4,prev_track_intances13:${MIN}"

# Construct inputs
INPUTS="max_obj_id:${DAT_PATH}/max_obj_id.dat,img_metas_can_bus:${DAT_PATH}/img_metas_can_bus.dat,img_metas_lidar2img:${DAT_PATH}/img_metas_lidar2img.dat,img:${DAT_PATH}/img.dat,use_prev_bev:${DAT_PATH}/use_prev_bev.dat,prev_bev:${DAT_PATH}/prev_bev.dat,command:${DAT_PATH}/command.dat,timestamp:${DAT_PATH}/timestamp.dat,l2g_r_mat:${DAT_PATH}/l2g_r_mat.dat,l2g_t:${DAT_PATH}/l2g_t.dat,prev_track_intances0:${DAT_PATH}/prev_track_intances0.dat,prev_track_intances1:${DAT_PATH}/prev_track_intances1.dat,prev_track_intances3:${DAT_PATH}/prev_track_intances3.dat,prev_track_intances4:${DAT_PATH}/prev_track_intances4.dat,prev_track_intances5:${DAT_PATH}/prev_track_intances5.dat,prev_track_intances6:${DAT_PATH}/prev_track_intances6.dat,prev_track_intances8:${DAT_PATH}/prev_track_intances8.dat,prev_track_intances9:${DAT_PATH}/prev_track_intances9.dat,prev_track_intances11:${DAT_PATH}/prev_track_intances11.dat,prev_track_intances12:${DAT_PATH}/prev_track_intances12.dat,prev_track_intances13:${DAT_PATH}/prev_track_intances13.dat,prev_timestamp:${DAT_PATH}/prev_timestamp.dat,prev_l2g_r_mat:${DAT_PATH}/prev_l2g_r_mat.dat,prev_l2g_t:${DAT_PATH}/prev_l2g_t.dat"

# Execute trtexec
LD_LIBRARY_PATH=${TRT_PATH}/TensorRT-${TRT_VERSION}/lib:$LD_LIBRARY_PATH \
${TRT_PATH}/TensorRT-${TRT_VERSION}/bin/trtexec \
  --onnx=${ONNX_PATH}/${MODEL}.onnx \
  --saveEngine=${ENGINE_PATH}/${MODEL}_trt${TRT_VERSION}_${FP}.plan \
  --plugins=${PLUGINS_PATH}/tensorrt_plugin_trt${TRT_VERSION}/lib/libtensorrt_ops.so \
  --verbose \
  --dumpLayerInfo \
  --dumpProfile \
  --separateProfileRun \
  --profilingVerbosity=detailed \
  --useCudaGraph \
  --minShapes=${SHAPES//${MIN}/${MIN}} \
  --optShapes=${SHAPES//${MIN}/${OPT}} \
  --maxShapes=${SHAPES//${MIN}/${MAX}} \
  --loadInputs=${INPUTS} \
  2>&1 | tee ${LOGS_PATH}/${MODEL}_trt${TRT_VERSION}_${FP}.log