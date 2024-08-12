MODEL_DIR="/mnt/models"
LOGS_DIR="${MODEL_DIR}/logs_rtx3090_trt10.0.1.6"
mkdir -p $LOGS_DIR
MODEL_NAMES=(
  bevformer_tiny_epoch_24_cp2_op13_post
)

PLUGIN_PATH="/workspace/BEVFormer_tensorrt/TensorRT/lib/libtensorrt_ops.so"
export CUDA_MODULE_LOADING=LAZY

len=${#MODEL_NAMES[@]}
echo "Evaluating ${len} models!"

for (( i=0; i<$len; i++ )); do
  MODEL_NAME=${MODEL_NAMES[$i]}
  echo "======== ${MODEL_NAME} ======== "

  # Baseline
  echo "Baseline:"
  for PRECISION in best fp16; do
    echo "  - ${PRECISION}"
    trtexec --onnx=${MODEL_DIR}/${MODEL_NAME}.onnx \
            --staticPlugins=$PLUGIN_PATH \
            --${PRECISION} &> ${LOGS_DIR}/${MODEL_NAME}_${PRECISION}_trtexec.log
    wait
  done

  # Quantize model
  echo "Quantize model"
  echo "  - Default:"
  python -m modelopt.onnx.quantization --onnx_path=${MODEL_DIR}/${MODEL_NAME}.onnx \
    --output_path=${MODEL_DIR}/${MODEL_NAME}_DEFAULT.quant.onnx \
    --trt_plugins=$PLUGIN_PATH
  wait
  for PRECISION in best; do
    echo "    - ${PRECISION}"
    trtexec --onnx=${MODEL_DIR}/${MODEL_NAME}_DEFAULT.quant.onnx \
            --staticPlugins=$PLUGIN_PATH \
            --${PRECISION} &> ${LOGS_DIR}/${MODEL_NAME}_qat_${PRECISION}_DEFAULT_trtexec.log
    wait
  done

  echo "  - Exclude MatMul:"
  python -m modelopt.onnx.quantization --onnx_path=${MODEL_DIR}/${MODEL_NAME}.onnx \
    --output_path=${MODEL_DIR}/${MODEL_NAME}_EXCLUDE_MatMul.quant.onnx \
    --trt_plugins=$PLUGIN_PATH \
    --op_types_to_exclude MatMul
  wait
  for PRECISION in best; do
    echo "    - ${PRECISION}"
    trtexec --onnx=${MODEL_DIR}/${MODEL_NAME}_EXCLUDE_MatMul.quant.onnx \
            --staticPlugins=$PLUGIN_PATH \
            --${PRECISION} &> ${LOGS_DIR}/${MODEL_NAME}_qat_${PRECISION}_EXCLUDE_MatMul_trtexec.log
    wait
  done

done

chmod 777 -R ${LOGS_DIR}/*
tail -n +1 $LOGS_DIR/*_trtexec.log | grep "==>\|GPU Compute Time: min"
