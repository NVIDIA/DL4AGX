MODEL_DIR="/mnt/models"
TRT_VERSION="10.9.0.34"
DEVICE="A40"
LOGS_DIR="${MODEL_DIR}/logs_${DEVICE}_trt${TRT_VERSION}"
BEVFORMER_REPO=/workspace/BEVFormer_tensorrt

MODEL_NAMES=(
  bevformer_tiny_epoch_24_cp_op13
  bevformer_tiny_epoch_24_cp2_op13
)

PLUGIN_PATH="/workspace/BEVFormer_tensorrt/TensorRT/lib/libtensorrt_ops.so"

len=${#MODEL_NAMES[@]}
echo "Evaluating ${len} models!"

cd $BEVFORMER_REPO

for (( i=0; i<$len; i++ )); do
  MODEL_NAME=${MODEL_NAMES[$i]}
  echo "======== ${MODEL_NAME} ======== "

  echo "Baseline:"
  for PRECISION in fp16 fp32; do
    echo "  - ${PRECISION}"
    ENGINE_PATH=${LOGS_DIR}/${MODEL_NAME}_${PRECISION}
    python tools/bevformer/evaluate_trt.py \
           configs/bevformer/plugin/bevformer_tiny_trt_p2.py \
           ${ENGINE_PATH}.engine \
           --trt_plugins=$PLUGIN_PATH | tee ${ENGINE_PATH}_acc.log
    wait
  done

  echo "Quantized - ModelOpt PTQ (Explicit Quantization):"
  for PRECISION in best; do
    echo "  - ${PRECISION}"
    ENGINE_PATH=${LOGS_DIR}/${MODEL_NAME}_qat_${PRECISION}
    python tools/bevformer/evaluate_trt.py \
            configs/bevformer/plugin/bevformer_tiny_trt_p2.py \
            ${ENGINE_PATH}.engine \
            --trt_plugins=$PLUGIN_PATH | tee ${ENGINE_PATH}_acc.log
    wait
  done

  echo "Quantized - TensorRT PTQ (Implicit Quantization):"
  ENGINE_PATH=${LOGS_DIR}/${MODEL_NAME}_IQ_PTQ
  python tools/bevformer/evaluate_trt.py \
          configs/bevformer/plugin/bevformer_tiny_trt_p2.py \
          ${ENGINE_PATH}.engine \
          --trt_plugins=$PLUGIN_PATH | tee ${ENGINE_PATH}_acc.log
  wait
done

echo "\n======== SUMMARY ========"
tail -n 4 $LOGS_DIR/*_acc.log
