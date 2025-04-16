MODEL_DIR="/mnt/models"
CALIB_PATH="/workspace/BEVFormer_tensorrt/data/nuscenes/calib_data.npz"
TRT_VERSION="10.3.0.26"
DEVICE="A40"

LOGS_DIR="${MODEL_DIR}/logs_${DEVICE}_trt${TRT_VERSION}"
mkdir -p $LOGS_DIR

BEVFORMER_REPO=/workspace/BEVFormer_tensorrt
ROOT_DIR=$(pwd)

MODEL_NAMES=(
  bevformer_tiny_epoch_24_cp_op13
  bevformer_tiny_epoch_24_cp2_op13
)

PLUGIN_PATH="/workspace/BEVFormer_tensorrt/TensorRT/lib/libtensorrt_ops.so"
export CUDA_MODULE_LOADING=LAZY

len=${#MODEL_NAMES[@]}
echo "Evaluating ${len} models!"

for (( i=0; i<$len; i++ )); do
  MODEL_NAME=${MODEL_NAMES[$i]}
  echo "======== ${MODEL_NAME} ======== "

  echo "Baseline:"
  for PRECISION in fp16 fp32; do
    echo "  - ${PRECISION}"
    PRECISION_FLAG=""
    if [[ "$PRECISION" != "fp32" ]]; then
        PRECISION_FLAG="--${PRECISION}"
    fi
    trtexec --onnx=${MODEL_DIR}/${MODEL_NAME}.onnx \
            --saveEngine=${LOGS_DIR}/${MODEL_NAME}_${PRECISION}.engine \
            --staticPlugins=$PLUGIN_PATH \
            ${PRECISION_FLAG} &> ${LOGS_DIR}/${MODEL_NAME}_${PRECISION}_trtexec.log
    wait
  done

  echo "Quantize model - ModelOpt PTQ (Explicit Quantization):"
  python /mnt/tools/quantize_model.py \
    --onnx_path=${MODEL_DIR}/${MODEL_NAME}.onnx \
    --output_path=${MODEL_DIR}/${MODEL_NAME}.quant.onnx \
    --trt_plugins=$PLUGIN_PATH \
    --op_types_to_exclude MatMul \
    --calibration_data_path=$CALIB_PATH \
    --simplify
  wait
  for PRECISION in best; do
    echo "    - ${PRECISION}"
    trtexec --onnx=${MODEL_DIR}/${MODEL_NAME}.quant.onnx \
            --staticPlugins=$PLUGIN_PATH \
            --saveEngine=${LOGS_DIR}/${MODEL_NAME}_qat_${PRECISION}.engine \
            --${PRECISION} &> ${LOGS_DIR}/${MODEL_NAME}_qat_${PRECISION}_trtexec.log
    wait
  done

  echo "Quantize model - TensorRT PTQ (Implicit Quantization):"
  cd $BEVFORMER_REPO
  IQ_ENGINE_PATH=${LOGS_DIR}/${MODEL_NAME}_IQ_PTQ
  python /mnt/results/onnx2trt_calib_npz.py configs/bevformer/plugin/bevformer_tiny_trt_p2.py \
        --onnx_path=${MODEL_DIR}/${MODEL_NAME}.onnx \
        --output=${IQ_ENGINE_PATH}.engine \
        --int8 --fp16 \
        --calibrator entropy \
        --calibration_data_path=$CALIB_PATH \
        --trt_plugins=$PLUGIN_PATH
  wait
  # Obtain engine's runtime
  trtexec --loadEngine=${IQ_ENGINE_PATH}.engine \
          --staticPlugins=$PLUGIN_PATH &> ${IQ_ENGINE_PATH}_trtexec.log
  wait
  cd $ROOT_DIR

done

chmod 777 -R ${LOGS_DIR}/*
tail -n +1 $LOGS_DIR/*_trtexec.log | grep "==>\|GPU Compute Time: min"
