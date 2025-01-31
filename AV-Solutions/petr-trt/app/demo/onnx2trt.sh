export TRT_EXEC=$TRT_ROOT/bin/trtexec
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_ROOT/lib

for fname in PETRv1.extract_feat.onnx PETRv1.pts_bbox_head.forward.onnx PETRv2.extract_feat.onnx PETRv2.pts_bbox_head.forward.onnx; do
$TRT_EXEC --onnx=onnx_files/$fname \
          --profilingVerbosity=detailed \
          --dumpLayerInfo --dumpProfile \
          --separateProfileRun --fp16 \
          --saveEngine=engines/$fname.fp16.engine
done
