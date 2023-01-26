
// RUN: iree-opt --iree-transform-dialect-interpreter --transform-dialect-drop-schedule %s | \
// RUN: iree-compile - --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=5 | \
// RUN: nvprof --print-gpu-trace iree-run-module --entry_function=conv_2d_nchw_fchw --device=cuda --function_input="127x47x16x16xf32=1" --function_input="127x16x14x14xf32=0"

!input_tensor_t = tensor<127x47x16x16xf32>
!weight_tensor_t = tensor<16x47x3x3xf32>
!output_tensor_t = tensor<127x16x14x14xf32>
func.func @conv_2d_nchw_fchw(%arg0: !input_tensor_t, %arg2: !output_tensor_t) 
  -> !output_tensor_t 
{
  %c0 = arith.constant dense<0.1> : !weight_tensor_t
  %0 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %c0: !input_tensor_t, !weight_tensor_t)
    outs(%arg2: !output_tensor_t) -> !output_tensor_t
  return %0 : !output_tensor_t
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %module_op
    : (!pdl.operation) -> !transform.op<"linalg.conv_2d_nchw_fchw">
  
  transform.structured.pack_greedily %conv
      gemm_packed_sizes = [8, 16, 32] gemm_inner_dims_order = [1, 2, 0]
    : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.op<"linalg.generic">

  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">) 
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.generic">)

  %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.unpack">
  transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">) 
    -> (!transform.op<"tensor.empty">, 
        !transform.op<"linalg.generic">,
        !transform.op<"tensor.collapse_shape">,
        !transform.op<"tensor.extract_slice">)
}
