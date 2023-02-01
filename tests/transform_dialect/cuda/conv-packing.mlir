
// RUN: iree-opt --iree-transform-dialect-interpreter --transform-dialect-drop-schedule %s | \
// RUN: iree-compile - --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=5 | \
// RUN: nvprof --print-gpu-trace iree-run-module --entry_function=conv_2d_nchw_fchw --device=cuda --function_input="127x47x16x16xf32=1" --function_input="127x16x14x14xf32=0"

func.func @conv_2d_nchw_fchw(%arg0: tensor<127x47x16x16xf32>, %arg2: tensor<127x16x14x14xf32>) -> tensor<127x16x14x14xf32> {
  %c0 = arith.constant dense<0.1> : tensor<16x47x3x3xf32>
  %0 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %c0: tensor<127x47x16x16xf32>, tensor<16x47x3x3xf32>)
    outs(%arg2: tensor<127x16x14x14xf32>) -> tensor<127x16x14x14xf32>
  return %0 : tensor<127x16x14x14xf32>
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
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)

  %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.unpack">
  transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">) 
    -> (!transform.op<"tensor.empty">, 
        !transform.op<"linalg.transpose">,
        !transform.op<"tensor.collapse_shape">,
        !transform.op<"tensor.extract_slice">)
}
