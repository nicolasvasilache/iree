func.func @double_matmul(%lhs : tensor<16x16xf16>, %rhs : tensor<16x16xf16>, %second : tensor<16x8xf16>) -> tensor<16x8xf16> {
  %c0 = arith.constant 0.0 : f16
  %0 = tensor.empty() : tensor<16x16xf16>
  %1 = linalg.fill ins(%c0 : f16) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>)
      outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %3 = tensor.empty() : tensor<16x8xf16>
  %4 = linalg.fill ins(%c0 : f16) outs(%3 : tensor<16x8xf16>) -> tensor<16x8xf16>
  %5 = linalg.matmul ins(%2, %second : tensor<16x16xf16>, tensor<16x8xf16>)
      outs(%4 : tensor<16x8xf16>) -> tensor<16x8xf16>
  return %5 : tensor<16x8xf16>
}

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-hal-cuda-llvm-target-arch=sm_80 \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-preloaded-transforms=%p/double_mma_layout_analysis_dispatch_spec.mlir \
// RUN:     --iree-preloaded-transforms=%p/double_mma_layout_analysis_codegen_spec.mlir | \
// RUN: iree-run-module --module=- --function=double_matmul --device=cuda \
// RUN: --input="16x16xf16=[[0.0999755859375,0.2249755859375,0.07501220703125,0.0,0.07501220703125,0.2249755859375,0.175048828125,0.07501220703125,0.175048828125,0.07501220703125,0.024993896484375,0.1500244140625,0.1500244140625,0.2249755859375,0.199951171875,0.1500244140625],[0.1500244140625,0.199951171875,0.0999755859375,0.07501220703125,0.1500244140625,0.2249755859375,0.024993896484375,0.0999755859375,0.0999755859375,0.024993896484375,0.2249755859375,0.2249755859375,0.2249755859375,0.0,0.024993896484375,0.04998779296875],[0.07501220703125,0.0,0.125,0.125,0.04998779296875,0.2249755859375,0.024993896484375,0.199951171875,0.199951171875,0.07501220703125,0.1500244140625,0.2249755859375,0.024993896484375,0.175048828125,0.07501220703125,0.125],[0.04998779296875,0.024993896484375,0.0,0.2249755859375,0.07501220703125,0.024993896484375,0.024993896484375,0.0,0.07501220703125,0.1500244140625,0.1500244140625,0.175048828125,0.2249755859375,0.1500244140625,0.07501220703125,0.0999755859375],[0.125,0.0,0.199951171875,0.04998779296875,0.199951171875,0.04998779296875,0.175048828125,0.125,0.0,0.0,0.199951171875,0.024993896484375,0.2249755859375,0.1500244140625,0.024993896484375,0.0],[0.04998779296875,0.2249755859375,0.0999755859375,0.07501220703125,0.2249755859375,0.07501220703125,0.2249755859375,0.07501220703125,0.2249755859375,0.199951171875,0.125,0.07501220703125,0.04998779296875,0.199951171875,0.125,0.1500244140625],[0.1500244140625,0.125,0.175048828125,0.04998779296875,0.125,0.1500244140625,0.1500244140625,0.125,0.0999755859375,0.0,0.199951171875,0.024993896484375,0.175048828125,0.199951171875,0.125,0.0999755859375],[0.0999755859375,0.199951171875,0.0999755859375,0.0999755859375,0.2249755859375,0.0,0.175048828125,0.0999755859375,0.125,0.07501220703125,0.07501220703125,0.175048828125,0.07501220703125,0.0,0.2249755859375,0.2249755859375],[0.07501220703125,0.024993896484375,0.199951171875,0.024993896484375,0.175048828125,0.199951171875,0.0999755859375,0.024993896484375,0.0,0.0999755859375,0.0,0.0999755859375,0.2249755859375,0.175048828125,0.0,0.0],[0.024993896484375,0.0999755859375,0.2249755859375,0.2249755859375,0.125,0.2249755859375,0.04998779296875,0.04998779296875,0.04998779296875,0.024993896484375,0.0999755859375,0.2249755859375,0.024993896484375,0.024993896484375,0.0,0.07501220703125],[0.0,0.1500244140625,0.175048828125,0.1500244140625,0.2249755859375,0.024993896484375,0.1500244140625,0.0999755859375,0.024993896484375,0.0,0.125,0.04998779296875,0.125,0.199951171875,0.024993896484375,0.199951171875],[0.024993896484375,0.04998779296875,0.199951171875,0.0,0.07501220703125,0.199951171875,0.2249755859375,0.04998779296875,0.175048828125,0.0,0.199951171875,0.199951171875,0.1500244140625,0.199951171875,0.125,0.199951171875],[0.1500244140625,0.125,0.04998779296875,0.0999755859375,0.04998779296875,0.175048828125,0.04998779296875,0.0999755859375,0.2249755859375,0.199951171875,0.125,0.1500244140625,0.0999755859375,0.07501220703125,0.07501220703125,0.0999755859375],[0.0,0.04998779296875,0.125,0.024993896484375,0.04998779296875,0.199951171875,0.04998779296875,0.0999755859375,0.199951171875,0.07501220703125,0.1500244140625,0.125,0.199951171875,0.199951171875,0.0,0.125],[0.024993896484375,0.07501220703125,0.0,0.199951171875,0.024993896484375,0.024993896484375,0.024993896484375,0.175048828125,0.04998779296875,0.04998779296875,0.04998779296875,0.07501220703125,0.07501220703125,0.1500244140625,0.175048828125,0.199951171875],[0.0,0.125,0.0,0.07501220703125,0.125,0.125,0.07501220703125,0.1500244140625,0.04998779296875,0.04998779296875,0.125,0.125,0.2249755859375,0.0999755859375,0.07501220703125,0.07501220703125]]" \
// RUN: --input="16x16xf16=[[0.175048828125,0.07501220703125,0.199951171875,0.0,0.175048828125,0.125,0.199951171875,0.04998779296875,0.0999755859375,0.175048828125,0.07501220703125,0.04998779296875,0.125,0.125,0.07501220703125,0.2249755859375],[0.024993896484375,0.199951171875,0.0,0.1500244140625,0.175048828125,0.0999755859375,0.175048828125,0.1500244140625,0.2249755859375,0.07501220703125,0.199951171875,0.0999755859375,0.0999755859375,0.2249755859375,0.0999755859375,0.0999755859375],[0.2249755859375,0.2249755859375,0.125,0.175048828125,0.0,0.07501220703125,0.04998779296875,0.0,0.199951171875,0.1500244140625,0.024993896484375,0.2249755859375,0.024993896484375,0.1500244140625,0.2249755859375,0.199951171875],[0.1500244140625,0.125,0.024993896484375,0.07501220703125,0.125,0.125,0.07501220703125,0.1500244140625,0.04998779296875,0.175048828125,0.125,0.175048828125,0.175048828125,0.07501220703125,0.024993896484375,0.125],[0.2249755859375,0.125,0.2249755859375,0.1500244140625,0.0,0.0,0.1500244140625,0.125,0.024993896484375,0.125,0.0,0.024993896484375,0.175048828125,0.175048828125,0.024993896484375,0.125],[0.2249755859375,0.024993896484375,0.04998779296875,0.0,0.0,0.1500244140625,0.07501220703125,0.2249755859375,0.1500244140625,0.024993896484375,0.0,0.0999755859375,0.125,0.1500244140625,0.2249755859375,0.0],[0.125,0.0999755859375,0.0,0.0999755859375,0.199951171875,0.125,0.175048828125,0.175048828125,0.1500244140625,0.2249755859375,0.04998779296875,0.125,0.1500244140625,0.0,0.0,0.0999755859375],[0.125,0.07501220703125,0.175048828125,0.1500244140625,0.175048828125,0.0,0.04998779296875,0.125,0.125,0.024993896484375,0.0999755859375,0.175048828125,0.024993896484375,0.0,0.024993896484375,0.0],[0.2249755859375,0.024993896484375,0.0999755859375,0.04998779296875,0.125,0.07501220703125,0.0999755859375,0.024993896484375,0.125,0.125,0.125,0.024993896484375,0.125,0.04998779296875,0.0999755859375,0.07501220703125],[0.0999755859375,0.175048828125,0.199951171875,0.0999755859375,0.175048828125,0.07501220703125,0.024993896484375,0.125,0.07501220703125,0.0,0.125,0.07501220703125,0.07501220703125,0.0,0.199951171875,0.175048828125],[0.07501220703125,0.0999755859375,0.175048828125,0.07501220703125,0.125,0.1500244140625,0.0,0.0999755859375,0.2249755859375,0.199951171875,0.04998779296875,0.0,0.0,0.1500244140625,0.199951171875,0.2249755859375],[0.024993896484375,0.2249755859375,0.04998779296875,0.1500244140625,0.2249755859375,0.2249755859375,0.175048828125,0.0999755859375,0.024993896484375,0.199951171875,0.125,0.199951171875,0.175048828125,0.2249755859375,0.175048828125,0.0999755859375],[0.125,0.0999755859375,0.04998779296875,0.125,0.199951171875,0.07501220703125,0.199951171875,0.0,0.024993896484375,0.04998779296875,0.0,0.04998779296875,0.04998779296875,0.199951171875,0.1500244140625,0.0999755859375],[0.199951171875,0.0,0.125,0.04998779296875,0.07501220703125,0.175048828125,0.0999755859375,0.175048828125,0.024993896484375,0.07501220703125,0.0,0.1500244140625,0.07501220703125,0.024993896484375,0.07501220703125,0.175048828125],[0.1500244140625,0.125,0.0999755859375,0.175048828125,0.04998779296875,0.0,0.04998779296875,0.1500244140625,0.024993896484375,0.125,0.125,0.175048828125,0.125,0.0999755859375,0.175048828125,0.1500244140625],[0.07501220703125,0.199951171875,0.024993896484375,0.0999755859375,0.175048828125,0.07501220703125,0.1500244140625,0.04998779296875,0.0,0.024993896484375,0.07501220703125,0.07501220703125,0.1500244140625,0.04998779296875,0.2249755859375,0.1500244140625]]" \
// RUN: --input="16x8xf16=[[0.1500244140625,0.07501220703125,0.1500244140625,0.0,0.199951171875,0.125,0.0,0.175048828125],[0.04998779296875,0.07501220703125,0.04998779296875,0.125,0.2249755859375,0.04998779296875,0.04998779296875,0.2249755859375],[0.0,0.07501220703125,0.04998779296875,0.175048828125,0.0999755859375,0.1500244140625,0.04998779296875,0.199951171875],[0.125,0.175048828125,0.04998779296875,0.07501220703125,0.199951171875,0.07501220703125,0.024993896484375,0.1500244140625],[0.175048828125,0.0,0.0,0.0999755859375,0.0999755859375,0.1500244140625,0.07501220703125,0.024993896484375],[0.1500244140625,0.199951171875,0.0999755859375,0.0999755859375,0.125,0.175048828125,0.199951171875,0.0],[0.175048828125,0.0999755859375,0.024993896484375,0.175048828125,0.125,0.07501220703125,0.175048828125,0.175048828125],[0.175048828125,0.175048828125,0.2249755859375,0.125,0.175048828125,0.0,0.04998779296875,0.175048828125],[0.175048828125,0.024993896484375,0.125,0.1500244140625,0.1500244140625,0.07501220703125,0.0,0.04998779296875],[0.125,0.0999755859375,0.024993896484375,0.199951171875,0.175048828125,0.0999755859375,0.04998779296875,0.125],[0.199951171875,0.04998779296875,0.1500244140625,0.0999755859375,0.04998779296875,0.07501220703125,0.199951171875,0.125],[0.1500244140625,0.0,0.125,0.175048828125,0.024993896484375,0.07501220703125,0.199951171875,0.0999755859375],[0.175048828125,0.04998779296875,0.07501220703125,0.125,0.024993896484375,0.2249755859375,0.0,0.0],[0.024993896484375,0.0999755859375,0.1500244140625,0.07501220703125,0.125,0.2249755859375,0.0,0.0],[0.04998779296875,0.125,0.175048828125,0.04998779296875,0.125,0.0999755859375,0.0999755859375,0.04998779296875],[0.125,0.175048828125,0.0,0.2249755859375,0.199951171875,0.175048828125,0.1500244140625,0.1500244140625]]" |\
// RUN: FileCheck %s --check-prefix=EXEC

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 16x8xf16=[0.465332 0.345703 0.341064 0.440674 0.495605 0.424805 0.296875 0.393555][0.420166 0.324707 0.314941 0.414551 0.470215 0.408447 0.268555 0.361084][0.404785 0.305176 0.304688 0.390381 0.438232 0.378174 0.262207 0.349609][0.33252 0.25708 0.235596 0.327393 0.364258 0.320312 0.222168 0.287109][0.326172 0.256592 0.235107 0.332031 0.377686 0.316895 0.204224 0.300537][0.484131 0.361328 0.346436 0.46875 0.525391 0.442871 0.303467 0.425293][0.422852 0.324219 0.311035 0.41626 0.472412 0.398926 0.26709 0.374023][0.447754 0.33252 0.314697 0.439209 0.487305 0.414551 0.285156 0.395508][0.30127 0.23938 0.229736 0.297363 0.343994 0.293701 0.192749 0.272705][0.355225 0.271729 0.270752 0.34668 0.391846 0.334717 0.227539 0.305664][0.375 0.286133 0.267822 0.369141 0.416992 0.348877 0.237549 0.334473][0.454834 0.349365 0.334717 0.44165 0.502441 0.426758 0.292725 0.390381][0.404541 0.302002 0.296631 0.390137 0.437012 0.379639 0.260742 0.34668][0.348877 0.269043 0.263428 0.335205 0.386719 0.33252 0.22522 0.300293][0.297363 0.216431 0.211182 0.283203 0.311523 0.266113 0.194946 0.25708][0.320068 0.242188 0.235107 0.30835 0.349365 0.299072 0.203125 0.275879]
