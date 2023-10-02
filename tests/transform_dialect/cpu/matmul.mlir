
!A_size = tensor<3x5xf32>
!B_size = tensor<5x3xf32>
!C_size = tensor<3x3xf32>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}

// RUN: iree-opt %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-abi-transformation-pipeline \
// RUN:   --iree-flow-transformation-pipeline \
// RUN:   --iree-stream-transformation-pipeline \
// RUN:   --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' \
// RUN:   --iree-preloaded-transforms=%p/matmul_codegen_default_spec.mlir | \
// RUN: FileCheck %s --check-prefixes=CODEGEN-DEFAULT

// CODEGEN-DEFAULT:     hal.executable.export public @matmul_static_dispatch_0_matmul_3x3x5
// CODEGEN-DEFAULT:         %[[C2:.+]] = arith.constant 2 : index
// CODEGEN-DEFAULT:         %[[C1:.+]] = arith.constant 1 : index
// CODEGEN-DEFAULT:         hal.return %[[C2]], %[[C1]], %[[C1]]

// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-preloaded-transforms=%p/matmul_codegen_default_spec.mlir | \
// RUN: iree-run-module --module=- --function=matmul_static \
// RUN:   --input="3x5xf32=1" \
// RUN:   --input="5x3xf32=2" \
// RUN:   --input="3x3xf32=42" | \
// RUN: FileCheck %s --check-prefixes=EXEC

// EXEC: 3x3xf32=[52 52 52][52 52 52][52 52 52]
