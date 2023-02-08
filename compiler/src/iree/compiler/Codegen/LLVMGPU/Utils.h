// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace vector {
class ContractionOp;
}  // namespace vector

namespace iree_compiler {

/// Pick an unrolling order that will allow tensorcore operation to reuse LHS
/// register. This is needed to get good performance on sm_80 target.
Optional<SmallVector<int64_t>> gpuMmaUnrollOrder(
    vector::ContractionOp contract);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_H_
