// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/MatmulTensorCoreStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Strategies.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::blockX;
using iree_compiler::blockY;
using iree_compiler::blockZ;
using iree_compiler::buildPad;
using iree_compiler::buildTileFuseDistToForallWithNumThreads;
using iree_compiler::buildTileFuseDistToForallWithTileSizes;
using iree_compiler::TileToForallAndFuseAndDistributeResult;
using iree_compiler::gpu::buildBufferize;
using iree_compiler::gpu::buildConvertToAsyncCopies;
using iree_compiler::gpu::buildDistributeOnePadOrCopyWithNumThreads;
using iree_compiler::gpu::buildDistributeOnePadOrCopyWithTileSizes;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using transform::MatchOp;
using transform_ext::RegisterMatchCallbacksOp;

static std::tuple<Value, Value> buildPadStrategyBlockDistribution(
    ImplicitLocOpBuilder &b, Value variantH) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [padH] = unpackRegisteredMatchCallback<1>(
      b, "pad", transform::FailurePropagationMode::Propagate, variantH);

  // Step 2. Create the block/mapping tiling level.
  MLIRContext *ctx = b.getContext();
  auto [tiledPadH, forallH] = buildDistributeOnePadOrCopyWithTileSizes(
      b, variantH, padH, /*tileSizes=*/{64, 64},
      /*threadDimMapping=*/{blockY(ctx), blockX(ctx)}, /*foldIfBranch=*/true);

  // Step 3.Handle the workgroup count region.
  b.create<IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp>(forallH);
  return std::make_tuple(tiledPadH, forallH);
}

void iree_compiler::gpu::buildPadStrategy(ImplicitLocOpBuilder &b,
                                          Value variantH) {
  MLIRContext *ctx = b.getContext();
  // Step 1. Apply block-level part of the strategy.
  auto [padBlockH, forallBlockH] =
      buildPadStrategyBlockDistribution(b, variantH);

  // Step 2. Apply thread-level part of the strategy.
  auto padThreadH = buildDistributeOnePadOrCopyWithNumThreads(
      b, variantH, padBlockH, /*numThreads=*/{16, 16},
      /*threadDimMapping=*/{threadY(ctx), threadX(ctx)}, /*foldIfBranch=*/true);

  // Step 3. Masked vectorization.
  b.create<transform::MaskedVectorizeOp>(padThreadH, ValueRange(), false,
                                         ArrayRef<int64_t>{4, 4});

  // Step 4. Lower all masked vector transfers at this point, as they make
  // canonicalization generate incorrect IR.
  // TODO: don't rematch, apply on the variant op directly.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildLowerMaskedTransfersAndCleanup(b, funcH);

  // Step 5. Vectorize the rest of func normally.
  funcH = buildVectorize(b, funcH, /*applyCleanups=*/true);

  // Step 6. Bufferize and drop HAL descriptor from memref ops.
  variantH = buildBufferize(b, variantH);

  // Step 7. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, needs hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, /*blockSize=*/{16, 16, 1});

  // TODO: Multi-buffering and async copies in cases where HW supports it.

  // Step 8. Lower masks before returning to the default lowering pipeline.
  buildLowerVectorMasksAndCleanup(b, funcH);
}
