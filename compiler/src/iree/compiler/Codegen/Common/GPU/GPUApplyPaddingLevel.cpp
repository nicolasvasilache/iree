// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-padding-level"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYPADDINGLEVELPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUApplyPaddingLevelPass final
    : impl::GPUApplyPaddingLevelPassBase<GPUApplyPaddingLevelPass> {
  using GPUApplyPaddingLevelPassBase::GPUApplyPaddingLevelPassBase;
  void runOnOperation() override;
};
} // namespace

using namespace mlir::linalg;

/// Compute the padded shape of the given operand. The operand is padded to a
/// static bounding box according to the specified padding options.
static LogicalResult computePaddedShape(OpBuilder &b, linalg::LinalgOp opToPad,
                                        OpOperand *opOperand,
                                        const LinalgPaddingOptions &options,
                                        SmallVector<int64_t> &paddedShape,
                                        SmallVectorImpl<Value> &dynDims,
                                        bool &alreadyHasRequestedShape) {
  AffineMap indexingMap = opToPad.getMatchingIndexingMap(opOperand);
  ArrayRef<int64_t> shape = opToPad.getShape(opOperand);

  // Collect the shape dimensions that are a function of "paddingDimensions",
  // along with the multiple that they should be padded to ("1" if none).
  alreadyHasRequestedShape = true;
  DenseMap<int64_t, int64_t> shapeDimToMultiple;
  for (const auto &dimEn : enumerate(options.paddingDimensions)) {
    for (const auto &en : enumerate(indexingMap.getResults())) {
      if (en.value().isFunctionOfDim(dimEn.value())) {
        int64_t dimSize = shape[en.index()];
        if (options.padToMultipleOf.has_value()) {
          shapeDimToMultiple[en.index()] =
              (*options.padToMultipleOf)[dimEn.index()];
        } else {
          shapeDimToMultiple[en.index()] = 1;
        }
        if (ShapedType::isDynamic(dimSize)) {
          alreadyHasRequestedShape = false;
        } else if (dimSize % shapeDimToMultiple[en.index()] != 0) {
          alreadyHasRequestedShape = false;
        }
      }
    }
  }

  // Helper function to round a number up to a given multiple.
  auto ceil = [](int64_t val, int64_t multiple) {
    return ((val + multiple - 1) / multiple) * multiple;
  };

  // Upper bound the sizes to obtain a static bounding box.
  paddedShape.assign(shape.begin(), shape.end());
  for (int64_t i = 0, e = shape.size(); i < e; ++i) {
    LLVM_DEBUG(DBGS() << "--compute padded size for dim " << i << "\n");
    // Skip dimensions that do not require padding.
    if (!shapeDimToMultiple.contains(i)) {
      LLVM_DEBUG(DBGS() << "----dim does not require padding, SKIP\n");
      continue;
    }
    if (!ShapedType::isDynamic(shape[i])) {
      // Otherwise, try to compute a constant upper bound for the size value.
      FailureOr<int64_t> upperBound =
          ValueBoundsConstraintSet::computeConstantBound(
              presburger::BoundType::UB,
              {opOperand->get(),
               /*dim=*/i},
              /*stopCondition=*/nullptr, /*closedUB=*/true);
      if (failed(upperBound)) {
        LLVM_DEBUG(
            DBGS() << "----could not compute a bounding box for padding");
        return failure();
      }
      paddedShape[i] = ceil(*upperBound, shapeDimToMultiple[i]);
      LLVM_DEBUG(DBGS() << "----new dim size: " << paddedShape[i] << "\n");
    } else {
      paddedShape[i] = ShapedType::kDynamic;
      AffineExpr size;
      bindDims(b.getContext(), size);
      AffineExpr ceil =
          ((size + shapeDimToMultiple[i] - 1).floorDiv(shapeDimToMultiple[i])) *
          shapeDimToMultiple[i];
      Value sz = b.create<tensor::DimOp>(opOperand->get().getLoc(),
                                         opOperand->get(), i);
      dynDims.push_back(affine::makeComposedAffineApply(
          b, opOperand->get().getLoc(), ceil, {sz}));
    }
  }

  return success();
}

static Value makeComposedPadHighOp(OpBuilder &b, Location loc,
                                   RankedTensorType type, Value source,
                                   Value pad, bool nofold,
                                   ArrayRef<Value> dynDims) {
  // Exit if `source` is not defined by an ExtractSliceOp.
  auto sliceOp = source.getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   llvm::to_vector(dynDims));

  // Search the `source` use-def chain for padded LinalgOps.
  Value current = sliceOp.getSource();
  while (current) {
    auto linalgOp = current.getDefiningOp<LinalgOp>();
    if (!linalgOp)
      break;
    OpResult opResult = cast<OpResult>(current);
    current = linalgOp.getDpsInitOperand(opResult.getResultNumber())->get();
  }
  auto padOp = current ? current.getDefiningOp<tensor::PadOp>() : nullptr;

  // Exit if the search fails to match a tensor::PadOp at the end of the matched
  // LinalgOp sequence.
  if (!padOp)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   llvm::to_vector(dynDims));

  // Exit if the padded result type does not match.
  if (sliceOp.getSource().getType() != type)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   llvm::to_vector(dynDims));

  // Exit if the LinalgOps are not high padded.
  if (llvm::any_of(padOp.getMixedLowPad(), [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) != static_cast<int64_t>(0);
      }))
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   llvm::to_vector(dynDims));

  // Exit if `padOpSliceOp`, which defines the slice used by
  // `padOp`, is rank-reducing.
  auto padOpSliceOp = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!padOpSliceOp ||
      sliceOp.getMixedSizes().size() != padOpSliceOp.getMixedSizes().size())
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   llvm::to_vector(dynDims));

  // Exit if the sizes of the dynamic sizes of `sliceOp` do not match the size
  // of the slice padded by `padOp`.
  if (llvm::any_of(
          llvm::zip(sliceOp.getMixedSizes(), padOpSliceOp.getMixedSizes()),
          [](std::tuple<OpFoldResult, OpFoldResult> it) {
            return !isEqualConstantIntOrValue(std::get<0>(it), std::get<1>(it));
          }))
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   llvm::to_vector(dynDims));

  // Exit if the padding values do not match.
  Attribute padOpPadAttr, padAttr;
  Value padOpPad = padOp.getConstantPaddingValue();
  if (!padOpPad || !matchPattern(padOpPad, m_Constant(&padOpPadAttr)) ||
      !matchPattern(pad, m_Constant(&padAttr)) || padOpPadAttr != padAttr)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b,
                                   llvm::to_vector(dynDims));

  // Return the padded result if the padding values and sizes match.
  return sliceOp.getSource();
}

/// Pad the `opOperand` in the "paddingDimensions" using the padding value and
/// the nofold flag found in "paddingValues" and "nofoldFlags", respectively.
///
/// Exit early and return the `opOperand` value if it already has the requested
/// shape. i.e.:
/// - static shape
/// - nofold is not set
/// - dim sizes are multiples of "padToMultipleOf"
///
/// Otherwise, try to pad the shape dimensions that match the iterator
/// dimensions "paddingDimensions" and return the tensor::PadOp result if
/// padding succeeds or failure otherwise.
static FailureOr<Value> padOperandToSmallestStaticBoundingBox(
    RewriterBase &rewriter, linalg::LinalgOp opToPad, OpOperand *opOperand,
    const LinalgPaddingOptions &options) {
  assert(
      (!options.padToMultipleOf.has_value() ||
       options.padToMultipleOf->size() == options.paddingDimensions.size()) &&
      "invalid number of elements in padToMultipleOf");

  // Fail if `paddingValues` specifies no padding value.
  if (opOperand->getOperandNumber() >= options.paddingValues.size()) {
    return rewriter.notifyMatchFailure(opToPad, "--no padding value specified");
  }

  // Compute padded shape.
  SmallVector<int64_t> paddedShape;
  bool alreadyHasRequestedShape = false;
  SmallVector<Value> dynDims;
  if (failed(computePaddedShape(rewriter, opToPad, opOperand, options,
                                paddedShape, dynDims,
                                alreadyHasRequestedShape)))
    return rewriter.notifyMatchFailure(opToPad,
                                       "--failed to compute padded shape");

  // Return the unpadded operand if padding to a static shape is not needed and
  // if the nofold flag is not set.
  bool nofold = opOperand->getOperandNumber() < options.nofoldFlags.size()
                    ? bool(options.nofoldFlags[opOperand->getOperandNumber()])
                    : false;
  if (!nofold && alreadyHasRequestedShape)
    return opOperand->get();
  Attribute paddingAttr = options.paddingValues[opOperand->getOperandNumber()];

  Value paddingValue;
  if (auto complexTy = dyn_cast<ComplexType>(
          getElementTypeOrSelf(opOperand->get().getType()))) {
    auto complexAttr = cast<ArrayAttr>(paddingAttr);
    paddingValue = rewriter.create<complex::ConstantOp>(opToPad.getLoc(),
                                                        complexTy, complexAttr);
  } else {
    paddingValue = rewriter.create<arith::ConstantOp>(
        opToPad.getLoc(), cast<TypedAttr>(paddingAttr));
  }

  // Pad the operand to the bounding box defined by `paddedShape`.
  auto paddedTensorType = RankedTensorType::get(
      paddedShape, getElementTypeOrSelf(opOperand->get()));
  LLVM_DEBUG(DBGS() << "--SUCCESS, makeComposedPadHighOp with type: "
                    << paddedTensorType);
  return mlir::iree_compiler::makeComposedPadHighOp(
      rewriter, opToPad->getLoc(), paddedTensorType, opOperand->get(),
      paddingValue, nofold, dynDims);
}

static LogicalResult rewriteAsPaddedOp(RewriterBase &rewriter, LinalgOp opToPad,
                                       const LinalgPaddingOptions &constOptions,
                                       LinalgOp &paddedOp,
                                       SmallVector<Value> &replacements,
                                       SmallVector<tensor::PadOp> &padOps) {
  LLVM_DEBUG(DBGS() << "Start rewriteAsPaddedOp : " << opToPad << "\n");
  Location loc = opToPad->getLoc();

  LinalgPaddingOptions options(constOptions);
  if (options.paddingValues.empty()) {
    SmallVector<Type> types(opToPad->getOperandTypes());
    llvm::append_range(types, opToPad->getResultTypes());
    for (Type t : types) {
      options.paddingValues.push_back(
          rewriter.getZeroAttr(getElementTypeOrSelf(t)));
    }
  }

  // TODO: there are cases where we may still want to pad to larger sizes.
  if (!opToPad.hasPureTensorSemantics())
    return rewriter.notifyMatchFailure(opToPad,
                                       "expected operation on tensors");

  OpBuilder::InsertionGuard g(rewriter);
  // Set IP after op because we also take the dims of the original output.
  rewriter.setInsertionPointAfter(opToPad);

  // Make a copy of the shaped operands and update it.
  SmallVector<Value> newOperands;
  newOperands.reserve(opToPad->getNumOperands());
  for (OpOperand &opOperand : opToPad->getOpOperands()) {
    FailureOr<Value> paddedOperand = padOperandToSmallestStaticBoundingBox(
        rewriter, opToPad, &opOperand, options);
    // Exit if `paddingDimensions` cannot be bounded statically.
    if (failed(paddedOperand)) {
      LLVM_DEBUG(DBGS() << "--operand cannot be bound statically : "
                        << opOperand.get() << " -> FAIL\n");
      return rewriter.notifyMatchFailure(opToPad,
                                         "operand cannot be bound statically");
    }
    newOperands.push_back(*paddedOperand);
    if (auto padOp = paddedOperand->getDefiningOp<tensor::PadOp>())
      padOps.push_back(padOp);
  }

  ReifiedRankedShapedTypeDims reifiedResultShapes;
  if (failed(reifyResultShapes(rewriter, opToPad, reifiedResultShapes))) {
    LLVM_DEBUG(DBGS() << "--failed to reify result shapes -> FAIL\n");
    return rewriter.notifyMatchFailure(opToPad,
                                       "failed to reify result shapes");
  }
  assert(reifiedResultShapes.size() == opToPad->getNumResults() &&
         "expected same number of results");

  // Clone `opToPad` to operate on the statically padded shapes.
  auto resultTensorTypes =
      ValueRange(newOperands).take_back(opToPad.getNumDpsInits()).getTypes();
  // clone **should** properly notify the rewriter.
  paddedOp = clone(rewriter, opToPad, resultTensorTypes, newOperands);
  LLVM_DEBUG(DBGS() << "--cloned padded op: " << paddedOp << "\n");

  // Recover the slice out of the new static results. This keeps the original
  // linalg op around because it uses the dims of the original results.
  SmallVector<Value> paddedSubtensorResults;
  paddedSubtensorResults.reserve(opToPad->getNumResults());
  for (const auto &en : llvm::enumerate(paddedOp->getResults())) {
    Value paddedResult = en.value();
    int64_t resultNumber = en.index();
    int64_t rank = cast<RankedTensorType>(paddedResult.getType()).getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    paddedSubtensorResults.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, paddedResult, offsets, reifiedResultShapes[resultNumber],
        strides));
  }

  assert(options.copyBackOp == LinalgPaddingOptions::CopyBackOp::None);
  replacements = std::move(paddedSubtensorResults);
  return success();
}

static llvm::SmallDenseSet<TilingInterface>
getTiledOps(Operation *funcOp, IREE::GPU::TilingLevel tilingLevel) {
  llvm::SmallDenseSet<TilingInterface> targets;
  unsigned opaqueLevel = llvm::to_underlying(tilingLevel);
  funcOp->walk([&](TilingInterface target) {
    // TODO: This would probably be easier with a lowering config interface
    // method that checks whether a particular level is tiled.
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(target)) {
      if (loweringConfig.hasTilingLevel(opaqueLevel)) {
        targets.insert(target);
      }
    }
  });
  return targets;
}

static LogicalResult applyPaddingLevel(RewriterBase &rewriter,
                                       TilingInterface tilingInterfaceOp,
                                       IREE::GPU::TilingLevel tilingLevel) {
  SmallVector<int64_t> tileSizes =
      getLoweringConfig(tilingInterfaceOp)
          .getStaticTilingLevelSizes(llvm::to_underlying(tilingLevel),
                                     tilingInterfaceOp);

  // Pad the tile sizes with zero.
  int64_t numLoops = tilingInterfaceOp.getLoopIteratorTypes().size();
  if (tileSizes.size() > numLoops) {
    tilingInterfaceOp.emitWarning("tileSizes.size() > numLoops");
    return failure();
  }
  while (tileSizes.size() < numLoops) {
    tileSizes.push_back(0);
  }

  SmallVector<int64_t> padSizes = llvm::map_to_vector(
      tileSizes, [](int64_t tileSize) { return tileSize == 0 ? 1 : tileSize; });

  SmallVector<int64_t> paddingDims =
      llvm::to_vector(llvm::seq<int64_t>(0, numLoops));

  auto options =
      linalg::LinalgPaddingOptions()
          .setPaddingDimensions(paddingDims)
          .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None)
          .setPadToMultipleOf(padSizes);

  if (auto linalgOp =
          dyn_cast<linalg::LinalgOp>(tilingInterfaceOp.getOperation())) {
    linalg::LinalgOp paddedOp;
    SmallVector<Value> newResults;
    SmallVector<tensor::PadOp> padOps;
    if (failed(mlir::iree_compiler::rewriteAsPaddedOp(
            rewriter, linalgOp, options, paddedOp, newResults, padOps))) {
      linalgOp.emitWarning("failed to pad ops");
      return failure();
    }
    rewriter.replaceOp(linalgOp, paddedOp);
  } else {
    tilingInterfaceOp.emitWarning("not a linalg op");
    return failure();
  }

  return success();
}

void GPUApplyPaddingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  llvm::SmallDenseSet<TilingInterface> targetOps =
      getTiledOps(funcOp, tilingLevel);

  IRRewriter rewriter(funcOp);
  for (TilingInterface op : targetOps) {
    // If some op does not get padded, that is fine for now.
    (void)applyPaddingLevel(rewriter, op, tilingLevel);
  }
}

} // namespace mlir::iree_compiler
