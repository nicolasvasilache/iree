// RUN: iree-opt %s

// transform.sequence failures(propagate) {
//   transform.do_this
//   transform.sequence failures(suppress) {
//     transform.do_this_too
//     transform.emit_silenceable_failure
//     transform.dont_do_this
//   }
//   transform.keep_doing
// }

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %generic = transform.structured.match ops{["linalg.generic"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  // Step 1. Tile to foreach_thread with tile_sizes [1, 1] and fuse fill.
  // ========================================================================
  %foreach_thread, %tiled_generic =
    transform.iree.tile_to_foreach_thread_and_workgroup_count_region %generic tile_sizes [1, 1]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.structured.fuse_into_containing_op %fill into %foreach_thread

  // Step 2. Tile outer loops sequentially by 1 so we get to loops around a large wmma-able op.
  //         Also fuse fill.
  // =========================================================================================
  %tiled_generic_2, %loops:5 = transform.structured.fuse %tiled_generic 
    {tile_sizes = [1, 1, 16, 16, 16], tile_interchange = [0, 1]}
  // TODO: fix types
  //   : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)


  // Step 2. Rank-reduce and vectorize.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %func_2 = transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector }
  %func_3 = transform.structured.vectorize %func_2

  // Step 2. Bufferize and drop HAL decriptor from memref ops.
  // =========================================================
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %variant_op_3 = transform.iree.bufferize %variant_op_2
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func

  // Step 3. Post-bufferization mapping workgroup.
  // =========================================================
  // %memref_func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  // transform.iree.foreach_thread_to_workgroup %memref_func_2



  // %func_x = transform.structured.match ops{["func.func"]} in %variant_op_3
  //   : (!pdl.operation) -> !pdl.operation
  // %func_x_2 = transform.iree.apply_patterns %func_x {  unroll_vectors_gpu_mma }
  // transform.print %func_x_2: !pdl.operation

}
