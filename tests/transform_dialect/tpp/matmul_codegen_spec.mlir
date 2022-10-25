transform.sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):
    // This has to be done at the Flow level to:
    //   1. allow linalgx..pack/unpack) swap patterns across the whole program.
    //   2. generally avoid huge allocs within dispatch regions.
    // %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    // transform.structured.packing %matmul { blocking_factors = [16, 32, 64] }

    // Tile to one level of outer-parallelism, just because we can. 
    %generic = transform.structured.match ops{["linalg.generic"]} in %variant_op
    // transform.structured.tile_to_foreach_thread_op %generic tile_sizes [1]
    %variant_op_2 = transform.iree.bufferize %variant_op
    %generic_2 = transform.structured.match ops{["linalg.generic"]} in %variant_op_2
    transform.structured.map_to_brgemm %generic_2

    %func = transform.structured.match ops{["func.func"]} in %variant_op_2
    // %func_2 = transform.iree.foreach_thread_to_workgroup %func
    
    // TODO: xsmm.ternary.dispatch currently does not hoist, investigate.
    %func_3 = transform.iree.apply_patterns %func
      { linalg_to_tpp, tpp_to_xsmm, xsmm_to_func }
    transform.iree.apply_patterns %func_3 { simplify_memref_metadata }
}
