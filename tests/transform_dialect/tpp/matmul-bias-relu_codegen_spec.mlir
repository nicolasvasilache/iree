transform.structured.canonicalized_sequence failures(propagate){
  ^bb0(%variant: !pdl.operation):
    // Tile to one level of outer-parallelism, just because we can. 
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant
    transform.structured.tile_to_foreach_thread_op %matmul tile_sizes [16]

    %variant_2 = transform.iree.bufferize %variant
    %generic = transform.structured.match ops{["linalg.generic"]} in %variant_2
    transform.structured.map_to_brgemm %generic

    %func = transform.structured.match ops{["func.func"]} in %variant_2
    %func_2 = transform.iree.foreach_thread_to_workgroup %func
    // TODO: xsmm.ternary.dispatch currently does not hoist, investigate.
    %func_3 = transform.iree.apply_patterns %func_2 
      { linalg_to_tpp, tpp_to_xsmm, xsmm_to_func }
    transform.iree.apply_patterns %func_3 { simplify_memref_metadata }
}
