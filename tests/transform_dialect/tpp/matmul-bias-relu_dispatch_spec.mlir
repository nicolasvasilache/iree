transform.structured.canonicalized_sequence failures(propagate){
^bb1(%func: !pdl.operation):
  // This has to be done a the Flow level to:
  //   1. allow linalgx.relayout (i.e. tensor.pack/unpack) swap patterns 
  //      across the whole program.
  //   2. generally avoid huge allocs within dispatch regions.
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %func
  %generic = transform.structured.pack %matmul { blocking_factors = [32, 32, 32] }
 
  %func_2 = transform.iree.apply_patterns %func { swapping_relayout_patterns }
  %relayouts = transform.structured.match ops{["linalgx.pack", "linalgx.unpack"]} in %func_2
  %generics = transform.structured.match ops{["linalg.generic"]} in %func_2

  // For now we need to be explicit and split out everything relayout into its
  // own region to avoid gynormous allocas.
  %relayout_regions = transform.iree.wrap_in_dispatch_region %relayouts
  transform.iree.region_to_workgroups %relayout_regions

  // We can be a little smarter with generic and force the fusion.
  // %packed_matmul, %packed_bias, %packed_relu = transform.split_handles %generics in [3]
  // %packed_relu_region = transform.iree.wrap_in_dispatch_region %packed_relu
  // transform.iree.move_preceding_op_into_dispatch_region %packed_bias into %packed_relu_region
  // transform.iree.move_preceding_op_into_dispatch_region %packed_matmul into %packed_relu_region
  // transform.iree.region_to_workgroups %packed_relu_region
  //
  // Or just IREE handle this case, since it knows how.
}
