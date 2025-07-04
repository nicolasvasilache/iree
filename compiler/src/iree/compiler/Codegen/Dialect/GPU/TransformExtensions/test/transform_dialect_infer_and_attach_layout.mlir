// RUN: iree-opt -iree-transform-dialect-interpreter --verify-diagnostics -transform-dialect-drop-schedule -canonicalize -cse --split-input-file %s | FileCheck %s


#matmat_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]

#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-DAG: #[[nested:.*]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [4, 16], element_tile = [4, 1], subgroup_strides = [0, 0], thread_strides = [16, 1]>
// CHECK-DAG: #[[nested1:.*]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [1, 1], outer_tile = [1, 1], thread_tile = [16, 4], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [1, 16]>
//
//     CHECK: func.func @contract
//     CHECK:   iree_vector_ext.to_layout %{{.*}} to layout(#[[nested]]) : vector<16x16xf32>
//     CHECK:   iree_vector_ext.to_layout %{{.*}} to layout(#[[nested]]) : vector<16x16xf32>
//     CHECK:   iree_vector_ext.to_layout %{{.*}} to layout(#[[nested1]]) : vector<16x16xf32>
//     CHECK:   vector.contract {{.*}} {iree.amdgpu.mma = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>}
//     CHECK:   iree_vector_ext.to_layout %{{.*}} to layout(#[[nested]]) : vector<16x16xf32>
func.func @contract(%va: vector<16x16xf32>, %vb: vector<16x16xf32>, %vc: vector<16x16xf32>) -> vector<16x16xf32> {
  %vres = vector.contract #matmat_trait %va, %vb, %vc
    : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
  return %vres : vector<16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %producer = transform.structured.match ops{["vector.contract"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %mma_attr = transform.param.constant #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16> -> !transform.any_param
    %2 = transform.iree.infer_and_attach_vector_contract_layout mma_attr(%mma_attr) to %producer
      : (!transform.any_param, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
