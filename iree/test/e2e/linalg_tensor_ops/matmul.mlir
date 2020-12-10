func @tensor() -> tensor<2x2xf32> attributes { iree.module.export } {
  %A = iree.unfoldable_constant dense<[[1.0, 2.0], [4.0, 5.0]]> : tensor<2x2xf32>
  %B = iree.unfoldable_constant dense<[[1.0, 2.0], [5.0, 6.0]]> : tensor<2x2xf32>
  %C = iree.unfoldable_constant dense<1000.0> : tensor<2x2xf32>
  %E = linalg.matmul ins(%A, %B: tensor<2x2xf32>, tensor<2x2xf32>)
                    init(%C: tensor<2x2xf32>) -> tensor<2x2xf32>
  // check.expect_almost_eq_const(%D, dense<[[1011.0, 1014.0],
  //                                         [1029.0, 1038.0]]> : tensor<2x2xf32>)
  //   : tensor<2x2xf32>
  return %E : tensor<2x2xf32>
}

