

#include "Kuiper_Example_TensorCore.h"

inline

__device__
void Kuiper_Example_TensorCore_test(half_t *m1, half_t *m2, half_t *m3)
{
  auto fa = KPR_FRAGMENT_INIT(half, wmma::matrix_a, 16, 16, 16, wmma::row_major);
  auto fb = KPR_FRAGMENT_INIT(half, wmma::matrix_b, 16, 16, 16, wmma::row_major);
  auto fc = KPR_FRAGMENT_INIT_C(half, wmma::accumulator, 16, 16, 16);
  wmma::load_matrix_sync(fa, m1, 16);
  wmma::load_matrix_sync(fb, m2, 16);
  wmma::fill_fragment(fc, (half_t)0.0f);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(m3, fc, 16, wmma::mem_row_major);
}

