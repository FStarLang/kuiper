

#include "Kuiper_Example_TensorCore.h"

inline

__device__
void Kuiper_Example_TensorCore_test(half_t *m1, half_t *m2, half_t *m3)
{
  auto
  fa =
    KPR_FRAGMENT_INIT(half,
      wmma::matrix_a,
      (size_t)16U,
      (size_t)16U,
      (size_t)16U,
      wmma::row_major);
  auto
  fb =
    KPR_FRAGMENT_INIT(half,
      wmma::matrix_b,
      (size_t)16U,
      (size_t)16U,
      (size_t)16U,
      wmma::row_major);
  auto fc = KPR_FRAGMENT_INIT_C(half, wmma::accumulator, (size_t)16U, (size_t)16U, (size_t)16U);
  wmma::load_matrix_sync(fa, m1, (size_t)16U);
  wmma::load_matrix_sync(fb, m2, (size_t)16U);
  wmma::fill_fragment(fc, (half_t)0.0f);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(m3, fc, (size_t)16U, wmma::mem_row_major);
}

