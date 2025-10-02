

#include "Kuiper_Example_TensorCore.h"

inline

__device__
void Kuiper_Example_TensorCore_test(half_t *m1, half_t *m2, half_t *m3)
{
  auto& fa = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto& fb = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto& fc = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16));
  wmma::load_matrix_sync(fa, kpr_offset(m1, 0U), 16U);
  wmma::load_matrix_sync(fb, kpr_offset(m2, 0U), 16U);
  wmma::fill_fragment(fc, 0.0f);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(kpr_offset(m3, 0U), fc, 16U, wmma::mem_row_major);
}

inline

__device__
void Kuiper_Example_TensorCore_test2(half_t *m1, half_t *m2, half_t *m3)
{
  auto& fa = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_a, 16, 16, 16, wmma::row_major));
  auto& fb = KPR_INIT(KPR_FRAGMENT_TYPE(half, wmma::matrix_b, 16, 16, 16, wmma::row_major));
  auto& fc = KPR_INIT(KPR_FRAGMENT_TYPE_C(half, wmma::accumulator, 16, 16, 16));
  half_t *t2 = m2;
  half_t *t3 = m3;
  wmma::load_matrix_sync(fa, kpr_offset(m1, 784U), 48U);
  wmma::load_matrix_sync(fb, kpr_offset(t2, 784U), 48U);
  wmma::fill_fragment(fc, 0.0f);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(kpr_offset(t3, 784U), fc, 48U, wmma::mem_row_major);
}

