
#include "Kuiper_Example_TensorCore.h"

inline
    __device__
    void Kuiper_Example_TensorCore_test(half_t *m1, half_t *m2, half_t *m3)
{
    auto & fa =
        KPR_INIT(kpr_fragment
                 (wmma::matrix_a, 16, 16, 16, half, wmma::row_major));
    auto & fb =
        KPR_INIT(kpr_fragment
                 (wmma::matrix_b, 16, 16, 16, half, wmma::row_major));
    auto & fc = KPR_INIT(kpr_fragment(wmma::accumulator, 16, 16, 16, half));
    wmma::load_matrix_sync(fa, m1, 16U);
    wmma::load_matrix_sync(fb, m2, 16U);
    wmma::fill_fragment(fc, 0.0f);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(m3, fc, 16U, wmma::mem_row_major);
}

inline
    __device__
    void Kuiper_Example_TensorCore_test2(half_t *m1, half_t *m2, half_t *m3)
{
    auto & fa =
        KPR_INIT(kpr_fragment
                 (wmma::matrix_a, 16, 16, 16, half, wmma::row_major));
    auto & fb =
        KPR_INIT(kpr_fragment
                 (wmma::matrix_b, 16, 16, 16, half, wmma::row_major));
    auto & fc = KPR_INIT(kpr_fragment(wmma::accumulator, 16, 16, 16, half));
    half_t *t2 = m2;
    half_t *t3 = m3;
    wmma::load_matrix_sync(fa, m1 + 784U, 48U);
    wmma::load_matrix_sync(fb, t2 + 784U, 48U);
    wmma::fill_fragment(fc, 0.0f);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(t3 + 784U, fc, 48U, wmma::mem_row_major);
}
