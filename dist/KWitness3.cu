
#include "KWitness3.h"

__global__
/**
  hoisted when extracting matmul_f32
*/
static void
__hoisted_matmul_f32_0(uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB,
                       float *gC)
{
    uint32_t g = 0U;
    for (; g < m * n; g++) {
        uint32_t g0 = g;
        uint32_t row = g0 / n;
        uint32_t col = g0 % n;
        uint32_t ri = row;
        uint32_t ci = col;
        uint32_t nt = k / 16U + (uint32_t) (k % 16U != 0U);
        uint32_t t = 0U;
        float sum = 0.0f;
        for (; t < nt; t++) {
            uint32_t k0 = t * 16U;
            uint32_t hi = k - k0 < 16U ? k : k0 + 16U;
            float acc = 0.0f;
            uint32_t jj = k0;
            for (; jj < hi; jj++) {
                uint32_t jc = jj;
                acc += gA[ri * k + jc] * gB[jc * n + ci];
            }
            sum += acc;
        }
        gC[row * n + col] = sum;
    }
}

void KWitness3_matmul_f32(uint32_t m, uint32_t n, uint32_t k, float *gA,
                          float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_matmul_f32_0, 1U, 1U, 0U, s, m, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
