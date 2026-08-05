
#include "KWitness7.h"

__global__
/**
  hoisted when extracting matmul_f32
*/
static void
__hoisted_matmul_f32_0(uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB,
                       float *gC)
{
    uint32_t g = 0U;
    for (; g < n * m; g++) {
        uint32_t g0 = g;
        uint32_t row = g0 / m;
        uint32_t col = g0 % m;
        uint32_t idx = 0U;
        float s1 = 0.0f;
        for (; idx < k; idx++) {
            uint32_t i0 = idx;
            s1 += gA[col * k + i0] * gB[i0 * n + row];
        }
        gC[row * m + col] = s1;
    }
}

void KWitness7_matmul_f32(uint32_t m, uint32_t n, uint32_t k, float *gA,
                          float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_matmul_f32_0, 1U, 1U, 0U, s, m, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
