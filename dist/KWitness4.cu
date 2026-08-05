
#include "KWitness4.h"

__global__
/**
  hoisted when extracting matmul_f32
*/
static void
__hoisted_matmul_f32_0(uint32_t m, uint32_t n, uint32_t k, float *gA, float *gB,
                       float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float acc = 0.0f;
        float c = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            float yc = gA[trow * k + __anf0] * gB[__anf0 * n + tcol] - c;
            float t = acc + yc;
            c = t - acc - yc;
            acc = t;
        }
        gC[trow * n + tcol] = acc;
    }
}

void KWitness4_matmul_f32(uint32_t m, uint32_t n, uint32_t k, float *gA,
                          float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_matmul_f32_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, s, m, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
