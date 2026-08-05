
#include "KWitness8.h"

__global__
/**
  hoisted when extracting matmul_f32
*/
static void
__hoisted_matmul_f32_0(uint32_t n,
                       uint32_t k,
                       float alpha, float beta, float *gA, float *gB, float *gC)
{
    uint32_t trow = blockIdx.x / n;
    uint32_t tcol = blockIdx.x % n;
    uint32_t k1 = 0U;
    float sum = 0.0f;
    for (; k1 < k; k1++) {
        uint32_t vk = k1;
        sum += gA[trow * k + vk] * gB[vk * n + tcol];
    }
    float s1 = sum;
    gC[trow * n + tcol] = alpha * s1 + beta * gC[trow * n + tcol];
}

void
KWitness8_matmul_f32(uint32_t m,
                     uint32_t n,
                     uint32_t k,
                     float alpha, float beta, float *gA, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_matmul_f32_0, m * n, 1U, 0U, s, n, k, alpha, beta, gA,
              gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
