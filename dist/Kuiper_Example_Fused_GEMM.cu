
#include "Kuiper_Example_Fused_GEMM.h"

__global__
/**
  hoisted when extracting gemm_sqrt_fused
*/
static void
__hoisted_gemm_sqrt_fused_0(uint32_t m, uint32_t n, uint32_t k, half *gA,
                            half *gB, half *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float sum = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t vk = k1;
            float __anf2 = sum;
            half __anf1 = gA[trow * k + vk];
            half __anf0 = gB[vk * n + tcol];
            sum = __anf2 + sqrtf(__half2float(__anf1)) * __half2float(__anf0);
        }
        gC[trow * n + tcol] = __float2half_rn(sum);
    }
}

/**
C <- sqrt(A) @ B, with A, B, C in fp16 and the accumulation done in fp32.
*/
void
Kuiper_Example_Fused_GEMM_gemm_sqrt_fused(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          half *gA, half *gB, half *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gemm_sqrt_fused_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U),
              1024U, 0U, s, m, n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
