
#include "Klas_Reduce.h"

__global__
/**
  hoisted when extracting mean
*/
static void __hoisted_mean_0(uint32_t n, float *a, float *out)
{
    float *sa1 = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < n; idx += 1024U)
        acc += a[idx];
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < 1024U; n1++) {
        uint32_t __anf01 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) -
                 1U) == 0U)
                sa1[threadIdx.x] += sa1[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa1;
}

float Klas_Reduce_mean(uint32_t n, float *a)
{
    float *out = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute
         (__hoisted_mean_0, cudaFuncAttributeMaxDynamicSharedMemorySize,
          4096U));
    KPR_KCALL(__hoisted_mean_0, 1U, 1024U, 4096U, n, a, out);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout / (float)(int64_t) (uint32_t) n;
}
