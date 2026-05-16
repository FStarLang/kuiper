
#include "Kuiper_Example_OffsetMemcpy.h"

uint64_t Kuiper_Example_OffsetMemcpy_main(void)
{
    uint64_t *src = (uint64_t *) KRML_HOST_CALLOC(8U, sizeof(uint64_t));
    *src = 10ULL;
    src[1U] = 20ULL;
    src[2U] = 30ULL;
    src[3U] = 40ULL;
    src[4U] = 50ULL;
    src[5U] = 60ULL;
    src[6U] = 70ULL;
    src[7U] = 80ULL;
    uint64_t *ga = (uint64_t *) KPR_GPU_ALLOC(sizeof(uint64_t), 8U);
    uint64_t *zeros = (uint64_t *) KRML_HOST_CALLOC(8U, sizeof(uint64_t));
    MUST(cudaMemcpy
         (ga, zeros, (uint32_t) sizeof(uint64_t) * 8U, cudaMemcpyHostToDevice));
    KRML_HOST_FREE(zeros);
    MUST(cudaMemcpy
         (ga + 2U, src + 1U, (uint32_t) sizeof(uint64_t) * 3U,
          cudaMemcpyHostToDevice));
    KRML_HOST_FREE(src);
    uint64_t *dst = (uint64_t *) KRML_HOST_CALLOC(8U, sizeof(uint64_t));
    MUST(cudaMemcpy
         (dst + 3U, ga + 2U, (uint32_t) sizeof(uint64_t) * 3U,
          cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    uint64_t r0 = dst[3U];
    uint64_t r1 = dst[4U];
    uint64_t r2 = dst[5U];
    KRML_HOST_FREE(dst);
    return r0 + r1 + r2;
}
