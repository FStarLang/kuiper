
#include "Kuiper_OffsetMemcpy.h"

uint64_t Kuiper_OffsetMemcpy_main(void)
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
    uint64_t *ga = (uint64_t *) KPR_GPU_ALLOC(8U, 8U);
    uint64_t *zeros = (uint64_t *) KRML_HOST_CALLOC(8U, sizeof(uint64_t));
    MUST(cudaMemcpy(ga, zeros, 64U, cudaMemcpyHostToDevice));
    KRML_HOST_FREE(zeros);
    MUST(cudaMemcpy(ga + 2U, src + 1U, 24U, cudaMemcpyHostToDevice));
    KRML_HOST_FREE(src);
    uint64_t *dst = (uint64_t *) KRML_HOST_CALLOC(8U, sizeof(uint64_t));
    MUST(cudaMemcpy(dst + 3U, ga + 2U, 24U, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
    uint64_t r0 = dst[3U];
    uint64_t r1 = dst[4U];
    uint64_t r2 = dst[5U];
    KRML_HOST_FREE(dst);
    return r0 + r1 + r2;
}
