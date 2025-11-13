#include <cuda_fp16.h>

void
tensorcore2d_host(
    uint32_t rows,
    uint32_t shared,
    uint32_t cols,
    half *gA,
    half *gB,
    half *gC
);
