
#include "Kuiper_GraphDist.h"

bool Kuiper_GraphDist_uu___is_D(uint16_t projectee)
{
    KRML_MAYBE_UNUSED_VAR(projectee);
    return true;
}

__device__ static uint16_t add_(uint16_t x, uint16_t y)
{
    if (x == 0U || y != 0U && y < x)
        return y;
    else
        return x;
}

__device__ static uint16_t mult(uint16_t x, uint16_t y)
{
    if (x == 0U || y == 0U)
        return 0U;
    else
        return (uint32_t) x + (uint32_t) y;
}

__global__
/**
  hoisted when extracting matmul_dist_gpu
*/
static void __hoisted_matmul_dist_gpu_0(uint32_t size, uint16_t *a, uint16_t *b)
{
    if (1024U * blockIdx.x + threadIdx.x < size * size) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / size;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % size;
        uint32_t k = 0U;
        uint16_t sum = 0U;
        for (; k < size; k++) {
            uint16_t __anf4 = sum;
            sum = add_(__anf4, mult(a[trow * size + k], a[k * size + tcol]));
        }
        uint16_t s = sum;
        b[trow * size + tcol] = add_(b[trow * size + tcol], s);
    }
}

void Kuiper_GraphDist_matmul_dist_gpu(uint32_t size, uint16_t *a, uint16_t *b)
{
    KPR_KCALL(__hoisted_matmul_dist_gpu_0,
              size * size / 1024U + (uint32_t) (size * size % 1024U != 0U),
              1024U, 0U, size, a, b);
    MUST(cudaDeviceSynchronize());
}
