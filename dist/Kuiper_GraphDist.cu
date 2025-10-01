

#include "Kuiper_GraphDist.h"

bool Kuiper_GraphDist_uu___is_D(uint16_t projectee)
{
  KRML_MAYBE_UNUSED_VAR(projectee);
  return true;
}

__device__
static uint16_t add_(uint16_t x, uint16_t y)
{
  if (x == 0U || !(y == 0U) && y < x)
    return y;
  else
    return x;
}

__device__
static uint16_t mult(uint16_t x, uint16_t y)
{
  if (x == 0U || y == 0U)
    return 0U;
  else
    return (uint32_t)x + (uint32_t)y;
}

__global__
/**
  hoisted when extracting matmul_dist_gpu
*/
static void __hoisted_0(uint32_t size, uint16_t *a, uint16_t *b)
{
  if ((uint32_t)1024U * blockIdx.x + threadIdx.x < size * size)
  {
    uint32_t k = (uint32_t)0U;
    uint16_t sum = 0U;
    while (k < size)
    {
      uint16_t vsum = sum;
      sum =
        add_(vsum,
          mult(a[((uint32_t)1024U * blockIdx.x + threadIdx.x) / size * size + k],
            a[k * size + ((uint32_t)1024U * blockIdx.x + threadIdx.x) % size]));
      k += (uint32_t)1U;
    }
    uint16_t s = sum;
    b[(uint32_t)1024U * blockIdx.x + threadIdx.x] =
      add_(b[(uint32_t)1024U * blockIdx.x + threadIdx.x],
        s);
  }
}

void Kuiper_GraphDist_matmul_dist_gpu(uint32_t size, uint16_t *a, uint16_t *b)
{
  KPR_KCALL(__hoisted_0,
    (size * size + (uint32_t)1023U) / (uint32_t)1024U,
    (uint32_t)1024U,
    (uint32_t)0U,
    size,
    a,
    b);
  cudaDeviceSynchronize();
}

