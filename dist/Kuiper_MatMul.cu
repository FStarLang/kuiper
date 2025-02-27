

#include "Kuiper_MatMul.h"

__global__

static void kernel(size_t shared, size_t columns, uint64_t *ga1, uint64_t *ga2, uint64_t *r)
{
  size_t tid = blockIdx_x();
  size_t trow = tid / columns;
  size_t tcol = tid % columns;
  size_t i = (size_t)0U;
  uint64_t sum = 0ULL;
  while (i < shared)
  {
    size_t v = i;
    sum += ga1[trow * shared + v] * ga2[v * columns + tcol];
    i = v + (size_t)1U;
  }
  r[tid] = sum;
}

uint64_t
*Kuiper_MatMul_main(size_t rows, size_t shared, size_t columns, uint64_t *a, uint64_t *b)
{
  size_t size = rows * columns;
  KRML_CHECK_SIZE(sizeof (uint64_t), size);
  uint64_t *ar = (uint64_t *)KRML_HOST_CALLOC(size, sizeof (uint64_t));
  size_t rs = rows * shared;
  size_t sc = shared * columns;
  uint64_t *ga = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * rs);
  uint64_t *gb = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * sc);
  MUST(cudaMemcpy(ga, a, (size_t)8U * rs, cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gb, b, (size_t)8U * sc, cudaMemcpyHostToDevice));
  uint64_t *gr = (uint64_t *)KPR_GPU_ALLOC((size_t)8U * size);
  KPR_KCALL(kernel, size, 1U, shared, columns, ga, gb, gr);
  MUST(cudaMemcpy(ar, gr, (size_t)8U * size, cudaMemcpyDeviceToHost));
  MUST(cudaFree(ga));
  MUST(cudaFree(gb));
  MUST(cudaFree(gr));
  return ar;
}

