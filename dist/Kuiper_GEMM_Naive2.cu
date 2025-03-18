

#include "Kuiper_GEMM_Naive2.h"

__global__
static void
__hoisted_0(size_t rows, size_t shared, size_t cols, float_t *gA, float_t *gB, float_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    float_t sum = (float_t)0.0f;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[trow * shared + vk] * gB[vk * cols + tcol];
      k = vk + (size_t)1U;
    }
    gC[trow * cols + tcol] = sum;
  }
}

float_t
*Kuiper_GEMM_Naive2_matmul_f32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_0,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    for (uint32_t _i = 0U; _i < rows * cols; ++_i)
      c[_i] = (float_t)0.0f;
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
static void
__hoisted_1(size_t rows, size_t shared, size_t cols, double_t *gA, double_t *gB, double_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    double_t sum = (double_t)0.0l;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[trow * shared + vk] * gB[vk * cols + tcol];
      k = vk + (size_t)1U;
    }
    gC[trow * cols + tcol] = sum;
  }
}

double_t
*Kuiper_GEMM_Naive2_matmul_f64_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_1,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (double_t), rows * cols);
  double_t *c = (double_t *)KRML_HOST_MALLOC(sizeof (double_t) * (rows * cols));
  if (c != NULL)
    for (uint32_t _i = 0U; _i < rows * cols; ++_i)
      c[_i] = (double_t)0.0l;
  MUST(cudaMemcpy(c, gC, (size_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
static void
__hoisted_2(size_t rows, size_t shared, size_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    uint32_t sum = 0U;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[trow * shared + vk] * gB[vk * cols + tcol];
      k = vk + (size_t)1U;
    }
    gC[trow * cols + tcol] = sum;
  }
}

uint32_t
*Kuiper_GEMM_Naive2_matmul_u32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_2,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
static void
__hoisted_3(size_t rows, size_t shared, size_t cols, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    uint64_t sum = 0ULL;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[trow * shared + vk] * gB[vk * cols + tcol];
      k = vk + (size_t)1U;
    }
    gC[trow * cols + tcol] = sum;
  }
}

uint64_t
*Kuiper_GEMM_Naive2_matmul_u64_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_3,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (size_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
static void
__hoisted_4(size_t rows, size_t shared, size_t cols, float_t *gA, float_t *gB, float_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    float_t sum = (float_t)0.0f;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[vk * rows + trow] * gB[tcol * shared + vk];
      k = vk + (size_t)1U;
    }
    gC[tcol * rows + trow] = sum;
  }
}

float_t
*Kuiper_GEMM_Naive2_matmul_f32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_4,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    for (uint32_t _i = 0U; _i < rows * cols; ++_i)
      c[_i] = (float_t)0.0f;
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
static void
__hoisted_5(size_t rows, size_t shared, size_t cols, double_t *gA, double_t *gB, double_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    double_t sum = (double_t)0.0l;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[vk * rows + trow] * gB[tcol * shared + vk];
      k = vk + (size_t)1U;
    }
    gC[tcol * rows + trow] = sum;
  }
}

double_t
*Kuiper_GEMM_Naive2_matmul_f64_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_5,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (double_t), rows * cols);
  double_t *c = (double_t *)KRML_HOST_MALLOC(sizeof (double_t) * (rows * cols));
  if (c != NULL)
    for (uint32_t _i = 0U; _i < rows * cols; ++_i)
      c[_i] = (double_t)0.0l;
  MUST(cudaMemcpy(c, gC, (size_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
static void
__hoisted_6(size_t rows, size_t shared, size_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    uint32_t sum = 0U;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[vk * rows + trow] * gB[tcol * shared + vk];
      k = vk + (size_t)1U;
    }
    gC[tcol * rows + trow] = sum;
  }
}

uint32_t
*Kuiper_GEMM_Naive2_matmul_u32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((size_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_6,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (size_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
static void
__hoisted_7(size_t rows, size_t shared, size_t cols, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    uint64_t sum = 0ULL;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[vk * rows + trow] * gB[tcol * shared + vk];
      k = vk + (size_t)1U;
    }
    gC[tcol * rows + trow] = sum;
  }
}

uint64_t
*Kuiper_GEMM_Naive2_matmul_u64_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((size_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (size_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (size_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_KCALL(__hoisted_7,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (size_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
static void
__hoisted_8(size_t rows, size_t shared, size_t cols, float_t *gA, float_t *gB, float_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    float_t sum = (float_t)0.0f;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[trow * shared + vk] * gB[vk * cols + tcol];
      k = vk + (size_t)1U;
    }
    gC[trow * cols + tcol] = sum;
  }
}

void
Kuiper_GEMM_Naive2_g_matmul_f32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_KCALL(__hoisted_8,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
static void
__hoisted_9(size_t rows, size_t shared, size_t cols, double_t *gA, double_t *gB, double_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    double_t sum = (double_t)0.0l;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[trow * shared + vk] * gB[vk * cols + tcol];
      k = vk + (size_t)1U;
    }
    gC[trow * cols + tcol] = sum;
  }
}

void
Kuiper_GEMM_Naive2_g_matmul_f64_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_KCALL(__hoisted_9,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
static void
__hoisted_10(size_t rows, size_t shared, size_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    uint32_t sum = 0U;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[trow * shared + vk] * gB[vk * cols + tcol];
      k = vk + (size_t)1U;
    }
    gC[trow * cols + tcol] = sum;
  }
}

void
Kuiper_GEMM_Naive2_g_matmul_u32_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_KCALL(__hoisted_10,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
static void
__hoisted_11(size_t rows, size_t shared, size_t cols, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    uint64_t sum = 0ULL;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[trow * shared + vk] * gB[vk * cols + tcol];
      k = vk + (size_t)1U;
    }
    gC[trow * cols + tcol] = sum;
  }
}

void
Kuiper_GEMM_Naive2_g_matmul_u64_rrr(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_KCALL(__hoisted_11,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
static void
__hoisted_12(size_t rows, size_t shared, size_t cols, float_t *gA, float_t *gB, float_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    float_t sum = (float_t)0.0f;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[vk * rows + trow] * gB[tcol * shared + vk];
      k = vk + (size_t)1U;
    }
    gC[tcol * rows + trow] = sum;
  }
}

void
Kuiper_GEMM_Naive2_g_matmul_f32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_KCALL(__hoisted_12,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
static void
__hoisted_13(size_t rows, size_t shared, size_t cols, double_t *gA, double_t *gB, double_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    double_t sum = (double_t)0.0l;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[vk * rows + trow] * gB[tcol * shared + vk];
      k = vk + (size_t)1U;
    }
    gC[tcol * rows + trow] = sum;
  }
}

void
Kuiper_GEMM_Naive2_g_matmul_f64_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_KCALL(__hoisted_13,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
static void
__hoisted_14(size_t rows, size_t shared, size_t cols, uint32_t *gA, uint32_t *gB, uint32_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    uint32_t sum = 0U;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[vk * rows + trow] * gB[tcol * shared + vk];
      k = vk + (size_t)1U;
    }
    gC[tcol * rows + trow] = sum;
  }
}

void
Kuiper_GEMM_Naive2_g_matmul_u32_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_KCALL(__hoisted_14,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

__global__
static void
__hoisted_15(size_t rows, size_t shared, size_t cols, uint64_t *gA, uint64_t *gB, uint64_t *gC)
{
  if (blockIdx.x * (size_t)1024U + threadIdx.x < rows * cols)
  {
    size_t trow = (blockIdx.x * (size_t)1024U + threadIdx.x) / cols;
    size_t tcol = (blockIdx.x * (size_t)1024U + threadIdx.x) % cols;
    size_t k = (size_t)0U;
    uint64_t sum = 0ULL;
    while (k < shared)
    {
      size_t vk = k;
      sum += gA[vk * rows + trow] * gB[tcol * shared + vk];
      k = vk + (size_t)1U;
    }
    gC[tcol * rows + trow] = sum;
  }
}

void
Kuiper_GEMM_Naive2_g_matmul_u64_ccc(
  size_t rows,
  size_t shared,
  size_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_KCALL(__hoisted_15,
    (rows * cols + (size_t)1024U - (size_t)1U) / (size_t)1024U,
    (size_t)1024U,
    (size_t)1U,
    (size_t)0U,
    rows,
    shared,
    cols,
    gA,
    gB,
    gC);
  cudaDeviceSynchronize();
}

