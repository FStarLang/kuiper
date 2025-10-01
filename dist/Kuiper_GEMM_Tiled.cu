

#include "Kuiper_GEMM_Tiled.h"

__global__
/**
  hoisted when extracting matmul_f32_rrr
*/
static void
__hoisted_0(
  uint32_t tile,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared + vbk * tile + k] *
          tB[(vbk * tile + k) * cols + blockIdx.x % mcols * tile + threadIdx.x % tile];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
    blockIdx.x % mcols * tile + threadIdx.x % tile]
  = sum;
}

float_t
*Kuiper_GEMM_Tiled_matmul_f32_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_0,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    shared,
    cols,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (float_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f64_rrr
*/
static void
__hoisted_1(
  uint32_t tile,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared + vbk * tile + k] *
          tB[(vbk * tile + k) * cols + blockIdx.x % mcols * tile + threadIdx.x % tile];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
    blockIdx.x % mcols * tile + threadIdx.x % tile]
  = sum;
}

double_t
*Kuiper_GEMM_Tiled_matmul_f64_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_1,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    shared,
    cols,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (double_t), rows * cols);
  double_t *c = (double_t *)KRML_HOST_MALLOC(sizeof (double_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (double_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u32_rrr
*/
static void
__hoisted_2(
  uint32_t tile,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared + vbk * tile + k] *
          tB[(vbk * tile + k) * cols + blockIdx.x % mcols * tile + threadIdx.x % tile];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
    blockIdx.x % mcols * tile + threadIdx.x % tile]
  = sum;
}

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_2,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    shared,
    cols,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u64_rrr
*/
static void
__hoisted_3(
  uint32_t tile,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared + vbk * tile + k] *
          tB[(vbk * tile + k) * cols + blockIdx.x % mcols * tile + threadIdx.x % tile];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
    blockIdx.x % mcols * tile + threadIdx.x % tile]
  = sum;
}

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_3,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    shared,
    cols,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f32_ccc
*/
static void
__hoisted_4(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile + threadIdx.x / tile] *
          tB[(blockIdx.x % mcols * tile + threadIdx.x % tile) * shared + vbk * tile + k];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
    blockIdx.x / mcols * tile + threadIdx.x / tile]
  = sum;
}

float_t
*Kuiper_GEMM_Tiled_matmul_f32_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_4,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    rows,
    shared,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (float_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f64_ccc
*/
static void
__hoisted_5(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile + threadIdx.x / tile] *
          tB[(blockIdx.x % mcols * tile + threadIdx.x % tile) * shared + vbk * tile + k];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
    blockIdx.x / mcols * tile + threadIdx.x / tile]
  = sum;
}

double_t
*Kuiper_GEMM_Tiled_matmul_f64_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_5,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    rows,
    shared,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (double_t), rows * cols);
  double_t *c = (double_t *)KRML_HOST_MALLOC(sizeof (double_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (double_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u32_ccc
*/
static void
__hoisted_6(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile + threadIdx.x / tile] *
          tB[(blockIdx.x % mcols * tile + threadIdx.x % tile) * shared + vbk * tile + k];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
    blockIdx.x / mcols * tile + threadIdx.x / tile]
  = sum;
}

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_6,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    rows,
    shared,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u64_ccc
*/
static void
__hoisted_7(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile + threadIdx.x / tile] *
          tB[(blockIdx.x % mcols * tile + threadIdx.x % tile) * shared + vbk * tile + k];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
    blockIdx.x / mcols * tile + threadIdx.x / tile]
  = sum;
}

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_7,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    rows,
    shared,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f32_tile32_rrr
*/
static void
__hoisted_8(
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * shared +
          vbk * (uint32_t)32U + k]
        *
          tB[(vbk * (uint32_t)32U + k) * cols +
            blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * cols +
    blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U]
  = sum;
}

float_t
*Kuiper_GEMM_Tiled_matmul_f32_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_8,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (float_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f64_tile32_rrr
*/
static void
__hoisted_9(
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * shared +
          vbk * (uint32_t)32U + k]
        *
          tB[(vbk * (uint32_t)32U + k) * cols +
            blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * cols +
    blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U]
  = sum;
}

double_t
*Kuiper_GEMM_Tiled_matmul_f64_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_9,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (double_t), rows * cols);
  double_t *c = (double_t *)KRML_HOST_MALLOC(sizeof (double_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (double_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u32_tile32_rrr
*/
static void
__hoisted_10(
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * shared +
          vbk * (uint32_t)32U + k]
        *
          tB[(vbk * (uint32_t)32U + k) * cols +
            blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * cols +
    blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U]
  = sum;
}

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_10,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u64_tile32_rrr
*/
static void
__hoisted_11(
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * shared +
          vbk * (uint32_t)32U + k]
        *
          tB[(vbk * (uint32_t)32U + k) * cols +
            blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * cols +
    blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U]
  = sum;
}

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_11,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f32_tile32_ccc
*/
static void
__hoisted_12(
  uint32_t rows,
  uint32_t shared,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)32U + k) * rows +
          blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * shared +
            vbk * (uint32_t)32U + k];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * rows +
    blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
  = sum;
}

float_t
*Kuiper_GEMM_Tiled_matmul_f32_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_12,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (float_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f64_tile32_ccc
*/
static void
__hoisted_13(
  uint32_t rows,
  uint32_t shared,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)32U + k) * rows +
          blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * shared +
            vbk * (uint32_t)32U + k];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * rows +
    blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
  = sum;
}

double_t
*Kuiper_GEMM_Tiled_matmul_f64_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_13,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (double_t), rows * cols);
  double_t *c = (double_t *)KRML_HOST_MALLOC(sizeof (double_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (double_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u32_tile32_ccc
*/
static void
__hoisted_14(
  uint32_t rows,
  uint32_t shared,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)32U + k) * rows +
          blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * shared +
            vbk * (uint32_t)32U + k];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * rows +
    blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
  = sum;
}

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_14,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u64_tile32_ccc
*/
static void
__hoisted_15(
  uint32_t rows,
  uint32_t shared,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)32U + k) * rows +
          blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * shared +
            vbk * (uint32_t)32U + k];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * rows +
    blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
  = sum;
}

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_15,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f32_tile16_rrr
*/
static void
__hoisted_16(
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * shared +
          vbk * (uint32_t)16U + k]
        *
          tB[(vbk * (uint32_t)16U + k) * cols +
            blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * cols +
    blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U]
  = sum;
}

float_t
*Kuiper_GEMM_Tiled_matmul_f32_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_16,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (float_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f64_tile16_rrr
*/
static void
__hoisted_17(
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * shared +
          vbk * (uint32_t)16U + k]
        *
          tB[(vbk * (uint32_t)16U + k) * cols +
            blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * cols +
    blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U]
  = sum;
}

double_t
*Kuiper_GEMM_Tiled_matmul_f64_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_17,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (double_t), rows * cols);
  double_t *c = (double_t *)KRML_HOST_MALLOC(sizeof (double_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (double_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u32_tile16_rrr
*/
static void
__hoisted_18(
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * shared +
          vbk * (uint32_t)16U + k]
        *
          tB[(vbk * (uint32_t)16U + k) * cols +
            blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * cols +
    blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U]
  = sum;
}

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_18,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u64_tile16_rrr
*/
static void
__hoisted_19(
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * shared +
          vbk * (uint32_t)16U + k]
        *
          tB[(vbk * (uint32_t)16U + k) * cols +
            blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * cols +
    blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U]
  = sum;
}

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_19,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f32_tile16_ccc
*/
static void
__hoisted_20(
  uint32_t rows,
  uint32_t shared,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)16U + k) * rows +
          blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * shared +
            vbk * (uint32_t)16U + k];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * rows +
    blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
  = sum;
}

float_t
*Kuiper_GEMM_Tiled_matmul_f32_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *a,
  float_t *b
)
{
  float_t *gA = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  float_t *gB = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  float_t *gC = (float_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_20,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (float_t), rows * cols);
  float_t *c = (float_t *)KRML_HOST_MALLOC(sizeof (float_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (float_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_f64_tile16_ccc
*/
static void
__hoisted_21(
  uint32_t rows,
  uint32_t shared,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)16U + k) * rows +
          blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * shared +
            vbk * (uint32_t)16U + k];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * rows +
    blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
  = sum;
}

double_t
*Kuiper_GEMM_Tiled_matmul_f64_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *a,
  double_t *b
)
{
  double_t *gA = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  double_t *gB = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  double_t *gC = (double_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_21,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (double_t), rows * cols);
  double_t *c = (double_t *)KRML_HOST_MALLOC(sizeof (double_t) * (rows * cols));
  if (c != NULL)
    memset(c, 0U, rows * cols * sizeof (double_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u32_tile16_ccc
*/
static void
__hoisted_22(
  uint32_t rows,
  uint32_t shared,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)16U + k) * rows +
          blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * shared +
            vbk * (uint32_t)16U + k];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * rows +
    blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
  = sum;
}

uint32_t
*Kuiper_GEMM_Tiled_matmul_u32_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *a,
  uint32_t *b
)
{
  uint32_t *gA = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * shared);
  uint32_t *gB = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, shared * cols);
  uint32_t *gC = (uint32_t *)KPR_GPU_ALLOC((uint32_t)4U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)4U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)4U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_22,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint32_t), rows * cols);
  uint32_t *c = (uint32_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint32_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)4U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting matmul_u64_tile16_ccc
*/
static void
__hoisted_23(
  uint32_t rows,
  uint32_t shared,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)16U + k) * rows +
          blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * shared +
            vbk * (uint32_t)16U + k];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * rows +
    blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
  = sum;
}

uint64_t
*Kuiper_GEMM_Tiled_matmul_u64_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *a,
  uint64_t *b
)
{
  uint64_t *gA = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * shared);
  uint64_t *gB = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, shared * cols);
  uint64_t *gC = (uint64_t *)KPR_GPU_ALLOC((uint32_t)8U, rows * cols);
  MUST(cudaMemcpy(gA, a, (uint32_t)8U * (rows * shared), cudaMemcpyHostToDevice));
  MUST(cudaMemcpy(gB, b, (uint32_t)8U * (shared * cols), cudaMemcpyHostToDevice));
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_23,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
  KRML_CHECK_SIZE(sizeof (uint64_t), rows * cols);
  uint64_t *c = (uint64_t *)KRML_HOST_CALLOC(rows * cols, sizeof (uint64_t));
  MUST(cudaMemcpy(c, gC, (uint32_t)8U * (rows * cols), cudaMemcpyDeviceToHost));
  MUST(cudaFree(gA));
  MUST(cudaFree(gB));
  MUST(cudaFree(gC));
  return c;
}

__global__
/**
  hoisted when extracting g_matmul_f32_rrr
*/
static void
__hoisted_24(
  uint32_t tile,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared + vbk * tile + k] *
          tB[(vbk * tile + k) * cols + blockIdx.x % mcols * tile + threadIdx.x % tile];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
    blockIdx.x % mcols * tile + threadIdx.x % tile]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_24,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    shared,
    cols,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f64_rrr
*/
static void
__hoisted_25(
  uint32_t tile,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared + vbk * tile + k] *
          tB[(vbk * tile + k) * cols + blockIdx.x % mcols * tile + threadIdx.x % tile];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
    blockIdx.x % mcols * tile + threadIdx.x % tile]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_25,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    shared,
    cols,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u32_rrr
*/
static void
__hoisted_26(
  uint32_t tile,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared + vbk * tile + k] *
          tB[(vbk * tile + k) * cols + blockIdx.x % mcols * tile + threadIdx.x % tile];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
    blockIdx.x % mcols * tile + threadIdx.x % tile]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_26,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    shared,
    cols,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u64_rrr
*/
static void
__hoisted_27(
  uint32_t tile,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * tile + threadIdx.x / tile) * shared + vbk * tile + k] *
          tB[(vbk * tile + k) * cols + blockIdx.x % mcols * tile + threadIdx.x % tile];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * tile + threadIdx.x / tile) * cols +
    blockIdx.x % mcols * tile + threadIdx.x % tile]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_rrr(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_27,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    shared,
    cols,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f32_ccc
*/
static void
__hoisted_28(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile + threadIdx.x / tile] *
          tB[(blockIdx.x % mcols * tile + threadIdx.x % tile) * shared + vbk * tile + k];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
    blockIdx.x / mcols * tile + threadIdx.x / tile]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_28,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    rows,
    shared,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f64_ccc
*/
static void
__hoisted_29(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile + threadIdx.x / tile] *
          tB[(blockIdx.x % mcols * tile + threadIdx.x % tile) * shared + vbk * tile + k];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
    blockIdx.x / mcols * tile + threadIdx.x / tile]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_29,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    rows,
    shared,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u32_ccc
*/
static void
__hoisted_30(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile + threadIdx.x / tile] *
          tB[(blockIdx.x % mcols * tile + threadIdx.x % tile) * shared + vbk * tile + k];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
    blockIdx.x / mcols * tile + threadIdx.x / tile]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_30,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    rows,
    shared,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u64_ccc
*/
static void
__hoisted_31(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < tile; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * tile + k) * rows + blockIdx.x / mcols * tile + threadIdx.x / tile] *
          tB[(blockIdx.x % mcols * tile + threadIdx.x % tile) * shared + vbk * tile + k];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * tile + threadIdx.x % tile) * rows +
    blockIdx.x / mcols * tile + threadIdx.x / tile]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_ccc(
  uint32_t tile,
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_GUARD(rows % tile == (uint32_t)0U);
  KPR_GUARD(shared % tile == (uint32_t)0U);
  KPR_GUARD(cols % tile == (uint32_t)0U);
  uint32_t mrows = rows / tile;
  uint32_t mshared = shared / tile;
  uint32_t mcols = cols / tile;
  KPR_ASSERT(tile > (uint32_t)0U);
  KPR_KCALL(__hoisted_31,
    mrows * mcols,
    tile * tile,
    (uint32_t)0U,
    tile,
    rows,
    shared,
    gA,
    gB,
    gC,
    mshared,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile32_rrr
*/
static void
__hoisted_32(
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * shared +
          vbk * (uint32_t)32U + k]
        *
          tB[(vbk * (uint32_t)32U + k) * cols +
            blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * cols +
    blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_32,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile32_rrr
*/
static void
__hoisted_33(
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * shared +
          vbk * (uint32_t)32U + k]
        *
          tB[(vbk * (uint32_t)32U + k) * cols +
            blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * cols +
    blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_33,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile32_rrr
*/
static void
__hoisted_34(
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * shared +
          vbk * (uint32_t)32U + k]
        *
          tB[(vbk * (uint32_t)32U + k) * cols +
            blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * cols +
    blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_34,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile32_rrr
*/
static void
__hoisted_35(
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * shared +
          vbk * (uint32_t)32U + k]
        *
          tB[(vbk * (uint32_t)32U + k) * cols +
            blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U) * cols +
    blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile32_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_35,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile32_ccc
*/
static void
__hoisted_36(
  uint32_t rows,
  uint32_t shared,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)32U + k) * rows +
          blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * shared +
            vbk * (uint32_t)32U + k];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * rows +
    blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_36,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile32_ccc
*/
static void
__hoisted_37(
  uint32_t rows,
  uint32_t shared,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)32U + k) * rows +
          blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * shared +
            vbk * (uint32_t)32U + k];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * rows +
    blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_37,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile32_ccc
*/
static void
__hoisted_38(
  uint32_t rows,
  uint32_t shared,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)32U + k) * rows +
          blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * shared +
            vbk * (uint32_t)32U + k];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * rows +
    blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_38,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile32_ccc
*/
static void
__hoisted_39(
  uint32_t rows,
  uint32_t shared,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < (uint32_t)32U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)32U + k) * rows +
          blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * shared +
            vbk * (uint32_t)32U + k];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)32U + threadIdx.x % (uint32_t)32U) * rows +
    blockIdx.x / mcols * (uint32_t)32U + threadIdx.x / (uint32_t)32U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile32_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)32U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)32U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)32U;
  KPR_KCALL(__hoisted_39,
    rows / (uint32_t)32U * mcols,
    (uint32_t)1024U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)32U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile16_rrr
*/
static void
__hoisted_40(
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * shared +
          vbk * (uint32_t)16U + k]
        *
          tB[(vbk * (uint32_t)16U + k) * cols +
            blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * cols +
    blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_40,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile16_rrr
*/
static void
__hoisted_41(
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * shared +
          vbk * (uint32_t)16U + k]
        *
          tB[(vbk * (uint32_t)16U + k) * cols +
            blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * cols +
    blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_41,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile16_rrr
*/
static void
__hoisted_42(
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * shared +
          vbk * (uint32_t)16U + k]
        *
          tB[(vbk * (uint32_t)16U + k) * cols +
            blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * cols +
    blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_42,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile16_rrr
*/
static void
__hoisted_43(
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * shared +
          vbk * (uint32_t)16U + k]
        *
          tB[(vbk * (uint32_t)16U + k) * cols +
            blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U) * cols +
    blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile16_rrr(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_43,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    shared,
    cols,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f32_tile16_ccc
*/
static void
__hoisted_44(
  uint32_t rows,
  uint32_t shared,
  float_t *gA,
  float_t *gB,
  float_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  float_t *gTile = gC;
  float_t sum = (float_t)0.0f;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    float_t *tA = gA;
    float_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    float_t sum1 = (float_t)0.0f;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)16U + k) * rows +
          blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * shared +
            vbk * (uint32_t)16U + k];
    float_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * rows +
    blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f32_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  float_t *gA,
  float_t *gB,
  float_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_44,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_f64_tile16_ccc
*/
static void
__hoisted_45(
  uint32_t rows,
  uint32_t shared,
  double_t *gA,
  double_t *gB,
  double_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  double_t *gTile = gC;
  double_t sum = (double_t)0.0l;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    double_t *tA = gA;
    double_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    double_t sum1 = (double_t)0.0l;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)16U + k) * rows +
          blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * shared +
            vbk * (uint32_t)16U + k];
    double_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * rows +
    blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_f64_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  double_t *gA,
  double_t *gB,
  double_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_45,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u32_tile16_ccc
*/
static void
__hoisted_46(
  uint32_t rows,
  uint32_t shared,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint32_t *gTile = gC;
  uint32_t sum = 0U;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint32_t *tA = gA;
    uint32_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint32_t sum1 = 0U;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)16U + k) * rows +
          blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * shared +
            vbk * (uint32_t)16U + k];
    uint32_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * rows +
    blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u32_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint32_t *gA,
  uint32_t *gB,
  uint32_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_46,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
}

__global__
/**
  hoisted when extracting g_matmul_u64_tile16_ccc
*/
static void
__hoisted_47(
  uint32_t rows,
  uint32_t shared,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC,
  uint32_t mshared,
  uint32_t mcols
)
{
  uint64_t *gTile = gC;
  uint64_t sum = 0ULL;
  uint32_t bk = (uint32_t)0U;
  while (bk < mshared)
  {
    uint32_t vbk = bk;
    uint64_t *tA = gA;
    uint64_t *tB = gB;
    uint32_t k = (uint32_t)0U;
    uint64_t sum1 = 0ULL;
    for (; k < (uint32_t)16U; k += (uint32_t)1U)
      sum1 +=
        tA[(vbk * (uint32_t)16U + k) * rows +
          blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
        *
          tB[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * shared +
            vbk * (uint32_t)16U + k];
    uint64_t s_ = sum1;
    sum += s_;
    bk = vbk + (uint32_t)1U;
  }
  gTile[(blockIdx.x % mcols * (uint32_t)16U + threadIdx.x % (uint32_t)16U) * rows +
    blockIdx.x / mcols * (uint32_t)16U + threadIdx.x / (uint32_t)16U]
  = sum;
}

void
Kuiper_GEMM_Tiled_g_matmul_u64_tile16_ccc(
  uint32_t rows,
  uint32_t shared,
  uint32_t cols,
  uint64_t *gA,
  uint64_t *gB,
  uint64_t *gC
)
{
  KPR_GUARD(rows % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(shared % (uint32_t)16U == (uint32_t)0U);
  KPR_GUARD(cols % (uint32_t)16U == (uint32_t)0U);
  uint32_t mcols = cols / (uint32_t)16U;
  KPR_KCALL(__hoisted_47,
    rows / (uint32_t)16U * mcols,
    (uint32_t)256U,
    (uint32_t)0U,
    rows,
    shared,
    gA,
    gB,
    gC,
    shared / (uint32_t)16U,
    mcols);
  cudaDeviceSynchronize();
}

