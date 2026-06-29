
#include "Klas_Softmax.h"

__global__
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void __hoisted_softmax_gpu_n_f16_0(uint32_t lena, half *a, uint32_t nthm,
                                          half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = a[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, a[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = kpr_hfmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void
__hoisted_softmax_gpu_n_f16_1(uint32_t nth, uint32_t lena, half m, half *x_,
                              half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        half v_ = hexp(__hsub(x_[idx], m));
        acc = __hadd(acc, v_);
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f16
*/
static void __hoisted_softmax_gpu_n_f16_2(uint32_t lena, half *a, half m,
                                          half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            __hdiv(hexp(__hsub(a[1024U * blockIdx.x + threadIdx.x], m)), sum);
}

void Klas_Softmax_softmax_gpu_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    uint32_t nthm = nth <= lena ? nth : lena;
    half *out1 = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    KPR_SHMEM_FITS(2U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_0, 1U, nthm, 2U * nthm, lena, a, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out1, sizeof(half), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    half m = hout;
    half *x_ = a;
    half *out0 = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    half *out = out0;
    KPR_SHMEM_FITS(2U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f16_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nth));
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_1, 1U, nth, 2U * nth, nth, lena, m,
              x_, out);
    MUST(cudaDeviceSynchronize());
    half *local_out = (half *) KRML_HOST_MALLOC(sizeof(half));
    if (local_out != NULL)
        *local_out = __float2half_rn(0.0f);
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(half), cudaMemcpyDeviceToHost));
    half res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    half sum = res;
    KPR_KCALL(__hoisted_softmax_gpu_n_f16_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, m, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f32
*/
static void __hoisted_softmax_gpu_n_f32_0(uint32_t lena, float *a,
                                          uint32_t nthm, float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = a[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, a[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f32
*/
static void
__hoisted_softmax_gpu_n_f32_1(uint32_t nth, uint32_t lena, float m, float *x_,
                              float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        float v_ = expf(x_[idx] - m);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f32
*/
static void __hoisted_softmax_gpu_n_f32_2(uint32_t lena, float *a, float m,
                                          float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            expf(a[1024U * blockIdx.x + threadIdx.x] - m) / sum;
}

void Klas_Softmax_softmax_gpu_n_f32(uint32_t nth, uint32_t lena, float *a)
{
    uint32_t nthm = nth <= lena ? nth : lena;
    float *out1 = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    KPR_SHMEM_FITS(4U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_0, 1U, nthm, 4U * nthm, lena, a, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out1, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    float m = hout;
    float *x_ = a;
    float *out0 = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    float *out = out0;
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f32_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nth));
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_1, 1U, nth, 4U * nth, nth, lena, m,
              x_, out);
    MUST(cudaDeviceSynchronize());
    float *local_out = (float *)KRML_HOST_MALLOC(sizeof(float));
    if (local_out != NULL)
        *local_out = 0.0f;
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(float), cudaMemcpyDeviceToHost));
    float res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    float sum = res;
    KPR_KCALL(__hoisted_softmax_gpu_n_f32_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, m, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f64
*/
static void __hoisted_softmax_gpu_n_f64_0(uint32_t lena, double *a,
                                          uint32_t nthm, double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = a[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, a[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f64
*/
static void
__hoisted_softmax_gpu_n_f64_1(uint32_t nth, uint32_t lena, double m, double *x_,
                              double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        double v_ = exp(x_[idx] - m);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_n_f64
*/
static void __hoisted_softmax_gpu_n_f64_2(uint32_t lena, double *a, double m,
                                          double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            exp(a[1024U * blockIdx.x + threadIdx.x] - m) / sum;
}

void Klas_Softmax_softmax_gpu_n_f64(uint32_t nth, uint32_t lena, double *a)
{
    uint32_t nthm = nth <= lena ? nth : lena;
    double *out1 = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    KPR_SHMEM_FITS(8U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_0, 1U, nthm, 8U * nthm, lena, a, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0;
    MUST(cudaMemcpy(&hout, out1, sizeof(double), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    double m = hout;
    double *x_ = a;
    double *out0 = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    double *out = out0;
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_n_f64_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nth));
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_1, 1U, nth, 8U * nth, nth, lena, m,
              x_, out);
    MUST(cudaDeviceSynchronize());
    double *local_out = (double *)KRML_HOST_MALLOC(sizeof(double));
    if (local_out != NULL)
        *local_out = 0.0;
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(double), cudaMemcpyDeviceToHost));
    double res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    double sum = res;
    KPR_KCALL(__hoisted_softmax_gpu_n_f64_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, m, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting softmax_gpu_f16
*/
static void __hoisted_softmax_gpu_f16_0(uint32_t lena, half *a, uint32_t nthm,
                                        half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = a[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, a[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = kpr_hfmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f16
*/
static void __hoisted_softmax_gpu_f16_1(uint32_t lena, half m, half *x_,
                                        half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        half v_ = hexp(__hsub(x_[idx], m));
        acc = __hadd(acc, v_);
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f16
*/
static void __hoisted_softmax_gpu_f16_2(uint32_t lena, half *a, half m,
                                        half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            __hdiv(hexp(__hsub(a[1024U * blockIdx.x + threadIdx.x], m)), sum);
}

void Klas_Softmax_softmax_gpu_f16(uint32_t lena, half *a)
{
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    half *out1 = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    KPR_SHMEM_FITS(2U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_f16_0, 1U, nthm, 2U * nthm, lena, a, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out1, sizeof(half), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    half m = hout;
    half *x_ = a;
    half *out0 = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    half *out = out0;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f16_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_softmax_gpu_f16_1, 1U, 1024U, 2048U, lena, m, x_, out);
    MUST(cudaDeviceSynchronize());
    half *local_out = (half *) KRML_HOST_MALLOC(sizeof(half));
    if (local_out != NULL)
        *local_out = __float2half_rn(0.0f);
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(half), cudaMemcpyDeviceToHost));
    half res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    half sum = res;
    KPR_KCALL(__hoisted_softmax_gpu_f16_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, m, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting softmax_gpu_f32
*/
static void __hoisted_softmax_gpu_f32_0(uint32_t lena, float *a, uint32_t nthm,
                                        float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = a[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, a[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f32
*/
static void __hoisted_softmax_gpu_f32_1(uint32_t lena, float m, float *x_,
                                        float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        float v_ = expf(x_[idx] - m);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f32
*/
static void __hoisted_softmax_gpu_f32_2(uint32_t lena, float *a, float m,
                                        float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            expf(a[1024U * blockIdx.x + threadIdx.x] - m) / sum;
}

void Klas_Softmax_softmax_gpu_f32(uint32_t lena, float *a)
{
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    float *out1 = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    KPR_SHMEM_FITS(4U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_f32_0, 1U, nthm, 4U * nthm, lena, a, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out1, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    float m = hout;
    float *x_ = a;
    float *out0 = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    float *out = out0;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f32_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_softmax_gpu_f32_1, 1U, 1024U, 4096U, lena, m, x_, out);
    MUST(cudaDeviceSynchronize());
    float *local_out = (float *)KRML_HOST_MALLOC(sizeof(float));
    if (local_out != NULL)
        *local_out = 0.0f;
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(float), cudaMemcpyDeviceToHost));
    float res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    float sum = res;
    KPR_KCALL(__hoisted_softmax_gpu_f32_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, m, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting softmax_gpu_f64
*/
static void __hoisted_softmax_gpu_f64_0(uint32_t lena, double *a, uint32_t nthm,
                                        double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = a[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, a[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f64
*/
static void __hoisted_softmax_gpu_f64_1(uint32_t lena, double m, double *x_,
                                        double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        double v_ = exp(x_[idx] - m);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_gpu_f64
*/
static void __hoisted_softmax_gpu_f64_2(uint32_t lena, double *a, double m,
                                        double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            exp(a[1024U * blockIdx.x + threadIdx.x] - m) / sum;
}

void Klas_Softmax_softmax_gpu_f64(uint32_t lena, double *a)
{
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    double *out1 = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    KPR_SHMEM_FITS(8U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nthm));
    KPR_KCALL(__hoisted_softmax_gpu_f64_0, 1U, nthm, 8U * nthm, lena, a, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0;
    MUST(cudaMemcpy(&hout, out1, sizeof(double), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    double m = hout;
    double *x_ = a;
    double *out0 = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    double *out = out0;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_gpu_f64_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_softmax_gpu_f64_1, 1U, 1024U, 8192U, lena, m, x_, out);
    MUST(cudaDeviceSynchronize());
    double *local_out = (double *)KRML_HOST_MALLOC(sizeof(double));
    if (local_out != NULL)
        *local_out = 0.0;
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(double), cudaMemcpyDeviceToHost));
    double res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    double sum = res;
    KPR_KCALL(__hoisted_softmax_gpu_f64_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, a, m, sum);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void __hoisted_softmax_n_f16_0(uint32_t lena, half *ga, uint32_t nthm,
                                      half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = ga[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, ga[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = kpr_hfmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void __hoisted_softmax_n_f16_1(uint32_t nth, uint32_t lena, half m,
                                      half *x_, half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        half v_ = hexp(__hsub(x_[idx], m));
        acc = __hadd(acc, v_);
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f16
*/
static void __hoisted_softmax_n_f16_2(uint32_t lena, half *ga, half m, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            __hdiv(hexp(__hsub(ga[1024U * blockIdx.x + threadIdx.x], m)), sum);
}

void Klas_Softmax_softmax_n_f16(uint32_t nth, uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC(sizeof(half), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(half) * lena, cudaMemcpyHostToDevice));
    uint32_t nthm = nth <= lena ? nth : lena;
    half *out1 = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    KPR_SHMEM_FITS(2U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nthm));
    KPR_KCALL(__hoisted_softmax_n_f16_0, 1U, nthm, 2U * nthm, lena, ga, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out1, sizeof(half), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    half m = hout;
    half *x_ = ga;
    half *out0 = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    half *out = out0;
    KPR_SHMEM_FITS(2U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f16_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nth));
    KPR_KCALL(__hoisted_softmax_n_f16_1, 1U, nth, 2U * nth, nth, lena, m, x_,
              out);
    MUST(cudaDeviceSynchronize());
    half *local_out = (half *) KRML_HOST_MALLOC(sizeof(half));
    if (local_out != NULL)
        *local_out = __float2half_rn(0.0f);
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(half), cudaMemcpyDeviceToHost));
    half res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    half sum = res;
    KPR_KCALL(__hoisted_softmax_n_f16_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, m, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_n_f32
*/
static void __hoisted_softmax_n_f32_0(uint32_t lena, float *ga, uint32_t nthm,
                                      float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = ga[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, ga[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f32
*/
static void
__hoisted_softmax_n_f32_1(uint32_t nth, uint32_t lena, float m, float *x_,
                          float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        float v_ = expf(x_[idx] - m);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f32
*/
static void __hoisted_softmax_n_f32_2(uint32_t lena, float *ga, float m,
                                      float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            expf(ga[1024U * blockIdx.x + threadIdx.x] - m) / sum;
}

void Klas_Softmax_softmax_n_f32(uint32_t nth, uint32_t lena, float *a)
{
    float *ga = (float *)KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(float) * lena, cudaMemcpyHostToDevice));
    uint32_t nthm = nth <= lena ? nth : lena;
    float *out1 = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    KPR_SHMEM_FITS(4U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nthm));
    KPR_KCALL(__hoisted_softmax_n_f32_0, 1U, nthm, 4U * nthm, lena, ga, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out1, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    float m = hout;
    float *x_ = ga;
    float *out0 = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    float *out = out0;
    KPR_SHMEM_FITS(4U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f32_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nth));
    KPR_KCALL(__hoisted_softmax_n_f32_1, 1U, nth, 4U * nth, nth, lena, m, x_,
              out);
    MUST(cudaDeviceSynchronize());
    float *local_out = (float *)KRML_HOST_MALLOC(sizeof(float));
    if (local_out != NULL)
        *local_out = 0.0f;
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(float), cudaMemcpyDeviceToHost));
    float res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    float sum = res;
    KPR_KCALL(__hoisted_softmax_n_f32_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, m, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_n_f64
*/
static void __hoisted_softmax_n_f64_0(uint32_t lena, double *ga, uint32_t nthm,
                                      double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = ga[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, ga[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f64
*/
static void
__hoisted_softmax_n_f64_1(uint32_t nth, uint32_t lena, double m, double *x_,
                          double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += nth) {
        double v_ = exp(x_[idx] - m);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nth; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_n_f64
*/
static void __hoisted_softmax_n_f64_2(uint32_t lena, double *ga, double m,
                                      double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            exp(ga[1024U * blockIdx.x + threadIdx.x] - m) / sum;
}

void Klas_Softmax_softmax_n_f64(uint32_t nth, uint32_t lena, double *a)
{
    double *ga = (double *)KPR_GPU_ALLOC(sizeof(double), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(double) * lena, cudaMemcpyHostToDevice));
    uint32_t nthm = nth <= lena ? nth : lena;
    double *out1 = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    KPR_SHMEM_FITS(8U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nthm));
    KPR_KCALL(__hoisted_softmax_n_f64_0, 1U, nthm, 8U * nthm, lena, ga, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0;
    MUST(cudaMemcpy(&hout, out1, sizeof(double), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    double m = hout;
    double *x_ = ga;
    double *out0 = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    double *out = out0;
    KPR_SHMEM_FITS(8U * nth);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_n_f64_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nth));
    KPR_KCALL(__hoisted_softmax_n_f64_1, 1U, nth, 8U * nth, nth, lena, m, x_,
              out);
    MUST(cudaDeviceSynchronize());
    double *local_out = (double *)KRML_HOST_MALLOC(sizeof(double));
    if (local_out != NULL)
        *local_out = 0.0;
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(double), cudaMemcpyDeviceToHost));
    double res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    double sum = res;
    KPR_KCALL(__hoisted_softmax_n_f64_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, m, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_f16
*/
static void __hoisted_softmax_f16_0(uint32_t lena, half *ga, uint32_t nthm,
                                    half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = ga[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = kpr_hfmax(acc, ga[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = kpr_hfmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_f16
*/
static void __hoisted_softmax_f16_1(uint32_t lena, half m, half *x_, half *out)
{
    half *sa = (half *) KPR_SHMEM_AT(0U);
    half acc = __float2half_rn(0.0f);
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        half v_ = hexp(__hsub(x_[idx], m));
        acc = __hadd(acc, v_);
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = __hadd(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_f16
*/
static void __hoisted_softmax_f16_2(uint32_t lena, half *ga, half m, half sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            __hdiv(hexp(__hsub(ga[1024U * blockIdx.x + threadIdx.x], m)), sum);
}

void Klas_Softmax_softmax_f16(uint32_t lena, half *a)
{
    half *ga = (half *) KPR_GPU_ALLOC(sizeof(half), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(half) * lena, cudaMemcpyHostToDevice));
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    half *out1 = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    KPR_SHMEM_FITS(2U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f16_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2U * nthm));
    KPR_KCALL(__hoisted_softmax_f16_0, 1U, nthm, 2U * nthm, lena, ga, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    half hout = __float2half_rn(0.0f);
    MUST(cudaMemcpy(&hout, out1, sizeof(half), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    half m = hout;
    half *x_ = ga;
    half *out0 = (half *) KPR_GPU_ALLOC(sizeof(half), 1U);
    half *out = out0;
    KPR_SHMEM_FITS(2048U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f16_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              2048U));
    KPR_KCALL(__hoisted_softmax_f16_1, 1U, 1024U, 2048U, lena, m, x_, out);
    MUST(cudaDeviceSynchronize());
    half *local_out = (half *) KRML_HOST_MALLOC(sizeof(half));
    if (local_out != NULL)
        *local_out = __float2half_rn(0.0f);
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(half), cudaMemcpyDeviceToHost));
    half res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    half sum = res;
    KPR_KCALL(__hoisted_softmax_f16_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, m, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(half) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_softmax_f32_0(uint32_t lena, float *ga, uint32_t nthm,
                                    float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = ga[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmaxf(acc, ga[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_softmax_f32_1(uint32_t lena, float m, float *x_,
                                    float *out)
{
    float *sa = (float *)KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        float v_ = expf(x_[idx] - m);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_f32
*/
static void __hoisted_softmax_f32_2(uint32_t lena, float *ga, float m,
                                    float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            expf(ga[1024U * blockIdx.x + threadIdx.x] - m) / sum;
}

void Klas_Softmax_softmax_f32(uint32_t lena, float *a)
{
    float *ga = (float *)KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(float) * lena, cudaMemcpyHostToDevice));
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    float *out1 = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    KPR_SHMEM_FITS(4U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f32_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4U * nthm));
    KPR_KCALL(__hoisted_softmax_f32_0, 1U, nthm, 4U * nthm, lena, ga, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out1, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    float m = hout;
    float *x_ = ga;
    float *out0 = (float *)KPR_GPU_ALLOC(sizeof(float), 1U);
    float *out = out0;
    KPR_SHMEM_FITS(4096U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f32_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              4096U));
    KPR_KCALL(__hoisted_softmax_f32_1, 1U, 1024U, 4096U, lena, m, x_, out);
    MUST(cudaDeviceSynchronize());
    float *local_out = (float *)KRML_HOST_MALLOC(sizeof(float));
    if (local_out != NULL)
        *local_out = 0.0f;
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(float), cudaMemcpyDeviceToHost));
    float res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    float sum = res;
    KPR_KCALL(__hoisted_softmax_f32_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, m, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(float) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_softmax_f64_0(uint32_t lena, double *ga, uint32_t nthm,
                                    double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = ga[threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < lena; idx += nthm)
        acc = fmax(acc, ga[idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] = fmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_softmax_f64_1(uint32_t lena, double m, double *x_,
                                    double *out)
{
    double *sa = (double *)KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        double v_ = exp(x_[idx] - m);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x & (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) -
                 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_f64
*/
static void __hoisted_softmax_f64_2(uint32_t lena, double *ga, double m,
                                    double sum)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        ga[1024U * blockIdx.x + threadIdx.x] =
            exp(ga[1024U * blockIdx.x + threadIdx.x] - m) / sum;
}

void Klas_Softmax_softmax_f64(uint32_t lena, double *a)
{
    double *ga = (double *)KPR_GPU_ALLOC(sizeof(double), lena);
    MUST(cudaMemcpy
         (ga, a, (uint32_t) sizeof(double) * lena, cudaMemcpyHostToDevice));
    uint32_t nthm = 1024U <= lena ? 1024U : lena;
    double *out1 = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    KPR_SHMEM_FITS(8U * nthm);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f64_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8U * nthm));
    KPR_KCALL(__hoisted_softmax_f64_0, 1U, nthm, 8U * nthm, lena, ga, nthm,
              out1);
    MUST(cudaDeviceSynchronize());
    double hout = 0.0;
    MUST(cudaMemcpy(&hout, out1, sizeof(double), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out1));
    double m = hout;
    double *x_ = ga;
    double *out0 = (double *)KPR_GPU_ALLOC(sizeof(double), 1U);
    double *out = out0;
    KPR_SHMEM_FITS(8192U);
    MUST(cudaFuncSetAttribute(__hoisted_softmax_f64_1,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              8192U));
    KPR_KCALL(__hoisted_softmax_f64_1, 1U, 1024U, 8192U, lena, m, x_, out);
    MUST(cudaDeviceSynchronize());
    double *local_out = (double *)KRML_HOST_MALLOC(sizeof(double));
    if (local_out != NULL)
        *local_out = 0.0;
    MUST(cudaMemcpy
         (local_out, out0, (uint32_t) sizeof(double), cudaMemcpyDeviceToHost));
    double res = *local_out;
    KRML_HOST_FREE(local_out);
    MUST(cudaFree(out0));
    double sum = res;
    KPR_KCALL(__hoisted_softmax_f64_2,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U),
              1024U, 0U, lena, ga, m, sum);
    MUST(cudaDeviceSynchronize());
    MUST(cudaMemcpy
         (a, ga, (uint32_t) sizeof(double) * lena, cudaMemcpyDeviceToHost));
    MUST(cudaFree(ga));
}
