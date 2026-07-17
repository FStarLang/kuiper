
#include "Kuiper_Example_OnlineSoftmax.h"

__global__
/**
  hoisted when extracting _test
*/
static void __hoisted__test_0(uint32_t len, float *a, float *b)
{
    if (1024U * blockIdx.x + threadIdx.x < len) {
        uint32_t i = 0U;
        float sum = 0.0f;
        float max = 0.0f - INFINITY;
        while (i < len) {
            float x = a[i];
            float max_ = fmaxf(max, x);
            float y1 = expf(max - max_);
            float y2 = expf(x - max_);
            float sum_ = sum * y1 + y2;
            max = max_;
            sum = sum_;
            i++;
            if (i == 1U) {

            }
        }
        float __anf0 = sum;
        b[1024U * blockIdx.x + threadIdx.x] =
            expf(a[1024U * blockIdx.x + threadIdx.x] - max) / __anf0;
    }
}

void Kuiper_Example_OnlineSoftmax__test(uint32_t len, float *a, float *b)
{
    KPR_KCALL(__hoisted__test_0,
              len / 1024U + (uint32_t) (len % 1024U != 0U),
              1024U, 0U, len, a, b);
    MUST(cudaDeviceSynchronize());
}

__global__
/**
  hoisted when extracting _testh
*/
static void __hoisted__testh_0(uint32_t len, half *a, half *b)
{
    if (1024U * blockIdx.x + threadIdx.x < len) {
        uint32_t i = 0U;
        half sum = __float2half_rn(0.0f);
        half max = __hsub(__float2half_rn(0.0f), HLF_INFINITY);
        while (i < len) {
            half x = a[i];
            half max_ = kpr_hfmax(max, x);
            half y1 = hexp(__hsub(max, max_));
            half y2 = hexp(__hsub(x, max_));
            half sum_ = __hadd(__hmul(sum, y1), y2);
            max = max_;
            sum = sum_;
            i++;
            if (i == 1U) {

            }
        }
        half __anf0 = sum;
        b[1024U * blockIdx.x + threadIdx.x] =
            __hdiv(hexp(__hsub(a[1024U * blockIdx.x + threadIdx.x], max)),
                   __anf0);
    }
}

void Kuiper_Example_OnlineSoftmax__testh(uint32_t len, half *a, half *b)
{
    KPR_KCALL(__hoisted__testh_0,
              len / 1024U + (uint32_t) (len % 1024U != 0U),
              1024U, 0U, len, a, b);
    MUST(cudaDeviceSynchronize());
}
