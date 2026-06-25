
#include "Kuiper_Example_TMap.h"

__global__
/**
  hoisted when extracting incr_all_1d
*/
static void __hoisted_incr_all_1d_0(uint32_t *a)
{
    if (1024U * blockIdx.x + threadIdx.x < 1024U)
        a[1024U * blockIdx.x + threadIdx.x]++;
}

void Kuiper_Example_TMap_incr_all_1d(uint32_t *a)
{
    KPR_KCALL(__hoisted_incr_all_1d_0, 1U, 1024U, 0U, a);
    MUST(cudaDeviceSynchronize());
}

typedef struct __uint32_t__uint32_t_______s {
    uint32_t fst;
    uint32_t snd;
} __uint32_t__uint32_t______;

__global__
/**
  hoisted when extracting incr_all_1d2
*/
static void __hoisted_incr_all_1d2_0(uint32_t *a)
{
    if (1024U * blockIdx.x + threadIdx.x < 1048576U)
        a[(KRML_CLITERAL(__uint32_t__uint32_t______) {
           .fst = (1024U * blockIdx.x + threadIdx.x) / 1024U,.snd =
           (1024U * blockIdx.x + threadIdx.x) % 1024U}
          ).fst * 1024U + (KRML_CLITERAL(__uint32_t__uint32_t______) {
                           .fst =
                           (1024U * blockIdx.x + threadIdx.x) / 1024U,.snd =
                           (1024U * blockIdx.x + threadIdx.x) % 1024U}
          ).snd]++;
}

void Kuiper_Example_TMap_incr_all_1d2(uint32_t *a)
{
    KPR_KCALL(__hoisted_incr_all_1d2_0, 1024U, 1024U, 0U, a);
    MUST(cudaDeviceSynchronize());
}
