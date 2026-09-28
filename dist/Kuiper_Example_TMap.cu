
#include "Kuiper_Example_TMap.h"

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting incr_all_1d
*/
static void
__hoisted_incr_all_1d_0(uint32_t *a)
{
    if (1024U * blockIdx.x + threadIdx.x < 1024U)
        a[1024U * blockIdx.x + threadIdx.x]++;
}

__global__ __launch_bounds__(1024)
/**
  hoisted when extracting incr_all_1d2
*/
static void
__hoisted_incr_all_1d2_0(uint32_t *a)
{
    if (1024U * blockIdx.x + threadIdx.x < 1048576U)
        a[(KRML_CLITERAL(
               FStar_Pervasives_Native_tuple2__uint32_t_FStar_Pervasives_Native_tuple2__uint32_t___){
               ._1 = (1024U * blockIdx.x + threadIdx.x) / 1024U,
               ._2 = (1024U * blockIdx.x + threadIdx.x) % 1024U})
                    ._1 *
                1024U +
            (KRML_CLITERAL(
                 FStar_Pervasives_Native_tuple2__uint32_t_FStar_Pervasives_Native_tuple2__uint32_t___){
                 ._1 = (1024U * blockIdx.x + threadIdx.x) / 1024U,
                 ._2 = (1024U * blockIdx.x + threadIdx.x) % 1024U})
                ._2]++;
}

void Kuiper_Example_TMap_incr_all_1d(uint32_t *a)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_incr_all_1d_0, 1U, 1024U, 0U, s, a);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void Kuiper_Example_TMap_incr_all_1d2(uint32_t *a)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_incr_all_1d2_0, 1024U, 1024U, 0U, s, a);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
