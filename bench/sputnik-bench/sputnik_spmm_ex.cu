/*
 * Sputnik SpMM translation unit for the benchmark.
 *
 * Sputnik only explicitly instantiates CudaSpmmEx<Config> for the configs
 * listed under SPUTNIK_BUILD_TEST in cuda_spmm.cu.cc, and none of them have
 * kBlockItemsY == 1 with float4 sparse loads, which is what Kuiper's kernel
 * corresponds to (see spmm_bench_config.h).  Rather than patching the vendored
 * checkout, we compile Sputnik's implementation here and add the explicit
 * instantiation the benchmark needs.
 *
 * This file *replaces* a plain compile of cuda_spmm.cu.cc -- do not link both,
 * they define the same symbols.
 */

#include "spmm_bench_config.h"

#include "sputnik/spmm/cuda_spmm.cu.cc"

namespace sputnik {

template cudaError_t CudaSpmmEx<BENCH_SPUTNIK_CONFIG>(
    int, int, int, int, const int *, const float *, const int *, const int *,
    const float *, const float *, float *, cudaStream_t);

}  // namespace sputnik
