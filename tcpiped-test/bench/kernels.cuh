#pragma once

#include "1_naive.cuh"
#include "2_kernel_global_mem_coalesce.cuh"
#include "3_kernel_shared_mem_blocking.cuh"
#include "4_kernel_1D_blocktiling.cuh"
#include "5_kernel_2D_blocktiling.cuh"
#include "6_kernel_vectorize.cuh"
#include "7_kernel_resolve_bank_conflicts.cuh"
#include "8_kernel_bank_extra_col.cuh"
#include "9_kernel_autotuned.cuh"
#include "10_kernel_warptiling.cuh"
#include "11_kernel_double_buffering.cuh"
// #include "12_kernel_double_buffering.cuh"
