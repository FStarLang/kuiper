#ifndef KUIPER_TENSORCORE_MMA_ZEROA_H
#define KUIPER_TENSORCORE_MMA_ZEROA_H

#include <stddef.h>
#include <kuiper/mma.h>

typedef __nv_bfloat16 Kuiper_BFloat16_t;

/* Caller establishes convergent live lanes and publication. Trailing pointers
   are erased proof arguments and are never read. */
__device__ __forceinline__ void Kuiper_TensorCore_MMA_ZeroA_mma_sync7(
    const __nv_bfloat16 *hidden, size_t step, kpr_mma_fragment &d,
    void *, void *, void *)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned lane = kpr_mma_lane();
    unsigned group = lane / 4, pair = (lane % 4) * 2;
    size_t col = step * 16 + pair;
    uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
    uint32_t b0 = group < 7
        ? kpr_mma_pack(hidden[group * 1024 + col],
            hidden[group * 1024 + col + 1]) : 0u;
    uint32_t b1 = group < 7
        ? kpr_mma_pack(hidden[group * 1024 + col + 8],
            hidden[group * 1024 + col + 9]) : 0u;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(__CUDA_ARCH__)
    __kuiper_mma_requires_sm_80__();
#endif
}

#endif
