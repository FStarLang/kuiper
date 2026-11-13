#ifndef KUIPER_TENSORCORE_MMA_GATHER_H
#define KUIPER_TENSORCORE_MMA_GATHER_H

#include <stddef.h>
#include <kuiper/mma.h>

typedef __nv_bfloat16 Kuiper_BFloat16_t;

/* Same native instruction and fragment as the installed MMA boundary.
   The caller supplies convergent live lanes and publishes input writes.
   Trailing pointers are erased proof arguments; they are never read. */
__device__ __forceinline__ void Kuiper_TensorCore_MMA_Gather_score(
    const __nv_bfloat16 *input, size_t head, size_t step, kpr_mma_fragment &d,
    void *, void *, void *)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned lane = kpr_mma_lane();
    unsigned group = lane / 4, pair = (lane % 4) * 2;
    size_t base = step * 16;
    __nv_bfloat16 a[8], b[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned row = group + ((i / 2) % 2) * 8;
        size_t col = base + pair + i % 2 + (i / 4) * 8;
        a[i] = row < 14
            ? input[(row / 2) * 4096 + head * 512 + (row % 2) * 128 + col]
            : __float2bfloat16(0.0f);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
        b[i] = group < 7
            ? input[group * 4096 + head * 512 + 256 + base + pair
                    + i % 2 + (i / 2) * 8]
            : __float2bfloat16(0.0f);
    uint32_t a0 = kpr_mma_pack(a[0], a[1]);
    uint32_t a1 = kpr_mma_pack(a[2], a[3]);
    uint32_t a2 = kpr_mma_pack(a[4], a[5]);
    uint32_t a3 = kpr_mma_pack(a[6], a[7]);
    uint32_t b0 = kpr_mma_pack(b[0], b[1]);
    uint32_t b1 = kpr_mma_pack(b[2], b[3]);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(__CUDA_ARCH__)
    __kuiper_mma_requires_sm_80__();
#endif
}

__device__ __forceinline__ void Kuiper_TensorCore_MMA_Gather_value(
    const __nv_bfloat16 *input, const __nv_bfloat16 *probability,
    size_t head, size_t block, size_t step, kpr_mma_fragment &d,
    void *, void *, void *, void *, void *)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned lane = kpr_mma_lane();
    unsigned group = lane / 4, pair = (lane % 4) * 2;
    size_t col = step * 16 + pair;
    uint32_t a0 = kpr_mma_pack(
        probability[group * 64 + col], probability[group * 64 + col + 1]);
    uint32_t a1 = kpr_mma_pack(probability[(group + 8) * 64 + col],
        probability[(group + 8) * 64 + col + 1]);
    uint32_t a2 = kpr_mma_pack(
        probability[group * 64 + col + 8], probability[group * 64 + col + 9]);
    uint32_t a3 = kpr_mma_pack(probability[(group + 8) * 64 + col + 8],
        probability[(group + 8) * 64 + col + 9]);
    __nv_bfloat16 b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        size_t row = col + i % 2 + (i / 2) * 8;
        b[i] = row < 7
            ? input[row * 4096 + head * 512 + 384 + block * 8 + group]
            : __float2bfloat16(0.0f);
    }
    uint32_t b0 = kpr_mma_pack(b[0], b[1]);
    uint32_t b1 = kpr_mma_pack(b[2], b[3]);
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
