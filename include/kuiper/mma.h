#ifndef KUIPER_MMA_H
#define KUIPER_MMA_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

struct kpr_mma_fragment {
    float x[4];
};

__device__ __forceinline__ unsigned kpr_mma_lane()
{
    return (threadIdx.x +
               blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) %
           32;
}

__device__ __forceinline__ void kpr_mma_fill(kpr_mma_fragment &d, float x)
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
        d.x[i] = x;
}

__device__ __forceinline__ void kpr_mma_load_accum(
    kpr_mma_fragment &d, const float *c, uint32_t stride)
{
    unsigned lane = kpr_mma_lane();
    unsigned row = lane / 4, col = (lane % 4) * 2;
    d.x[0] = c[row * stride + col];
    d.x[1] = c[row * stride + col + 1];
    d.x[2] = c[(row + 8) * stride + col];
    d.x[3] = c[(row + 8) * stride + col + 1];
}

__device__ __forceinline__ void kpr_mma_store(
    const kpr_mma_fragment &d, float *c, uint32_t stride)
{
    unsigned lane = kpr_mma_lane();
    unsigned row = lane / 4, col = (lane % 4) * 2;
    c[row * stride + col] = d.x[0];
    c[row * stride + col + 1] = d.x[1];
    c[(row + 8) * stride + col] = d.x[2];
    c[(row + 8) * stride + col + 1] = d.x[3];
}

__device__ __forceinline__ uint32_t kpr_mma_pack(
    __nv_bfloat16 lo, __nv_bfloat16 hi)
{
    return uint32_t(__bfloat16_as_ushort(lo)) |
           (uint32_t(__bfloat16_as_ushort(hi)) << 16);
}

extern "C" __device__ void __kuiper_mma_requires_sm_80__();

// PTX m16n8k16 BF16 fragment layout: A is row-major, B column-major.
// The caller supplies convergent participation by all 32 live warp lanes
// and publishes any input writes before entering this operation.
__device__ __forceinline__ void kpr_mma_mma_sync(const __nv_bfloat16 *a,
    uint32_t lda, const __nv_bfloat16 *b, uint32_t ldb, kpr_mma_fragment &d)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned lane = kpr_mma_lane();
    unsigned group = lane / 4, pair = (lane % 4) * 2;
    uint32_t a0 =
        kpr_mma_pack(a[group * lda + pair], a[group * lda + pair + 1]);
    uint32_t a1 = kpr_mma_pack(
        a[(group + 8) * lda + pair], a[(group + 8) * lda + pair + 1]);
    uint32_t a2 =
        kpr_mma_pack(a[group * lda + pair + 8], a[group * lda + pair + 9]);
    uint32_t a3 = kpr_mma_pack(
        a[(group + 8) * lda + pair + 8], a[(group + 8) * lda + pair + 9]);
    uint32_t b0 =
        kpr_mma_pack(b[group * ldb + pair], b[group * ldb + pair + 1]);
    uint32_t b1 =
        kpr_mma_pack(b[group * ldb + pair + 8], b[group * ldb + pair + 9]);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(__CUDA_ARCH__)
    __kuiper_mma_requires_sm_80__();
#endif
}

#endif /* KUIPER_MMA_H */
