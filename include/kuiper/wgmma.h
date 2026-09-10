#ifndef KUIPER_WGMMA_H
#define KUIPER_WGMMA_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// One thread's four registers of a warpgroup's 64x8 FP32 accumulator.
struct kpr_wgmma_fragment {
    float x[4];
};

__device__ __forceinline__ void kpr_wgmma_fill(kpr_wgmma_fragment &d, float x)
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
        d.x[i] = x;
}

// PTX m64nNk16 accumulator mapping, N=8. Use the linear thread index so
// multidimensional thread blocks and multiple warpgroups work too.
__device__ __forceinline__ unsigned kpr_wgmma_thread()
{
    return (threadIdx.x +
            blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) %
           128;
}

__device__ __forceinline__ void
kpr_wgmma_load_accum(kpr_wgmma_fragment &d, const float *c, uint32_t stride)
{
    unsigned t = kpr_wgmma_thread();
    unsigned row = (t / 32) * 16 + (t % 32) / 4;
    unsigned col = (t % 4) * 2;
    d.x[0] = c[row * stride + col];
    d.x[1] = c[row * stride + col + 1];
    d.x[2] = c[(row + 8) * stride + col];
    d.x[3] = c[(row + 8) * stride + col + 1];
}

__device__ __forceinline__ void kpr_wgmma_store(const kpr_wgmma_fragment &d,
                                                float *c, uint32_t stride)
{
    unsigned t = kpr_wgmma_thread();
    unsigned row = (t / 32) * 16 + (t % 32) / 4;
    unsigned col = (t % 4) * 2;
    c[row * stride + col] = d.x[0];
    c[row * stride + col + 1] = d.x[1];
    c[(row + 8) * stride + col] = d.x[2];
    c[(row + 8) * stride + col + 1] = d.x[3];
}

// K-major, no swizzle, LBO=128 bytes, SBO=256 bytes. The caller supplies
// aligned shared memory in Kuiper.TensorCore.WGMMA.Layout's packed layout.
__device__ __forceinline__ uint64_t kpr_wgmma_descriptor(const __nv_bfloat16 *p)
{
    uint64_t address = __cvta_generic_to_shared(p);
    return ((address & 0x3ffff) >> 4) | (uint64_t(128 >> 4) << 16) |
           (uint64_t(256 >> 4) << 32);
}

// Like CUDA's PTX wrappers, use an undefined device symbol to diagnose an
// unsupported target at device link time. nvcc -arch=sm_90a also emits a
// generic compute_90 PTX pass, so a preprocessing-time error/static_assert
// would reject supported builds as well.
extern "C" __device__ void __kuiper_wgmma_requires_sm_90a__();

// Synchronous interface to one native WGMMA operation. The caller must
// arrange convergent participation by all 128 threads and publish shared
// writes (including the producer-side async proxy fence and synchronization).
// No block barrier is hidden here. Waiting before return also keeps operand
// lifetimes and register accesses consistent with the sequential Pulse spec.
__device__ __forceinline__ void kpr_wgmma_mma_sync(const __nv_bfloat16 *a,
                                                   const __nv_bfloat16 *b,
                                                   kpr_wgmma_fragment &d)
{
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
    uint64_t da = kpr_wgmma_descriptor(a);
    uint64_t db = kpr_wgmma_descriptor(b);
    asm volatile("{\n"
                 "  .reg .pred accumulate;\n"
                 "  setp.ne.b32 accumulate, 1, 0;\n"
                 "  wgmma.fence.sync.aligned;\n"
                 "  wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                 "{%0, %1, %2, %3}, %4, %5, accumulate, 1, 1, 0, 0;\n"
                 "  wgmma.commit_group.sync.aligned;\n"
                 "  wgmma.wait_group.sync.aligned 0;\n"
                 "}\n"
                 : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
                 : "l"(da), "l"(db)
                 : "memory");
#elif defined(__CUDA_ARCH__)
    __kuiper_wgmma_requires_sm_90a__();
#endif
}

#endif /* KUIPER_WGMMA_H */
