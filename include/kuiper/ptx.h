#ifndef KUIPER_PTX_H
#define KUIPER_PTX_H 1

static __device__ __forceinline__ float kpr_ptx_mul_rn_ftz_f32(float x, float y)
{
    float result;
    asm volatile("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(x), "f"(y));
    return result;
}

static __device__ __forceinline__ float kpr_ptx_ex2_approx_ftz_f32(float x)
{
    float result;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

#endif /* KUIPER_PTX_H */
