#pragma once

/* Single-precision complex support for Kuiper-generated kernels.
   Kuiper.Complex32.Base.t extracts to cuFloatComplex (= float2); its arithmetic
   extracts to the cuComplex.h intrinsics (cuCaddf, cuCmulf, cuCsubf, cuCdivf,
   cuConjf, make_cuFloatComplex, cuCrealf, cuCimagf). The two helpers below back
   the comparison operations of the `scalar` class (cuComplex has no operator==,
   and complex is unordered so the order comparisons are always false). */

#include <cuComplex.h>

static __host__ __device__ __inline__ bool kpr_cceqf(cuFloatComplex a,
                                                      cuFloatComplex b)
{
    return cuCrealf(a) == cuCrealf(b) && cuCimagf(a) == cuCimagf(b);
}

static __host__ __device__ __inline__ bool kpr_cltf(cuFloatComplex a,
                                                    cuFloatComplex b)
{
    (void) a;
    (void) b;
    return false;
}

/* Double-precision (cuDoubleComplex) versions of the comparison helpers. */
static __host__ __device__ __inline__ bool kpr_cceq(cuDoubleComplex a,
                                                    cuDoubleComplex b)
{
    return cuCreal(a) == cuCreal(b) && cuCimag(a) == cuCimag(b);
}

static __host__ __device__ __inline__ bool kpr_clt(cuDoubleComplex a,
                                                   cuDoubleComplex b)
{
    (void) a;
    (void) b;
    return false;
}
