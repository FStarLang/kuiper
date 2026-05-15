#ifndef KUIPER_MATH_H
#define KUIPER_MATH_H 1

/*
 * Half-precision math wrappers.
 *
 * On device, we dispatch to native CUDA half intrinsics when they exist
 * (hsqrt, hsin, hcos, hceil, hfloor, hexp2, hlog2, hrsqrt, __habs,
 * __hfma).  On host (and for ops without a native intrinsic) we
 * round-trip through float.
 *
 * Convention: every wrapper is named kpr_h<op>.  The extraction plugin
 * emits these names so they must stay in sync.
 *
 * TODO: would static inline functions be better?
 */

#include <math.h>
#include <cuda_fp16.h>

static inline __host__ __device__
bool kpr_hisvalid(half x)
{
        float f = __half2float(x);
        return !isnan(f) && !isinf(f);
}

static inline __host__ __device__
bool kpr_fisvalid(float x)
{
        return !isnan(x) && !isinf(x);
}

static inline __host__ __device__
bool kpr_disvalid(double x)
{
        return !isnan(x) && !isinf(x);
}

/* ---- dispatch helpers ------------------------------------------------ */

/*
 * KPR_HWRAP{1,2,3}: use the native __device__ intrinsic on device,
 *                    fall back to float round-trip on host.
 * KPR_HFALL{1,2,3}: no native intrinsic — always round-trip.
 */

#ifdef __CUDA_ARCH__
#define KPR_HWRAP1(native, f32op, x)          native(x)
#define KPR_HWRAP2(native, f32op, x, y)       native(x, y)
#define KPR_HWRAP3(native, f32op, x, y, z)    native(x, y, z)
#else
#define KPR_HWRAP1(native, f32op, x)          (__float2half(f32op(__half2float(x))))
#define KPR_HWRAP2(native, f32op, x, y)       (__float2half(f32op(__half2float(x), __half2float(y))))
#define KPR_HWRAP3(native, f32op, x, y, z)    (__float2half(f32op(__half2float(x), __half2float(y), __half2float(z))))
#endif

#define KPR_HFALL1(f32op, x)                  (__float2half(f32op(__half2float(x))))
#define KPR_HFALL2(f32op, x, y)               (__float2half(f32op(__half2float(x), __half2float(y))))
#define KPR_HFALL3(f32op, x, y, z)            (__float2half(f32op(__half2float(x), __half2float(y), __half2float(z))))

/* ---- unary ops (native intrinsic available) -------------------------- */

#define kpr_hsqrt(f)      KPR_HWRAP1(hsqrt,  sqrtf,  f)
#define kpr_hrsqrt(f)     KPR_HWRAP1(hrsqrt, rsqrtf, f)
#define kpr_hsin(f)       KPR_HWRAP1(hsin,   sinf,   f)
#define kpr_hcos(f)       KPR_HWRAP1(hcos,   cosf,   f)
#define kpr_hceil(f)      KPR_HWRAP1(hceil,  ceilf,  f)
#define kpr_hfloor(f)     KPR_HWRAP1(hfloor, floorf, f)
#define kpr_hlog2(f)      KPR_HWRAP1(hlog2,  log2f,  f)
#define kpr_hexp2(f)      KPR_HWRAP1(hexp2,  exp2f,  f)
#define kpr_hfabs(f)      KPR_HWRAP1(__habs,  fabsf,  f)
#define kpr_hfma(x, y, z) KPR_HWRAP3(__hfma,  fmaf,   x, y, z)

/* ---- unary ops (no native intrinsic — float fallback) ---------------- */

#define kpr_htan(f)       KPR_HFALL1(tanf,   f)
#define kpr_hasin(f)      KPR_HFALL1(asinf,  f)
#define kpr_hacos(f)      KPR_HFALL1(acosf,  f)
#define kpr_hatan(f)      KPR_HFALL1(atanf,  f)
#define kpr_hsinh(f)      KPR_HFALL1(sinhf,  f)
#define kpr_hcosh(f)      KPR_HFALL1(coshf,  f)
#define kpr_htanh(f)      KPR_HFALL1(tanhf,  f)
#define kpr_hround(f)     KPR_HFALL1(roundf, f)
#define kpr_herf(f)       KPR_HFALL1(erff,   f)
#define kpr_hlog10(f)     KPR_HFALL1(log10f, f)

/* ---- binary ops (no native intrinsic) -------------------------------- */

#define kpr_hpow(f, g)       KPR_HFALL2(powf,      f, g)
#define kpr_hatan2(f, g)     KPR_HFALL2(atan2f,    f, g)
#define kpr_hfmin(f, g)      KPR_HFALL2(fminf,     f, g)
#define kpr_hfmax(f, g)      KPR_HFALL2(fmaxf,     f, g)
#define kpr_hfmod(f, g)      KPR_HFALL2(fmodf,     f, g)
#define kpr_hcopysign(f, g)  KPR_HFALL2(copysignf, f, g)

#endif /* KUIPER_MATH_H */
