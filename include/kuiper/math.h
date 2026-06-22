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
#include <cuda_bf16.h>

#define HLF_MIN      __float2half(6.10352e-5f)
#define HLF_MAX      __float2half(65504.0f)
#define HLF_INFINITY __float2half(INFINITY)
#define __hrcp(x)    ((half)1.0 / (x))

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

/* ======================================================================== */
/* BFloat16 math wrappers                                                   */
/* ======================================================================== */

/* ---- dispatch helpers for bf16 --------------------------------------- */

#ifdef __CUDA_ARCH__
#define KPR_BF16WRAP1(native, f32op, x)          native(x)
#define KPR_BF16WRAP2(native, f32op, x, y)       native(x, y)
#define KPR_BF16WRAP3(native, f32op, x, y, z)    native(x, y, z)
#else
#define KPR_BF16WRAP1(native, f32op, x)          (__float2bfloat16(f32op(__bfloat162float(x))))
#define KPR_BF16WRAP2(native, f32op, x, y)       (__float2bfloat16(f32op(__bfloat162float(x), __bfloat162float(y))))
#define KPR_BF16WRAP3(native, f32op, x, y, z)    (__float2bfloat16(f32op(__bfloat162float(x), __bfloat162float(y), __bfloat162float(z))))
#endif

#define KPR_BF16FALL1(f32op, x)                  (__float2bfloat16(f32op(__bfloat162float(x))))
#define KPR_BF16FALL2(f32op, x, y)               (__float2bfloat16(f32op(__bfloat162float(x), __bfloat162float(y))))
#define KPR_BF16FALL3(f32op, x, y, z)            (__float2bfloat16(f32op(__bfloat162float(x), __bfloat162float(y), __bfloat162float(z))))

/* ---- unary ops (native intrinsic available on device) ---------------- */

#define kpr_bf16sqrt(f)      KPR_BF16WRAP1(hsqrt,  sqrtf,  f)
#define kpr_bf16rsqrt(f)     KPR_BF16WRAP1(hrsqrt, rsqrtf, f)
#define kpr_bf16sin(f)       KPR_BF16WRAP1(hsin,   sinf,   f)
#define kpr_bf16cos(f)       KPR_BF16WRAP1(hcos,   cosf,   f)
#define kpr_bf16ceil(f)      KPR_BF16WRAP1(hceil,  ceilf,  f)
#define kpr_bf16floor(f)     KPR_BF16WRAP1(hfloor, floorf, f)
#define kpr_bf16log2(f)      KPR_BF16WRAP1(hlog2,  log2f,  f)
#define kpr_bf16exp2(f)      KPR_BF16WRAP1(hexp2,  exp2f,  f)
#define kpr_bf16fabs(f)      KPR_BF16WRAP1(__habs,  fabsf,  f)
#define kpr_bf16exp(f)       KPR_BF16WRAP1(hexp,   expf,   f)
#define kpr_bf16log(f)       KPR_BF16WRAP1(hlog,   logf,   f)
#define kpr_bf16log10(f)     KPR_BF16WRAP1(hlog10, log10f, f)
#define kpr_bf16tanh(f)      KPR_BF16WRAP1(htanh,  tanhf,  f)
#define kpr_bf16fma(x, y, z) KPR_BF16WRAP3(__hfma,  fmaf,   x, y, z)

/* ---- binary ops (native intrinsic available on device) ---------------- */

#define add(f,g) ((f)+(g))
#define sub(f,g) ((f)-(g))
#define mul(f,g) ((f)*(g))
#define div(f,g) ((f)/(g))

// On older versions of CUDA, the native bf16 intrinsics don't compile on a device
// without native bf16 support. The operator overloads, by contrast, are still defined,
// and will be compiled to round-trip casts on f32 iff the device does not support the native ops.
// However, some operator overloads like += are missing, so
// we force the use of an opaque bf16add/sub/mul/etc. macro 
// so that karamel does not try to simplify into += etc.

#define kpr_bf16add(f,g) KPR_BF16WRAP2(add,add,f,g) 
#define kpr_bf16sub(f,g) KPR_BF16WRAP2(sub,sub,f,g) 
#define kpr_bf16mul(f,g) KPR_BF16WRAP2(mul,mul,f,g) 
#define kpr_bf16div(f,g) KPR_BF16WRAP2(div,div,f,g) 

/* ---- unary ops (no native intrinsic — float fallback) ---------------- */

#define kpr_bf16tan(f)       KPR_BF16FALL1(tanf,   f)
#define kpr_bf16asin(f)      KPR_BF16FALL1(asinf,  f)
#define kpr_bf16acos(f)      KPR_BF16FALL1(acosf,  f)
#define kpr_bf16atan(f)      KPR_BF16FALL1(atanf,  f)
#define kpr_bf16sinh(f)      KPR_BF16FALL1(sinhf,  f)
#define kpr_bf16cosh(f)      KPR_BF16FALL1(coshf,  f)
#define kpr_bf16round(f)     KPR_BF16FALL1(roundf, f)
#define kpr_bf16erf(f)       KPR_BF16FALL1(erff,   f)

/* ---- binary ops (no native intrinsic) -------------------------------- */

#define kpr_bf16pow(f, g)       KPR_BF16FALL2(powf,      f, g)
#define kpr_bf16atan2(f, g)     KPR_BF16FALL2(atan2f,    f, g)
#define kpr_bf16fmin(f, g)      KPR_BF16FALL2(fminf,     f, g)
#define kpr_bf16fmax(f, g)      KPR_BF16FALL2(fmaxf,     f, g)
#define kpr_bf16fmod(f, g)      KPR_BF16FALL2(fmodf,     f, g)
#define kpr_bf16copysign(f, g)  KPR_BF16FALL2(copysignf, f, g)

#endif /* KUIPER_MATH_H */
