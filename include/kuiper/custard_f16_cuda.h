#ifndef KUIPER_CUSTARD_F16_CUDA_H
#define KUIPER_CUSTARD_F16_CUDA_H 1

/* Kuiper's override of Custard's portable 16-bit floats (custard.md section
   66).  Custard emits custard_f16/custard_bf16 as opaque two-byte structs
   with the arithmetic done via float, which is correct everywhere and is the
   right default.  On a GPU it is the wrong choice twice over: the arithmetic
   is native, and -- the reason this file exists -- wmma::fragment is a
   template over __half and __nv_bfloat16, so a fragment holding a struct is
   not the type any wmma overload accepts.

   The representation is the format's own bit pattern either way, so this is
   layout-compatible with what it replaces by construction.

   This has to be included BEFORE the Custard-generated header, because the
   support block is emitted above the [custard_c_header] includes -- so a
   header reached through an F* declaration is already too late.  -include is
   currently the only way in. */

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CUSTARD_FLOAT16_DEFINED

typedef __half custard_f16;
typedef __nv_bfloat16 custard_bf16;

#if defined(__CUDACC__)
#define CUSTARD_FN static __host__ __device__ inline
#else
#define CUSTARD_FN static inline
#endif

/* A literal arrives as the format's bit pattern, which is what Custard
   computed by rounding the source literal's exact rational value once. */
#define CUSTARD_F16_LIT(b)   (__ushort_as_half((unsigned short)(b)))
#define CUSTARD_BF16_LIT(b)  (kpr_bf16_from_bits((unsigned short)(b)))
#define CUSTARD_F16_INIT(b)  (__ushort_as_half((unsigned short)(b)))
#define CUSTARD_BF16_INIT(b) (kpr_bf16_from_bits((unsigned short)(b)))

CUSTARD_FN __nv_bfloat16 kpr_bf16_from_bits(unsigned short b) {
  __nv_bfloat16 r;
  memcpy(&r, &b, sizeof r);
  return r;
}

/* On the device the formats are native: __half and __nv_bfloat16 carry
   operator overloads that compile to a single hardware instruction, and
   routing through float instead costs a cvt each way and an extra live
   register per operand.  Off the device (and on architectures with no
   native support) the same overloads are unavailable or emulated, so there
   we do the round-trip explicitly.  This is the dispatch kuiper/math.h
   already uses for kpr_bf16mul and friends; the two must agree, since a
   kernel can reach both.

   The operator overloads are used rather than __hmul/__hadd because the
   intrinsics do not compile on devices without native bf16 support, while
   the overloads degrade to the f32 round-trip by themselves. */
#ifdef __CUDA_ARCH__
#define KPR_F16_BIN(nm, op)                                                    \
  CUSTARD_FN custard_f16 custard_f16_##nm(custard_f16 a, custard_f16 b) {      \
    return a op b;                                                             \
  }                                                                            \
  CUSTARD_FN custard_bf16 custard_bf16_##nm(custard_bf16 a, custard_bf16 b) {  \
    return a op b;                                                             \
  }
#else
#define KPR_F16_BIN(nm, op)                                                    \
  CUSTARD_FN custard_f16 custard_f16_##nm(custard_f16 a, custard_f16 b) {      \
    return __float2half(__half2float(a) op __half2float(b));                   \
  }                                                                            \
  CUSTARD_FN custard_bf16 custard_bf16_##nm(custard_bf16 a, custard_bf16 b) {  \
    return __float2bfloat16(__bfloat162float(a) op __bfloat162float(b));       \
  }
#endif

KPR_F16_BIN(add, +)
KPR_F16_BIN(sub, -)
KPR_F16_BIN(mul, *)
KPR_F16_BIN(div, /)

#ifdef __CUDA_ARCH__
#define KPR_F16_CMP(nm, op)                                                    \
  CUSTARD_FN bool custard_f16_##nm(custard_f16 a, custard_f16 b) {             \
    return a op b;                                                             \
  }                                                                            \
  CUSTARD_FN bool custard_bf16_##nm(custard_bf16 a, custard_bf16 b) {          \
    return a op b;                                                             \
  }
#else
#define KPR_F16_CMP(nm, op)                                                    \
  CUSTARD_FN bool custard_f16_##nm(custard_f16 a, custard_f16 b) {             \
    return __half2float(a) op __half2float(b);                                 \
  }                                                                            \
  CUSTARD_FN bool custard_bf16_##nm(custard_bf16 a, custard_bf16 b) {          \
    return __bfloat162float(a) op __bfloat162float(b);                         \
  }
#endif

KPR_F16_CMP(eq, ==)
KPR_F16_CMP(lt, <)
KPR_F16_CMP(le, <=)
KPR_F16_CMP(gt, >)
KPR_F16_CMP(ge, >=)

CUSTARD_FN custard_f16 custard_f16_of_i64(int64_t v) {
  return __float2half((float)v);
}
CUSTARD_FN custard_f16 custard_f16_of_f32(float v) { return __float2half(v); }
CUSTARD_FN custard_f16 custard_f16_of_f64(double v) {
  return __float2half((float)v);
}
CUSTARD_FN float custard_f16_to_f32(custard_f16 v) { return __half2float(v); }

CUSTARD_FN custard_bf16 custard_bf16_of_i64(int64_t v) {
  return __float2bfloat16((float)v);
}
CUSTARD_FN custard_bf16 custard_bf16_of_f32(float v) {
  return __float2bfloat16(v);
}
CUSTARD_FN custard_bf16 custard_bf16_of_f64(double v) {
  return __float2bfloat16((float)v);
}
CUSTARD_FN float custard_bf16_to_f32(custard_bf16 v) {
  return __bfloat162float(v);
}

#endif /* KUIPER_CUSTARD_F16_CUDA_H */
