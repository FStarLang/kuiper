module Kuiper.Example.MathPrimitives

(* Smoke test: call every math primitive on every floating type so
   extraction + CUDA compilation are exercised. *)

#lang-pulse

open Kuiper

let test_sqrt_f16     (x : f16)     : f16 = sqrt x
let test_rsqrt_f16    (x : f16)     : f16 = rsqrt x
let test_sin_f16      (x : f16)     : f16 = sin x
let test_cos_f16      (x : f16)     : f16 = cos x
let test_tan_f16      (x : f16)     : f16 = tan x
let test_asin_f16     (x : f16)     : f16 = asin x
let test_acos_f16     (x : f16)     : f16 = acos x
let test_atan_f16     (x : f16)     : f16 = atan x
let test_sinh_f16     (x : f16)     : f16 = sinh x
let test_cosh_f16     (x : f16)     : f16 = cosh x
let test_tanh_f16     (x : f16)     : f16 = tanh x
let test_ceil_f16     (x : f16)     : f16 = ceil x
let test_floor_f16    (x : f16)     : f16 = floor x
let test_round_f16    (x : f16)     : f16 = round x
let test_fabs_f16     (x : f16)     : f16 = fabs x
let test_erf_f16      (x : f16)     : f16 = erf x
let test_log2_f16     (x : f16)     : f16 = log2 x
let test_log10_f16    (x : f16)     : f16 = log10 x
let test_exp2_f16     (x : f16)     : f16 = exp2 x
let test_pow_f16      (x y : f16)   : f16 = pow x y
let test_atan2_f16    (x y : f16)   : f16 = atan2 x y
let test_fmin_f16     (x y : f16)   : f16 = fmin x y
let test_fmax_f16     (x y : f16)   : f16 = fmax x y
let test_fmod_f16     (x y : f16)   : f16 = fmod x y
let test_copysign_f16 (x y : f16)   : f16 = copysign x y
let test_fma_f16      (x y z : f16) : f16 = fma x y z
let test_largest_f16  ()            : f16 = largest
let test_infinity_f16 ()            : f16 = infinity

let test_sqrt_f32     (x : f32)     : f32 = sqrt x
let test_rsqrt_f32    (x : f32)     : f32 = rsqrt x
let test_sin_f32      (x : f32)     : f32 = sin x
let test_cos_f32      (x : f32)     : f32 = cos x
let test_tan_f32      (x : f32)     : f32 = tan x
let test_asin_f32     (x : f32)     : f32 = asin x
let test_acos_f32     (x : f32)     : f32 = acos x
let test_atan_f32     (x : f32)     : f32 = atan x
let test_sinh_f32     (x : f32)     : f32 = sinh x
let test_cosh_f32     (x : f32)     : f32 = cosh x
let test_tanh_f32     (x : f32)     : f32 = tanh x
let test_ceil_f32     (x : f32)     : f32 = ceil x
let test_floor_f32    (x : f32)     : f32 = floor x
let test_round_f32    (x : f32)     : f32 = round x
let test_fabs_f32     (x : f32)     : f32 = fabs x
let test_erf_f32      (x : f32)     : f32 = erf x
let test_log2_f32     (x : f32)     : f32 = log2 x
let test_log10_f32    (x : f32)     : f32 = log10 x
let test_exp2_f32     (x : f32)     : f32 = exp2 x
let test_pow_f32      (x y : f32)   : f32 = pow x y
let test_atan2_f32    (x y : f32)   : f32 = atan2 x y
let test_fmin_f32     (x y : f32)   : f32 = fmin x y
let test_fmax_f32     (x y : f32)   : f32 = fmax x y
let test_fmod_f32     (x y : f32)   : f32 = fmod x y
let test_copysign_f32 (x y : f32)   : f32 = copysign x y
let test_fma_f32      (x y z : f32) : f32 = fma x y z
let test_largest_f32  ()            : f32 = largest
let test_infinity_f32 ()            : f32 = infinity

let test_sqrt_f64     (x : f64)     : f64 = sqrt x
let test_rsqrt_f64    (x : f64)     : f64 = rsqrt x
let test_sin_f64      (x : f64)     : f64 = sin x
let test_cos_f64      (x : f64)     : f64 = cos x
let test_tan_f64      (x : f64)     : f64 = tan x
let test_asin_f64     (x : f64)     : f64 = asin x
let test_acos_f64     (x : f64)     : f64 = acos x
let test_atan_f64     (x : f64)     : f64 = atan x
let test_sinh_f64     (x : f64)     : f64 = sinh x
let test_cosh_f64     (x : f64)     : f64 = cosh x
let test_tanh_f64     (x : f64)     : f64 = tanh x
let test_ceil_f64     (x : f64)     : f64 = ceil x
let test_floor_f64    (x : f64)     : f64 = floor x
let test_round_f64    (x : f64)     : f64 = round x
let test_fabs_f64     (x : f64)     : f64 = fabs x
let test_erf_f64      (x : f64)     : f64 = erf x
let test_log2_f64     (x : f64)     : f64 = log2 x
let test_log10_f64    (x : f64)     : f64 = log10 x
let test_exp2_f64     (x : f64)     : f64 = exp2 x
let test_pow_f64      (x y : f64)   : f64 = pow x y
let test_atan2_f64    (x y : f64)   : f64 = atan2 x y
let test_fmin_f64     (x y : f64)   : f64 = fmin x y
let test_fmax_f64     (x y : f64)   : f64 = fmax x y
let test_fmod_f64     (x y : f64)   : f64 = fmod x y
let test_copysign_f64 (x y : f64)   : f64 = copysign x y
let test_fma_f64      (x y z : f64) : f64 = fma x y z
let test_largest_f64  ()            : f64 = largest
let test_infinity_f64 ()            : f64 = infinity
