module Kuiper.Example.MathPrimitives

(* Smoke test: call every math primitive on every floating type so
   extraction + CUDA compilation are exercised. *)

#lang-pulse

open Kuiper

module F16 = Kuiper.Float16
module F32 = Kuiper.Float32
module F64 = Kuiper.Float64

(* ---- Float32 unary ---- *)

fn test_sqrt_f32 (x : f32) returns f32 { F32.sqrt x }
fn test_rsqrt_f32 (x : f32) returns f32 { F32.rsqrt x }
fn test_sin_f32 (x : f32) returns f32 { F32.sin x }
fn test_cos_f32 (x : f32) returns f32 { F32.cos x }
fn test_tan_f32 (x : f32) returns f32 { F32.tan x }
fn test_asin_f32 (x : f32) returns f32 { F32.asin x }
fn test_acos_f32 (x : f32) returns f32 { F32.acos x }
fn test_atan_f32 (x : f32) returns f32 { F32.atan x }
fn test_sinh_f32 (x : f32) returns f32 { F32.sinh x }
fn test_cosh_f32 (x : f32) returns f32 { F32.cosh x }
fn test_tanh_f32 (x : f32) returns f32 { F32.tanh x }
fn test_ceil_f32 (x : f32) returns f32 { F32.ceil x }
fn test_floor_f32 (x : f32) returns f32 { F32.floor x }
fn test_round_f32 (x : f32) returns f32 { F32.round x }
fn test_fabs_f32 (x : f32) returns f32 { F32.fabs x }
fn test_erf_f32 (x : f32) returns f32 { F32.erf x }
fn test_log2_f32 (x : f32) returns f32 { F32.log2 x }
fn test_log10_f32 (x : f32) returns f32 { F32.log10 x }
fn test_exp2_f32 (x : f32) returns f32 { F32.exp2 x }

(* ---- Float32 binary ---- *)

fn test_pow_f32 (x y : f32) returns f32 { F32.pow x y }
fn test_atan2_f32 (x y : f32) returns f32 { F32.atan2 x y }
fn test_fmin_f32 (x y : f32) returns f32 { F32.fmin x y }
fn test_fmax_f32 (x y : f32) returns f32 { F32.fmax x y }
fn test_fmod_f32 (x y : f32) returns f32 { F32.fmod x y }
fn test_copysign_f32 (x y : f32) returns f32 { F32.copysign x y }

(* ---- Float32 ternary ---- *)

fn test_fma_f32 (x y z : f32) returns f32 { F32.fma x y z }

(* ---- Float64 unary ---- *)

fn test_sqrt_f64 (x : f64) returns f64 { F64.sqrt x }
fn test_rsqrt_f64 (x : f64) returns f64 { F64.rsqrt x }
fn test_sin_f64 (x : f64) returns f64 { F64.sin x }
fn test_cos_f64 (x : f64) returns f64 { F64.cos x }
fn test_tan_f64 (x : f64) returns f64 { F64.tan x }
fn test_asin_f64 (x : f64) returns f64 { F64.asin x }
fn test_acos_f64 (x : f64) returns f64 { F64.acos x }
fn test_atan_f64 (x : f64) returns f64 { F64.atan x }
fn test_sinh_f64 (x : f64) returns f64 { F64.sinh x }
fn test_cosh_f64 (x : f64) returns f64 { F64.cosh x }
fn test_tanh_f64 (x : f64) returns f64 { F64.tanh x }
fn test_ceil_f64 (x : f64) returns f64 { F64.ceil x }
fn test_floor_f64 (x : f64) returns f64 { F64.floor x }
fn test_round_f64 (x : f64) returns f64 { F64.round x }
fn test_fabs_f64 (x : f64) returns f64 { F64.fabs x }
fn test_erf_f64 (x : f64) returns f64 { F64.erf x }
fn test_log2_f64 (x : f64) returns f64 { F64.log2 x }
fn test_log10_f64 (x : f64) returns f64 { F64.log10 x }
fn test_exp2_f64 (x : f64) returns f64 { F64.exp2 x }

(* ---- Float64 binary ---- *)

fn test_pow_f64 (x y : f64) returns f64 { F64.pow x y }
fn test_atan2_f64 (x y : f64) returns f64 { F64.atan2 x y }
fn test_fmin_f64 (x y : f64) returns f64 { F64.fmin x y }
fn test_fmax_f64 (x y : f64) returns f64 { F64.fmax x y }
fn test_fmod_f64 (x y : f64) returns f64 { F64.fmod x y }
fn test_copysign_f64 (x y : f64) returns f64 { F64.copysign x y }

(* ---- Float64 ternary ---- *)

fn test_fma_f64 (x y z : f64) returns f64 { F64.fma x y z }

(* ---- Float16 unary ---- *)

fn test_sqrt_f16 (x : f16) returns f16 { F16.sqrt x }
fn test_rsqrt_f16 (x : f16) returns f16 { F16.rsqrt x }
fn test_sin_f16 (x : f16) returns f16 { F16.sin x }
fn test_cos_f16 (x : f16) returns f16 { F16.cos x }
fn test_tan_f16 (x : f16) returns f16 { F16.tan x }
fn test_asin_f16 (x : f16) returns f16 { F16.asin x }
fn test_acos_f16 (x : f16) returns f16 { F16.acos x }
fn test_atan_f16 (x : f16) returns f16 { F16.atan x }
fn test_sinh_f16 (x : f16) returns f16 { F16.sinh x }
fn test_cosh_f16 (x : f16) returns f16 { F16.cosh x }
fn test_tanh_f16 (x : f16) returns f16 { F16.tanh x }
fn test_ceil_f16 (x : f16) returns f16 { F16.ceil x }
fn test_floor_f16 (x : f16) returns f16 { F16.floor x }
fn test_round_f16 (x : f16) returns f16 { F16.round x }
fn test_fabs_f16 (x : f16) returns f16 { F16.fabs x }
fn test_erf_f16 (x : f16) returns f16 { F16.erf x }
fn test_log2_f16 (x : f16) returns f16 { F16.log2 x }
fn test_log10_f16 (x : f16) returns f16 { F16.log10 x }
fn test_exp2_f16 (x : f16) returns f16 { F16.exp2 x }

(* ---- Float16 binary ---- *)

fn test_pow_f16 (x y : f16) returns f16 { F16.pow x y }
fn test_atan2_f16 (x y : f16) returns f16 { F16.atan2 x y }
fn test_fmin_f16 (x y : f16) returns f16 { F16.fmin x y }
fn test_fmax_f16 (x y : f16) returns f16 { F16.fmax x y }
fn test_fmod_f16 (x y : f16) returns f16 { F16.fmod x y }
fn test_copysign_f16 (x y : f16) returns f16 { F16.copysign x y }

(* ---- Float16 ternary ---- *)

fn test_fma_f16 (x y z : f16) returns f16 { F16.fma x y z }

(* ---- valid / min_val / max_val ---- *)

fn test_valid_f32  (x : f32)  returns bool { F32.valid x }
fn test_min_val_f32 (_:unit)  returns f32  { F32.min_val }
fn test_max_val_f32 (_:unit)  returns f32  { F32.max_val }

fn test_valid_f64  (x : f64)  returns bool { F64.valid x }
fn test_min_val_f64 (_:unit)  returns f64  { F64.min_val }
fn test_max_val_f64 (_:unit)  returns f64  { F64.max_val }

fn test_valid_f16  (x : f16)  returns bool { F16.valid x }
fn test_min_val_f16 (_:unit)  returns f16  { F16.min_val }
fn test_max_val_f16 (_:unit)  returns f16  { F16.max_val }
