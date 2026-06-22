module Kuiper.Example.FloatCasts

(* Smoke test: call every float_cast variant so extraction + CUDA
   compilation are exercised. *)

#lang-pulse

open Kuiper
open Kuiper.Float.Casts

let test_cast_f16_f16 (x : f16) : f16 = fcast x
let test_cast_f16_f32 (x : f16) : f32 = fcast x
let test_cast_f16_f64 (x : f16) : f64 = fcast x

let test_cast_f32_f16 (x : f32) : f16 = fcast x
let test_cast_f32_f32 (x : f32) : f32 = fcast x
let test_cast_f32_f64 (x : f32) : f64 = fcast x

let test_cast_f64_f16 (x : f64) : f16 = fcast x
let test_cast_f64_f32 (x : f64) : f32 = fcast x
let test_cast_f64_f64 (x : f64) : f64 = fcast x

let test_cast_bf16_bf16 (x : bf16) : bf16 = fcast x
let test_cast_bf16_f32  (x : bf16) : f32  = fcast x
let test_cast_f32_bf16  (x : f32)  : bf16 = fcast x

let test_cast_f16_bf16  (x : f16)  : bf16 = fcast x
let test_cast_bf16_f16  (x : bf16) : f16  = fcast x
let test_cast_bf16_f64  (x : bf16) : f64  = fcast x
let test_cast_f64_bf16  (x : f64)  : bf16 = fcast x
