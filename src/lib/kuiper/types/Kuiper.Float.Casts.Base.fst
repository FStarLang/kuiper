module Kuiper.Float.Casts.Base

friend Kuiper.Float16.Base
friend Kuiper.BFloat16.Base
friend Kuiper.Float32.Base
friend Kuiper.Float64.Base
friend Kuiper.Float16
friend Kuiper.BFloat16
friend Kuiper.Float32
friend Kuiper.Float64

open Kuiper.Float.Model
module R = Kuiper.Float.Casts.Raw
module F16 = Kuiper.Float16
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module F64 = Kuiper.Float64

let cast_f16_to_f32 (x:F16.t) : F32.t =
  { repr = R.cast_f16_to_f32 x.repr; ideal = x.ideal }
let cast_f16_to_f32_ok x r = ()

let cast_f16_to_f64 (x:F16.t) : F64.t =
  { repr = R.cast_f16_to_f64 x.repr; ideal = x.ideal }
let cast_f16_to_f64_ok x r = ()

let cast_f32_to_f16 (x:F32.t) : F16.t =
  { repr = R.cast_f32_to_f16 x.repr; ideal = x.ideal }
let cast_f32_to_f16_ok x r = ()

let cast_f32_to_f64 (x:F32.t) : F64.t =
  { repr = R.cast_f32_to_f64 x.repr; ideal = x.ideal }
let cast_f32_to_f64_ok x r = ()

let cast_bf16_to_f32 (x:BF16.t) : F32.t =
  { repr = R.cast_bf16_to_f32 x.repr; ideal = x.ideal }
let cast_bf16_to_f32_ok x r = ()

let cast_f32_to_bf16 (x:F32.t) : BF16.t =
  { repr = R.cast_f32_to_bf16 x.repr; ideal = x.ideal }
let cast_f32_to_bf16_ok x r = ()

let cast_f16_to_bf16 (x:F16.t) : BF16.t =
  { repr = R.cast_f16_to_bf16 x.repr; ideal = x.ideal }
let cast_f16_to_bf16_ok x r = ()

let cast_bf16_to_f16 (x:BF16.t) : F16.t =
  { repr = R.cast_bf16_to_f16 x.repr; ideal = x.ideal }
let cast_bf16_to_f16_ok x r = ()

let cast_bf16_to_f64 (x:BF16.t) : F64.t =
  { repr = R.cast_bf16_to_f64 x.repr; ideal = x.ideal }
let cast_bf16_to_f64_ok x r = ()

let cast_f64_to_bf16 (x:F64.t) : BF16.t =
  { repr = R.cast_f64_to_bf16 x.repr; ideal = x.ideal }
let cast_f64_to_bf16_ok x r = ()

let cast_f64_to_f16 (x:F64.t) : F16.t =
  { repr = R.cast_f64_to_f16 x.repr; ideal = x.ideal }
let cast_f64_to_f16_ok x r = ()

let cast_f64_to_f32 (x:F64.t) : F32.t =
  { repr = R.cast_f64_to_f32 x.repr; ideal = x.ideal }
let cast_f64_to_f32_ok x r = ()
