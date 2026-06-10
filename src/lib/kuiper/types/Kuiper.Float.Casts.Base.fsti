module Kuiper.Float.Casts.Base

(* These are all assumptions and are extracted primitively. *)

#lang-pulse
open Kuiper
open Kuiper.Approximates.Base

module F16 = Kuiper.Float16
module F32 = Kuiper.Float32
module F64 = Kuiper.Float64

val cast_f16_to_f32 : F16.t -> F32.t
val cast_f16_to_f32_ok :
  x:F16.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f16_to_f32 x %~ y)
                             [SMTPat (cast_f16_to_f32 x %~ y)]

val cast_f16_to_f64 : F16.t -> F64.t
val cast_f16_to_f64_ok :
  x:F16.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f16_to_f64 x %~ y)
                             [SMTPat (cast_f16_to_f64 x %~ y)]

val cast_f32_to_f16 : F32.t -> F16.t
val cast_f32_to_f16_ok :
  x:F32.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f32_to_f16 x %~ y)
                             [SMTPat (cast_f32_to_f16 x %~ y)]

val cast_f32_to_f64 : F32.t -> F64.t
val cast_f32_to_f64_ok :
  x:F32.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f32_to_f64 x %~ y)
                             [SMTPat (cast_f32_to_f64 x %~ y)]

val cast_f64_to_f16 : F64.t -> F16.t
val cast_f64_to_f16_ok :
  x:F64.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f64_to_f16 x %~ y)
                             [SMTPat (cast_f64_to_f16 x %~ y)]

val cast_f64_to_f32 : F64.t -> F32.t
val cast_f64_to_f32_ok :
  x:F64.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f64_to_f32 x %~ y)
                             [SMTPat (cast_f64_to_f32 x %~ y)]
