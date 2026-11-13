module Kuiper.Float.Casts.Base

(* These are all assumptions and are extracted primitively. *)

#lang-pulse
open Kuiper
open Kuiper.Approximates.Base

module F16 = Kuiper.Float16
module BF16 = Kuiper.BFloat16
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

val cast_bf16_to_f32 : BF16.t -> F32.t
val cast_bf16_to_f32_ok :
  x:BF16.t -> y:real -> Lemma (requires x %~ y) (ensures cast_bf16_to_f32 x %~ y)
                             [SMTPat (cast_bf16_to_f32 x %~ y)]

(* Exact native widening; the small integer endpoint is representable in
   both source and destination. This is independent of approximation. *)
val cast_bf16_to_f32_magnitude
  (bound:Kuiper.Floating.Bounds.bound{0 < bound /\ bound <= 128}) (x:BF16.t)
  : Lemma
      (requires Kuiper.Floating.Bounds.magnitude bound x)
      (ensures Kuiper.Floating.Bounds.magnitude bound (cast_bf16_to_f32 x))

val cast_f32_to_bf16 : F32.t -> BF16.t
val cast_f32_to_bf16_ok :
  x:F32.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f32_to_bf16 x %~ y)
                             [SMTPat (cast_f32_to_bf16 x %~ y)]

val cast_f16_to_bf16 : F16.t -> BF16.t
val cast_f16_to_bf16_ok :
  x:F16.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f16_to_bf16 x %~ y)
                             [SMTPat (cast_f16_to_bf16 x %~ y)]

val cast_bf16_to_f16 : BF16.t -> F16.t
val cast_bf16_to_f16_ok :
  x:BF16.t -> y:real -> Lemma (requires x %~ y) (ensures cast_bf16_to_f16 x %~ y)
                             [SMTPat (cast_bf16_to_f16 x %~ y)]

val cast_bf16_to_f64 : BF16.t -> F64.t
val cast_bf16_to_f64_ok :
  x:BF16.t -> y:real -> Lemma (requires x %~ y) (ensures cast_bf16_to_f64 x %~ y)
                             [SMTPat (cast_bf16_to_f64 x %~ y)]

val cast_f64_to_bf16 : F64.t -> BF16.t
val cast_f64_to_bf16_ok :
  x:F64.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f64_to_bf16 x %~ y)
                             [SMTPat (cast_f64_to_bf16 x %~ y)]

val cast_f64_to_f16 : F64.t -> F16.t
val cast_f64_to_f16_ok :
  x:F64.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f64_to_f16 x %~ y)
                             [SMTPat (cast_f64_to_f16 x %~ y)]

val cast_f64_to_f32 : F64.t -> F32.t
val cast_f64_to_f32_ok :
  x:F64.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f64_to_f32 x %~ y)
                             [SMTPat (cast_f64_to_f32 x %~ y)]
