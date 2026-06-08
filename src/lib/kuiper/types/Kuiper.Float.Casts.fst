module Kuiper.Float.Casts

#lang-pulse
open Kuiper
open Kuiper.Approximates.Base

module F16 = Kuiper.Float16
module F32 = Kuiper.Float32
module F64 = Kuiper.Float64

assume val cast_f16_to_f32 : F16.t -> F32.t
assume val cast_f16_to_f32_ok :
  x:F16.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f16_to_f32 x %~ y)
                             [SMTPat (cast_f16_to_f32 x %~ y)]

assume val cast_f16_to_f64 : F16.t -> F64.t
assume val cast_f16_to_f64_ok :
  x:F16.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f16_to_f64 x %~ y)
                             [SMTPat (cast_f16_to_f64 x %~ y)]

assume val cast_f32_to_f16 : F32.t -> F16.t
assume val cast_f32_to_f16_ok :
  x:F32.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f32_to_f16 x %~ y)
                             [SMTPat (cast_f32_to_f16 x %~ y)]

assume val cast_f32_to_f64 : F32.t -> F64.t
assume val cast_f32_to_f64_ok :
  x:F32.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f32_to_f64 x %~ y)
                             [SMTPat (cast_f32_to_f64 x %~ y)]

assume val cast_f64_to_f16 : F64.t -> F16.t
assume val cast_f64_to_f16_ok :
  x:F64.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f64_to_f16 x %~ y)
                             [SMTPat (cast_f64_to_f16 x %~ y)]

assume val cast_f64_to_f32 : F64.t -> F32.t
assume val cast_f64_to_f32_ok :
  x:F64.t -> y:real -> Lemma (requires x %~ y) (ensures cast_f64_to_f32 x %~ y)
                             [SMTPat (cast_f64_to_f32 x %~ y)]

inline_for_extraction noextract
instance c_16_16 : float_cast F16.t F16.t = { cast = id }
inline_for_extraction noextract
instance c_16_32 : float_cast F16.t F32.t = { cast = cast_f16_to_f32 }
inline_for_extraction noextract
instance c_16_64 : float_cast F16.t F64.t = { cast = cast_f16_to_f64 }
inline_for_extraction noextract
instance c_32_16 : float_cast F32.t F16.t = { cast = cast_f32_to_f16 }
inline_for_extraction noextract
instance c_32_32 : float_cast F32.t F32.t = { cast = id }
inline_for_extraction noextract
instance c_32_64 : float_cast F32.t F64.t = { cast = cast_f32_to_f64 }
inline_for_extraction noextract
instance c_64_16 : float_cast F64.t F16.t = { cast = cast_f64_to_f16 }
inline_for_extraction noextract
instance c_64_32 : float_cast F64.t F32.t = { cast = cast_f64_to_f32 }
inline_for_extraction noextract
instance c_64_64 : float_cast F64.t F64.t = { cast = id }
