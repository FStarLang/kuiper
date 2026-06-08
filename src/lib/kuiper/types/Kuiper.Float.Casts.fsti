module Kuiper.Float.Casts

#lang-pulse
open Kuiper
module F16 = Kuiper.Float16
module F32 = Kuiper.Float32
module F64 = Kuiper.Float64
open Kuiper.Approximates.Base

inline_for_extraction noextract
class float_cast
  (a b : Type) {| scalar a, scalar b, real_like a, real_like b |}
= {
  cast : a -> b;

  #[Tactics.Easy.easy_fill()]
  cast_approx :
    x:a -> y:real ->
    Lemma (requires x %~ y)
          (ensures cast x %~ y)
          [SMTPat (cast x %~ y)];
}

inline_for_extraction noextract
instance val c_16_16 : float_cast F16.t F16.t
inline_for_extraction noextract
instance val c_16_32 : float_cast F16.t F32.t
inline_for_extraction noextract
instance val c_16_64 : float_cast F16.t F64.t
inline_for_extraction noextract
instance val c_32_16 : float_cast F32.t F16.t
inline_for_extraction noextract
instance val c_32_32 : float_cast F32.t F32.t
inline_for_extraction noextract
instance val c_32_64 : float_cast F32.t F64.t
inline_for_extraction noextract
instance val c_64_16 : float_cast F64.t F16.t
inline_for_extraction noextract
instance val c_64_32 : float_cast F64.t F32.t
inline_for_extraction noextract
instance val c_64_64 : float_cast F64.t F64.t
