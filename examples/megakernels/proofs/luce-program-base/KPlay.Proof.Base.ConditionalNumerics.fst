module KPlay.Proof.Base.ConditionalNumerics

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Exp = FStar.Math.Exp

inline_for_extraction noextract
let widen (scope:prop) (value:BF.t) (logical:real)
  : Lemma
      (requires scope ==> Approx.v_approximates value logical)
      (ensures scope ==> Approx.v_approximates (Cast.cast_bf16_to_f32 value) logical) =
  let under (_:scope)
    : Lemma (Approx.v_approximates (Cast.cast_bf16_to_f32 value) logical) =
    Cast.cast_bf16_to_f32_ok value logical
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let multiply (scope:prop) (left right:FP.t) (x y:real)
  : Lemma
      (requires scope ==> Approx.v_approximates left x /\ Approx.v_approximates right y)
      (ensures scope ==> Approx.v_approximates (F.mul left right) (x *. y)) =
  let under (_:scope)
    : Lemma (Approx.v_approximates (F.mul left right) (x *. y)) =
    Approx.a_mul left right x y
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let exponential (scope:prop) (value:FP.t) (logical:real)
  : Lemma
      (requires scope ==> Approx.v_approximates value logical)
      (ensures scope ==> Approx.v_approximates (F.fexp value) (Exp.exp logical)) =
  let under (_:scope)
    : Lemma (Approx.v_approximates (F.fexp value) (Exp.exp logical)) =
    Approx.exp_approx value logical
  in
  FStar.Classical.impl_intro under
