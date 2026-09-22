module Kuiper.Float32

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open FStar.Tactics.Typeclasses { solve }
open Kuiper.Sized
open Kuiper.Canonical
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Real
module Pow = FStar.Math.Pow

open Kuiper.Float32.Base

let t = Float32.Base.t

inline_for_extraction noextract
instance is_sized : sized t = { size = 4sz; default = zero }

assume CanonicalSized : canonical is_sized

inline_for_extraction noextract
instance _ : scalar t = {
  is_sized = solve;
  add; mul; zero; one; lt; lte; eq;
}

inline_for_extraction noextract
instance is_floating : floating t = {
  is_scalar = solve;
  sub; div; bit_eq;
  of_int; of_literal; of_int_zero; of_int_one;
  kind; is_zero;
  largest; infinity;
  kind_one; kind_zero; kind_largest; kind_infinity;
  zero_is_zero; one_is_nonzero;
  fexp; flog; sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
  sinh; cosh; tanh; ceil; floor; round; fabs; erf; log2;
  log10; exp2; pow; atan2; fmin; fmax; fmod; copysign;
  fma;
}

(* Approximation semantics is assumed. *)
instance is_real_like          : real_like t = magic()
instance is_floating_real_like : floating_real_like t = magic()

inline_for_extraction noextract
let fexpm1 = Kuiper.Float32.Base.fexpm1

inline_for_extraction noextract
let flog1p = Kuiper.Float32.Base.flog1p

(* Approximation semantics for these CUDA math operations is assumed. *)
let expm1_approx
  (x : t)
  (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (fexpm1 x) (exp r -. 1.0R))
= admit()

let log1p_approx
  (x : t)
  (r : real { r >. 0.0R -. 1.0R })
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (flog1p x) (log (1.0R +. r)))
= admit()

(* Extracted primitively; the approximation contracts are trusted. *)
noextract
fn mul_rn_ftz (x y : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr : real).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (xr *. yr))
{
  admit()
}

noextract
fn exp2_approx_ftz (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real).
      v_approximates x xr ==>
      v_approximates result (Pow.exp2 xr))
{
  admit()
}

let lem_sizeof () = ()
